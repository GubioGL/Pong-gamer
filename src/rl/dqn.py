"""Deep Q-learning (JAX + optax) shared by every agent: MLP, qubit VQC and continuous-variable QNN.

The comparison is fair only if the three models are trained by exactly this code: same replay
buffer, target network, epsilon schedule, loss and number of environment steps.

    from rl.dqn import DQNConfig, train
    params, history = train(lambda s: PongEnv(seed=s, points_to_win=1), MLPQNet(6, 3),
                            DQNConfig(total_steps=20_000), seed=0, log_path="data/.../seed_0.csv")

A model is an object with
    init(key) -> params (a pytree)
    apply(params, obs) -> Q-values, shape (n_actions,), for ONE observation (vmapped here)
    labels(params) -> same pytree with "base" or "out" per leaf (optional; "out" leaves get
                      cfg.lr_out, as the output scale in Skolik et al. 2022)
    config -> dict saved next to the results
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass
class DQNConfig:
    total_steps: int = 50_000        # environment steps
    gamma: float = 0.99
    lr: float = 1e-3
    lr_out: float | None = None      # learning rate of "out" leaves (None = lr)
    batch_size: int = 32
    buffer_size: int = 20_000
    warmup: int = 1_000              # random steps before learning starts
    train_every: int = 1             # gradient step every n environment steps
    target_update: int = 500         # copy online -> target every n gradient steps
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 20_000    # linear decay
    double_dqn: bool = True
    grad_clip: float = 10.0
    eval_every: int = 0              # environment steps between greedy evaluations (0 = off)
    eval_episodes: int = 5


class ReplayBuffer:
    def __init__(self, size, n_obs, rng):
        self.obs = np.zeros((size, n_obs), np.float32)
        self.next_obs = np.zeros((size, n_obs), np.float32)
        self.act = np.zeros(size, np.int32)
        self.rew = np.zeros(size, np.float32)
        self.done = np.zeros(size, np.float32)
        self.size, self.ptr, self.full, self.rng = size, 0, False, rng

    def add(self, o, a, r, o2, d):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i], self.next_obs[i], self.done[i] = o, a, r, o2, d
        self.ptr = (i + 1) % self.size
        self.full = self.full or self.ptr == 0

    def __len__(self):
        return self.size if self.full else self.ptr

    def sample(self, n):
        idx = self.rng.integers(0, len(self), n)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]


def count_parameters(params) -> int:
    return int(sum(np.size(x) for x in jax.tree_util.tree_leaves(params)))


def epsilon_at(step, cfg: DQNConfig):
    frac = min(1.0, step / max(1, cfg.eps_decay_steps))
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def make_optimizer(model, params, cfg: DQNConfig):
    lr_out = cfg.lr if cfg.lr_out is None else cfg.lr_out
    labels = model.labels(params) if hasattr(model, "labels") else jax.tree_util.tree_map(lambda _: "base", params)
    return optax.chain(
        optax.clip_by_global_norm(cfg.grad_clip),
        optax.multi_transform({"base": optax.adam(cfg.lr), "out": optax.adam(lr_out)}, labels),
    )


def make_functions(model, optimizer, cfg: DQNConfig):
    q_batch = jax.vmap(model.apply, in_axes=(None, 0))

    @jax.jit
    def act(params, obs):
        return jnp.argmax(model.apply(params, obs))

    @jax.jit
    def update(params, target, opt_state, o, a, r, o2, d):
        def loss_fn(p):
            q = q_batch(p, o)[jnp.arange(o.shape[0]), a]
            q_next_target = q_batch(target, o2)
            if cfg.double_dqn:
                a2 = jnp.argmax(q_batch(p, o2), axis=1)
            else:
                a2 = jnp.argmax(q_next_target, axis=1)
            q2 = q_next_target[jnp.arange(o.shape[0]), a2]
            y = jax.lax.stop_gradient(r + cfg.gamma * (1.0 - d) * q2)
            return optax.huber_loss(q, y).mean()

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    return act, update


def evaluate(env_factory, model, params, episodes=5, seed=10_000, act=None):
    """Mean return of the greedy policy (epsilon = 0) on fresh environments."""
    act = act or jax.jit(lambda p, o: jnp.argmax(model.apply(p, o)))
    returns = []
    for k in range(episodes):
        env = env_factory(seed + k)
        obs, done, total = env.reset(), False, 0.0
        while not done:
            obs, r, done, _ = env.step(int(act(params, jnp.asarray(obs, jnp.float32))))
            total += r
        returns.append(total)
    return float(np.mean(returns))


def train(env_factory, model, cfg: DQNConfig, seed=0, log_path=None, verbose=False):
    """Train one agent; returns (params, history). history = list of per-episode dicts.

    env_factory(seed) -> environment with reset()/step(), N_ACTIONS and OBS_SIZE.
    """
    rng = np.random.default_rng(seed)
    env = env_factory(seed)
    params = model.init(jax.random.PRNGKey(seed))
    target = params
    optimizer = make_optimizer(model, params, cfg)
    opt_state = optimizer.init(params)
    act, update = make_functions(model, optimizer, cfg)
    buf = ReplayBuffer(cfg.buffer_size, env.OBS_SIZE, rng)

    history, obs = [], env.reset()
    ep_return, ep_steps, episode, grad_steps, last_loss, eval_score = 0.0, 0, 0, 0, float("nan"), None
    t0 = time.perf_counter()
    for step in range(1, cfg.total_steps + 1):
        eps = epsilon_at(step, cfg)
        if step <= cfg.warmup or rng.random() < eps:
            action = int(rng.integers(env.N_ACTIONS))
        else:
            action = int(act(params, jnp.asarray(obs, jnp.float32)))
        next_obs, r, done, info = env.step(action)
        terminal = done and not info.get("truncated", False)      # truncation still bootstraps
        buf.add(obs, action, r, next_obs, float(terminal))
        obs, ep_return, ep_steps = next_obs, ep_return + r, ep_steps + 1

        if step > cfg.warmup and step % cfg.train_every == 0 and len(buf) >= cfg.batch_size:
            params, opt_state, loss = update(params, target, opt_state, *buf.sample(cfg.batch_size))
            grad_steps += 1
            last_loss = float(loss)
            if grad_steps % cfg.target_update == 0:
                target = params

        if cfg.eval_every and step % cfg.eval_every == 0:
            eval_score = evaluate(env_factory, model, params, cfg.eval_episodes, act=act)

        if done:
            episode += 1
            history.append({"episode": episode, "env_steps": step, "return": round(ep_return, 4),
                            "steps": ep_steps, "epsilon": round(eps, 4), "loss": last_loss,
                            "eval_return": eval_score, "wall_time": round(time.perf_counter() - t0, 3)})
            if verbose and episode % 10 == 0:
                print(f"ep {episode:5d}  step {step:7d}  return {ep_return:7.2f}  eps {eps:.2f}")
            obs, ep_return, ep_steps = env.reset(), 0.0, 0

    if log_path is not None:
        write_history(history, log_path, cfg, model, params, seed)
    return params, history


def write_history(history, path, cfg, model, params, seed):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if history:
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
    meta = {"seed": seed, "parameters": count_parameters(params), "model": type(model).__name__,
            "model_config": getattr(model, "config", {}), "dqn": asdict(cfg)}
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
