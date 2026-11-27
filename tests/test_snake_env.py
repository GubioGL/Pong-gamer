import pytest

from games.snake.env import SnakeEnv, greedy_policy


def test_reset_observation():
    env = SnakeEnv(seed=0)
    obs = env.reset()
    assert len(obs) == SnakeEnv.OBS_SIZE
    assert all(-1.0 <= v <= 1.0 for v in obs)
    assert env.length == env.cfg.start_length
    assert obs[6:] == [1.0, 0.0]                 # starts moving right


def test_moves_and_reverse_is_ignored():
    env = SnakeEnv(seed=0)
    hx, hy = env.head
    env.step(SnakeEnv.LEFT)                      # opposite of RIGHT: ignored
    assert env.head == (hx + 1, hy)
    env.step(SnakeEnv.UP)
    assert env.head == (hx + 1, hy - 1)


def test_eating_grows_and_rewards():
    env = SnakeEnv(seed=0)
    hx, hy = env.head
    env.food = (hx + 1, hy)
    _, r, done, info = env.step(SnakeEnv.RIGHT)
    assert r == 1.0 and info["event"] == "eat" and not done
    assert env.length == env.cfg.start_length + 1
    assert env.food not in env.cells


def test_wall_kills():
    env = SnakeEnv(seed=0)
    done = False
    while not done:
        _, r, done, info = env.step(SnakeEnv.UP)
    assert r == -1.0 and info["event"] == "wall"


def test_biting_itself_kills():
    env = SnakeEnv(seed=0, start_length=5)
    env.food = (0, 0)                                # keep food out of the way
    for a in (SnakeEnv.UP, SnakeEnv.LEFT, SnakeEnv.DOWN):
        _, r, done, info = env.step(a)
    assert done and info["event"] == "self" and r == -1.0


def test_following_own_tail_is_allowed():
    env = SnakeEnv(seed=0, start_length=4)           # 2x2 loop: head enters the cell the tail leaves
    env.food = (0, 0)
    for a in (SnakeEnv.UP, SnakeEnv.LEFT, SnakeEnv.DOWN, SnakeEnv.RIGHT, SnakeEnv.UP):
        _, _, done, _ = env.step(a)
        assert not done


def test_danger_flags():
    env = SnakeEnv(seed=0)
    env.body.clear(); env.cells.clear()
    for cell in [(0, 0), (1, 0)]:
        env.body.append(cell); env.cells.add(cell)
    env.direction = SnakeEnv.LEFT
    obs = env.observation()
    assert obs[2] == 1.0 and obs[4] == 1.0           # wall above and to the left
    assert obs[3] == -1.0                            # free below


def test_idle_truncation_and_determinism():
    env = SnakeEnv(seed=1, max_idle=10)
    env.food = (0, env.cfg.grid_h - 1)
    acts = [SnakeEnv.UP, SnakeEnv.RIGHT, SnakeEnv.DOWN, SnakeEnv.LEFT] * 5
    for a in acts:
        _, _, done, info = env.step(a)
        if done:
            break
    assert done and info["truncated"]

    def play(seed):
        e, out = SnakeEnv(seed=seed), []
        o, d = e.reset(), False
        while not d:
            o, r, d, _ = e.step(greedy_policy(o))
            out.append(r)
        return out
    assert play(7) == play(7)


def test_greedy_beats_random():
    import random
    rnd = random.Random(0)
    def score(policy, seed):
        e = SnakeEnv(seed=seed); o, d = e.reset(), False
        while not d:
            o, _, d, _ = e.step(policy(o))
        return e.score
    g = sum(score(greedy_policy, s) for s in range(10))
    r = sum(score(lambda o: rnd.randrange(4), s) for s in range(10))
    assert g > r and g >= 30


def test_invalid_action():
    with pytest.raises(ValueError):
        SnakeEnv(seed=0).step(9)


def test_headless_window_runs():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from games.snake.play import main
    main(["--frames", "60", "--agente", "guloso", "--seed", "0", "--grade", "12x12"])
