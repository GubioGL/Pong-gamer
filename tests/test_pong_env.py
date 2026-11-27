import pytest

from games.pong.env import PongEnv, follow_ball_policy


def test_reset_observation_shape_and_range():
    env = PongEnv(seed=0)
    obs = env.reset()
    assert len(obs) == PongEnv.OBS_SIZE
    assert all(-1.0 <= v <= 1.0 for v in obs)


def test_actions_move_player_and_walls_clamp():
    env = PongEnv(seed=0)
    y0 = env.player_y
    env.step(PongEnv.UP)
    assert env.player_y < y0
    for _ in range(200):
        env.step(PongEnv.UP)
    assert env.player_y == env.cfg.wall
    for _ in range(200):
        env.step(PongEnv.DOWN)
    assert env.player_y == env.cfg.height - env.cfg.wall - env.cfg.paddle_h


def test_ball_bounces_on_top_wall():
    env = PongEnv(seed=0)
    env.ball_y, env.ball_vy, env.ball_vx = env.cfg.wall + 1, -5.0, 0.5
    env.step(PongEnv.STAY)
    assert env.ball_vy > 0


def test_player_hit_reverses_ball_and_rewards():
    env = PongEnv(seed=0)
    c = env.cfg
    env.player_y = (c.height - c.paddle_h) / 2
    env.ball_x = env.player_x - c.ball_size - 1
    env.ball_y = (c.height - c.ball_size) / 2
    env.ball_vx, env.ball_vy = 7.0, 0.0
    _, reward, _, info = env.step(PongEnv.STAY)
    assert info["event"] == "player_hit"
    assert env.ball_vx < 0
    assert reward == pytest.approx(c.hit_reward)


def test_missed_ball_gives_point_to_cpu():
    env = PongEnv(seed=0)
    c = env.cfg
    env.player_y = c.wall                          # paddle far away at the top
    env.ball_x, env.ball_y = c.width - 3, c.height - 100
    env.ball_vx, env.ball_vy = 7.0, 0.0
    _, reward, _, info = env.step(PongEnv.STAY)
    assert info["score_cpu"] == 1 and reward == -1.0
    assert env.ball_x == pytest.approx((c.width - c.ball_size) / 2)   # served again


def test_same_seed_same_game():
    def play(seed):
        env = PongEnv(seed=seed)
        out = []
        for t in range(500):
            obs, r, done, _ = env.step(t % 3)
            out.append((tuple(obs), r))
            if done:
                break
        return out
    assert play(42) == play(42)


def test_episode_ends_and_follow_policy_beats_idle():
    def final(policy, seed):
        env = PongEnv(seed=seed, points_to_win=3)
        obs, done, total = env.reset(), False, 0.0
        while not done:
            obs, r, done, _ = env.step(policy(obs))
            total += r
        return env, total
    env, _ = final(lambda o: PongEnv.STAY, 1)
    assert env.done and env.winner == "cpu"
    env, _ = final(follow_ball_policy, 1)
    assert env.done and env.score_player > 0


def test_invalid_action_and_step_after_done():
    env = PongEnv(seed=0, max_steps=1)
    with pytest.raises(ValueError):
        env.step(5)
    env.step(0)
    with pytest.raises(RuntimeError):
        env.step(0)


def test_headless_window_runs():
    import os
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    from games.pong.play import main
    main(["--frames", "60", "--agente", "seguidor", "--seed", "0"])
