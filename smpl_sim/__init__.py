from . import envs as domains

__version__ = "0.0.1"

_MAX_EPISODE_LENGTH = 500
# Horizontal speeds above which move reward is 1.
_WALK_SPEED = 1
_RUN_SPEED = 10

def make(cfg, motions):
    task = cfg["task"]
    max_episode_length = cfg.get("max_episode_length", _MAX_EPISODE_LENGTH)
    seed = cfg.get("seed", 0)
    data_dir = cfg["data_dir"]
    initial_position = cfg.get("initial_position", "hybrid")
    enablePID = cfg.get("enablePID", False)

    if task == "smpl_stand":
        env = domains.SMPLHumanoidMove(
                motions=motions,
                pid_controlled=enablePID,
                data_dir=data_dir,
                max_episode_length=max_episode_length,
                move_speed=0,
                initial_position=initial_position,
                seed=seed)
    elif task == "smpl_walk":
        env = domains.SMPLHumanoidMove(
                motions=motions,
                pid_controlled=enablePID,
                data_dir=data_dir,
                max_episode_length=max_episode_length,
                move_speed=_WALK_SPEED,
                initial_position=initial_position,
                seed=seed)
    elif task == "smpl_run":
        env = domains.SMPLHumanoidMove(
                motions=motions,
                pid_controlled=enablePID,
                data_dir=data_dir,
                max_episode_length=max_episode_length,
                move_speed=_RUN_SPEED,
                initial_position=initial_position,
                seed=seed)
    else:
        raise ValueError(f"Unknown task {task}")

    return env

def make_dmc(cfg, motions):
    from .envs import wrappers
    env = make(cfg, motions)
    return wrappers.DMCSuiteWrapper(env)
