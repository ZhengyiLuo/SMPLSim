import argparse

import sys
import pickle
import time
import joblib
import glob
import pdb
import os.path as osp
import os

os.environ["OMP_NUM_THREADS"] = "1"
sys.path.append(os.getcwd())

import torch
import numpy as np
import wandb

from smpl_sim.utils.flags import flags
from smpl_sim.agents import agent_dict
from omegaconf import DictConfig, OmegaConf

try:
    # Python < 3.9
    from importlib_resources import files
except ImportError:
    from importlib.resources import files
import hydra


@hydra.main(
    version_base=None,
    config_path=str(files("smpl_sim").joinpath("data/cfg")),
    config_name="config",
)
def main(cfg: DictConfig) -> None:
    print(
        f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}"
    )
    cfg.output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if (not cfg.no_log) and (not cfg.test):
        group = cfg.get("group", cfg.learning.agent_name)
        exp_name = (
            f"{cfg.learning.agent_name}_{cfg.env.task}"
            if (cfg.exp_name is None) or (cfg.exp_name == "none")
            else cfg.exp_name
        )
        wandb.init(
            project="phc_mjx",
            group=group,
            resume=not cfg.resume_str is None,
            id=cfg.resume_str,
            notes=cfg.notes,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False),
        )
        # wandb.config.update(dict(cfg), allow_val_change=True)
        wandb.run.name = cfg.exp_name
        wandb.run.save()

    dtype = torch.float32
    torch.set_default_dtype(dtype)
    device = (
        torch.device("cuda", index=0)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(f"Using: {device}")
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    agent = agent_dict[cfg.learning.agent_name](
        cfg, dtype, device, training=True, checkpoint_epoch=cfg.epoch
    )

    if cfg.test:
        agent.run_policy()
    else:
        agent.optimize_policy()
        print("training done!")


if __name__ == "__main__":
    main()
