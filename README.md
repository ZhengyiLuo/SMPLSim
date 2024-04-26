[Repo still under construction]
# SMPLSim: Simulating SMPL/SMPLX Humanoids in MUJOCO and Isaac Gym

`smpl_sim` is a pip-installable library containing a modelization of the SMPL humanoid in different simulators (MUJOCO and Isaac Gym). It is a minimal library to support simple humanoid tasks, and is the basis library for doing more complicated tasks such as motion imitation. 

<div float="center">
  <img src="assets/smplsim_teaser.gif" />
</div>



## Core Features
This repo supports creating the xml files for SMPL/SMPLH/SMPLX compatible humanoid models in MUJOCO>=3. It also supports creating the humanoid models in Isaac Gym. It also provides a simple PPO implementation for training the humanoid models in MUJOCO. 

### Commands:

```
python examples/env_humanoid_test.py headless=False
python smpl_sim/run.py env=speed exp_name=speed env.self_obs_v=2  robot.create_vel_sensors=True
python smpl_sim/run.py env=getup exp_name=speed env.self_obs_v=2 robot.create_vel_sensors=True
python smpl_sim/run.py env=reach exp_name=speed env.self_obs_v=2 robot.create_vel_sensors=True
```


## Citation
If you find this work useful for your research, please cite our paper:
```
PULSE:
@inproceedings{
luo2024universal,
title={Universal Humanoid Motion Representations for Physics-Based Control},
author={Zhengyi Luo and Jinkun Cao and Josh Merel and Alexander Winkler and Jing Huang and Kris M. Kitani and Weipeng Xu},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=OrOd8PxOO2}
}

PHC:
@inproceedings{Luo2023PerpetualHC,
    author={Zhengyi Luo and Jinkun Cao and Alexander W. Winkler and Kris Kitani and Weipeng Xu},
    title={Perpetual Humanoid Control for Real-time Simulated Avatars},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2023}
}           

UHC:
@inproceedings{Luo2021DynamicsRegulatedKP,
  title={Dynamics-Regulated Kinematic Policy for Egocentric Pose Estimation},
  author={Zhengyi Luo and Ryo Hachiuma and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2021}
}
```


Also consider citing these prior works that are used in this project:

```
@inproceedings{Luo2022EmbodiedSH,
  title={Embodied Scene-aware Human Pose Estimation},
  author={Zhengyi Luo and Shun Iwase and Ye Yuan and Kris Kitani},
  booktitle={Advances in Neural Information Processing Systems},
  year={2022}
}


@inproceedings{rempeluo2023tracepace,
    author={Rempe, Davis and Luo, Zhengyi and Peng, Xue Bin and Yuan, Ye and Kitani, Kris and Kreis, Karsten and Fidler, Sanja and Litany, Or},
    title={Trace and Pace: Controllable Pedestrian Animation via Guided Trajectory Diffusion},
    booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2023}
}     

```
