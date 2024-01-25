# Porting of UHC and PHC to MUJOCO>=3

`mj_smpl` is a pip-installable library containing a modelization of the SMPL humanoid.

Go to the [README](README_smpl.md) for installation instruction


## ROADMAP


- [ ] Add PID controller
- [ ] Adjust parameters of torque control and PID control
- [ ] Define zero position (stand pose). This is needed in the [reset](https://github.com/teopir/PerpetualHumanoidControl-MJX/blob/81140054b664ec7e044527043d7d9fef20586724/new_library/mj_smpl/mj_smpl/envs/smplenv.py#L274) function of the environment
- [ ] Initialize the falling position, in particular how to set a valid initial random position. See [current implementation](https://github.com/teopir/PerpetualHumanoidControl-MJX/blob/81140054b664ec7e044527043d7d9fef20586724/new_library/mj_smpl/mj_smpl/envs/smplenv.py#L279).
- [ ] Make sure internal collision is correct. 
- [ ] Compute qvel from mocap. See [missing code here](https://github.com/teopir/PerpetualHumanoidControl-MJX/blob/81140054b664ec7e044527043d7d9fef20586724/new_library/mj_smpl/mj_smpl/envs/smplenv.py#L345).
- [ ] Porting the PHC formulation of Nvidia Isaac Gym, i.e. single morphology neutral with height adaptation.
- [ ] MotionLib ?? 
 
### Commands:

```
python examples/env_humanoid_test.py headless=False

python scripts/run.py  env=speed  exp_name=humanoid_speed 
python scripts/run.py  env=reach  exp_name=humanoid_reach
```