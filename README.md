[Repo still under construction]
# Porting of UHC and PHC to MUJOCO>=3

`smpl_sim` is a pip-installable library containing a modelization of the SMPL humanoid in different simulators (MUJOCO and Isaac Gym). It is a minimal library to support simple humanoid tasks, and is the basis library for doing more complicated tasks such as motion imitation. 

### Commands:

```
python examples/env_humanoid_test.py headless=False
python smpl_sim/run.py env=speed
python smpl_sim/run.py env=getup
python smpl_sim/run.py env=reach

```

Authors: [@Zhengyi](https://github.com/ZhengyiLuo)  [@Matteo](https://github.com/teopir)