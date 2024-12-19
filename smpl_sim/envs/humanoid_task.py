import numpy as np
from smpl_sim.envs.humanoid_env import HumanoidEnv

class HumanoidTask(HumanoidEnv):
        
    def reset(self, seed=None, options=None):
        # First reset humanoid, then reset task, then reset the simulation. 
        self.reset_task(options = options)
        return super().reset(seed=seed, options=options)

    def get_task_obs_size(self):
        raise NotImplementedError
    
    def get_obs_size(self):
        return self.get_self_obs_size() + self.get_task_obs_size()

    def update_task(self):
        raise NotImplementedError
    
    def reset_task(self, options=None):
        raise NotImplementedError
    
    def compute_task_obs(self):
        raise NotImplementedError

    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        self.update_task()
    
    def render(self):
        if not self.headless:
            self.draw_task()
        return super().render() # this may return an RGB image (if render_mode="rgb_array")
    
    def draw_task(self):
        pass

    def create_task_visualization(self):
        pass

    def compute_observations(self):
        prop_obs = self.compute_proprioception()
        task_obs = self.compute_task_obs()
        return np.concatenate([prop_obs, task_obs])
        # we need to change the definition of obs space if we want to use dictionaries
        # prop_obs.update(task_obs)
        # return prop_obs

    def create_viewer(self):
        super().create_viewer()
        self.create_task_visualization()
        # after creating the viewer be sure that the elements of the task are up-to-date
        self.draw_task()