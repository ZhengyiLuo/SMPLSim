import numpy as np

import gymnasium as gym
import warnings
import mujoco
import time
from typing import Optional
import logging
import imageio


class BaseEnv(gym.Env):
    """
    BaseEnv Class
    -------------
    This module contains the BaseEnv class, a base environment for Mujoco simulations. Setup simulation and rendering. 
    """
    
    # see https://gymnasium.farama.org/api/env/#gymnasium.Env.render for information
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}


    def __init__(self, cfg):
        self.clip_actions = cfg.env.clip_actions
        self.render_mode = cfg.env.render_mode
        assert self.render_mode is None or self.render_mode in self.metadata["render_modes"]

        self.headless = cfg.headless
        self.sim_timestep_inv = cfg.env.sim_timestep_inv
        self.sim_timestep = 1.0 / self.sim_timestep_inv
        self.control_freq_inv = cfg.env.control_frequency_inv
        self.cur_t = 0  # Current simulation time step
        self.dt = self.sim_timestep * self.control_freq_inv

        if not np.isclose(self.sim_timestep_inv / self.control_freq_inv, 30.0):
            warnings.warn(
                f"Control frequency is {self.sim_timestep_inv/self.control_freq_inv:.2f}Hz. Please be sure that data is generated with the same frequency"
            )

        """
        If human-rendering is used, `self.window` will be a reference
        to the window that we draw to. `self.clock` will be a clock that is used
        to ensure that the environment is rendered at the correct framerate in
        human-mode. They will remain `None` until human-mode is used for the
        first time.
        """
        self.viewer = None
        self.renderer = None
        self.camera = cfg.env.get("camera", -1)
        self.paused = False
        self.disable_reset = False
        self.follow = False
        self.recording, self.recording_state_change = False, False
        if not self.headless:
            if self.render_mode == "human":
                logging.info(f"Using viewer for interactive rendering")
            else:
                logging.info(f"Using RGB rendered with camera {self.camera}")

    ############################################################################################
    # MAIN EXTERNAL FUNCTION (reset, step, render, close, ..)
    ############################################################################################

    def reset(self, seed=None, options=None):
        # You may assume that the step method will not be called before reset has been called
        # reset should be called whenever a done signal has been issued
        # seed keyword to reset to initialize any random number generator that is used by the environment to a deterministic state
        # self.np_random that is provided by the environment’s base class

        super().reset(seed=seed, options=options)
        self.cur_t = 0

        observation = self.compute_observations()
        info = self.compute_info()

        if self.render_mode == "human":
            self.render()

        return observation, info
    
    def step(self, action):
        # apply actions
        self.pre_physics_step(action)

        # step physics and render each frame
        self.physics_step(action)

        # compute observations, rewards, resets, ...
        observation, reward, terminated, truncated, info = self.post_physics_step(action)

        # if humand render update the visualizer
        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
    
    def render(self):
        return self._render_frame()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
    
    def seed(self, seed: Optional[int] = None):
        super().reset(seed=seed)
    
    ############################################################################################
    # Observations, infos, termination conditions
    ############################################################################################

    def compute_observations(self):
        # We choose to represent observations in the form of dictionaries
        raise NotImplementedError

    def compute_info(self):
        raise NotImplementedError
    
    ############################################################################################
    # Step-related functions
    ############################################################################################

    def physics_step(self, action):
        raise NotImplementedError

    def post_physics_step(self, action):
        raise NotImplementedError
    
    def pre_physics_step(self, action):
        raise NotImplementedError

    ############################################################################################
    # Mujoco functions
    ############################################################################################
    def create_sim(self, xml: str):
        self.mj_model = mujoco.MjModel.from_xml_string(xml)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_timestep

    ############################################################################################
    # Visualization
    ############################################################################################

    def _render_frame(self):
        if not self.headless:
            if self.viewer is None and self.renderer is None:
                self.create_viewer()
            
            if self.render_mode == "human":
                self.viewer.sync()
                if self.follow:
                    self.viewer.cam.lookat = self.mj_data.qpos[:3]
                time.sleep(1. / self.metadata["render_fps"])
            
            if self.render_mode == "rgb_array":
                self.renderer.update_scene(self.mj_data, camera=self.camera)
                pixels = self.renderer.render()
                return pixels
            
            if self.recording:
                self.cur_t
                self.record_states()
                
    def _create_renderer(self):
        self.renderer = mujoco.Renderer(self.mj_model)  # MJ offline renderer
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self.renderer.update_scene(self.mj_data)

    def create_viewer(self):
        if not self.headless and self.render_mode == "human":
            print("human")
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data, key_callback=self.key_callback)
            
        if not self.headless and self.render_mode == "rgb_array":
            self._create_renderer()

    def key_callback(self, keycode):
        print(keycode)
        if chr(keycode) == " ":
            self.paused = not self.paused
            print(f"Paused {self.paused}")
        elif chr(keycode) == "R":
            self.reset()
        elif chr(keycode) == "M":
            self.disable_reset = not self.disable_reset
            print(f"Disable reset {self.disable_reset}")
        elif chr(keycode) == "F":
            self.follow = not self.follow
            print(f"Follow {self.follow}")
        elif chr(keycode) == "L":
            self.recording = not self.recording
            print(f"Record {self.recording}")


    # Recording states 
    def record_states(self):
        raise NotImplementedError