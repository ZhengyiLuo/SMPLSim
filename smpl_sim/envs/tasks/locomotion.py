from typing import Any, Sequence
import numpy as np
from ..smplenv import SMPLHumanoid
from ...utils import tolerance


def locomotion_reward(obs, control, move_speed) -> float:
    head_height = obs["Head_xpos"][-1]
    stand_height = (
        obs["full_height"] * 0.86
    )  # TODO: check this (does it make sense 86% of total height)
    standing = tolerance.tolerance(
        head_height, bounds=(stand_height, float("inf")), margin=stand_height / 4
    )
    chest_upright = obs["Chest_upright"]
    upright = tolerance.tolerance(
        chest_upright,
        bounds=(0.9, float("inf")),
        sigmoid="linear",
        margin=1.9,
        value_at_margin=0,
    )
    stand_reward = standing * upright
    small_control = tolerance.tolerance(
        control, margin=1, value_at_margin=0, sigmoid="quadratic"
    ).mean()
    small_control = (4 + small_control) / 5
    center_of_mass_velocity = obs["center_of_mass_velocity"]
    if move_speed == 0:
        horizontal_velocity = center_of_mass_velocity[[0, 1]]
        dont_move = tolerance.tolerance(horizontal_velocity, margin=2).mean()
        return small_control * stand_reward * dont_move
    else:
        com_velocity = np.linalg.norm(center_of_mass_velocity[[0, 1]])
        move = tolerance.tolerance(
            com_velocity,
            bounds=(move_speed, float("inf")),
            margin=move_speed,
            value_at_margin=0,
            sigmoid="linear",
        )
        move = (5 * move + 1) / 6
        return small_control * stand_reward * move


class SMPLHumanoidMove(SMPLHumanoid):
    def __init__(
        self,
        motions: Any,
        max_episode_length: int,
        pid_controlled: bool = True,
        data_dir: str = "",
        physics_steps_per_control_step: int = 6,  # 30Hz
        move_speed: float = 0,
        sim_timestep: float = 1.0 / 180.0,
        initial_position: str = "random",
        probabilities_of_random_methods: Sequence[float] = [
            0.5,
            0.5,
            0,
        ],  # only for hybrid mode, [mocap, random, stand]
        seed: int = 0,
    ):
        super().__init__(
            motions=motions,
            max_episode_length=max_episode_length,
            pid_controlled=pid_controlled,
            data_dir=data_dir,
            physics_steps_per_control_step=physics_steps_per_control_step,
            sim_timestep=sim_timestep,
            initial_position=initial_position,
            probabilities_of_random_methods=probabilities_of_random_methods,
            seed=seed,
        )
        self._move_speed = move_speed

    def get_reward(self, obs, control) -> float:
        return locomotion_reward(obs=obs, control=control, move_speed=self._move_speed)
