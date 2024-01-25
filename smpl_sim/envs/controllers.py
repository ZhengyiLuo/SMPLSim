import numpy as np
import mujoco
from scipy.linalg import cho_solve, cho_factor


class SimpleTorqueController:
    """
    The controller computes the force as

    ..math::
        \tau^n = \min( \max (scale \dot action, -torque\_lim), torque\_lim))


    Attributes
    ----------
    scale : np.ndarray
        action scale
    torque_lim : np.ndarray
        the output is clip in [-torque_lim, torque_lim]
    """

    def __init__(self, scale: np.ndarray, torque_lim: np.ndarray) -> None:
        self._scale = scale
        self._torque_lim = torque_lim

    def control(
        self, action: np.ndarray, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> np.ndarray:
        """Computes the clipped torque :math:`\tau^n`.

        Parameters
        ----------
        action : np.ndarray
            action in [-1,1]
        mj_model : mujoco.MjModel
            The mujoco model
        mj_data : mujoco.MjData
            The mujoco data

        Returns
        -------
        np.ndarray
            the torque to be applied
        """
        torque = action * self._scale
        torque = np.clip(torque, -self._torque_lim, self._torque_lim)
        return torque


class StablePDController:
    """
    Stable PD (SPD) computes the control forces using the next time step

    .. math::
        \tau^n = -k_p (q^{n+1}-\bar{q}^{n+1}) - k_d \dot{q}^{n+1}

    where :math:`q^n` and :math:`\dot{q}^n` are the position and velocity of the state at time :math:`n`.

    Since :math:`q^{n+1}` and :math:`\dot{q}^{n+1}` are unknown, they are computed using a Taylor expansion

    .. math::
        \tau^n = -k_p (q^{n}+\Delta t \dot{q}^n -\bar{q}^{n+1}) - k_d (\dot{q}^{n+1} + \Delta t \ddot{q}^{n})

    For a nonlinear dynamic systems with multiple degrees of freedo, we compute :math:`\ddot{q}^n` by solving the equation

    .. math::
        M(q) \ddot{q} + C(q,\dot{q}) = \tau_{init} + \tau_{ext}

    where :math:`q`, :math:`\dot{q}` and :math:`\ddot{q}` are vectors of positions, velocities and accelerations of the degrees of freedom, respectively.
    :math:`M(q)` is the mass matrix and :math:`C(q,\dot{q})` is the centrifugal force. :math:`\tau_{ext}` indicates other external force such as gravity.

    See https://www.jie-tan.net/project/spd.pdf for more precise description.


    Note that the desired position is computed from a normalized :math:`action \in [-1, 1]^d`, i.e.,

    .. math::
        \bar{q}^{n+1} = action * action\_scale + action\_offset

    The output is clipped in the range :math:`[torque\_lim, torque\_lim]`.


    Attributes
    ----------
    pd_action_scale : np.ndarray
        scale for action normalization
    pd_action_offset : np.ndarray
        offset for action normalization
    qvel_lim : np.ndarray
        used to extract the matrix :math:`M(q)`
    torque_lim : np.ndarray
        the output is clip in [-torque_lim, torque_lim]
    jkp : np.ndarray
        the :math:`k_p` parameters
    jkd : np.ndarray
        the :math:`k_d` parameters

    """

    def __init__(
        self,
        pd_action_scale: np.ndarray,
        pd_action_offset: np.ndarray,
        qvel_lim: np.ndarray,
        torque_lim: np.ndarray,
        jkp: np.ndarray,
        jkd: np.ndarray,
    ) -> None:
        self.pd_action_scale = pd_action_scale
        self.pd_action_offset = pd_action_offset
        self.qvel_lim = qvel_lim
        self.torque_lim = torque_lim
        self.jkp = jkp
        self.jkd = jkd

    def control(
        self, action: np.ndarray, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> np.ndarray:
        """Computes the clipped torque :math:`\tau^n`.

        Parameters
        ----------
        action : np.ndarray
            action in [-1,1]
        mj_model : mujoco.MjModel
            The mujoco model
        mj_data : mujoco.MjData
            The mujoco data

        Returns
        -------
        np.ndarray
            the torque to be applied
        """
        # scale ctrl to qpos.range
        target_pos = action * self.pd_action_scale + self.pd_action_offset
        
        torque = self._compute_torque(target_pos, mj_model, mj_data)
        torque = np.clip(torque, -self.torque_lim, self.torque_lim)
        return torque

    def _compute_torque(
        self, setpoint: np.ndarray, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> np.ndarray:
        qpos = mj_data.qpos.copy()
        qvel = mj_data.qvel.copy()
        dt = mj_model.opt.timestep
        k_p = np.zeros(qvel.shape[0])
        k_d = np.zeros(qvel.shape[0])
        curr_jkp = self.jkp
        curr_jkd = self.jkd
        k_p[6:] = curr_jkp
        k_d[6:] = curr_jkd
        
        qpos_err = np.concatenate((np.zeros(6), qpos[7:] + qvel[6:] * dt - setpoint))
        
        qvel_err = qvel
        q_accel = self._compute_desired_accel(
            qpos_err, qvel_err, k_p, k_d, mj_model, mj_data
        )
        qvel_err += q_accel * dt
        torque = -curr_jkp * qpos_err[6:] - curr_jkd * qvel_err[6:]
        return torque

    def _compute_desired_accel(
        self,
        qpos_err: np.ndarray,
        qvel_err: np.ndarray,
        k_p: np.ndarray,
        k_d: np.ndarray,
        mj_model: mujoco.MjModel,
        mj_data: mujoco.MjData,
    ) -> np.ndarray:
        dt = mj_model.opt.timestep
        nv = mj_model.nv

        M = np.zeros((nv, nv))
        mujoco.mj_fullM(mj_model, M, mj_data.qM)
        M.resize(nv, nv)
        M = M[: self.qvel_lim, : self.qvel_lim]
        C = mj_data.qfrc_bias.copy()[: self.qvel_lim]
        K_p = np.diag(k_p)
        K_d = np.diag(k_d)
        q_accel = cho_solve(
            cho_factor(M + K_d * dt, overwrite_a=True, check_finite=False),
            -C[:, None] - K_p.dot(qpos_err[:, None]) - K_d.dot(qvel_err[:, None]),
            overwrite_b=True,
            check_finite=False,
        )
        return q_accel.squeeze()


class SimplePID:
    """
    Based on
    https://github.com/m-lundberg/simple-pid/blob/master/simple_pid/pid.py
    """

    def __init__(
        self,
        Kp: np.ndarray,
        Ki: np.ndarray,
        Kd: np.ndarray,
        dt: float,
        output_lim: np.ndarray,
        pd_action_scale,
        pd_action_offset,
        proportional_on_measurement: bool = False,
        differential_on_measurement: bool = False,
    ) -> None:
        self._Kp, self._Ki, self._Kd = Kp, Ki, Kd
        self._proportional_on_measurement = proportional_on_measurement
        self._differential_on_measurement = differential_on_measurement
        self.pd_action_scale = pd_action_scale
        self.pd_action_offset = pd_action_offset
        self._output_lim = output_lim
        self._last_input = None
        self._last_error = None
        self._proportional = np.zeros(self._output_lim.shape[0])
        self._integral = np.zeros(self._output_lim.shape[0])
        self._derivative = np.zeros(self._output_lim.shape[0])
        self._dt = dt

    def control(self, action, mj_model, mj_data):
        feedback = mj_data.qpos[7:].copy()
        setpoint = action * self.pd_action_scale + self.pd_action_offset
        error = setpoint - feedback
        d_input = feedback - (
            self._last_input if (self._last_input is not None) else feedback
        )
        d_error = error - (
            self._last_error if (self._last_error is not None) else error
        )

        # Compute the proportional term
        if not self._proportional_on_measurement:
            # Regular proportional-on-error, simply set the proportional term
            self._proportional = self._Kp * error
        else:
            # Add the proportional error on measurement to error_sum
            self._proportional -= self._Kp * d_input

        # Compute integral and derivative terms
        self._integral += self._Ki * error * self._dt
        self._integral = np.clip(
            self._integral, -self._output_lim, self._output_lim
        )  # Avoid integral windup

        if self._differential_on_measurement:
            self._derivative = -self._Kd * d_input / self._dt
        else:
            self._derivative = self._Kd * d_error / self._dt

        # Compute final output
        output = self._proportional + self._integral + self._derivative
        output = np.clip(output, -self._output_lim, self._output_lim)

        # Keep track of state
        self._last_input = feedback
        self._last_error = error

        return output


class PIDController:
    """
    PD computes the control forces as

    .. math::
        \tau^n = -k_p (q^{n}-\bar{q}^{n}) - k_d \dot{q}^{n} - k_i \int_y (q^{y}-\bar{q}^{y})

    where :math:`q^n` and :math:`\dot{q}^n` are the position and velocity of the state at time :math:`n`.

    Note that the desired position is computed from a normalized :math:`action \in [-1, 1]^d`, i.e.,

    .. math::
        \bar{q}^{n} = action * action\_scale + action\_offset

    The output is clipped in the range :math:`[-torque\_lim, torque\_lim]`.


    Attributes
    ----------
    pd_action_scale : np.ndarray
        scale for action normalization
    pd_action_offset : np.ndarray
        offset for action normalization
    torque_lim : np.ndarray
        the output is clip in [-torque_lim, torque_lim]
    jkp : np.ndarray
        the :math:`k_p` parameters
    jkd : np.ndarray
        the :math:`k_d` parameters
    jki : np.ndarray
        the :math:`k_i` parameters

    """

    def __init__(
        self,
        pd_action_scale: np.ndarray,
        pd_action_offset: np.ndarray,
        torque_lim: np.ndarray,
        jkp: np.ndarray,
        jkd: np.ndarray,
        jki: np.ndarray,
    ) -> None:
        self.pd_action_scale = pd_action_scale
        self.pd_action_offset = pd_action_offset
        self.torque_lim = torque_lim
        self.jkp = jkp
        self.jkd = jkd
        self.jki = jki
        self._integral = 0  # TODO: remove the integral part if not used

    def control(
        self, action: np.ndarray, mj_model: mujoco.MjModel, mj_data: mujoco.MjData
    ) -> np.ndarray:
        """Computes the clipped torque :math:`\tau^n`.

        Parameters
        ----------
        action : np.ndarray
            action in [-1,1]
        mj_model : mujoco.MjModel
            The mujoco model
        mj_data : mujoco.MjData
            The mujoco data

        Returns
        -------
        np.ndarray
            the torque to be applied
        """
        dt = mj_model.opt.timestep
        target_pos = action * self.pd_action_scale + self.pd_action_offset
        qpos = mj_data.qpos.copy()[7:]
        qvel = mj_data.qvel.copy()[6:]
        error = qpos - target_pos
        self._integral = self._integral + error * dt
        self._integral = np.clip(
            self._integral, -self.torque_lim, self.torque_lim
        )  # Avoid integral windup
        torque = -self.jkp * error - self.jkd * qvel - self.jki * self._integral
        torque = np.clip(torque, -self.torque_lim, self.torque_lim)
        return torque

    def reset(self) -> None:
        self._integral = 0
