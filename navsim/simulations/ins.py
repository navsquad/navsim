"""
|============================================= ins.py =============================================|
|                                                                                                  |
|   Property of NAVSQUAD. Unauthorized copying of this file via any medium would be super sad and  |
|   unfortunate for us. Proprietary and confidential.                                              |
|                                                                                                  |
| ------------------------------------------------------------------------------------------------ |
|                                                                                                  |
|   @file     navsim/simulations/ins.py                                                            |
|   @brief    Generates INS path and measurements based on imu-level input commands                |
|               - A fork of Acenia's GNSS-INS-SIM modified for NAVSIM                              |
|   @author   Daniel Sturdivant <sturdivant20@gmail.com>                                           |
|   @date     February 2024                                                                        |
|                                                                                                  |
|==================================================================================================|
"""

__all__ = ["INSSimulation"]

import os, warnings
import numpy as np
import scipy.io as sio
import pandas as pd
import pathlib as pl
from datetime import datetime
from tqdm import tqdm
from dataclasses import dataclass
from numba import njit

try:
    is_readline_available = True
    import readline as rl
except ImportError:
    is_readline_available = False
    from tkinter import filedialog as fd

from navsim.configuration import select_file
from navsim.error_models.imu import get_imu_allan_variance_values, IMU
from navsim.configuration import SimulationConfiguration, IMUConfiguration
from navtools.measurements.radii_of_curvature import radiiOfCurvature
from navtools.measurements.gravity import nedg
from navtools.conversions.coordinates import lla2enu, lla2ecef, ned2ecefv
from navtools.conversions.attitude import wrapEulerAngles
from navtools.constants import GNSS_OMEGA_EARTH

try:
    is_log_utils_available = True
    from log_utils import *
except:
    is_log_utils_available = False
# is_log_utils_available = False

R2D = 180 / np.pi
D2R = np.pi / 180
LLA_R2D = np.array([R2D, R2D, 1])
LLA_D2R = np.array([D2R, D2R, 1])
I3 = np.eye(3)


class INSSimulation:
    @property
    def imu_model(self) -> IMU:
        return self.__imu_model

    @property
    def ecef_position(self) -> np.ndarray:
        return self.__ecef_pos

    @property
    def ecef_velocity(self) -> np.ndarray:
        return self.__ecef_vel

    @property
    def specific_force(self) -> np.ndarray:
        return self.__meas_spc_frc

    @property
    def angular_velocity(self) -> np.ndarray:
        return self.__meas_ang_vel

    @property
    def true_specific_force(self) -> np.ndarray:
        return self.__spc_frc

    @property
    def true_angular_velocity(self) -> np.ndarray:
        return self.__ang_vel

    @property
    def euler_angles(self) -> np.ndarray:
        return self.__eul_ang

    @property
    def geodetic_position(self):
        return self.__lla_pos

    @property
    def tangent_position(self):
        return self.__enu_pos

    @property
    def tangent_velocity(self):
        return self.__enu_vel

    @property
    def time(self):
        return self.__time

    def __init__(self, config: SimulationConfiguration, disable_progress: bool = False, use_config_fsim: bool = False) -> None:
        # generate output filename
        now = datetime.now().strftime(format="%Y%m%d-%H%M%S")
        sim_now = datetime(
            year=config.time.year,
            month=config.time.month,
            day=config.time.day,
            hour=config.time.hour,
            minute=config.time.minute,
            second=config.time.second,
        ).strftime(format="%Y%m%d-%H%M%S")
        self.__output_file_stem = f"{now}_NAVSIM_{sim_now}_{int(config.time.duration)}_{config.time.fsim}"

        # tqdm boolean
        self.__disable_progress = disable_progress

        # initialize imu errors
        if config.imu is None or config.imu.model.casefold() == "perfect":
            self.__is_error_simulated = False
            self.__imu_model = get_imu_allan_variance_values("perfect")
        else:
            self.__is_error_simulated = True
            self.__imu_model = get_imu_allan_variance_values(config.imu.model)
        if use_config_fsim:
            self.__imu_model.f = config.time.fsim
        self.__mobility = np.array(config.imu.mobility)
        self.__osr = config.imu.osr
        self.__vibration_model = config.imu.vibration_model

    # --------------------------------------------------------------------------------------------------#
    #! === Input Motion Commands ===
    def motion_commands(self, motion_def: str) -> None:
        """select CSV file that contains motion definition

        Parameters
        ----------
        motion_def : str
            File that contains CSV commands for the simulator
              - Row 1: Header line for the initial states
              - Row 2: Initial states
                -> Col 1-3: initial LLA position [deg, deg, m]
                -> Col 4-6: initial body frame velocity [m/s]
                -> Col 7-9: initial euler angles (roll, pitch, yaw) [deg]
              - Row 3: header line for the motion commands
              - Row 4+: motion commands:
                -> Col 1: Command type
                    + 1: body frame velocity and attitude rates
                    + 2: absolute velocity and attitude to reach
                    + 3: relative velocity and attitude change
                    + 4: relative velocity, absolute attitude
                    + 5: absolute velocity, relative attitude
                -> Col 2-4: angular velocity/attitude command
                -> Col 5-7: acceleration/velocity command
                -> Col 8: Maximum time allowed for command segment
                -> Col 9: just put zeros here
        """
        # select proper file
        if is_log_utils_available:
            prompt_string = default_logger.GenerateSring("[navsim] select an IMU motion definition: ", Level.Info, Color.Info)
        else:
            prompt_string = "[navsim] select an IMU motion definition: "

        if is_readline_available:
            config_file_name = select_file(
                directory_path=motion_def,
                prompt_string=prompt_string,
            )
            motion_def_file_path = motion_def / config_file_name
        else:
            filetypes = (("csv", "*.csv"), ("xlxs", "*.xlxs*"))
            motion_def_file_path = fd.askopenfilename(
                initialdir=motion_def,
                title=prompt_string,
                filetypes=filetypes,
            )

        # generate motion parameters from file
        self.__init_pva = np.genfromtxt(motion_def_file_path, delimiter=",", skip_header=1, max_rows=1)
        self.__motion_def = np.genfromtxt(motion_def_file_path, delimiter=",", skip_header=3)

    #! === Simulate IMU and Generate Path ===
    def simulate(self) -> None:
        # --- simulation variables ---
        pos_n = self.__init_pva[:3] * LLA_D2R
        self.__vel_b = self.__init_pva[3:6]
        self.__att = self.__init_pva[6:] * D2R
        self.__C_b_n = eulzyx2dcm(self.__att).T  # body to nav
        vel_n = self.__C_b_n @ self.__vel_b
        pos_delta_n = np.zeros(3, dtype=np.float64)
        self.__vel_dot_b = np.zeros(3, dtype=np.float64)
        self.__att_dot = np.zeros(3, dtype=np.float64)
        acc_sum = np.zeros(3, dtype=np.float64)
        gyr_sum = np.zeros(3, dtype=np.float64)

        sim_count = 0
        out_idx = 0
        self.__mobility[1:] *= D2R  # convert angular acceleration to rad
        self.__motion_def[:, 1:4] *= D2R  # covert angular commands to rad
        out_freq = int(self.__imu_model.f)  # imu frequency
        sim_freq = self.__osr * out_freq  # simulation frequency
        dt = 1.0 / sim_freq  # simulation period

        # --- pathgen LPF to smooth trajectory ---
        alpha = 0.9
        filt_a = alpha * I3
        filt_b = (1.0 - alpha) * I3
        kp = 5.0
        kd = 10.0
        att_converge_threshold = 1e-6
        vel_converge_threshold = 1e-4
        max_acc = self.__mobility[0]  # max acceleration
        max_dw = self.__mobility[1]  # max angular acceleration
        max_w = self.__mobility[2]  # max angular velocity

        # --- convert time duration commands to maximum number of simulation iterations ---
        sim_count_max = 0
        for i in range(self.__motion_def.shape[0]):
            seg_count = self.__motion_def[i, 7] * out_freq  # max output iterations for this segment
            sim_count_max += int(np.ceil(seg_count))  # add to total length
            self.__motion_def[i, 7] = int(np.round(seg_count * self.__osr))  # simulation iterations

        # --- output variables ---
        time = np.zeros(sim_count_max, dtype=np.float64)
        spc_frc = np.zeros((sim_count_max, 3), dtype=np.float64)
        ang_vel = np.zeros((sim_count_max, 3), dtype=np.float64)
        lla_pos = np.zeros((sim_count_max, 3), dtype=np.float64)
        enu_pos = np.zeros((sim_count_max, 3), dtype=np.float64)
        enu_vel = np.zeros((sim_count_max, 3), dtype=np.float64)
        eul_ang = np.zeros((sim_count_max, 3), dtype=np.float64)
        ecef_pos = np.zeros((sim_count_max, 3), dtype=np.float64)
        ecef_vel = np.zeros((sim_count_max, 3), dtype=np.float64)

        # if is_log_utils_available:
        #     prompt_string = default_logger.GenerateSring("[navsim] ins path generation ", Level.Info, Color.Info)
        # else:
        prompt_string = "[\u001b[35;1mnavsim\u001b[0m] ins path generation "

        # --- begin simulation loop ---
        for i in tqdm(
            range(self.__motion_def.shape[0]),
            total=self.__motion_def.shape[0],
            desc=prompt_string,
            disable=self.__disable_progress,
            ascii=".>#",
            bar_format="{desc}{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{rate_fmt}]",
            ncols=120,
        ):
            com_type = np.round(self.__motion_def[i, 0])  # motion command type for this segment

            # parse motion command for this segment
            motion_com = self.__parse_motion_def(self.__motion_def[i, :])
            if com_type == 1:
                att_dot_com = motion_com[0]
                vel_dot_com = motion_com[1]
            else:
                att_com = motion_com[0]
                vel_com_b = motion_com[1]

            # reset filter states to most recent attitude and velocity
            att_com_filt = self.__att
            vel_com_b_filt = self.__vel_b

            # generate trajectory based on command
            sim_count_max = int(sim_count + self.__motion_def[i, 7])  # max cycles to execute in this segment
            com_complete = 0  # completion boolean
            while (sim_count < sim_count_max) and (com_complete == 0):
                # filter the motion command
                if com_type == 1:
                    self.__att_dot = filt_a @ self.__att_dot + filt_b @ att_dot_com
                    self.__vel_dot_b = filt_a @ self.__vel_dot_b + filt_b @ vel_dot_com
                else:
                    att_com_filt = filt_a @ att_com_filt + filt_b @ att_com
                    vel_com_b_filt = filt_a @ vel_com_b_filt + filt_b @ vel_com_b

                    # Close the loop. Velocity change is acceleration and sudden change in acceleration
                    # is reasonable. Attitude change is angular velocity and sudden change in angular
                    # velocity means infinite torque, which is unreasonable. So a simple PD controller
                    # is used here to track the commanded attitude.

                    self.__vel_dot_b = (vel_com_b_filt - self.__vel_b) / dt
                    self.__vel_dot_b[self.__vel_dot_b > max_acc] = max_acc
                    self.__vel_dot_b[self.__vel_dot_b < -max_acc] = -max_acc

                    att_dot_dot = kp * (att_com - self.__att) + kd * (0 - self.__att_dot)  # feedback control
                    att_dot_dot[att_dot_dot > max_dw] = max_dw  # limit w change rate
                    att_dot_dot[att_dot_dot < -max_dw] = -max_dw
                    self.__att_dot = self.__att_dot + att_dot_dot * dt
                    self.__att_dot[self.__att_dot > max_w] = max_w  # limit self.__att change rate
                    self.__att_dot[self.__att_dot < -max_w] = -max_w

                    # check to see if segment command has been completed
                    if (
                        np.linalg.norm(self.__att - att_com) < att_converge_threshold
                        and np.linalg.norm(self.__vel_b - vel_com_b) < vel_converge_threshold
                    ):
                        com_complete = 1

                # compute IMU readings based on pos/vel/self.__att changes
                acc, gyr, _, pos_dot_n = self.__calc_true_sensor_output(pos_n + pos_delta_n)
                acc_sum = acc_sum + acc
                gyr_sum = gyr_sum + gyr

                # write simulation results
                if not (sim_count % self.__osr):
                    time[out_idx] = out_idx / out_freq
                    if sim_count == 0:
                        spc_frc[out_idx, :] = acc_sum
                        ang_vel[out_idx, :] = gyr_sum
                    else:
                        spc_frc[out_idx, :] = acc_sum / self.__osr
                        ang_vel[out_idx, :] = gyr_sum / self.__osr
                    lla_pos[out_idx, :] = (pos_n + pos_delta_n) * LLA_R2D
                    enu_pos[out_idx, :] = lla2enu(pos_n + pos_delta_n, pos_n)
                    enu_vel[out_idx, :] = np.array([vel_n[1], vel_n[0], -vel_n[2]])
                    eul_ang[out_idx, :] = wrapEulerAngles(self.__att.copy()[::-1]) * R2D
                    ecef_pos[out_idx, :] = lla2ecef(pos_n + pos_delta_n)
                    # ecef_vel[out_idx, :] = ned2ecefv(vel_n, pos_n)
                    ecef_vel[out_idx, :] = ned2ecefv(vel_n, pos_n + pos_delta_n)
                    acc_sum = np.zeros(3, dtype=np.float64)
                    gyr_sum = np.zeros(3, dtype=np.float64)
                    out_idx += 1

                # accumulate pos/vel/att change
                pos_delta_n = pos_delta_n + pos_dot_n * dt  # accumulated pos change
                self.__vel_b = self.__vel_b + self.__vel_dot_b * dt
                self.__att = self.__att + self.__att_dot * dt
                self.__C_b_n = eulzyx2dcm(self.__att).T  # body to nav
                vel_n = self.__C_b_n @ self.__vel_b
                sim_count += 1

            # if command is completed, att_dot and vel_dot should be set to zero
            if com_complete == 1:
                self.__att_dot = np.zeros(3, dtype=np.float64)
                self.__vel_dot_b = np.zeros(3, dtype=np.float64)

        self.__time = time[:out_idx]
        self.__spc_frc = spc_frc[:out_idx, :]
        self.__ang_vel = ang_vel[:out_idx, :]
        self.__lla_pos = lla_pos[:out_idx, :]
        self.__enu_pos = enu_pos[:out_idx, :]
        self.__enu_vel = enu_vel[:out_idx, :]
        self.__eul_ang = eul_ang[:out_idx, :]
        self.__ecef_pos = ecef_pos[:out_idx, :]
        self.__ecef_vel = ecef_vel[:out_idx, :]

        # finished, add imu noise
        if self.__is_error_simulated:
            self.add_noise()
        else:
            self.__meas_spc_frc = self.__spc_frc
            self.__meas_ang_vel = self.__ang_vel

    def add_noise(self) -> None:
        """Add error to true IMU readings according to model parameters"""
        n = self.__spc_frc.shape[0]
        dt = 1 / self.__imu_model.f
        sqdt = np.sqrt(dt)
        beta_acc = dt / self.__imu_model.Tc_acc
        beta_gyr = dt / self.__imu_model.Tc_gyr

        # GNSS-INS-SIM
        n_gyr = np.zeros((n, 3))
        n_acc = np.zeros((n, 3))
        d_gyr = np.zeros((n, 3))
        d_acc = np.zeros((n, 3))
        for i in range(3):
            # white noise
            n_gyr[:, i] = self.__imu_model.N_gyr[i] / sqdt * np.random.randn(n)
            n_acc[:, i] = self.__imu_model.N_acc[i] / sqdt * np.random.randn(n)

            # drift
            a_gyr = 1 - beta_gyr[i]
            b_gyr = self.__imu_model.B_gyr[i] * np.sqrt(1.0 - np.exp(-2.0 * beta_gyr[i]))
            w_gyr = np.random.randn(n)
            a_acc = 1 - beta_acc[i]
            b_acc = self.__imu_model.B_acc[i] * np.sqrt(1.0 - np.exp(-2.0 * beta_acc[i]))
            w_acc = np.random.randn(n)
            for j in range(1, n):
                d_gyr[j, i] = a_gyr * d_gyr[j - 1, i] + b_gyr * w_gyr[j - 1]
                d_acc[j, i] = a_acc * d_acc[j - 1, i] + b_acc * w_acc[j - 1]

        self.__meas_spc_frc = self.__spc_frc + n_acc + d_acc
        self.__meas_ang_vel = self.__ang_vel + n_gyr + d_gyr

        # # exponentially correlated, fixed-variance, first-order, Markov process (Groves 14.85, pg. 593)
        # # A. G. Quinchia et. al, "A Comparison between Different Error Modeling of MEMS Applied to GPS/INS Integrated Systems"
        # for i in range(3):
        #     # noise model
        #     X_acc = np.zeros(2)
        #     Y_acc = np.zeros(self.__spc_frc.shape)
        #     F_acc = np.array([[1 - beta_acc[i], 0], [0, 1]])
        #     G_acc = np.array(
        #         [np.sqrt(1 - np.exp(-2 * beta_acc[i])) * self.__imu_model.B_acc[i], sqdt * self.__imu_model.K_acc[i]]
        #     )
        #     H_acc = np.array([1, 1])
        #     D_acc = self.__imu_model.N_acc[i] / sqdt

        #     X_gyr = np.zeros(2)
        #     Y_gyr = np.zeros(self.__ang_vel.shape)
        #     F_gyr = np.array([[1 - beta_gyr[i], 0], [0, 1]])
        #     G_gyr = np.array(
        #         [np.sqrt(1 - np.exp(-2 * beta_gyr[i])) * self.__imu_model.B_gyr[i], sqdt * self.__imu_model.K_gyr[i]]
        #     )
        #     H_gyr = np.array([1, 1])
        #     D_gyr = self.__imu_model.N_gyr[i] / sqdt

        #     # noise loop
        #     w_acc = np.random.randn(n)
        #     v_acc = np.random.randn(n)
        #     w_gyr = np.random.randn(n)
        #     v_gyr = np.random.randn(n)
        #     for k in range(n):
        #         X_acc = F_acc @ X_acc + G_acc * w_acc[k]
        #         Y_acc[k, i] = H_acc @ X_acc + D_acc * v_acc[k]

        #         X_gyr = F_gyr @ X_gyr + G_gyr * w_gyr[k]
        #         Y_gyr[k, i] = H_gyr @ X_gyr + D_gyr * v_gyr[k]

        # self.__meas_spc_frc = self.__spc_frc + Y_acc
        # self.__meas_ang_vel = self.__ang_vel + Y_gyr

    # --------------------------------------------------------------------------------------------------#\
    def to_hdf(self, output_dir_path: str):
        output_path = pl.Path(output_dir_path) / self.__output_file_stem

        pva_states_df = pd.DataFrame(
            np.block(
                [
                    self.__time[:, None],
                    self.__lla_pos,
                    self.__enu_pos,
                    self.__enu_vel,
                    self.__eul_ang,
                ]
            ),
            columns=[
                "time [s]",
                "lat [deg]",
                "lon [deg]",
                "hgt [m]",
                "east [m]",
                "north [m]",
                "up [m]",
                "ve [m/s]",
                "vn [m/s]",
                "vu [m/s]",
                "roll [deg]",
                "pitch [deg]",
                "yaw [deg]",
            ],
        )
        ecef_states_df = pd.DataFrame(
            np.block([self.__time[:, None], self.__ecef_pos, self.__ecef_vel]),
            columns=[
                "time [s]",
                "x [m]",
                "y [m]",
                "z [m]",
                "vx [m]",
                "vy [m]",
                "vz [m]",
            ],
        )
        imu_meas_df = pd.DataFrame(
            np.block([self.__time[:, None], self.__spc_frc, self.__ang_vel]),
            columns=[
                "time [s]",
                "fx [m/s^2]",
                "fy [m/s^2]",
                "fz [m/s^2]",
                "wx [rad/s]",
                "wy [rad/s]",
                "wz [rad/s]",
            ],
        )

        warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
        pva_states_df.to_hdf(output_path.with_suffix(".h5"), key="pva", mode="a")
        ecef_states_df.to_hdf(output_path.with_suffix(".h5"), key="ecef_pv", mode="a")
        imu_meas_df.to_hdf(output_path.with_suffix(".h5"), key="imu_meas", mode="a")

        if is_log_utils_available:
            default_logger.Info(f"[navsim] exported measurement-level results to {output_path.with_suffix('.h5')}")
        else:
            print(f"[navsim] exported measurement-level results to {output_path.with_suffix('.h5')}")

    def to_mat(self, output_dir_path: str):
        output_path = pl.Path(output_dir_path) / self.__output_file_stem

        d = {
            "pva": {
                "time [s]": self.__time[:, None],
                "lat [deg]": self.__lla_pos[:, 0],
                "lon [deg]": self.__lla_pos[:, 1],
                "alt [m]": self.__lla_pos[:, 2],
                "east [m]": self.__enu_pos[:, 0],
                "north [m]": self.__enu_pos[:, 1],
                "up [m]": self.__enu_pos[:, 2],
                "ve [m/s]": self.__enu_vel[:, 0],
                "vn [m/s]": self.__enu_vel[:, 1],
                "vu [m/s]": self.__enu_vel[:, 2],
                "roll [deg]": self.__eul_ang[:, 0],
                "pitch [deg]": self.__eul_ang[:, 1],
                "yaw [deg]": self.__eul_ang[:, 2],
            },
            "ecef_pv": {
                "time [s]": self.__time[:, None],
                "x [m]": self.__ecef_pos[:, 0],
                "y [m]": self.__ecef_pos[:, 1],
                "z [m]": self.__ecef_pos[:, 2],
                "vx [m/s]": self.__ecef_vel[:, 0],
                "vy [m/s]": self.__ecef_vel[:, 1],
                "vz [m/s]": self.__ecef_vel[:, 2],
            },
            "imu_meas": {
                "time [s]": self.__time[:, None],
                "fx [m/s^2]": self.__spc_frc[:, 0],
                "fy [m/s^2]": self.__spc_frc[:, 1],
                "fz [m/s^2]": self.__spc_frc[:, 2],
                "wx [rad/s]": self.__ang_vel[:, 0],
                "wy [rad/s]": self.__ang_vel[:, 1],
                "wz [rad/s]": self.__ang_vel[:, 2],
            },
        }
        sio.savemat(
            file_name=output_path.with_suffix(".mat"),
            mdict=d,
            do_compression=True,
        )

        if is_log_utils_available:
            default_logger.Info(f"[navsim] exported measurement-level results to {output_path.with_suffix('.mat')}")
        else:
            print(f"[navsim] exported measurement-level results to {output_path.with_suffix('.mat')}")

    # --------------------------------------------------------------------------------------------------#
    def __parse_motion_def(self, motion_def_seg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """parse the command of a segment in motion_def

        Parameters
        ----------
        motion_def_seg : np.ndarray
            1x7 a command/segment of motion_def
        self.__att : np.ndarray
            1x3 current attitude
        vel : np.ndarray
            1x3 current velocity

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            1x3 target attitude and 1x3 target velocity
        """
        match motion_def_seg[0]:
            case 1:
                att_com = motion_def_seg[1:4]
                vel_com = motion_def_seg[4:7]
            case 2:  # abs self.__att and abs vel
                att_com = motion_def_seg[1:4]
                vel_com = motion_def_seg[4:7]
            case 3:  # rel self.__att and rel vel
                att_com = self.__att + motion_def_seg[1:4]
                vel_com = self.__vel_b + motion_def_seg[4:7]
            case 4:  # abs self.__att and rel vel
                att_com = motion_def_seg[1:4]
                vel_com = self.__vel_b + motion_def_seg[4:7]
            case 5:  # rel self.__att and abs vel
                att_com = self.__att + motion_def_seg[1:4]
                vel_com = motion_def_seg[4:7]
        return att_com, vel_com

    def __calc_true_sensor_output(self, pos_n: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate the true IMU measurements based on the attitude/velocity change rates

        Parameters
        ----------
        pos_n : np.ndarray
            3x1 absolute LLA position [rad,rad,m]
        self.__vel_b : np.ndarray
            3x1 body frame velocity [m/s]
        self.__att : np.ndarray
            3x1 euler angles (yaw, pitch, roll) [rad]
        self.__C_b_n : np.ndarray
            3x1 DCM from  body to nav
        self.__vel_dot_b : np.ndarray
            3x1 body frame velocity rate [m/s^2]
        self.__att_dot : np.ndarray
            3x1 euler angle rate [rad/s]

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
            acc         3x1   true accelerometer specific force [m/s^2]
            gyr         3x1   true gyroscope angular velocity [rad/s]
            vel_dot_n   3x1   velocity rate in the nav frame [m/s^2]
            pos_dot_n   3x1   position rate in the nav frame [m/s]
        """

        # transpose inputs into the nav frame
        C_n_b = self.__C_b_n.T
        vel_n = self.__C_b_n @ self.__vel_b

        # calculate rotation rate of nav frame w.r.t inertial frame (i -> e -> n)
        phi, lam, h = pos_n
        vn, ve, vd = vel_n
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)

        Re, Rn, r_es_e = radiiOfCurvature(phi)
        gravity = nedg(pos_n)
        rm_eff = Rn + h
        rn_eff = Re + h
        w_en_n = np.array(
            [ve / rn_eff, -vn / rm_eff, -ve * sinphi / cosphi / rn_eff],
            dtype=np.float64,
        )
        w_ie_n = np.array(
            [GNSS_OMEGA_EARTH * cosphi, 0.0, -GNSS_OMEGA_EARTH * sinphi],
            dtype=np.float64,
        )

        # calculate rotation rate of body frame w.r.t nav frame
        sinyaw = np.sin(self.__att[0])
        cosyaw = np.cos(self.__att[0])
        w_nb_n = np.array(
            [
                -sinyaw * self.__att_dot[1] + self.__C_b_n[0, 0] * self.__att_dot[2],
                cosyaw * self.__att_dot[1] + self.__C_b_n[1, 0] * self.__att_dot[2],
                self.__att_dot[0] + self.__C_b_n[2, 0] * self.__att_dot[2],
            ],
            dtype=np.float64,
        )

        vel_dot_n = self.__C_b_n @ self.__vel_dot_b + np.cross(w_nb_n, vel_n)  # velocity derivative
        pos_dot_n = np.array([vn / rm_eff, ve / rn_eff / cosphi, -vd], dtype=np.float64)  # position derivative

        gyr = C_n_b @ (w_nb_n + w_en_n + w_ie_n)  # gyroscope output
        w_ie_b = C_n_b @ w_ie_n
        acc = self.__vel_dot_b + np.cross(w_ie_b + gyr, self.__vel_b) - C_n_b @ gravity  # accelerometer output

        return acc, gyr, vel_dot_n, pos_dot_n


# --------------------------------------------------------------------------------------------------#
@njit(cache=True, fastmath=True)
def eulzyx2dcm(v: np.ndarray) -> np.ndarray:
    """convert a set of euler angles (yaw, pitch, roll) into a rotation matrix
        ZYX rotation of ZYX ordered euler angles

    Parameters
    ----------
    v : np.ndarray
        euler angles

    Returns
    -------
    np.ndarray
        direction cosine matrix
    """
    sinpsi, sinth, sinphi = np.sin(v)
    cospsi, costh, cosphi = np.cos(v)

    return np.array(
        [
            [costh * cospsi, costh * sinpsi, -sinth],
            [
                sinphi * sinth * cospsi - cosphi * sinpsi,
                sinphi * sinth * sinpsi + cosphi * cospsi,
                costh * sinphi,
            ],
            [
                sinth * cosphi * cospsi + sinpsi * sinphi,
                sinth * cosphi * sinpsi - cospsi * sinphi,
                costh * cosphi,
            ],
        ]
    )
