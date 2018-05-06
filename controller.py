"""
PID Controller

components:
    follow attitude commands
    gps commands and yaw
    waypoint following
"""
import numpy as np
from frame_utils import euler2RM

DRONE_MASS_KG = 0.5
GRAVITY = -9.81
MOI = np.array([0.005, 0.005, 0.01])
MAX_THRUST = 10.0
MAX_TORQUE = 1.0

class NonlinearController(object):

    def __init__(self):
        """Initialize the controller object and control gains"""
        # Position control gains
        self.kpPosXY = 6.5 #1
        self.kpPosZ = 4 #1
        self.KiPosZ = 20 #20

        # Velocity control gains
        self.kpVelXY = 3.1 #4
        self.kpVelZ = 1.5 #1.5

        # Angle control gains
        self.kpBank = 8 #8
        self.kpYaw = 4.5 #4.5

        # Angle rate gains
        self.kpPQR = 20, 20, 5

        return

    def trajectory_control(self, position_trajectory, yaw_trajectory, time_trajectory, current_time):
        """Generate a commanded position, velocity and yaw based on the trajectory

        Args:
            position_trajectory: list of 3-element numpy arrays, NED positions
            yaw_trajectory: list yaw commands in radians
            time_trajectory: list of times (in seconds) that correspond to the position and yaw commands
            current_time: float corresponding to the current time in seconds

        Returns: tuple (commanded position, commanded velocity, commanded yaw)

        """

        ind_min = np.argmin(np.abs(np.array(time_trajectory) - current_time))
        time_ref = time_trajectory[ind_min]


        if current_time < time_ref:
            position0 = position_trajectory[ind_min - 1]
            position1 = position_trajectory[ind_min]

            time0 = time_trajectory[ind_min - 1]
            time1 = time_trajectory[ind_min]
            yaw_cmd = yaw_trajectory[ind_min - 1]

        else:
            yaw_cmd = yaw_trajectory[ind_min]
            if ind_min >= len(position_trajectory) - 1:
                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min]

                time0 = 0.0
                time1 = 1.0
            else:

                position0 = position_trajectory[ind_min]
                position1 = position_trajectory[ind_min + 1]
                time0 = time_trajectory[ind_min]
                time1 = time_trajectory[ind_min + 1]

        position_cmd = (position1 - position0) * \
                        (current_time - time0) / (time1 - time0) + position0
        velocity_cmd = (position1 - position0) / (time1 - time0)


        return (position_cmd, velocity_cmd, yaw_cmd)

    def lateral_position_control(self, local_position_cmd, local_velocity_cmd, local_position, local_velocity,
                               acceleration_ff = np.array([0.0, 0.0])):
        """Generate horizontal acceleration commands for the vehicle in the local frame

        Args:
            local_position_cmd: desired 2D position in local frame [north, east]
            local_velocity_cmd: desired 2D velocity in local frame [north_velocity, east_velocity]
            local_position: vehicle position in the local frame [north, east]
            local_velocity: vehicle velocity in the local frame [north_velocity, east_velocity]
            acceleration_cmd: feedforward acceleration command

        Returns: desired vehicle 2D acceleration in the local frame [north, east]
        """
        #return np.array([0.0, 0.0])
        return self.kpPosXY*(local_position_cmd - local_position)+self.kpVelXY*(local_velocity_cmd - local_velocity)+acceleration_ff

    def altitude_control(self, altitude_cmd, vertical_velocity_cmd, altitude, vertical_velocity, attitude, acceleration_ff=0.0):
        """Generate vertical acceleration (thrust) command

        Args:
            altitude_cmd: desired vertical position (+up)
            vertical_velocity_cmd: desired vertical velocity (+up)
            altitude: vehicle vertical position (+up)
            vertical_velocity: vehicle vertical velocity (+up)
            attitude: the vehicle's current attitude, 3 element numpy array (roll, pitch, yaw) in radians
            acceleration_ff: feedforward acceleration command (+up)

        Returns: thrust command for the vehicle (+up)
        """
        #return 0.0
        rot_mat = euler2RM(*attitude)
        u1_bar = self.kpPosZ*(altitude_cmd - altitude) + self.kpVelZ * (vertical_velocity_cmd - vertical_velocity) + acceleration_ff
        c = (u1_bar - 0.0*GRAVITY) / rot_mat[2,2] * DRONE_MASS_KG
        c = np.clip(c, 0.0, MAX_THRUST)
        return c


    def roll_pitch_controller(self, acceleration_cmd, attitude, thrust_cmd):
        """ Generate the rollrate and pitchrate commands in the body frame

        Args:
            target_acceleration: 2-element numpy array (north_acceleration_cmd,east_acceleration_cmd) in m/s^2
            attitude: 3-element numpy array (roll, pitch, yaw) in radians
            thrust_cmd: vehicle thruts command in Newton

        Returns: 2-element numpy array, desired rollrate (p) and pitchrate (q) commands in radians/s
        """
        #return np.array([0.0, 0.0])
        if thrust_cmd > 0:
            MAX_TILT = 1.0
            c = -thrust_cmd / DRONE_MASS_KG # why -ve? did not understand fully but followed comments in forum
            bxc,byc = np.clip(acceleration_cmd/c, -MAX_TILT, MAX_TILT)
            rot_mat = euler2RM(*attitude)
            R13 = rot_mat[0,2]
            R23 = rot_mat[1,2]
            R33 = rot_mat[2,2]
            R11 = rot_mat[0,0]
            R22 = rot_mat[1,1]
            R12 = rot_mat[0,1]
            R21 = rot_mat[1,0]
            b_x_c_dot = self.kpBank*(bxc-R13)
            b_y_c_dot = self.kpBank*(byc-R23)
            p_c, q_c = 1./R33* np.matmul(np.array([[R21, -R11],[R22, -R12]]),np.array([b_x_c_dot, b_y_c_dot]))
            return np.array([p_c, q_c])
        else:
            return np.array([0.0, 0.0])


    def body_rate_control(self, body_rate_cmd, body_rate):
        """ Generate the roll, pitch, yaw moment commands in the body frame

        Args:
            body_rate_cmd: 3-element numpy array (p_cmd,q_cmd,r_cmd) in radians/second^2
            body_rate: 3-element numpy array (p,q,r) in radians/second^2

        Returns: 3-element numpy array, desired roll moment, pitch moment, and yaw moment commands in Newtons*meters
        """
        #return np.array([0.0, 0.0, 0.0])
        moment_cmd = MOI * self.kpPQR * (body_rate_cmd - body_rate)
        tau = np.linalg.norm(moment_cmd)
        if tau > MAX_TORQUE:
            moment_cmd = moment_cmd * MAX_TORQUE / tau
        return moment_cmd

    def yaw_control(self, yaw_cmd, yaw):
        """ Generate the target yawrate

        Args:
            yaw_cmd: desired vehicle yaw in radians
            yaw: vehicle yaw in radians

        Returns: target yawrate in radians/sec
        """
        #return 0.0
        yaw_error = yaw_cmd - yaw
        while yaw_error > np.pi:
            yaw_error -= 2.0*np.pi
        while yaw_error < -np.pi:
            yaw_error += 2.0*np.pi

        return self.kpYaw * yaw_error

