
"""
Autonomous flight library for Crazyflie2
"""
import os
import sys
sys.path.append("../lib")
import time
from threading import Thread, Timer, Lock
from multiprocessing import Process, Pipe

import termios
import contextlib
import numpy as np
import logging
logging.basicConfig(level=logging.ERROR)

import Sensors

from cflib.crazyflie import Crazyflie
import cflib.crtp
# from math import sin, cos, sqrt

from mpc import mpc
import math 
# from lqr import lqr
import CBF.CBF as cbf
from CBF.GP import GP
from CBF.learner import controller_init
import copy
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

@contextlib.contextmanager
def raw_mode(file):
    """ Implement termios module for k eyboard detection """
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
         termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)


class Crazy_Auto:
    """ Basic calls and functions to enable autonomous flight """  
    def __init__(self, link_uri):
        """ Initialize crazyflie using  passed in link"""
        self._cf = Crazyflie()
        self.t = Sensors.logs(self)
        # the three function calls below setup threads for connection monitoring
        self._cf.disconnected.add_callback(self._disconnected) #first monitor thread checking for disconnections
        self._cf.connection_failed.add_callback(self._connection_failed) #second monitor thread for checking for back connection to crazyflie
        self._cf.connection_lost.add_callback(self._connection_lost) # third monitor thread checking for lost connection
        print("Connecting to %s" % link_uri)
        self._cf.open_link(link_uri) #connects to crazyflie and downloads TOC/Params
        self.is_connected = True

        self.daemon = True
        self.timePrint = 0.0
        self.is_flying = False

        # Control Parm
        self.g = 9.81  # gravitational acc. [m/s^ 2]
        self.m = 0.04  # mass [kg]
        self.pi = 3.14
        self.num_state = 6
        self.num_action = 3
        self.hover_thrust = 36850.0
        self.thrust2input = 115000
        self.input2thrust = 11e-6

        # trans calculated control to command input
        self.K_pitch = 1
        self.K_roll = 1
        # self.K_thrust = self.hover_thrust / (self.m * self.g)
        self.K_thrust = 4000

        # Logged states - ,
        # log.position, log.velocity and log.attitude are all in the body frame of reference
        self.position = [0.0, 0.0, 0.0]  # [m] in the global frame of reference
        self.velocity = [0.0, 0.0, 0.0]  # [m/s] in the global frame of reference
        self.attitude = [0.0, 0.0, 0.0]  # [rad] Attitude (p,r,y) with inverted roll (r)

        # References
        self.position_reference = [0.0, 0.0, 0.0]  # [m] in the global frame
        self.yaw_reference = 0.0  # [rad] in the global rame

        # Increments
        self.position_increments = [0.1, 0.1, 0.1]  # [m]
        self.micro_height_increments = 0.01
        self.yaw_increment = 0.1  # [rad]

        # Limits
        self.thrust_limit = (0, 63000)
        # self.roll_limit = (-30.0, 30.0)
        # self.pitch_limit = (-30.0, 30.0)
        # self.yaw_limit = (-200.0, 200.0)
        self.roll_limit = (-30, 30)
        self.pitch_limit = (-30, 30)
        self.max_hor_vel = 1
        self.max_vert_vel = 0.1
        # Controller settings
        self.isEnabled = True
        self.rate = 50 # Hz

        # ref_euler
        self.cur_euler = [0,0,0]

    def predict_f_g(self, obs,phi=0,theta=0,psi=0):
        # Params
        dt = 1.0/self.rate
        m = self.m
        g = self.g
        dO = self.num_state
        obs = obs.reshape(-1, dO)

        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z] = obs.T

        sample_num = obs.shape[0]
        # calculate f with size [-1,6,1]
        f = np.concatenate([
            np.array(vel_x).reshape(sample_num, 1),
            np.array(vel_y).reshape(sample_num, 1),
            np.array(vel_z - g * dt).reshape(sample_num, 1),
            np.zeros([sample_num, 1]),
            np.zeros([sample_num, 1]),
            -g * np.ones([sample_num, 1]).reshape(-1, 1),
        ], axis=1)
        f = f * dt + obs
        f = f.reshape([-1, dO, 1])
        # calculate g with size [-1,6,3]
        accel_x = np.concatenate([
            (np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)).reshape([-1, 1, 1]) / m,
            np.zeros([sample_num, 1, 2])
        ], axis=2)
        accel_y = np.concatenate([
            (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)).reshape([-1, 1, 1]) / m,
            np.zeros([sample_num, 1, 2])
        ], axis=2)
        accel_z = np.concatenate([
            (np.cos(theta) * np.cos(phi)).reshape([-1, 1, 1]) / m,
            np.zeros([sample_num, 1, 2])
        ], axis=2)
        
        g_mat = np.concatenate([
            accel_x * dt, 
            accel_y * dt,
            accel_z * dt,
            accel_x,
            accel_y,
            accel_z,
        ], axis=1)
        g_mat = dt * g_mat
        return f, g_mat, np.copy(obs) 

    def _run_controller(self,gp_pipe,data_pipe):
        """ Main control loop """
        # Controller parameters

        horizon = 20

        # Set the current reference to the current positional estimate, at a
        # slight elevation
        # time.sleep(2)
        #self.position_reference = [0., 0, 0.8]
        self.curve = lambda t: [0,0.4*t,(0.8*t)/(0.8*t+1)] if t <= 10 else [0,4,8.0/9.0]
        self.curve_vel = lambda t: [0,0.4,0.8/(0.8*t+1)**2] if t <= 10 else [0,0,0]
        #self.curve = lambda t: [0,0,0.8] if t <= 15 else [0,0,0.8]
        #self.curve_vel = lambda t: [0,0,0] if t <= 15 else [0,0,0]


        # Unlock the controller, BE CAREFLUE!!!
        self._cf.commander.send_setpoint(0, 0, 0, 0)

        self.reference_data = []
        self.state_data = []
        self.control_data = []
        self.roll_data = []
        self.pitch_data = []
        self.yaw_data = []
        save_dir = 'data1' # will cover the old ones
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            os.makedirs(save_dir + '/training_data')
        step_count = 0
        eInt_vert = 0
        roll_r, pitch_r, yaw_r = 0,0,0

        # init controller
        controller_init(self)
        t0 = time.time()
        while True:

            timeStart = time.time()

            # # tracking
            # x_r, y_r, z_r = [0., -2., 0.2]
            # dx_r, dy_r, dz_r = [0.0, 0.0, 0.0]
            cur_t = time.time() - t0
            x_r, y_r, z_r = self.curve(cur_t)
            dx_r, dy_r, dz_r = self.curve_vel(cur_t)

            # x_r, y_r, z_r = [0., 0., 1.]
            # dx_r, dy_r, dz_r = [0.0, 0.0, 0.5]

            target = [np.array([x_r, y_r, z_r, dx_r, dy_r, dz_r])]
            target.append(np.array(self.curve(cur_t+1.0/self.rate)+self.curve_vel(cur_t+1.0/self.rate)))
            target.append(np.array(self.curve(cur_t+2.0/self.rate)+self.curve_vel(cur_t+2.0/self.rate)))

            # print("position_references", self.position_reference)
            #print("target: ", target[0])
            # Get measurements from the log
            x, y, z = self.position
            dx, dy, dz = self.velocity
            pitch, roll, yaw = np.array(self.t.attitude)/180*self.pi
            #weight = 0.0
            #roll = weight * roll_r + (1-weight) * roll
            #pitch = weight * pitch_r + (1-weight) * pitch
            #yaw = weight * yaw_r + (1-weight) * yaw
            state = np.array([x, y, z, dx, dy, dz])
            #print("state: ", state)
            #print("euler: ", roll,pitch,yaw)

            self.state_data.append(state)
            self.roll_data.append(roll)
            self.pitch_data.append(pitch)
            self.yaw_data.append(yaw)

            self.reference_data.append(target[0])

            # Compute control signal - map errors to control signals
            if self.isEnabled:
                if step_count <= 100:
                    f,g,x = self.predict_f_g(state,roll,pitch,yaw)
                    std = np.zeros([6, 1])
                else:
                    [f, mean, g, x, std] = GP.get_GP_dynamics(self,state,roll,pitch,yaw)
                    #print(mean)
                    std = np.zeros([6, 1])
                t_cbf = time.time()
                u_bar_, v_t_pos = cbf.control_barrier(self, np.squeeze(state), f, g, x, std, target, roll, pitch, None)
                [thrust_r, pitch_r, roll_r] = u_bar_
                
                #print("cbf time: ", time.time()-t_cbf)
                #thrust_r= u_bar_[0]
                
                roll_r = self.saturate(roll_r / self.pi * 180, self.roll_limit)
                pitch_r = self.saturate(pitch_r / self.pi * 180, self.pitch_limit)
                thrust_r = self.saturate((thrust_r ) * self.thrust2input,
                                         self.thrust_limit)  # minus, see notebook


                ## PID ##
                # HORI_POS_P = 1
                # HORI_POS_D = 0
                # HORI_VEL_P = 5
                #
                # VERT_POS_P = 0.1
                # VERT_POS_D = 0
                # # VERT_VEL_P = 0.05
                #
                # pos = np.array([x, y])
                # vel = np.array([dx, dy])
                # exp_pos = np.array([x_r, y_r])
                # u_vel = (exp_pos - pos) * HORI_POS_P - vel * HORI_POS_D
                # u_vel_length = np.linalg.norm(u_vel)
                # if u_vel_length > self.max_hor_vel:
                #     u_vel = u_vel/u_vel_length
                #     u_vel = u_vel * self.max_hor_vel
                #     print("norm hori")
                # u_horizon = (u_vel - vel) * HORI_VEL_P
                # roll_r = u_horizon[0]
                # pitch_r = u_horizon[1]
                # eInt_vert += z_r - z
                # vert = (z_r - z) * VERT_POS_P - z * VERT_POS_D
                # vert_length = np.linalg.norm(vert)
                # if vert_length > self.max_vert_vel:
                #     vert = vert/vert_length
                #     vert = vert * self.max_vert_vel
                #     print("norm vert")
                # # thrust_r = (vert - dz) * VERT_VEL_P
                # thrust_r = vert
                # roll_r = self.saturate(roll_r, self.roll_limit)
                # pitch_r = self.saturate(pitch_r, self.pitch_limit)
                # thrust_r = self.saturate((thrust_r + self.m * self.g) * self.thrust2input, self.thrust_limit)  # minus, see notebook

            else:
                # If the controller is disabled, send a zero-thrust
                roll_r, pitch_r, thrust_r = (0, 0, 0)
            yaw_r = 0
            #print("roll_r: ", roll_r/180*self.pi)
            #print("pitch_r: ", pitch_r/180*self.pi)
            #print("thrust_r: ", int(thrust_r))
            self._cf.commander.send_setpoint(roll_r, -pitch_r, yaw_r, int(thrust_r)) # change!!!
            # self._cf.commander.send_setpoint(roll_r, pitch_r, yaw_r, int(thrust_r))

            self.control_data.append(np.array([float(thrust_r)/self.thrust2input, roll_r, -pitch_r]))
            # test height control
            # self._cf.commander.send_setpoint(0, 0, 0, int(thrust_r)) # change!!!
            t_c = time.time()
            data_pipe.send([state,self.control_data[-1],roll,pitch,yaw,self.GP_model])
            if gp_pipe.poll(0.0001):            
                self.GP_model = gp_pipe.recv()
                print('Receive GP model!')
            step_count += 1
            if step_count >= 800 and step_count%100==0:
                np.save(save_dir + '/training_data/state' + str(step_count) + '.npy ', self.state_data)
                np.save(save_dir + '/training_data/ref' + str(step_count) + '.npy', self.reference_data)
                np.save(save_dir + '/training_data/ctrl' + str(step_count) + '.npy', self.control_data)



            '''
            # Compute control errors
            ex = x_r - x
            ey = y_r - y
            ez = z_r - z
            dex = dx_r - dx
            dey = dy_r - dy
            dez = dz_r - dz

            Kzp, Krp, Kpp, Kyp = (200.0, 20.0, 20.0, 10.0)  # Pretty haphazard tuning
            Kzd, Krd, Kpd = (2 * 2 * sqrt(Kzp), 2 * sqrt(Krp), 2 * sqrt(Kpp))

            # Compute control signal - map errors to control signals
            if self.isEnabled:
                ux = +self.saturate(Krp * ex + Krd * dex, self.pitch_limit)
                uy = -self.saturate(Kpp * ey + Kpd * dey, self.roll_limit)
                pitch_r = cos(yaw) * ux - sin(yaw) * uy
                roll_r = sin(yaw) * ux + cos(yaw) * uy
                # thrust_r = + self.saturate((Kzp * ez + Kzd * dez + self.m * self.g) * (self.hover_thrust/(self.m*self.g)/ (cos(roll) * cos(pitch))), self.thrust_limit)
                thrust_r = + self.saturate((Kzp * ez + self.m * self.g) * (self.hover_thrust / (self.m * self.g) / (cos(roll) * cos(pitch))), self.thrust_limit)

            else:
                # If the controller is disabled, send a zero-thrust
                roll_r, pitch_r, yaw_r, thrust_r = (0, 0, 0, 0)

            # Communicate a reference value to the Crazyflie
            # self._cf.commander.send_setpoint(roll_r, pitch_r, 0, int(thrust_r))
            self._cf.commander.send_setpoint(0, 0, 0, int(thrust_r))
            '''
            self.loop_sleep(timeStart) # to make sure not faster than 200Hz

    def update_vals(self):
        self.position = self.t.position
        self.velocity = self.t.velocity  # [m/s] in the global frame of reference
        self.attitude = self.t.attitude  # [rad] Attitude (p,r,y) with inverted roll (r)

        Timer(.1, self.update_vals).start()

    def saturate(self, value, limits):
        """ Saturates a given value to reside on the the interval 'limits'"""
        if value < limits[0]:
            value = limits[0]
        elif value > limits[1]:
            value = limits[1]
        return value

    def print_at_period(self, period, message):
        """ Prints the message at a given period """
        if (time.time() - period) > self.timePrint:
            self.timePrint = time.time()
            print(message)

    def loop_sleep(self, timeStart):
        """ Sleeps the control loop to make it run at a specified rate """
        deltaTime = 1.0 / float(self.rate) - (time.time() - timeStart)
        if deltaTime > 0:
            print("RealTime")
            time.sleep(deltaTime)

    def set_reference(self, message):
        """ Enables an incremental change in the reference and defines the
        keyboard mapping (change to your preference, but if so, also make sure
        to change the valid_keys attribute in the interface thread)"""
        verbose = True
        if message == "s":
            self.position_reference[1] -= self.position_increments[1]
            if verbose: print('-y')
        if message == "w":
            self.position_reference[1] += self.position_increments[1]
            if verbose: print('+y')
        if message == "d":
            self.position_reference[0] += self.position_increments[1]
            if verbose: print('+x')
        if message == "a":
            self.position_reference[0] -= self.position_increments[1]
            if verbose: print('-x')
        if message == "k":
            self.position_reference[2] -= self.position_increments[2]
            if verbose: print('-z')
        if message == "i":
            self.position_reference[2] += self.position_increments[2]
            if verbose: print('+z')
        if message == "n":
            self.position_reference[2] += self.micro_height_increments
            if verbose: print('-z')
        if message == "m":
            self.position_reference[2] += self.micro_height_increments
            if verbose: print('+z')
        # if message == "j":
        #     self.yaw_reference += self.yaw_increment
        # if message == "l":
        #     self.yaw_reference -= self.yaw_increment
        if message == "q":
            self.isEnabled = False
        if message == "e":
            self.isEnabled = True


### ------------------------------------------------------- Callbacks ----------------------------------------------------------------------###
    def _connected(self, link_uri):
        """ This callback is called form the Crazyflie API when a Crazyflie
        has been connected and the TOCs have been downloaded."""

        # Start a separate thread to do the motor test.
        # Do not hijack the calling thread!
        Thread(target=self.update_vals).start()
        print("Waiting for logs to initalize...")
        gp_pipe_a, gp_pipe_b = Pipe()
        data_pipe_a, data_pipe_b = Pipe()

        Process(target=update_gp,args=(gp_pipe_b, data_pipe_b)).start()
        data_pipe_a.recv()

        Thread(target=self._run_controller,args=(gp_pipe_a, data_pipe_a)).start()
        master = inputThread(self)


    def _connection_failed(self, link_uri, msg):
        """Callback when connection initial connection fails (i.e no Crazyflie
        at the speficied address)"""
        print("Connection to %s failed: %s" % (link_uri, msg))
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        """Callback when disconnected after a connection has been made (i.e
        Crazyflie moves out of range)"""
        print("Connection to %s lost: %s" % (link_uri, msg))
        self.is_connected = False

    def _disconnected(self, link_uri):
        """Callback when the Crazyflie is disconnected (called in all cases)"""

        print("Disconnected from %s" % link_uri)
        self.is_connected = False
### ------------------------------------------------------ END CALLBACKS -------------------------------------------------------------------###



class inputThread(Thread):
    """Create an input thread which sending references taken in increments from
    the keys "valid_characters" attribute. With the incremental directions (+,-)
    a mapping is done such that

        x-translation - controlled by ("w","s")
        y-translation - controlled by ("a","d")
        z-translation - controlled by ("i","k")
        yaw           - controlled by ("j","l")

    Furthermore, the controller can be enabled and disabled by

        disable - controlled by "q"
        enable  - controlled by "e"
    """
    def __init__(self, controller):
        Thread.__init__(self)
        self.valid_characters = ["a","d","s", "w","i","j","l","k","q","e"]
        self.daemon = True
        self.controller = controller
        self.start()

    def run(self):
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if ch in self.valid_characters:
                        self.controller.set_reference(ch)

            except (KeyboardInterrupt, EOFError):
                sys.exit(0)
                pass


def update_gp(gp_pipe,data_pipe):
    '''
    j0 = 0
    print("Update GP Thread")
    time.sleep(0.4)
    while True:
        j = len(self.state_data)
        print("gp timestep:",j)
        if j-j0 <= 10:
            time.sleep(0.5)
            continue
        print("update timestep:",j)
        X = np.concatenate(self.state_data[j0:j]).reshape((-1, 6))
        U = np.concatenate(self.control_data[j0:j]).reshape((-1, 3))
        R = np.array(self.roll_data[j0:j]).reshape((-1, 1))
        P = np.array(self.pitch_data[j0:j]).reshape((-1, 1))
        Y = np.array(self.yaw_data[j0:j]).reshape((-1, 1))
        GP_bak = copy.deepcopy(self.GP_model)

        L = X.shape[0]
        err = np.zeros((L - 1, 6))
        for i in range(L - 1):
            [f, g, x1] = self.predict_f_g(X[i, :],R[i,:],P[i,:],Y[i,:])
            f = np.squeeze(f)
            g = np.squeeze(g)
            err[i, :] = X[i + 1, :] - f - np.squeeze(np.matmul(g, U[i, :].reshape([3,1])))

        S = X[0:L - 1, :6]
        t1 = time.time()
        GP_bak[0].fit(S, err[:,:6])
        self.GP_model = GP_bak
        j0 = j
    '''
    j0 = 0
    print("Update GP Process")
    #time.sleep(0.4)
    data_pipe.send(None)
    X = []
    U = []
    R = []
    P = []
    Y = []    
    while True:
        data = data_pipe.recv()
        X.append(data[0])
        U.append(data[1])
        R.append(data[2])
        P.append(data[3])
        Y.append(data[4])
        j0 += 1
        if j0 % 20 == 0:
            GP_bak = copy.deepcopy(data[-1])
            L = len(X)
            err = np.zeros((L - 1, 6))
            for i in range(L - 1):
                [f, g, x1] = predict_f_g(X[i],R[i],P[i],Y[i])
                f = np.squeeze(f)
                g = np.squeeze(g)
                err[i, :] = X[i + 1] - f - np.squeeze(np.matmul(g, U[i].reshape([3,1])))
            X = np.concatenate(X).reshape([-1,6])
            S = X[0:L - 1, :6]
            t1 = time.time()
            GP_bak[0].fit(S, err[:,:6])
            print("step ",j0," update GP time: ",time.time()-t1)
            X = []
            U = []
            R = []
            P = []
            Y = []  
            gp_pipe.send(GP_bak)

def predict_f_g(obs,phi=0,theta=0,psi=0):
    # Params
    dt = 1.0/50
    m = 0.04
    g = 9.81
    dO = 6
    obs = obs.reshape(-1, dO)

    [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z] = obs.T

    sample_num = obs.shape[0]
    # calculate f with size [-1,6,1]
    f = np.concatenate([
        np.array(vel_x).reshape(sample_num, 1),
        np.array(vel_y).reshape(sample_num, 1),
        np.array(vel_z - g * dt).reshape(sample_num, 1),
        np.zeros([sample_num, 1]),
        np.zeros([sample_num, 1]),
        -g * np.ones([sample_num, 1]).reshape(-1, 1),
    ], axis=1)
    f = f * dt + obs
    f = f.reshape([-1, dO, 1])
    # calculate g with size [-1,6,3]
    accel_x = np.concatenate([
        (np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)).reshape([-1, 1, 1]) / m,
        np.zeros([sample_num, 1, 2])
    ], axis=2)
    accel_y = np.concatenate([
        (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)).reshape([-1, 1, 1]) / m,
        np.zeros([sample_num, 1, 2])
    ], axis=2)
    accel_z = np.concatenate([
        (np.cos(theta) * np.cos(phi)).reshape([-1, 1, 1]) / m,
        np.zeros([sample_num, 1, 2])
    ], axis=2)
    
    g_mat = np.concatenate([
        accel_x * dt, 
        accel_y * dt,
        accel_z * dt,
        accel_x,
        accel_y,
        accel_z,
    ], axis=1)
    g_mat = dt * g_mat
    return f, g_mat, np.copy(obs) 


if __name__ == '__main__':
    # Initialize the low-level drivers (don't list the debug drivers)
    cflib.crtp.init_drivers(enable_debug_driver=False)
    # Scan for Crazyflies and use the first one found
    print("Scanning interfaces for Crazyflies...")
    available = cflib.crtp.scan_interfaces()
    print("Crazyflies found:")
    for i in available:
        print(i[0])

    le = Crazy_Auto('radio://0/120/2M/E7E7E7E750')
    while le.is_connected:
        time.sleep(1)

    #if len(available) > 0:
    #    le = Crazy_Auto(available[0][0])
    #    while le.is_connected:
    #        time.sleep(1)

    else:
        print("No Crazyflies found, cannot run example")
