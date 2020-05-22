#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import copy
import math

class Quad():

    def __init__(self, pos_x, pos_y, pos_z, vel_x,  vel_y, vel_z,
                 phi, theta, psi):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.vel_x = vel_x
        self.vel_y = vel_y
        self.vel_z = vel_z
        self.phi = phi
        self.theta = theta
        self.psi = psi

        self.dt = 0.02
        self.m = 0.6
        self.l = 0.2
        self.g = 9.81
        self.k1 = np.array([0.1, 0.1, 0.1])
        self.k2 = np.array([0.1, 0.1, 0.1])

    def next_state(self, state, act, cur_t,  act_noise=0.1):
        '''
        Dynamics for quad
        '''

        act = act.reshape([4,])
        state = state.reshape([9,])
        if act_noise:
            act = np.random.normal(act, act_noise)
        thrust = act[0]
        omega = act[1:]
        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, phi, theta, psi] = state
        wind = self.wind(pos_x, pos_y, pos_z, cur_t)
        Rotation = np.array([
            np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi),
            np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi),
            np.cos(theta) * np.cos(phi)
        ])
        omega2euler = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin((phi))],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])
        accel = np.array([0,0,-self.g]) + (Rotation * thrust + wind - self.k1 * (np.array([vel_x,vel_y,vel_z]))) / (self.m*1)
        vel = np.array([vel_x,vel_y,vel_z]) + accel * self.dt
        angle_dot = np.dot(omega2euler,omega).squeeze()

        return np.squeeze(np.concatenate([vel,accel,angle_dot], axis=0))

    def getState(self):
        return np.array(copy.deepcopy([self.pos_x, self.pos_y, self.pos_z,
        self.vel_x, self.vel_y, self.vel_z,
        self.phi, self.theta, self.psi]))

    # -2 * np.sin(0.5 * t), 4 + 3 * np.cos(0.5 * t), .4 * t
    # Pick wind speed function.
    def wind(self, x, y, z, t):
        v = [
            0.3*self.g*self.m,
            0.3*self.g*self.m,
            0.1*self.g*self.m,
        ]
        return np.array(v)

    def set_state(self, state):
        [self.pos_x, self.pos_y, self.pos_z,
         self.vel_x, self.vel_y, self.vel_z,
         self.phi, self.theta, self.psi] = state


class quad_env():
    def __init__(self):
        # set params for quad
        self.L = 20
        self.t = 0
        self.g = 9.81
        self.dt = 0.02
        self.state_dim = 9
        self.action_dim = 4

        # self.curve_pos = lambda t: np.array([
        #     np.min([np.max([1.5 * (t - 0.02 * 200), 0]), 1.5 * 0.02 * 100])
        #     + np.min([np.max([1.5 * (t - 0.02 * 500), 0]), 1.5 * 0.02 * 100])
        #     + np.min([np.max([1.5 * (t - 0.02 * 800), 0]), 1.5 * 0.02 * 200]),
        #
        #     np.min([1.5 * t, 1.5 * 0.02 * 200])
        #     + np.max([np.min([-1.5 * (t - 0.02 * 300), 0]), -1.5 * 0.02 * 200])
        #     + np.min([np.max([1.5 * (t - 0.02 * 600), 0]), 1.5 * 0.02 * 200]),
        #
        #     np.min([1.5 * t, 1.5 * 0.02 * 200])
        #     + np.max([np.min([-1.5 * (t - 0.02 * 300), 0]), -1.5 * 0.02 * 200])
        #     + np.min([np.max([1.5 * (t - 0.02 * 600), 0]), 1.5 * 0.02 * 200])])
        #
        # self.curve_vel = lambda t: np.array([
        #     ((np.min([np.max([1.5 * (t + self.dt - 0.02 * 200), 0]), 1.5 * 0.02 * 100])
        #       + np.min([np.max([1.5 * (t + self.dt - 0.02 * 500), 0]), 1.5 * 0.02 * 100])
        #       + np.min([np.max([1.5 * (t + self.dt - 0.02 * 800), 0]), 1.5 * 0.02 * 200]))
        #      - (np.min([np.max([1.5 * (t - 0.02 * 200), 0]), 1.5 * 0.02 * 100])
        #         + np.min([np.max([1.5 * (t - 0.02 * 500), 0]), 1.5 * 0.02 * 100])
        #         + np.min([np.max([1.5 * (t - 0.02 * 800), 0]), 1.5 * 0.02 * 200]))) / self.dt,
        #
        #     ((np.min([1.5 * (t + self.dt), 1.5 * 0.02 * 200])
        #       + np.max([np.min([-1.5 * (t + self.dt - 0.02 * 300), 0]), -1.5 * 0.02 * 200])
        #       + np.min([np.max([1.5 * (t + self.dt - 0.02 * 600), 0]), 1.5 * 0.02 * 200]))
        #      - (np.min([1.5 * t, 1.5 * 0.02 * 200])
        #         + np.max([np.min([-1.5 * (t - 0.02 * 300), 0]), -1.5 * 0.02 * 200])
        #         + np.min([np.max([1.5 * (t - 0.02 * 600), 0]), 1.5 * 0.02 * 200]))) / self.dt,
        #
        #     ((np.min([1.5 * (t + self.dt), 1.5 * 0.02 * 200])
        #       + np.max([np.min([-1.5 * (t + self.dt - 0.02 * 300), 0]), -1.5 * 0.02 * 200])
        #       + np.min([np.max([1.5 * (t + self.dt - 0.02 * 600), 0]), 1.5 * 0.02 * 200]))
        #      - (np.min([1.5 * t, 1.5 * 0.02 * 200])
        #         + np.max([np.min([-1.5 * (t - 0.02 * 300), 0]), -1.5 * 0.02 * 200])
        #         + np.min([np.max([1.5 * (t - 0.02 * 600), 0]), 1.5 * 0.02 * 200]))) / self.dt,
        # ])
        # #
        # self.curve_accel_0 = np.array([0, 0.5, 0.5])


        #self.curve_pos = lambda t: np.array([-2*np.sin(0.5*t),4 + 3*np.cos(0.5*t),.4*t])
        #self.curve_vel = lambda t: np.array([2*(-np.sin(0.5 * (t+self.dt)) + np.sin(0.5*t))/self.dt,
        #                                     3*(np.cos(0.5 * (t+self.dt)) - np.cos(0.5*t))/self.dt,
        #                                     .4])
        self.curve_accel_0 = np.array([0, 0, 0])
        
        self.curve_vel = lambda t: (self.curve_pos(t+self.dt)-self.curve_pos(t))/self.dt
        
        self.init_pos = self.curve_pos(0)
        self.init_pos = np.array([0,0,0])
        self.init_vel = self.curve_vel(0)
        self.init_vel = np.array([0,0,0])
        self.init_accel = np.array([0, 0, 0])

        self.psi_d = 0
        self.init_theta,self.init_phi = self.cal_theta_phi(self.init_accel, self.psi_d)
        self.quad = Quad(self.init_pos[0], self.init_pos[1], self.init_pos[2],
                         self.init_vel[0], self.init_vel[1], self.init_vel[2],
                         self.init_phi, self.init_theta, self.psi_d)
        self.cur_angle = np.array([self.init_phi, self.init_theta, self.psi_d])
        self.barrier = np.array([
            self.curve_pos(4),
            self.curve_pos(12),
            self.curve_pos(20),
        ])
        self.obstacles = []
        self.obstacle_without_center = []
        '''obstacle1'''
        # center1 = np.array(
        #     [[6, 0.15, 0.15]]).reshape([3, 1])
        # scale = 1
        # pts = np.min([np.random.random([3, 8]) * 3, 1.8 * np.ones([3, 8])], axis=0)
        # pts = np.max([pts, 0.8 * np.ones([3, 8])], axis=0)
        # pts = pts - np.mean(pts, axis=1)[:, np.newaxis]
        # pts1 = scale * pts + center1
        # self.obstacles.append(pts1)
        '''obstacle2'''

        center1 = np.array([2,7,4])
        # center2 = np.array([-2*np.sin(0.5*10*1.3),4.5 + 3*np.cos(0.5*10*1.2),.4*10*1.2])
        # # center2 = np.array([-2*np.sin(0.5*10*1.2),4 + 3*np.cos(0.5*10*1.2),.4*10*1.0])
        # center3 = np.array([-2*np.sin(0.5*10*1.45)-0.5,4 + 3*np.cos(0.5*10*1.45)+0.8,.4*10*1.45-0.2])
        # center4 = np.array([-2*np.sin(0.5*10*1.45)-0.5,4 + 3*np.cos(0.5*10*1.45)+0.8,.4*10*1.45-0.2])

        # center4 = np.array([-2*np.sin(0.5*10*1.5),4 + 3*np.cos(0.5*10*1.5)-0.5,.4*10*1.5])
        scale = 1
        pts3 = np.array([[2,4,3.5,2,2,2.5,2,2],[6,4,5,6,7,7,4,10],[0,0,0,10,10,10,7,2]])
        pts1 = np.array([[0.5,1.0,0.25,0.2,0.2,0.5,0,0.40],[2,2.5,4,6,2.6,2.4,3.4,5.4],[0,0,0,0,8,8,8,8]])
        pts2 = np.array([[0,0,2,2,0,0,2,2,0,0,2,2,0,0,2,2],[8.5,8,8,7.5,8,7.8,8,8,8.5,8.5,9.5,9.5,9.5,9,9,9],[0,0,0,0,8,8,8,8,0,0,0,0,6,6,6,6]])
        pts4 = np.array([[-3.0,-2.7,-3.0,-1.5,-2.5,-3.0,-3,-3.2,-2.9,-2.8,-2.5,-3.0],[1,2,7,1,3,7,2,2,1,3,1,1],[0,2,0,0,1,0,5,5,6,6,6,6]])
        pts5 = np.array([[-1,-2,-1,-2,-1,-2,-1,-2],[8,7,6.5,7,8.5,7.5,7,7],[0,0,8,8,0,0,8,8]])
        # pts5 = np.array([[-1,1,-1,1,-1,1,-1,1],[4,6,4,6,4,6,4,6],[0,0,8,8,0,0,8,8]])
        # pts1 = pts1 - np.mean(pts1, axis=1)[:, np.newaxis]
        # pts2 = pts2 - np.mean(pts2, axis=1)[:, np.newaxis]
        # pts3 = pts3 - np.mean(pts3, axis=1)[:, np.newaxis]
        # pts4 = pts4 - np.mean(pts4, axis=1)[:, np.newaxis]
        # pts1 = scale * pts4 + center1[:, np.newaxis]
        # pts2 = scale * pts2 + center2[:, np.newaxis]
        # pts3 = scale * pts3 + center3[:, np.newaxis]
        # pts4 = scale * pts4 + center4[:, np.newaxis]
        #self.obstacles.append(pts1)
        #self.obstacles.append(pts2)
        #self.obstacles.append(pts3)
        #self.obstacles.append(pts4)
        #self.obstacles.append(pts5)
        self.obstacles.append(np.array([[100],[100],[100]]))
    def cal_theta_phi(self, accel, psi_d):
        x_dd, y_dd, z_dd = accel
        belta_a = x_dd * np.cos(psi_d) + y_dd * np.sin(psi_d)
        belta_b = z_dd + self.g
        belta_c = x_dd * np.sin(psi_d) - y_dd * np.cos(psi_d)
        theta_d = np.arctan2(belta_a,belta_b)
        phi_d = np.arctan2(belta_c, np.sqrt(belta_a ** 2 + belta_b ** 2))
        return theta_d, phi_d
    def curve_pos(self, t):
        # return np.array([math.sqrt(3)*math.sin(t),math.sqrt(3)*(1-math.cos(t)),1*t])/2
        return np.array([1*t,0,2])/2


        # if t <= 10:
        #     return np.array([0.5*t,0,0.2])
        # elif t <= 20:
        #     return np.array([5+math.sin(0.5*t-5),1-math.cos(0.5*t-5),0.2*t-1])
        # elif t <= 25:
        #     return np.array([0.5*math.cos(5)*(t-20)+5+math.sin(5),0.5*math.sin(5)*(t-20)+1-math.cos(5),0.2*t-1])
        # else:
        #     return np.array([0.5*(t-25)+5/2*math.cos(5)+5+math.sin(5),0.5*(t-25)+5/2*math.sin(5)+1-math.cos(5),4])


    def getReward(self, action):
        s = self.quad.getState()
        r = np.sum(abs(action),axis=0)
        return r

    def reset(self, state=None):
        self.t = 0
        # self.obstacles[0] = self.obstacle_without_center[0] + self.curve_pos(max(0, 4 - self.t)).reshape(-1,1)
        # self.obstacles[1] = self.obstacle_without_center[1] + self.curve_pos(max(0, 12 - self.t)).reshape(-1,1)
        # self.obstacles[2] = self.obstacle_without_center[2] + self.curve_pos(max(0, 20 - self.t)).reshape(-1,1)
        # [a,b,c] = np.random.random([3,])
        # a,b,c = 0.25, 0.25, 0.25
        # self.curve_pos = lambda t: np.array([(a+b)*t,(c+b)*t,(a+c)*t])
        # self.curve_vel = lambda t: np.array([(a+b),(c+b),(a+c)])
        # self.curve_accel = lambda t: np.array([0, 0, 0])
        #self.init_pos = self.curve_pos(0)
        #self.init_vel = self.curve_vel(0)
        #self.init_accel = self.curve_accel_0
        self.psi_d = 0
        #self.init_theta,self.init_phi = self.cal_theta_phi(self.init_accel, self.psi_d)
        self.cur_angle = np.array([self.init_phi, self.init_theta, self.psi_d])
        if not state:
            self.quad.set_state([self.init_pos[0]+0., self.init_pos[1]+0., self.init_pos[2]-0.,
                                self.init_vel[0],self.init_vel[1],self.init_vel[2],0,0,0])
        else:
            state = np.squeeze(state)
            self.quad.set_state(state)
        return self.quad.getState()

    def step(self, action):
        # Runge-Kutta methods
        state = self.quad.getState()
        k1 = np.array(self.quad.next_state(state, action, self.t, 0) * self.quad.dt)
        # k2 = np.array(self.quad.next_state(state + k1/2.0, action) * self.quad.dt)
        # k3 = np.array(self.quad.next_state(state + k2/2.0, action) * self.quad.dt)
        # k4 = np.array(self.quad.next_state(state + k3, action) * self.quad.dt)
        r = self.getReward(action)
        self.t = self.t + self.quad.dt
        flag = False
        self.quad.set_state(state + (k1))# + 2.0 * (k2 + k3) + k4) / 6.0)

        # self.obstacles[0] = self.obstacle_without_center[0] + self.curve_pos(max(0, 4 - self.t)).reshape(-1,1)
        # self.obstacles[1] = self.obstacle_without_center[1] + self.curve_pos(max(0, 12 - self.t)).reshape(-1,1)
        # self.obstacles[2] = self.obstacle_without_center[2] + self.curve_pos(max(0, 20 - self.t)).reshape(-1,1)

        return self.quad.getState(), r, flag

    def predict_f_g(self, obs):
        # Params
        dt = self.quad.dt
        m = self.quad.m
        g = self.quad.g
        dO = self.state_dim
        obs = obs.reshape(-1, dO)

        [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z, phi, theta, psi] = obs.T

        sample_num = obs.shape[0]
        # calculate f with size [-1,9,1]
        f = np.concatenate([
            np.array(vel_x).reshape(sample_num, 1),
            np.array(vel_y).reshape(sample_num, 1),
            np.array(vel_z - g * dt).reshape(sample_num, 1),
            np.zeros([sample_num, 1]),
            np.zeros([sample_num, 1]),
            -g * np.ones([sample_num, 1]).reshape(-1, 1),
            np.zeros([sample_num,3]),
        ], axis=1)
        f = f * dt + obs
        f = f.reshape([-1, dO, 1])
        # calculate g with size [-1,9,4]
        accel_x = np.concatenate([
            (np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi)).reshape([-1, 1, 1]) / m,
            np.zeros([sample_num, 1, 3])
        ], axis=2)
        accel_y = np.concatenate([
            (np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi)).reshape([-1, 1, 1]) / m,
            np.zeros([sample_num, 1, 3])
        ], axis=2)
        accel_z = np.concatenate([
            (np.cos(theta) * np.cos(phi)).reshape([-1, 1, 1]) / m,
            np.zeros([sample_num, 1, 3])
        ], axis=2)
        phi_dot = np.concatenate([
            np.zeros([sample_num, 1, 1]),
            np.ones([sample_num, 1, 1]),
            (np.sin(phi)*np.tan(theta)).reshape([-1,1,1]),
            (np.cos(phi)*np.tan(theta)).reshape([-1,1,1])
        ], axis=2)
        theta_dot = np.concatenate([
            np.zeros([sample_num, 1, 2]),
            np.cos(phi).reshape([-1,1,1]),
            -np.sin(phi).reshape([-1,1,1])
        ], axis=2)
        psi_dot = np.concatenate([
            np.zeros([sample_num, 1, 2]),
            (np.sin(phi)/np.cos(theta)).reshape([-1,1,1]),
            (np.cos(phi)/np.cos(theta)).reshape([-1,1,1])
        ], axis=2)
        g_mat = np.concatenate([
            accel_x * dt,
            accel_y * dt,
            accel_z * dt,
            accel_x,
            accel_y,
            accel_z,
            phi_dot,
            theta_dot,
            psi_dot
        ], axis=1)
        g_mat = dt * g_mat
        return f, g_mat, np.copy(obs)

    def _predict_next_obs_uncertainty(self, obs, cur_acs,GP_model=None):
        f, g, x = self.predict_f_g(obs)
        cur_acs = np.asarray(cur_acs).reshape([-1, 4, 1])
        if GP_model:
            #for i in range(f.shape[0]):
                m1 = GP_model[0].predict(x[:,:6], return_std=False)
                m2 = GP_model[1].predict(x[:,:6], return_std=False)
                m3 = GP_model[2].predict(x[:,:6], return_std=False)
                m4 = GP_model[3].predict(x[:,:6], return_std=False)
                m5 = GP_model[4].predict(x[:,:6], return_std=False)
                m6 = GP_model[5].predict(x[:,:6], return_std=False)
                f[:,0,:] += m1.reshape([-1,1])
                f[:,1,:] += m2.reshape([-1,1])
                f[:,2,:] += m3.reshape([-1,1])
                f[:,3,:] += m4.reshape([-1,1])
                f[:,4,:] += m5.reshape([-1,1])
                f[:,5,:] += m6.reshape([-1,1])

        next_obs = f + np.matmul(g, cur_acs)

        return next_obs

    def cal_u1(self,state,state_dsr):
        vector = self.direction(state,state_dsr)[0]
        u1 = np.linalg.norm(vector)
        return u1

    def direction(self,state,state_dsr,gp_prediction=np.zeros(6,)):
        dt = self.quad.dt
        g = self.quad.g
        m = self.quad.m
        pos_direction = ((state_dsr[:3] - state[:3] - gp_prediction[:3] - state[3:6] * dt) / dt ** 2 + np.array([0, 0, g])) * m
        if state_dsr.shape[0] == 3:
            return pos_direction, None
        vel_direction = ((state_dsr[3:6] - gp_prediction[3:6] - state[3:6]) / dt + np.array([0, 0, g])) * m
        return pos_direction, vel_direction

    def cost_func(self, action, state, plan_hor, dO, dU, t0,GP_model=None):
        state = np.asarray(state).reshape((dO, ))
        action = np.asarray(action).reshape((-1, plan_hor, dU))

        init_obs = np.tile(state, (action.shape[0],1))
        init_cost = np.zeros((action.shape[0],))
        dt = self.quad.dt
        # t0 = int(t0//dt)
        # plan_hor = min(plan_hor, self.curve.shape[0] - t0)
        for i in range(plan_hor):
            cur_acs = action[:, i, :].reshape(-1, dU, 1)
            next_obs = self._predict_next_obs_uncertainty(init_obs, cur_acs,GP_model)
            init_obs = np.squeeze(next_obs)
            # Here to set the tracking curve
            # init_cost += np.sum(np.square(target[:3].T - init_obs[:,[0,1,2,]]), axis=1) * np.exp(-i*0.01)

            target = self.curve_pos(t0+i*dt)
            target_dot = self.curve_vel(t0+i*dt)
            k_p = 8.5
            k_v = 1.5
            init_cost += k_p*np.sum(np.square(target[:3] - init_obs[:, :3]), axis=1) + k_v*np.sum(np.square(target_dot[:3] - init_obs[:, 3:6]), axis=1)
            #for b in self.barrier:
            #    temp = (np.sqrt(np.sum(np.square(init_obs[:, :3] - b[:3]), axis=1)) <= b[3] + self.quad.l)
            #    init_cost += temp*100000
            #init_cost + (init_obs[:, 2] <= 0) * 100000
        cost = init_cost
        return cost
