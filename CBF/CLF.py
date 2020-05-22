#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from CBF.GP import GP
import math
import irispy

# Build barrier function model
def build_barrier(self):
    N = self.action_space
    self.P_angle = matrix(np.diag([1., 1., 1., 1., 1e20, 1e22]), tc='d')
    self.P_pos = matrix(np.diag([1., 1., 1., 1., 1e20, 1e18, 1e25]), tc='d')
    self.q_angle = matrix(np.zeros(N + 2))
    self.q_pos = matrix(np.zeros(N + 3))

def cal_angle(self, state, state_dsr,gp_prediction):
    pos_vec,vel_vec = self.env.direction(state, state_dsr,gp_prediction)
    pos_vec = pos_vec / np.linalg.norm(pos_vec)
    vel_vec = vel_vec / np.linalg.norm(vel_vec)
    psi = [0]
    phi = np.arcsin(-pos_vec[1])
    theta = np.arcsin((pos_vec[0] / np.cos(phi)))
    pos_angle = np.concatenate([phi, theta, psi])
    phi = np.arcsin(-vel_vec[1])
    theta = np.arcsin((vel_vec[0] / np.cos(phi)))
    vel_angle = np.concatenate([phi, theta, psi])
    return pos_angle, vel_angle

# Get compensatory action based on satisfaction of barrier function
def control_barrier(self, obs, u_rl, f, g, x, std, t):
    step_time = t
    dt = self.env.dt
    ''' Recorded curve '''
    # curve = self.env.curve
    # cur_pt = curve[t,:]
    # next_pt = curve[t+1,:]
    # third_pt = curve[t+2,:]
    ''' Parametric curve '''
    cur_pt = np.concatenate([self.env.curve_pos(step_time*dt), self.env.curve_vel(step_time*dt)])
    next_pt = np.concatenate([self.env.curve_pos((step_time + 1)*dt), self.env.curve_vel((step_time + 1)*dt)])
    third_pt = np.concatenate([self.env.curve_pos((step_time + 2)*dt), self.env.curve_vel((step_time + 2)*dt)])

    # Set up Quadratic Program to satisfy the Control Barrier Function
    kd = 2.0
    # calculate energy function V_t
    v_t_pos = np.abs(cur_pt - obs[:6])
    v_t_angle = np.abs(self.env.cur_angle - obs[6:9])
    N = self.observation_space
    M = self.action_space
    '''The main hyperparameter'''
    # gamma 越小（甚至为0）,能量衰减越慢, 震荡越小, 设置过小会导致跟踪收敛慢
    gamma_pos_clf = 0.1
    gamma_angle_clf = 0.3
    # gamma_pos_cbf 越小, 在安全域内越不允许往更不安全的地方探索, 在安全域外则更快收敛到安全域内, 因此应根据实际场景调整设置
    gamma_pos_cbf = 100
    gamma_angle_cbf = 1

    up_b = self.action_bound_up.reshape([M, 1])##
    low_b = self.action_bound_low.reshape([M, 1])##
    f = np.reshape(f, [N, 1])
    g = np.reshape(g, [N, M])
    # std = np.reshape(std, [N, 1])
    std = np.zeros([N, 1])
    u_rl = np.reshape(u_rl, [M, 1])#
    x = np.reshape(x, [N, 1])
    ''' q is the thrust direction'''
    q = np.array([math.cos(x[6])*math.sin(x[7]), -math.sin(x[6]), math.cos(x[6])*math.cos(x[7])]).reshape(([3,1]))

    '''QP 1 : use CLF & CBF to solve thrust'''
    '''v1(x) = |x-x_g| ,v1(y) = |y-y_g|, v1(z) = |z-z_g|'''
    '''h1(x) = 1 - (X^T * A * X + b^T * X + C)'''

    if (step_time) % 1 == 0:
        ''' 
            key 1: the setting of the bound, large bound will make the quad easier to avoid the moving obstacle 
            small bound will enable the quad to cross complex area, the quad will more flexible 
        '''
        bounds = irispy.Polyhedron.from_bounds(obs[:3] - 2.5*self.env.quad.l*np.ones(3,), obs[:3] + 2.5*self.env.quad.l*np.ones(3,))
        obstacles = self.env.obstacles
        start = obs[:3]
        region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)
        iter_result = debug.iterRegions()
        ellipsoid = list(iter_result)[-1][1]
        mapping_matrix = ellipsoid.getC() * 0.9
        center_pt = ellipsoid.getD().reshape([-1,1])
        inv_mapping_matrix = np.linalg.inv(mapping_matrix)
        A_mat = np.dot(inv_mapping_matrix.T, inv_mapping_matrix)
        b_vec = -2*A_mat.dot(center_pt)
        c = np.dot(center_pt.T, A_mat).dot(center_pt)
        '''
            key 2: the gamma is different when the quad inside or outside the safe region.
            When outside the safe region, the gamma should be quite large.
            When inside the safe region, the gamma should be quite small. Besides, the gamma will become larger when the
            quad leave the center.
        '''
        if (1 - x[:3,:].T.dot(A_mat).dot(x[:3,:]) - b_vec.T.dot(x[:3,:]) - c) < 0:
            gamma_pos_cbf = 100
            print('out of safe region')
        else:
            gamma_pos_cbf = (1 - x[:3,:].T.dot(A_mat).dot(x[:3,:]) - b_vec.T.dot(x[:3,:]) - c) / 5
        self.safe_region = [A_mat, b_vec, c]
        self.ellipsoid = [mapping_matrix, center_pt]

    [A_mat, b_vec, c] = self.safe_region
    print(1 - x[:3, :].T.dot(A_mat).dot(x[:3, :]) - b_vec.T.dot(x[:3, :]) - c)

    G_pos = np.concatenate([np.concatenate([
        g[0, :].reshape([1, -1]),
        -g[0, :].reshape([1, -1]),
        g[1, :].reshape([1, -1]),
        -g[1, :].reshape([1, -1]),
        g[2, :].reshape([1, -1]),
        -g[2, :].reshape([1, -1]),
        g[3, :].reshape([1, -1]),
        -g[3, :].reshape([1, -1]),
        g[4, :].reshape([1, -1]),
        -g[4, :].reshape([1, -1]),
        g[5, :].reshape([1, -1]),
        -g[5, :].reshape([1, -1]),
        (2 * A_mat.dot(x[:3,:]) + b_vec) * g[:3,:],
        np.eye(M), -np.eye(M)], axis=0),
        np.concatenate([-np.ones([6, 1]), np.zeros([6, 1]), np.zeros([3, 1]), np.zeros([2 * M, 1])], axis=0),
        np.concatenate([np.zeros([6, 1]), -np.ones([6, 1]), np.zeros([3, 1]), np.zeros([2 * M, 1])], axis=0),
        np.concatenate([np.zeros([6,1]), np.zeros([6, 1]), -np.ones([3, 1]), np.zeros([2 * M, 1])], axis=0)], axis=1)
    h_pos = np.concatenate([
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[0] - f[0] + next_pt[0]) - abs(kd*std[0]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[0] + f[0] - next_pt[0]) - abs(kd*std[0]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[1] - f[1] + next_pt[1]) - abs(kd*std[1]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[1] + f[1] - next_pt[1]) - abs(kd*std[1]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[2] - f[2] + next_pt[2]) - abs(kd*std[2]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[2] + f[2] - next_pt[2]) - abs(kd*std[2]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[3] - f[3] + next_pt[3]) - abs(kd*std[3]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[3] + f[3] - next_pt[3]) - abs(kd*std[3]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[4] - f[4] + next_pt[4]) - abs(kd*std[4]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[4] + f[4] - next_pt[4]) - abs(kd*std[4]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[5] - f[5] + next_pt[5]) - abs(kd*std[5]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[5] + f[5] - next_pt[5]) - abs(kd*std[5]),
        np.ones([1, 1]) * (gamma_pos_cbf * dt * (1 - x[:3,:].T.dot(A_mat).dot(x[:3,:]) - b_vec.T.dot(x[:3,:]) - c) - (2 * A_mat.dot(x[:3,:]) + b_vec) * (f[:3,:] - x[:3,:]) - abs((2 * A_mat.dot(x[:3,:]) + b_vec)*kd*std[:3,:])),
        up_b,
        - low_b], axis=0)
    G_pos = matrix(G_pos, tc='d')
    h_pos = matrix(h_pos, tc='d')
    # Solve QP
    solvers.options['show_progress'] = False
    sol = solvers.qp(P=self.P_pos, q=self.q_pos, G=G_pos, h=h_pos)
    thrust = sol['x'][0]
    # predict new pos and vel to calculate new angle
    predict_xyz = f + np.dot(g, thrust * np.ones([4, 1]))
    gp_prediction = GP.get_GP_prediction(self,predict_xyz[:6])
    # gp_prediction = np.zeros(6)
    pos_angle, vel_angle = cal_angle(self, predict_xyz[:6].squeeze(), third_pt, gp_prediction)
    weight = 0.7
    [phi_d, theta_d, psi_d] = weight*pos_angle + (1-weight)*vel_angle
    self.env.cur_angle = np.array([phi_d, theta_d, psi_d])

    '''QP 2 : use CLF to approximate the desired angle as well as CBF to escape from the obstacle'''
    '''v2(phi) = |phi-phi_d| ,v2(theta) = |theta-theta_d|, v2(psi) = |psi-psi_d|'''
    '''h2(x) = r^T * q - 1'''
    r = center_pt - x[:3,:]
    r_q = np.squeeze(np.dot(r.T, q))
    # print('r^T * q', r_q)
    r_dot = - (predict_xyz[:3, :] - x[:3, :]) / dt
    q_dot_mat = np.array([
        [-math.sin(x[6])*math.sin(x[7]), math.cos(x[6])*math.cos(x[7]), 0],
        [-math.cos(x[6]), 0, 0],
        [-math.sin(x[6])*math.cos(x[7]), -math.cos(x[6])*math.sin(x[7]), 0]
    ])
    r_q_dot_mat = np.dot(r.T, q_dot_mat)

    G_angle = np.concatenate([np.concatenate([
                                        g[6,:].reshape([1,-1]),
                                        -g[6, :].reshape([1, -1]),
                                        g[7,:].reshape([1,-1]),
                                        -g[7, :].reshape([1, -1]),
                                        g[8,:].reshape([1,-1]),
                                        -g[8, :].reshape([1, -1]),
                                        -np.dot(r_q_dot_mat, g[6:, :]).reshape([1, -1]),
                                        np.eye(M), -np.eye(M)], axis=0),
                        np.concatenate([-np.ones([6, 1]), np.zeros([1, 1]), np.zeros([2 * M, 1])], axis=0),
                        np.concatenate([np.zeros([6, 1]), -np.ones([1, 1]), np.zeros([2 * M, 1])], axis=0),
                        ], axis=1)
    h_angle = np.concatenate([
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[0] - f[6] + phi_d - np.dot(g[6,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[0] + f[6] - phi_d + np.dot(g[6,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[1] - f[7] + theta_d - np.dot(g[7,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[1] + f[7] - theta_d + np.dot(g[7,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[2] - f[8] + psi_d - np.dot(g[8,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((1-gamma_angle_clf) * v_t_angle[2] + f[8] - psi_d + np.dot(g[8,:].reshape([1,M]), u_rl)),
                        np.ones([1,1]) * ((gamma_angle_cbf * (r_q - 1) * dt + np.dot(r_dot.T, q)) * dt + np.dot(r_q_dot_mat, (f[6:,:] + np.dot(g[6:,:], u_rl) - x[6:,:]))),
                        -u_rl + up_b,
                        u_rl - low_b], axis=0)

    h_angle = np.squeeze(h_angle).astype(np.double)
    # Convert numpy arrays to cvx matrices to set up QP
    G_angle = matrix(G_angle, tc='d')
    h_angle = matrix(h_angle, tc='d')
    # Solve QP
    solvers.options['show_progress'] = False
    sol = solvers.qp(self.P_angle, self.q_angle, G=G_angle, h=h_angle)
    u_bar = np.squeeze(sol['x'])
    u_bar[0] = thrust - u_rl[0]

    return np.expand_dims(np.array(u_bar[:4]), 0),np.sum(v_t_pos),np.sum(v_t_angle)
