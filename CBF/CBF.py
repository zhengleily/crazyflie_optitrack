#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from CBF.GP import GP
import math

# Build barrier function model
def build_barrier(self):
    N = self.action_space
    self.P_angle = matrix(np.diag([1, 1., 1., 1e10]), tc='d')
    self.P_pos = matrix(np.diag([1., 1., 1., 1e20, 1e19]), tc='d')
    self.q_angle = matrix(np.zeros(N + 1))
    self.q_pos = matrix(np.zeros(N + 2))

def cal_angle(self, state, state_dsr,gp_prediction):
    pos_vec,vel_vec = direction(self, state, state_dsr,gp_prediction)
    pos_vec = pos_vec / np.linalg.norm(pos_vec)
    psi = 0
    phi = np.arcsin(-pos_vec[1])
    theta = np.arcsin((pos_vec[0] / np.cos(phi)))
    pos_angle = np.array([phi, theta, psi])
    if vel_vec is None:
        return pos_angle
    vel_vec = vel_vec / np.linalg.norm(vel_vec)
    phi = np.arcsin(-vel_vec[1])
    theta = np.arcsin((vel_vec[0] / np.cos(phi)))
    vel_angle = np.array([phi, theta, psi])
    return pos_angle, vel_angle

def direction(self, state, state_dsr, gp_prediction):
    dt = 1.0/self.rate
    g = self.g
    m = self.m
    pos_direction = ((state_dsr[:3] - state[:3] - gp_prediction[:3] - state[3:6] * dt) / dt ** 2 + np.array([0, 0, g])) * m
    if state_dsr.shape[0] == 3:
        return pos_direction, None
    vel_direction = ((state_dsr[3:6] - gp_prediction[3:6] - state[3:6]) / dt + np.array([0, 0, g])) * m
    return pos_direction, vel_direction

# Get compensatory action based on satisfaction of barrier function
def control_barrier(self, obs, f, g, x, std, target,is_pid=False):
    #step_time = t
    dt = 1.0/self.rate
    ''' Recorded curve '''
    # curve = self.env.curve
    # cur_pt = curve[t,:]
    # next_pt = curve[t+1,:]
    # third_pt = curve[t+2,:]
    ''' Parametric curve '''

    #cur_pt = np.concatenate([self.curve_pos(step_time), self.curve_vel(step_time)])
    #next_pt = np.concatenate([self.curve_pos((step_time) + 1*dt), self.curve_vel((step_time) + 1*dt)])
    #third_pt = np.concatenate([self.curve_pos((step_time) + 2*dt), self.curve_vel((step_time) + 2*dt)])
    cur_pt = target[0]
    next_pt = target[1]
    third_pt = target[2]

    # Set up Quadratic Program to satisfy the Control Barrier Function
    kd = 2.0
    # calculate energy function V_t
    v_t_pos = np.abs(cur_pt - obs[:6])
    N = self.observation_space
    M = self.action_space
    '''The main hyperparameter'''
    # gamma 越小（甚至为0）,能量衰减越慢, 震荡越小, 设置过小会导致跟踪收敛慢
    gamma_pos_clf = 0.06
    gamma_angle_clf = 1
    # gamma_pos_cbf 越小, 在安全域内越不允许往更不安全的地方探索, 在安全域外则更慢收敛到安全域内, 因此应根据实际场景调整设置
    gamma_vel_cbf = 10
    gamma_angle_cbf = 10000

    up_b = self.action_bound_up.reshape([M, 1])##
    low_b = self.action_bound_low.reshape([M, 1])##
    f = np.reshape(f, [N, 1])
    g = np.reshape(g, [N, M])
    std = np.reshape(std, [N, 1])
    # std = np.zeros([N, 1])
    x = np.reshape(x, [N, 1])
    ''' q is the thrust direction'''
    #q = np.array([math.cos(x[6])*math.sin(x[7]), -math.sin(x[6]), math.cos(x[6])*math.cos(x[7])]).reshape(([-1,1]))

    '''QP 1 : use CLF & CBF to solve thrust'''
    '''v1(x) = |x-x_g| ,v1(y) = |y-y_g|, v1(z) = |z-z_g|'''
    '''h1(x) = 1 - (X^T * A * X + b^T * X + C)'''
    #
    # if (step_time) % 1 == 0:
    #     '''
    #         key 1: the setting of the bound, large bound will make the quad easier to avoid the moving obstacle
    #         small bound will enable the quad to cross complex area, the quad will more flexible
    #     '''
    #     bounds = irispy.Polyhedron.from_bounds(obs[:3] - 3*self.env.quad.l*np.ones(3,), obs[:3] + 3*self.env.quad.l*np.ones(3,))
    #     obstacles = self.env.obstacles
    #     start = obs[:3]
    #     region, debug = irispy.inflate_region(obstacles, start, bounds=bounds, return_debug_data=True)
    #     iter_result = debug.iterRegions()
    #     ellipsoid = list(iter_result)[-1][1]
    #     mapping_matrix = ellipsoid.getC() * 0.9
    #     center_pt = ellipsoid.getD().reshape([-1,1])
    #     inv_mapping_matrix = np.linalg.inv(mapping_matrix)
    #     A_mat = np.dot(inv_mapping_matrix.T, inv_mapping_matrix)
    #     b_vec = -2*A_mat.dot(center_pt)
    #     c = np.dot(center_pt.T, A_mat).dot(center_pt)
    #     '''
    #         key 2: the gamma is different when the quad inside or outside the safe region.
    #         When outside the safe region, the gamma should be quite large.
    #         When inside the safe region, the gamma should be quite small. Besides, the gamma will become larger when the
    #         quad leave the center.
    #     '''
    #     if (1 - x[:3,:].T.dot(A_mat).dot(x[:3,:]) - b_vec.T.dot(x[:3,:]) - c) < 0:
    #         gamma_pos_cbf = 100
    #         print('out of safe region')
    #     else:
    #         gamma_pos_cbf = (1 - x[:3,:].T.dot(A_mat).dot(x[:3,:]) - b_vec.T.dot(x[:3,:]) - c) ** 0.5
    #         if step_time >= 20:
    #             gamma_angle_cbf = (1 - x[:3,:].T.dot(A_mat).dot(x[:3,:]) - b_vec.T.dot(x[:3,:]) - c)
    #     self.safe_region = [A_mat, b_vec, c]
    #     center_pt_dot = center_pt - self.ellipsoid[1] if step_time>0 else np.zeros([3,1])
    #     #center_pt_dot = np.zeros([3,1])
    #     self.ellipsoid = [mapping_matrix, center_pt]
    #
    # [A_mat, b_vec, c] = self.safe_region
    # print(1 - x[:3, :].T.dot(A_mat).dot(x[:3, :]) - b_vec.T.dot(x[:3, :]) - c)
    z_weight = 1
    G_pos = np.concatenate([np.concatenate([
        g[0, :].reshape([1, -1]),
        -g[0, :].reshape([1, -1]),
        g[1, :].reshape([1, -1]),
        -g[1, :].reshape([1, -1]),
        z_weight*g[2, :].reshape([1, -1]),
        -z_weight*g[2, :].reshape([1, -1]),
        g[3, :].reshape([1, -1]),
        -g[3, :].reshape([1, -1]),
        g[4, :].reshape([1, -1]),
        -g[4, :].reshape([1, -1]),
        g[5, :].reshape([1, -1]),
        -g[5, :].reshape([1, -1]),
        # (2 * A_mat.dot(x[:3,:]) + b_vec).T.dot(g[:3,:]),
        # (2 * x[3:6,:]).T.dot(g[3:6,:]),
        np.eye(M), -np.eye(M)], axis=0),
        # np.concatenate([-np.ones([6, 1]), np.zeros([6, 1]), np.zeros([2, 1]), np.zeros([2 * M, 1])], axis=0),
        # np.concatenate([np.zeros([6, 1]), -np.ones([6, 1]), np.zeros([2, 1]), np.zeros([2 * M, 1])], axis=0),
        # np.concatenate([np.zeros([6, 2]), np.zeros([6, 2]), -np.eye(2), np.zeros([2 * M, 2])], axis=0)], axis=1)
        np.concatenate([-np.ones([6, 1]), np.zeros([6, 1]), np.zeros([2 * M, 1])], axis=0),
        np.concatenate([np.zeros([6, 1]), -np.ones([6, 1]), np.zeros([2 * M, 1])], axis=0)], axis = 1)
    h_pos = np.concatenate([
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[0] - f[0] + next_pt[0]) - abs(kd*std[0]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[0] + f[0] - next_pt[0]) - abs(kd*std[0]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[1] - f[1] + next_pt[1]) - abs(kd*std[1]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[1] + f[1] - next_pt[1]) - abs(kd*std[1]),
        np.ones([1, 1]) * z_weight*((1 - gamma_pos_clf) * v_t_pos[2] - f[2] + next_pt[2]) - abs(kd*std[2]),
        np.ones([1, 1]) * z_weight*((1 - gamma_pos_clf) * v_t_pos[2] + f[2] - next_pt[2]) - abs(kd*std[2]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[3] - f[3] + next_pt[3]) - abs(kd*std[3]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[3] + f[3] - next_pt[3]) - abs(kd*std[3]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[4] - f[4] + next_pt[4]) - abs(kd*std[4]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[4] + f[4] - next_pt[4]) - abs(kd*std[4]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[5] - f[5] + next_pt[5]) - abs(kd*std[5]),
        np.ones([1, 1]) * ((1 - gamma_pos_clf) * v_t_pos[5] + f[5] - next_pt[5]) - abs(kd*std[5]),
        # np.ones([1, 1]) * (gamma_pos_cbf * dt * (1 - x[:3,:].T.dot(A_mat).dot(x[:3,:]) - b_vec.T.dot(x[:3,:]) - c) - (2 * A_mat.dot(x[:3,:]) + b_vec).T.dot(f[:3,:] - x[:3,:] - center_pt_dot) - abs(kd*(2 * A_mat.dot(x[:3,:]) + b_vec).T.dot(std[:3,:]))),
        # np.ones([1, 1]) * (gamma_vel_cbf * dt * (3 - x[3:6,:].T.dot(x[3:6,:])) - (2 * x[3:6,:]).T.dot(f[3:6,:] - x[3:6,:]) - abs(kd*(2 * x[3:6,:]).T.dot(std[3:6,:]))),
        up_b,
        -low_b], axis=0)
    G_pos = matrix(G_pos, tc='d')
    h_pos = matrix(h_pos, tc='d')
       
    # Solve QP
    if not is_pid:
        solvers.options['show_progress'] = False
        
        sol = solvers.qp(P=self.P_pos, q=self.q_pos, G=G_pos, h=h_pos)
        clf_sol = np.array(sol['x'][:3]).reshape([-1,1])
        pass
    ''' PID controller '''
    K_p = 13
    K_d = 5
    K_i = 0.18
    if is_pid:
        if step_time == 0:
            self.pos_err_intergral = 0
            self.angle_err_intergral = 0
        else:
            self.pos_err_intergral +=  q.T.dot(next_pt[:3].reshape(-1,1)-x[:3,:])
        clf_sol = K_p * q.T.dot(next_pt[:3].reshape(-1,1)-x[:3,:])+K_i*self.pos_err_intergral+ K_d* q.T.dot(next_pt[3:6].reshape(-1,1)-x[3:6,:]) 
        clf_sol = np.array([clf_sol[0][0],0,0,0]).reshape([-1,1])
        clf_sol = np.clip(clf_sol,low_b,up_b)

    # predict new pos and vel to calculate new angle
    predict_xyz = f + np.dot(g,clf_sol)
    #gp_prediction = GP.get_GP_prediction(self,predict_xyz[:6])
    gp_prediction = np.zeros(6)
    pos_angle, vel_angle = cal_angle(self, predict_xyz[:6].squeeze(), third_pt, gp_prediction)
    weight = 0.6
    [phi_d, theta_d, psi_d] = weight*pos_angle + (1-weight)*vel_angle
    
    u_bar = np.array([clf_sol[0][0],phi_d,theta_d]).reshape([-1,1])
    u_bar = np.squeeze(np.clip(u_bar,low_b,up_b))

    return u_bar,np.sum(v_t_pos[:3])
