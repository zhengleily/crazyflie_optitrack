#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from cvxopt import matrix
from cvxopt import solvers
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import time
import GPy

class GP():

    def build_GP_model(self):
        N = 1  # 6 GPs
        GP_list = []
        noise = 1e-10
        for i in range(N):
            kern = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
            gp = GaussianProcessRegressor(kernel=kern, alpha=noise, n_restarts_optimizer=10)
            GP_list.append(gp)
        self.GP_model = GP_list

    def get_GP_dynamics(self, obs, phi=0,theta=0,psi=0):
        s = obs.reshape(-1,6)
        [f_nom, g, x] = self.predict_f_g(obs,phi,theta,psi)
        f_nom = np.ravel(f_nom)
        g = np.ravel(g)
        x = np.ravel(x)[:6]
        f = np.copy(f_nom)

        [m0, std0] = self.GP_model[0].predict(x.reshape(1, -1), return_std=True)
        #[m1, std1] = self.GP_model[1].predict(x.reshape(1, -1), return_std=True)
        #[m2, std2] = self.GP_model[2].predict(x.reshape(1, -1), return_std=True)
        #[m3, std3] = self.GP_model[3].predict(x.reshape(1, -1), return_std=True)
        #[m4, std4] = self.GP_model[4].predict(x.reshape(1, -1), return_std=True)
        #[m5, std5] = self.GP_model[5].predict(x.reshape(1, -1), return_std=True)

        # [m0, std0] = [0,0]
        # [m1, std1] = [0,0]
        # [m2, std2] = [0,0]
        # [m3, std3] = [0,0]
        # [m4, std4] = [0,0]
        # [m5, std5] = [0,0]
        m0 = np.squeeze(m0)
        f[0] = f_nom[0] + m0[0]
        f[1] = f_nom[1] + m0[1]
        f[2] = f_nom[2] + m0[2]
        f[3] = f_nom[3] + m0[3]
        f[4] = f_nom[4] + m0[4]
        f[5] = f_nom[5] + m0[5]

        return [np.squeeze(f),m0, np.squeeze(g), np.squeeze(obs), np.array([np.squeeze(std0),
                                                                       np.squeeze(std0),
                                                                       np.squeeze(std0),
                                                                       np.squeeze(std0),
                                                                       np.squeeze(std0),
                                                                       np.squeeze(std0),])]

    def get_GP_prediction(self, obs):
        x = obs[:6] 
        [m0,m1,m2,m3,m4,m5] = self.GP_model[0].predict(x.reshape(1, -1), return_std=False)[0]
        #m1 = self.GP_model[1].predict(x.reshape(1, -1), return_std=False)[0]
        #m2 = self.GP_model[2].predict(x.reshape(1, -1), return_std=False)[0]
        #m3 = self.GP_model[3].predict(x.reshape(1, -1), return_std=False)[0]
        #m4 = self.GP_model[4].predict(x.reshape(1, -1), return_std=False)[0]
        #m5 = self.GP_model[5].predict(x.reshape(1, -1), return_std=False)[0]
        return np.array([m0,m1,m2,m3,m4,m5])
