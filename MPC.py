import os
import sys
sys.path.append("..")
import numpy as np
from scipy.io import savemat
from CEM import CEMOptimizer
from CBF.learner import LEARNER
import time
from quadenv import quad_env
from multiprocessing import Process, Value

class MPC(LEARNER):
    def __init__(self):
        env = quad_env()
        LEARNER.__init__(self, env)
        self.has_been_trained = False
        self.plan_hor = 20
        self.dO, self.dU = self.observation_space, self.action_space
        self.obs_cost_fn = None
        self.ac_cost_fn = None
        self.ac_ub = self.action_bound_up
        self.ac_lb = self.action_bound_low
        self.per = 1 #The numbers of action_step, a agent make in each decision
        self.prev_sol = np.tile((self.ac_lb + self.ac_ub)/2, [self.plan_hor])
        # self.prev_sol = np.tile(171.55*np.ones([4,]), [self.plan_hor])

        self.init_var = np.tile(np.square(self.ac_ub - self.ac_lb) / 2, [self.plan_hor])
        self.ac_buf = np.array([]).reshape(0, self.dU)
        self.optimizer = CEMOptimizer(self.plan_hor*self.dU, 5, 100, 10,
                                      np.tile(self.ac_ub, [self.plan_hor]),
                                      np.tile(self.ac_lb, [self.plan_hor]))
        self.optimizer.setup(self._compile_cost)


    def _compile_cost(self, ac_seqs, init_state,t0):
        ac_seqs = np.reshape(ac_seqs, ((-1, self.plan_hor, self.dU)))
        cost = self.cost(ac_seqs, init_state, self.plan_hor, self.dO, self.dU, t0,self.GP_model)
        return cost

    def act(self, obs):
        """Returns the action that this controller would take at time t given observation obs.

        Arguments:
            obs: The current observation
            t: The current timestep ？？js
            get_pred_cost: If True, returns the predicted cost for the action sequence found by
                the internal optimizer.

        Returns: An action (and possibly the predicted cost)
        # """
        if self.ac_buf.shape[0] > 0:
            action, self.ac_buf = self.ac_buf[0], self.ac_buf[1:]
            return action
        t = self.env.t
        soln, solvarn = np.squeeze(self.optimizer.obtain_solution(self.prev_sol, self.init_var, obs, t))
        self.ac_buf = soln[:self.per*self.dU].reshape(-1, self.dU)
        return self.act(obs)
