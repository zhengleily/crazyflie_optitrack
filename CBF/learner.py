import numpy as np
from .GP import GP
from .CBF import build_barrier
from barrier_comp import BARRIER

def controller_init(self):
    self.action_bound_up = np.array([0.55,30,30])
    self.action_bound_low = np.array([0,-30,-30])

    # Set up observation space and action space
    self.observation_space = self.num_state
    self.action_space = self.num_action
    # Build barrier function model
    build_barrier(self)
    # Build GP model of dynamics
    GP.build_GP_model(self)

    self.safe_region = []
    self.obs = []
    self.action = []

    print('Observation space', self.observation_space)
    print('Action space', self.action_space)

