from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import numpy as np
import scipy.stats as stats


class CEMOptimizer(object):
    """A Tensorflow-compatible CEM optimizer.
    """

    def __init__(self, sol_dim, max_iters, popsize, num_elites, upper_bound=None, lower_bound=None, epsilon=0.001, alpha=0.25):
        """Creates an instance of this class.

        Arguments:
            sol_dim (int): The dimensionality of the problem space, plan_horizon*action_dim
            max_iters (int): The maximum number of iterations to perform during optimization
            popsize (int): The number of candidate solutions to be sampled at every iteration
            num_elites (int): The number of top solutions that will be used to obtain the distribution
                at the next iteration.
            tf_session (tf.Session): (optional) Session to be used for this optimizer. Defaults to None,
                in which case any functions passed in cannot be tf.Tensor-valued.
            upper_bound (np.array): An array of upper bounds
            lower_bound (np.array): An array of lower bounds
            epsilon (float): A minimum variance. If the maximum variance drops below epsilon, optimization is
                stopped.
            alpha (float): Controls how much of the previous mean and variance is used for the next iteration.
                next_mean = alpha * old_mean + (1 - alpha) * elite_mean, and similarly for variance.
        """
        super().__init__()
        self.sol_dim, self.max_iters, self.popsize, self.num_elites = sol_dim, max_iters, popsize, num_elites
        self.ub, self.lb = upper_bound, lower_bound
        self.epsilon, self.alpha = epsilon, alpha

        if num_elites > popsize:
            raise ValueError("Number of elites must be at most the population size.")

        self.num_opt_iters, self.mean, self.var = None, None, None


    def setup(self, cost_function, GP_model=None,tf_compatible=None):
        """Sets up this optimizer using a given cost function.

        Arguments:
            cost_function (func): A function for computing costs over a batch of candidate solutions.
            tf_compatible (bool): True if the cost function provided is tf.Tensor-valued.

        Returns: None
        """
        self.cost_function = cost_function

    def reset(self):
        pass

    def obtain_solution(self, init_mean, init_var, cur_state, t0):
        """Optimizes the cost function using the provided initial candidate distribution

        Arguments:
            init_mean (np.ndarray): The mean of the initial candidate distribution.
            init_var (np.ndarray): The variance of the initial candidate distribution.
        """

        mean, var, t = init_mean, init_var, 0
        X = stats.truncnorm(-2, 2, loc=np.zeros_like(mean), scale=np.ones_like(mean))
        while (t < self.max_iters) and np.max(var) > self.epsilon:
        # while (t < self.max_iters):
            lb_dist, ub_dist = mean - self.lb, self.ub - mean
            constrained_var = np.minimum(np.minimum(np.square(lb_dist / 2), np.square(ub_dist / 2)), var) + 1 * np.tile(1, [self.sol_dim*1])
            # constrained_var = var

            samples = X.rvs(size=[self.popsize, self.sol_dim]) * np.sqrt(constrained_var) + mean
            samples = np.clip(samples, np.tile(self.lb,[self.popsize, 1]), np.tile(self.ub,[self.popsize, 1]))

            # samples = self.constrain(samples, cur_state)

            #If sampled policies are feasible(satisfy the constraints), they are sorted according to their gain(as in classic cross - entropy)


            costs = self.cost_function(samples, cur_state, t0)

            elites = np.squeeze(samples[np.argsort(costs, axis=0)][:self.num_elites])

            new_mean = np.mean(elites, axis=0)
            new_var = np.var(elites, axis=0)

            mean = self.alpha * mean + (1 - self.alpha) * new_mean
            var = self.alpha * var + (1 - self.alpha) * new_var

            t += 1
        sol, solvar = mean, var
        return sol,solvar
