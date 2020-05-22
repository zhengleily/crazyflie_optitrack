import sys                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      
sys.path.append("..")
from MPC import MPC
import tensorflow as tf
import CBF.CBF as cbf
import matplotlib.pyplot as plt
from matplotlib.colors import colorConverter
from matplotlib.animation import FuncAnimation
import mpl_toolkits.mplot3d as a3
import numpy as np
import argparse
import pprint as pp
from scipy.io import savemat
import scipy.spatial
import time
from quadenv import quad_env
from CBF.learner import LEARNER
from CBF.GP import GP
import datetime
import copy

# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=5e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

def draw_convhull(points, ax, **kwargs):
    kwargs.setdefault("edgecolor", "w")
    kwargs.setdefault("facecolor", "c")
    kwargs.setdefault("alpha", 0.2)
    kwargs["facecolor"] = colorConverter.to_rgba(kwargs["facecolor"], kwargs["alpha"])
    hull = scipy.spatial.ConvexHull(points)
    for simplex in hull.simplices:
        poly = a3.art3d.Poly3DCollection([points[simplex]], **kwargs)
        if "alpha" in kwargs:
            poly.set_alpha(kwargs["alpha"])
        ax.add_collection3d(poly)

# ===========================
#   Agent Training
# ===========================

def train(sess, env, args, actor_noise, reward_result, agent):
    # Needed to enable BatchNorm.
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    paths = list()
    sess.run(tf.global_variables_initializer())
    f_vec = []
    g_vec = []
    gpr_vec = []
    mean_vec = []
    std_vec = []
    for i in range(int(args['max_episodes'])):
        for el in range(1):
            obs, action, rewards, action_bar, action_BAR, action_cem = [], [], [], [], [], []
            ep_reward = 0
            ep_ave_max_q = 0
            state_x = []
            state_y = []
            state_z = []
            state_v_x = []
            state_v_y = []
            state_v_z = []
            state_phi = []
            state_theta = []
            state_psi = []
            state_phi_dot = []
            state_theta_dot = []
            state_psi_dot = []
            h1_cbf = []
            h2_cbf = []
            h3_cbf = []
            h4_cbf = []
            v_pos = []
            v_angle = []
            obstacle_list = []

            s1 = env.reset()
            path_plot = list()
            s = np.copy(s1)
            t0 = time.time()
            curve_pos = env.curve_pos
            curve_vel = env.curve_vel
            curve = np.array(list(map(curve_pos,np.linspace(0,int(args['max_episode_len'])*0.02,int(args['max_episode_len'])+1))))
            for j in range(int(args['max_episode_len'])):
                t1 = time.time()
                # env.render()
                # Added exploration noise
                # Incorporate barrier function
                #action_rl = agent.act(np.reshape(s, (1, agent.observation_space))) #+ actor_noise()
                action_rl = np.zeros(4)
                # Utilize compensation barrier function
                if (agent.firstIter == 1):
                    u_BAR_ = np.zeros(4)
                else:
                    # u_BAR_ = np.zeros(4)
                    u_BAR_ = agent.bar_comp.get_action(s)[0]

                action_RL = action_rl + u_BAR_
                # action_RL = np.ones(action_RL.shape)*0
                # [f,gpr, g, x, std] = GP.get_GP_dynamics(agent, s, action_RL)
                mean = GP.get_GP_prediction(agent,s)
                f,g,x = agent.env.predict_f_g(s)
                std = np.zeros([9,1])

                u_bar_, v_t_pos, v_t_angle = cbf.control_barrier(agent, np.squeeze(s), action_RL, f, g, x, std, j,None)
                # print('Thrust',u_bar_[0][0])
                #u_bar_ = np.zeros(4)
                action_ = np.squeeze(action_RL) + np.squeeze(u_bar_)
                # d1 = s[6] - s[9]
                # d2 = s[12] - s[9]
                # d3 = s[6] - s[12]
                s2, r, terminal = env.step(action_)
                v_pos.append(v_t_pos)
                # v_angle.append(v_t_angle)
                state_x.append(s2[0])
                state_y.append(s2[1])
                state_z.append(s2[3])
                state_v_x.append(s2[4])
                state_v_z.append(s2[5])
                state_phi.append(s2[6])
                state_theta.append(s2[7])
                state_psi.append(s2[8])
                # h1_cbf.append(-abs(s[0] - curve[j, 0]) + 0.1)
                # h2_cbf.append(-abs(s[1] - curve[j, 1]) + 0.1)
                # h3_cbf.append(-abs(s[2] - curve[j, 2]) + 0.1)
                action_bar.append(u_bar_)
                action_BAR.append(u_BAR_)
                obs.append(s)
                rewards.append(r)
                action.append(action_)
                action_cem.append(action_rl)
                f_vec.append(f)
                g_vec.append(g)
                std_vec.append(std)
                mean_vec.append(mean)
                #gpr_vec.append(gpr)
                obstacle_list.append(copy.deepcopy(agent.env.obstacles))

                s = np.copy(s2)
                ep_reward += r

                update_interval = 30
                if (j+1) % update_interval == 0:
                # if (j+1) % int(args['max_episode_len']) == 0:
                    # writer.add_summary(summary_str, i)
                    # writer.flush()
                    # plt.plot(DB_state_12)
                    # plt.show()
                    # print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f}'.format(int(ep_reward), i,
                    #                                                              (ep_ave_max_q / float(j))))
                    print(ep_reward)
                    print('Step time:',j)
                    # reward_result[i] = ep_reward
                    path = {"Observation": np.concatenate(obs[-update_interval:]).reshape((-1, 9)),
                            "Action": np.concatenate(action[-update_interval:]).reshape((-1, 4)),
                            "Action_bar": np.concatenate(action_bar[-update_interval:]).reshape([-1,4])+np.concatenate(action_BAR[-update_interval:]).reshape([-1,4]),
                            "Reward": np.asarray(rewards)}
                    paths.append(path)
                    # GP.update_GP_dynamics(agent, path)

                if (j + 1) % 2000 == 0 or j == int(args['max_episode_len'])-1:
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    ax.set_title("Tracking")
                    ax.set_xlabel('x')
                    ax.set_xlim(-0, 7)
                    ax.set_ylabel('y')
                    ax.set_ylim(-0, 7)
                    ax.set_zlabel('z')
                    ax.set_zlim(0, 4)
                    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], c='r')
                    obs2 = np.vstack(obs)
                    ax.plot(obs2[:, 0], obs2[:, 1], obs2[:, 2], c='b')
                    #for obstacle in agent.env.obstacles:
                    #    draw_convhull(obstacle.T, ax, edgecolor='k', facecolor='k', alpha=0.2)
                    # [mapping_mat, center_pt] = agent.ellipsoid
                    theta = np.linspace(0, 2 * np.pi, 20)
                    y = np.vstack((np.sin(theta), np.cos(theta), np.zeros_like(theta)))
                    for phi in np.linspace(0, np.pi, 10):
                        R = np.array([[1.0, 0.0, 0.0],
                                      [0.0, np.cos(phi), -np.sin(phi)],
                                      [0.0, np.sin(phi), np.cos(phi)]])
                        y = np.hstack((y, R.dot(y)))
                    # points = mapping_mat.dot(y) + center_pt
                    # points = points.T
                    # draw_convhull(points, ax)
                    # plt.show()

            if el <= 1:
                
                print('time', time.time() - t0)
                # np.save('err_0.8m.npy',np.vstack(v_pos))
                # np.save('action.npy',np.vstack(action))
                # np.save('f.npy',np.vstack(f_vec))
                # np.save('g.npy',np.vstack(g_vec))
                # np.save('obs.npy',np.vstack(obs))
                # np.save('mean.npy',np.vstack(mean_vec))
                # np.save('std.npy',np.vstack(std_vec))
                obs = np.vstack(obs)
                action = np.vstack(action)
                tail_len = 100
                curve_len = len(obs)
                L = agent.env.quad.l
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.set_title("Tracking")
                ax.set_xlabel('x')
                ax.set_xlim(-4, 4)
                ax.set_ylabel('y')
                ax.set_ylim(-2, 10)
                ax.set_zlabel('z')
                ax.set_zlim(0, 12)
                view_trans = np.linspace(0, 600, curve_len+2*tail_len)

                def init():
                    for obstacle in obstacle_list[0]:
                        draw_convhull(obstacle.T, ax, edgecolor='k', facecolor='k', alpha=0.2)
                    x, y, z, _, _, _, phi, theta, psi = obs[0, :]
                    rotation_x = np.array([
                        np.cos(phi) * np.cos(theta),
                        np.cos(theta) * np.sin(phi),
                        - np.sin(theta)
                    ])
                    rotation_y = np.array([
                        np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi),
                        np.cos(psi) * np.cos(phi) - np.sin(psi) * np.sin(phi) * np.sin(theta),
                        np.cos(theta) * np.sin(phi)
                    ])
                    pole1 = rotation_x * L + np.array([x, y, z])
                    pole2 = rotation_y * L + np.array([x, y, z])
                    pole1_v = 2 * np.array([x, y, z]) - pole1
                    pole2_v = 2 * np.array([x, y, z]) - pole2
                    pole_1_plt, = ax.plot([pole1_v[0], pole1[0]], [pole1_v[1], pole1[1]], [pole1_v[2], pole1[2]], 'b.-')
                    pole_2_plt, = ax.plot([pole2_v[0], pole2[0]], [pole2_v[1], pole2[1]], [pole2_v[2], pole2[2]], 'b.-')

                    curve_plt, = ax.plot(curve[:1, 0], curve[:1, 1], curve[:1, 2], 'r--')
                    obs_plt, = ax.plot(obs[:1, 0], obs[:1, 1], obs[:1, 2], 'b')

                def update(tt):
                    plt.cla()
                    ax = fig.gca(projection='3d')
                    ax.set_title("Tracking")
                    ax.set_xlabel('x')
                    ax.set_xlim(-4, 4)
                    ax.set_ylabel('y')
                    ax.set_ylim(-2, 10)
                    ax.set_zlabel('z')
                    ax.set_zlim(0, 12)
                    # horiAngle = 35 + view_trans[tt]
                    # vertAngle = 30
                    # ax.view_init(vertAngle, horiAngle)

                    tt = tt - tail_len
                    curve_plt, = ax.plot(curve[:1, 0], curve[:1, 1], curve[:1, 2], 'r--')
                    obs_plt, = ax.plot(obs[:1, 0], obs[:1, 1], obs[:1, 2], 'b')
                    if tt <= 0:
                        curve_plt.set_data(curve[:tt+tail_len, 0:2].T)
                        curve_plt.set_3d_properties(curve[:tt+tail_len, 2].T)
                        tt = 0
                    elif tt <= tail_len:
                        curve_plt.set_data(curve[:tt+tail_len, 0:2].T)
                        curve_plt.set_3d_properties(curve[:tt+tail_len, 2].T)
                        obs_plt.set_data(obs[:tt, 0:2].T)
                        obs_plt.set_3d_properties(obs[:tt, 2].T)
                    elif tt < curve_len:
                        curve_plt.set_data(curve[tt-tail_len:min(tt+tail_len,curve_len), 0:2].T)
                        curve_plt.set_3d_properties(curve[tt-tail_len:min(tt+tail_len,curve_len), 2].T)
                        obs_plt.set_data(obs[tt-tail_len:tt, 0:2].T)
                        obs_plt.set_3d_properties(obs[tt-tail_len:tt, 2].T)
                    else:
                        curve_plt.set_data(curve[tt - (curve_len + tail_len):, 0:2].T)
                        curve_plt.set_3d_properties(curve[tt - (curve_len + tail_len):, 2].T)
                        obs_plt.set_data(obs[tt - (curve_len + tail_len):, 0:2].T)
                        obs_plt.set_3d_properties(obs[tt - (curve_len + tail_len):, 2].T)

                    x,y,z,_,_,_,phi,theta,psi = obs[min(tt, curve_len-1), :]
                    rotation_x = np.array([
                        np.cos(phi) * np.cos(theta),
                        np.cos(theta) * np.sin(phi),
                        - np.sin(theta)
                    ])
                    rotation_y = np.array([
                        np.sin(psi) * np.sin(theta) * np.cos(phi) - np.cos(psi) * np.sin(phi),
                        np.cos(psi) * np.cos(phi) - np.sin(psi) * np.sin(phi) * np.sin(theta),
                        np.cos(theta) * np.sin(phi)
                    ])
                    pole1 = rotation_x * L + np.array([x,y,z])
                    pole2 = rotation_y * L + np.array([x,y,z])
                    pole1_v = 2*np.array([x,y,z]) - pole1
                    pole2_v = 2*np.array([x,y,z]) - pole2
                    pole_1_plt, = ax.plot([pole1_v[0], pole1[0]], [pole1_v[1], pole1[1]], [pole1_v[2], pole1[2]], 'b.-')
                    pole_2_plt, = ax.plot([pole2_v[0], pole2[0]], [pole2_v[1], pole2[1]], [pole2_v[2], pole2[2]], 'b.-')
                    # pole_1_plt.set_xdata([pole1_v[0],pole1[0]])
                    # pole_1_plt.set_ydata([pole1_v[1],pole1[1]])
                    # pole_1_plt.set_3d_properties([pole1_v[2],pole1[2]])
                    # pole_2_plt.set_xdata([pole2_v[0],pole2[0]])
                    # pole_2_plt.set_ydata([pole2_v[1],pole2[1]])
                    # pole_2_plt.set_3d_properties([pole2_v[2],pole2[2]])

                    for obstacle in obstacle_list[min(tt, curve_len-1)]:
                        draw_convhull(obstacle.T, ax, edgecolor='k', facecolor='k', alpha=0.2)
                    return ax
                anim = FuncAnimation(fig, update, init_func=init, frames=np.arange(0, curve_len + 2 * tail_len), interval=20)
                if i > 0:
                    anim.save('iris_moving.gif', dpi=80, writer='imagemagick')
                plt.show()
                plt.figure()
                plt.plot(v_pos)
                plt.plot(v_angle)
                plt.show()

                action_bar = np.vstack(action_bar)
                action_BAR = np.vstack(action_BAR)
                action = np.vstack(action_cem)
                plt.figure()
                plt.plot(action_bar[:,0],color='r',label='u_bar')
                plt.plot(action_BAR[:,0],color='g',label='u_nn')
                plt.plot(action[:,0],color='b',label='u_rl')
                plt.legend(loc='upper right')
                plt.title('Thrust')
                plt.xlabel('Time(s)')
                plt.ylabel('Action')
                plt.show()

                plt.figure()
                plt.plot(action_bar[:,1],color='r',label='u_bar')
                plt.plot(action_BAR[:,1],color='g',label='u_nn')
                plt.plot(action[:,1],color='b',label='u_rl')
                plt.legend(loc='upper right')
                plt.title('w1')
                plt.xlabel('Time(s)')
                plt.ylabel('Action')
                plt.show()

                plt.figure()
                plt.plot(action_bar[:,2],color='r',label='u_bar')
                plt.plot(action_BAR[:,2],color='g',label='u_nn')
                plt.plot(action[:,2],color='b',label='u_rl')
                plt.legend(loc='upper right')
                plt.title('w2')
                plt.xlabel('Time(s)')
                plt.ylabel('Action')
                plt.show()

                plt.figure()
                plt.plot(action_bar[:,3],color='r',label='u_bar')
                plt.plot(action_BAR[:,3],color='g',label='u_nn')
                plt.plot(action[:,3],color='b',label='u_rl')
                plt.legend(loc='upper right')
                plt.title('w3')
                plt.xlabel('Time(s)')
                plt.ylabel('Action')
                plt.show()
                #
                plt.figure()
                plt.plot(curve[:j+1,0]-obs[:,0],color='r',label='eX')
                plt.plot(curve[:j+1,1]-obs[:,1],color='g',label='eY')
                plt.plot(curve[:j+1,2]-obs[:,2],color='b',label='eZ')
                plt.legend(loc='upper right')
                plt.title('Tracking error')
                plt.xlabel('Time(s)')
                plt.ylabel('Tracking error')
                plt.show()
                #
                plt.figure()
                plt.plot(obs[:, 6], color='r', label='phi')
                plt.plot(obs[:, 7], color='g', label='theta')
                plt.plot(obs[:, 8], color='b', label='psi')
                plt.legend(loc='upper right')
                plt.title('Angle')
                plt.xlabel('Time(s)')
                plt.ylabel('Angle')
                plt.show()

                plt.figure()
                plt.plot(obs[:, 3], color='r', label='vx')
                plt.plot(obs[:, 4], color='g', label='vy')
                plt.plot(obs[:, 5], color='b', label='vz')
                plt.plot(abs(obs[:, 5])+abs(obs[:,4])+abs(obs[:,3]), color='m', label='v')
                plt.legend(loc='upper right')
                plt.title('vel')
                plt.xlabel('Time(s)')
                plt.ylabel('vel')
                plt.show()
            agent.bar_comp.get_training_rollouts(paths)
            barr_loss = agent.bar_comp.train()
            agent.firstIter = 0
        if i == 9:
            observation = np.squeeze(np.concatenate([path["Observation"] for path in paths]))
            action = np.concatenate([path["Action"] for path in paths])
            prediction = []
            for i in range(len(f_vec)):
                prediction.append(f_vec[i] + np.squeeze(np.dot(g_vec[i].reshape([9,4]),action[i,:].reshape([4,1]))))
            np.save('prediction.npy',np.vstack(prediction).reshape([-1,9]))
            np.save('gpr.npy',np.vstack(gpr_vec).reshape([-1,6]))
            np.save('obs.npy',observation)
            np.save('act.npy',action)
    return paths

def main(args, reward_result):
    with tf.Session() as sess:
        env = quad_env()
        tf.set_random_seed(int(args['random_seed']))
        np.random.seed(int(args['random_seed']))
        action_dim = 4
        agent = MPC()
        # Ensure action bound is symmetric

        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(action_dim))
        paths = train(sess, agent.env, args, actor_noise, reward_result, agent)
        return paths

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for cem_mpc agent')

    # agent parameters
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.00001)
    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=6781)  # 1234 default
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=200)  # 50000 default
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=500)  # 1000 default
    parser.add_argument('--render-env', help='render the gym env', action='store_false')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_false')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    pp.pprint(args)

    reward_result = np.zeros(3000)
    paths = main(args, reward_result)

    savemat('data1_' + datetime.datetime.now().strftime("%y-%m-%d-%H-%M") + '.mat',
            dict(data=paths, reward=reward_result))
