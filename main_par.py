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
from CBF.learner import LEARNER
from CBF.GP import GP
import datetime
import threading
import copy
share_lock = threading.Lock()

def control_process(args, agent):
    v_pos = []
    v_angle = []
    s0 = agent.env.reset()
    path_plot = list()
    s = np.copy(s0)
    t0 = time.time()
    curve_pos = agent.env.curve_pos
    curve_vel = agent.env.curve_vel
    curve = np.array(list(map(curve_pos,np.linspace(0,int(args['max_episode_len'])*0.02,int(args['max_episode_len'])+1))))
    for j in range(int(args['max_episode_len'])):
        share_lock.acquire()
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
        if j <= 50:
            f,g,x = agent.env.predict_f_g(s)
            std = np.zeros([9,1])
        else:
            [f, gpr, g, x, std] = GP.get_GP_dynamics(agent, s, action_RL)

        u_bar_, v_t_pos, v_t_angle = cbf.control_barrier(agent, np.squeeze(s), action_RL, f, g, x, std, j,None)
        # print('Thrust',u_bar_[0][0])
        #u_bar_ = np.zeros(4)
        action_ = np.squeeze(action_RL) + np.squeeze(u_bar_)
        s2, r, terminal = agent.env.step(action_)
        v_pos.append(v_t_pos)
        # v_angle.append(v_t_angle)
        agent.obs.append(s)
        agent.action.append(action_)
        s = np.copy(s2)
        share_lock.release()
        if j % 30 == 0:
            time.sleep(0.002)

    print(time.time()-t0)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title("Tracking")
    ax.set_xlabel('x')
    ax.set_xlim(-0, 10)
    ax.set_ylabel('y')
    ax.set_ylim(-1, 1)
    ax.set_zlabel('z')
    ax.set_zlim(0, 2)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], c='r')
    obs2 = np.vstack(agent.obs)
    ax.plot(obs2[:, 0], obs2[:, 1], obs2[:, 2], c='b')
    plt.show()

    plt.figure()
    plt.plot(v_pos)
    plt.show()

def gp_process(args, agent):
    j0 = 0
    global share_lock
    time.sleep(0.1)
    while 1:
        share_lock.acquire()
        j = int(agent.env.t // 0.02)
        print("update time:",j)
        path = {"Observation": np.concatenate(agent.obs[j0:j]).reshape((-1, 9)),
                "Action": np.concatenate(agent.action[j0:j]).reshape((-1, 4)),}
        agent_bak = copy.deepcopy(agent)
        share_lock.release()

        GP.update_GP_dynamics(agent_bak, path)

        agent.GP_model = copy.deepcopy(agent_bak.GP_model)
        j0 = j

        if j >= int(args['max_episode_len'])-1:
            break

def main(args):
    np.random.seed(int(args['random_seed']))
    agent = MPC()

    control = threading.Thread(target=control_process,args=(args, agent))
    gp_update = threading.Thread(target=gp_process,args=(args, agent))
    process_list = []
    process_list.append(control)
    process_list.append(gp_update)
    for process in process_list:
        process.start()
    for process in process_list:
        process.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='provide arguments for cem_mpc agent')

    # agent parameters
    parser.add_argument('--gamma', help='discount factor for critic updates', default=0.99)
    parser.add_argument('--tau', help='soft target update parameter', default=0.00001)
    # run parameters
    parser.add_argument('--random-seed', help='random seed for repeatability', default=6781)  # 1234 default
    parser.add_argument('--max-episodes', help='max num of episodes to do while training', default=200)  # 50000 default
    parser.add_argument('--max-episode-len', help='max length of 1 episode', default=1000)  # 1000 default
    parser.add_argument('--render-env', help='render the gym env', action='store_false')
    parser.add_argument('--use-gym-monitor', help='record gym results', action='store_false')

    parser.set_defaults(render_env=False)
    parser.set_defaults(use_gym_monitor=False)

    args = vars(parser.parse_args())

    pp.pprint(args)
    main(args)
