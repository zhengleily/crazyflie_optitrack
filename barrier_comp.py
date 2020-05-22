import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Barrier Function Compensator
class BARRIER():
    def __init__(self, sess, input_size, action_size):
        self.sess = sess
        self.input_size = input_size
        self.action_size = action_size
        self.build_model()

    # Input will be state : [batch_size, observation size]
    # Ouput will be control action
    def build_model(self):
        print('Initializing Barrier Compensation network')
        with tf.variable_scope('Compensator'):
            # Input will be observation
            self.x = tf.placeholder(tf.float32, [None, self.input_size], name='Obs')
            # Target will be control action
            self.target = tf.placeholder(tf.float32, [None, self.action_size], name='Target_comp')

            # Model is MLP composed of 2 hidden layers with 50, 40 relu units
            h1 = tf.layers.dense(self.x,200,activation=tf.nn.tanh, name='h1')
            h2 = tf.layers.dense(h1, 150,activation=tf.nn.tanh, name='h2')
            h3 = tf.layers.dense(h2, 150,activation=tf.nn.tanh, name='h3')
            self.value = tf.layers.dense(h3, self.action_size, name='ouput')

        tr_vrbs = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Compensator')
        for i in tr_vrbs:
            print(i.op.name)

        # Compute the loss and gradient of loss w.r.t. neural network weights
        self.loss = tf.reduce_mean(tf.square(self.target - self.value))
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)

        self.sess.run(tf.global_variables_initializer())

    def get_training_rollouts(self, paths):
        # Get observations and actions
        self.observation = np.squeeze(np.concatenate([path["Observation"] for path in paths]))
        self.action_bar = np.concatenate([path["Action_bar"] for path in paths])
        # self.action_BAR = np.concatenate([path["Action_BAR"] for path in paths])

        # Reshape observations & actions to use for training
        batch_s = self.observation.shape[0]
        self.action_bar = np.resize(self.action_bar, [batch_s, self.action_size])
        # self.action_BAR = np.resize(self.action_BAR, [batch_s, self.action_size])

    # Given current observation, get the neural network output (representing barrier compensator)
    def get_action(self, obs):
        observation = np.expand_dims(np.squeeze(obs), 0)
        feed_dict = {self.x: observation}
        u_bar = self.sess.run(self.value, feed_dict)
        return u_bar

    def train(self):
        # print('Training barrier function compensator')
        loss = []
        batch_size = self.observation.shape[0]
        for i in range(50):
            # Get the parameter values for gradient, etc...
            index = np.arange(self.observation.shape[0])
            np.random.shuffle(index)
            obs_batch = self.observation[index[:batch_size]]
            action_batch = self.action_bar[index[:batch_size]]
            self.sess.run(self.optimizer,feed_dict={self.x: obs_batch, self.target: action_batch})
            loss.append(self.sess.run(self.loss,feed_dict={self.x: obs_batch, self.target: action_batch}))
        # plt.figure()
        # plt.plot(loss)
        # plt.show()
        return sum(loss)
