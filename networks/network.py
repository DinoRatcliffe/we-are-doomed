from abc import ABCMeta, abstractmethod
from collections import deque
import numpy as np
import tensorflow as tf
import random
import os

class QNetwork(object):
    """ Network class for interacting with Neural Networks that estimate Q values 
    Attributes:
        model_dir: A string indicating the directory to save the model
        num_actions: The number of actions the nework can decide from
        sess: A tensorflow session
        saver: A tensorflow saver object
        network: A dict containing the network properties
                 {'input': input, 
                  'output': output, 
                  'target': target, 
                  'action': action, 
                  'training_step': training_step}
    """
    __metaclass__ = ABCMeta

    def __init__(self, model_dir, num_actions):
        """ Initialises network
        """
        self.model_dir = model_dir
        self.num_actions = num_actions

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def pick_random_action(self):
        """ Picks an action at random
        Returns:
            An Integer between 0 and num_actions to indicate
            the action chosen at random
        """
        return random.choice(range(self.num_actions))

    def save_model(self, t_step):
        """ Saves the current network to model dir
        Args:
            t_step: Integer indicating the current time step
        """
        self.saver.save(self.sess, self.model_dir + '/model.ckpt', t_step)

    def start_session(self):
        self.network = self.create_network();
        self.saver = tf.train.Saver();
        self.sess = tf.Session()

        self.summaries = tf.merge_all_summaries()
        self.train_writer = tf.train.SummaryWriter('logs/train', self.sess.graph)
        self.summary_writer = tf.train.SummaryWriter('logs/summary')

        self.sess.run(tf.initialize_all_variables())

        ckpt = tf.train.get_checkpoint_state(self.model_dir)

        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('loaded saved model at: ' + self.model_dir)
        else:
            print('No model found at: ' + self.model_dir)

    def end_session(self):
        self.sess.close()

    def infer(self, state):
        """ Given an input state returns the Q values for each action
        Args:
            state: a representation of the game state
        Returns:
            A vector (of length num_action) giving the estimated Q value for each action
        """
        q_values = self.sess.run(self.network['output'], feed_dict={self.network['input']: [state]})
        return q_values

    @abstractmethod
    def game_train_tick(self, state):
        """ Given an input state gives back chosen actions and trains if possible
        Args: 
            state: a representation of the game state
        Returns:
            An Integer between 0 and num_actions to indicate
            the chosen action
        """
        pass

    def pick_action(self, state, epsilon): 
        """ Pick an action given a game state
        Args: 
            state: a representation of the game state
            epsilon: the chance of the action being random
        Returns:
            An Integer between 0 and num_actions to indicate
            the chosen action
        """
        if random.random() > epsilon:
            return np.argmax(self.infer(state))
        else:
            return self.pick_random_action()

    @abstractmethod
    def create_network(self):
        """ Creates and initialises the network
        Returns:
            A dict containing the network properties
                 {'input': input, 
                  'output': output, 
                  'target': target, 
                  'action': action, 
                  'training_step': training_step}
        """
        pass

    @staticmethod 
    def weight_variable(shape):
        """ Creates tensorflow variables in a specified shape to be used for weights 
        Args:
            shape: a vector indicating the shape of the resulting tensor 2x2 being:
                   [2, 2]
        Returns:
            A tensor of the defined shape with small random values
        """
        initial = tf.truncated_normal(shape, stddev=1e-6)
        return tf.Variable(initial)

    @staticmethod
    def bias_variable(shape):
        """ Creates tensorflow variables in a specified shape to be used for bias 
        Args:
            shape: a vector indicating the shape of the resulting tensor 2x2 being:
                   [2, 2]
        Returns:
            A tensor of the defined shape
        """
        initial = tf.constant(1e-6, shape=shape)
        return tf.Variable(initial)

    @staticmethod
    # TODO(yo@dino.io): Add documentation
    def variable_summaries(var, name):
        with tf.name_scope("summaries"):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('stddev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)

    @staticmethod
    # TODO(yo@dino.io): Add documentation
    def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
        return tf.nn.conv2d(x, W, strides=strides, padding=padding)

    @staticmethod
    # TODO(yo@dino.io): Add documentation
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='SAME')


class FrameQueue:
    """ A class that holds the last n frames of the game state 
    Attributes:
        size: the number of frames to be stored 
        memory: A deque that stores the frames 
    """

    def __init__(self, size):
        """ Initialize the memory
        """
        self.size = size
        self.memory = deque(maxlen=size)
        self.zipped = []

    def add_frame(self, frame):
        """ Adds a frame to memory
        Args:
            frame: a 2D vector of the game frame
        """
        self.memory.append(frame)

    def filled(self):
        return len(self.memory) == self.size

    def zip(self):
        return np.reshape(np.dstack(self.memory).flatten(), (len(self.memory[0]), len(self.memory[0][0]), self.size))
            

class ReplayMemory:
    """ A class that represents the replay memory for a Reinforcement learning
    network
    Attributes:
        size: the number of experiance tuples to be stored
        memory: A deque that stores the experiances
    """

    def __init__(self, size):
        """ Initialize the memory
        """
        self.size = size
        self.memory = deque(maxlen=size)

    def get_batch(self, size):
        """ gets a random batch of experiances
        Args:
            size: the size of the batch to retrieve
        Returns:
            A list of experiances of size = argument size
        """
        return random.sample(self.memory, size)

    def add_memory(self, experiance):
        """ Adds an experiance to memory
        Args:
            experiance: a tuple containting all of the information for an experiance
        """
        self.memory.append(experiance)
