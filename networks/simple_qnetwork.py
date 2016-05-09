from network import QNetwork, ReplayMemory, FrameQueue
import numpy as np
import tensorflow as tf

class SimpleQNetwork(QNetwork):
    """ Network class that takes images of a game and learns best actions
    Attributes:
        image_height: An integer indicating the height of the input image
        image_width: An integer indicating the width of the input image
        model_dir: A string indicating the directory to save the model
        num_actions: An integer indicating the number of actions the nework can decide from
        input_frame_length: An integer indicating the number of frames to stack for the network input
        batch_size: An integer indicating how many experiances to train on every train tick
        gamma: A float indicating the discount factor for rewards propogating to sibling states
        observe_ticks: An integer indicating the umber of ticks to observe before training

        t: An integer indicating the number of game ticks processed by this network
        replay_memory: The replay memory for this network
        previous_sa_pair: A dict of the last state action pair the network produced
                          {'state': state, 'action': action}
        frame_queue: stores n number of previous states, where n = input_frame_length

        start_epsilon: A float indicating the starting epsilon value
        end_epsilon: A float indicating the final epsilon value
        epsilon_degrade_steps: An integer indicating how many steps to degrade epsilon over
        current_epsilon: A float indicating the current epsilon value
    """

    def __init__(self, model_dir, num_actions, image_height, image_width,  
            memorysize = 20000, input_frame_length = 4, observe_ticks = 10000, batch_size = 32):
        """ Initialises network
        """
        super(SimpleQNetwork, self).__init__(model_dir, num_actions)
        self.image_height = image_height
        self.image_width = image_width
        self.input_frame_length = input_frame_length
        self.batch_size = batch_size
        self.gamma = 0.99
        self.observe_ticks = observe_ticks

        self.t = 0
        self.replay_memory = ReplayMemory(memorysize)
        self.previous_sa_pair = ()
        self.frame_queue = FrameQueue(input_frame_length)

        self.start_epsilon = 0
        self.end_epsilon = 0
        self.epsilon_degrade_steps = 0
        self.current_epsilon = self.start_epsilon 

        self.start_session()

    def generate_experiance(self, state):
        """ generates an experiance from the current and previous states
        Args:
            state: a representation of the game state
        Returns:
            A dict that represents the experiance in the form:
            {'state_t'  : the previous state,
             'action'   : the action taken in that state,
             'reward'   : the reward recieved after taking the action,
             'state_t+1': the resulting state after taking the action,
             'terminal' : booling indicating if this was the end state of an episonde}
        """
        state_t = self.frame_queue.zip()
        # TODO(yo@dino.io): tidy up flatten
        self.frame_queue.add_frame(np.array(state['frame']))
        state_t1 = self.frame_queue.zip()

        if self.previous_sa_pair['state']['terminal'] is True:
            reward = 0
        else:
            reward = state['reward']

        action = [0] * self.num_actions
        action[self.previous_sa_pair['action']] = 1

        return {'state_t': state_t,
                'action': action,
                'reward': reward,
                'state_t+1': state_t1,
                'terminal': state['terminal']} 
    
    def game_train_tick(self, state):
        """ performs a game tick and does a round of training if possible
        Args:
            state: a representation of the game state
        Returns:
            An Integer between 0 and num_actions to indicate
            the action chosen at random
        """
        self.t += 1
        if self.t % 1000 == 0:
            self.save_model(self.t)

        if self.t < self.observe_ticks:
            if self.t < self.input_frame_length+1:
                # TODO(yo@dino.io): tidy up flatten
                self.frame_queue.add_frame(np.array(state['frame']))
            else:
                self.replay_memory.add_memory(self.generate_experiance(state))

            action = self.pick_random_action()
            self.previous_sa_pair = {'state': state, 'action': action}
            return action
        else:
            self.replay_memory.add_memory(self.generate_experiance(state))

            stacked_state = self.frame_queue.zip()
            action = self.pick_action(stacked_state, self.current_epsilon)
            self.previous_sa_pair = {'state': state, 'action': action}

            if self.current_epsilon > self.end_epsilon:
                self.current_epsilon -= (self.start_epsilon - self.end_epsilon) / self.epsilon_degrade_steps
            
            self.train_step()
            return action

    def game_tick(self, state):
        """ performs a game tick without doing any training
        Args:
            state: a representation of the game state
        Returns:
            An Integer between 0 and num_actions to indicate
            the action chosen at random
        """
        if self.frame_queue.filled():
            self.frame_queue.add_frame(np.array(state['frame']))
            return self.pick_action(self.frame_queue.zip(), 0)
        else:
            self.frame_queue.add_frame(np.array(state['frame']))
            return self.pick_random_action()

    def train_step(self):
        """ performs a train step by picking a minibatch of experiances from memory
        """
        batch = self.replay_memory.get_batch(self.batch_size)
        state_t_batch = []
        action_batch = []
        reward_batch = []
        state_t1_batch = []

        for experiance in batch:
            state_t_batch.append(experiance['state_t'])
            action_batch.append(experiance['action'])
            reward_batch.append(experiance['reward'])
            state_t1_batch.append(experiance['state_t+1'])

        target_batch = []

        yj = self.sess.run(self.network['output'], feed_dict={self.network['input']: state_t1_batch})

        for i in range(0, len(batch)):
            if batch[i]['terminal'] is True:
                target_batch.append(reward_batch[i])
            else:
                target_batch.append(reward_batch[i] + self.gamma * np.max(yj[i]))

        output, summary = self.sess.run([self.network['output'], self.summaries], feed_dict={
            self.network['input']: state_t_batch,
            self.network['action']: action_batch,
            self.network['target']: target_batch})

        self.summary_writer.add_summary(summary, self.t)


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
        # IO Placeholders
        input = tf.placeholder(tf.float32, shape=[None, 
            self.image_height, self.image_width, self.input_frame_length])

        target = tf.placeholder(tf.float32, shape=[None])
        action = tf.placeholder(tf.float32, shape=[None, self.num_actions])

        # First Layer
        W_conv1 = QNetwork.weight_variable([5, 5, self.input_frame_length, 32])
        b_conv1 = QNetwork.bias_variable([32])

        h_conv1 = tf.nn.relu(QNetwork.conv2d(input, W_conv1) + b_conv1)
        h_pool1 = QNetwork.max_pool_2x2(h_conv1)

        # Second Layer
        W_conv2 = QNetwork.weight_variable([5, 5, 
            32, 64])
        b_conv2 = QNetwork.bias_variable([64])

        h_conv2 = tf.nn.relu(QNetwork.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = QNetwork.max_pool_2x2(h_conv2)

        # Third Layer
        W_fc1 = QNetwork.weight_variable([((self.image_height / 4) * (self.image_width / 4) * 64), 1024])
        b_fc1 = QNetwork.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, ((self.image_height / 4) * (self.image_width / 4)) * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # Fourth Layer
        W_fc2 = QNetwork.weight_variable([1024, self.num_actions])
        b_fc2 = QNetwork.bias_variable([self.num_actions])
        
        output = tf.matmul(h_fc1, W_fc2) + b_fc2

        # Train and Eval Steps 
        action_value = tf.reduce_sum(tf.mul(output, action), reduction_indices = 1)
        error = tf.reduce_mean(tf.square(target - action_value))
        training_step = tf.train.AdamOptimizer(1e-6).minimize(error)
        
        QNetwork.variable_summaries(output, 'output')
        QNetwork.variable_summaries(error, 'error')
        QNetwork.variable_summaries(W_fc2, 'final_weights')
        tf.scalar_summary('action/picked', np.argmax(output))
        
        return {'input': input, 
                'output': output, 
                'target': target, 
                'action': action, 
                'training_step': training_step}
