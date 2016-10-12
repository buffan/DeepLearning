import gym
import tensorflow as tf
import numpy as np
from collections import deque
from random import random, sample



MIN_EPSILON = 0.05
MAX_REPLAYS = 100
BATCH_SIZE = 50
UPDATE = 200
DROPOUT_RATE = 0.5
GAMMA = 0.99


class network(object):
    """
    Represents a neural network
    """
    def __init__(self, obs_space, act_space, hidden_nodes):
        """
        Initializes the network
        """
        self.obs_space = obs_space
        self.act_space = act_space
        self.replay_memory = deque(maxlen=MAX_REPLAYS)
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.num_steps = 0  
        self.input_width = obs_space.shape[0]
        self.output_width = act_space.n
        self.hidden_nodes = hidden_nodes

        self.weights = self.initialize_weights()
        self.weights_ = self.initialize_weights()
        self.biases = self.initialize_biases()
        self.biases_ = self.initialize_biases()

        self.x = tf.placeholder(tf.float32, [None, self.input_width])
        self.Q_ = self.Q = tf.placeholder(tf.float32, [None, self.output_width])

 
    def build_network(self, weights, biases): 
        """
        Constructs the network from the initialized dictionaries of weights & biases

        Returns: Final layer of the neural network
        """
        hidden1 = tf.nn.relu(tf.matmul(self.x, weights['w1']) + biases['b1'])
        hidden2 = tf.nn.relu(tf.matmul(hidden1, weights['w2']) + biases['b2'])
        hidden3 = tf.nn.relu(tf.matmul(hidden2, weights['w3']) + biases['b3'])
        # add dropout
        hidden3 = tf.nn.dropout(hidden3, DROPOUT_RATE)
        out = tf.matmul(hidden3, weights['out']) + biases['out']

        return out


    def initialize_weights(self):
        """
        Initializes weights of the network with normalized random variables.
        Network contains 3 hidden layers and an output layer

        Returns: dictionary containing initialized weights of all layers
        """
        return {
            'w1': tf.Variable(tf.random_normal([self.input_width, self.hidden_nodes])),
            'w2': tf.Variable(tf.random_normal([self.hidden_nodes, self.hidden_nodes])),
            'w3': tf.Variable(tf.random_normal([self.hidden_nodes, self.hidden_nodes])),
            'out': tf.Variable(tf.random_normal([self.hidden_nodes, self.output_width]))
        }


    def initialize_biases(self):
        """
        Initializes biases of the network with normalized random variables.
        Network contains 3 hidden layers and an output layer

        Returns: dictionary containing initialized biases of all layers
        """
        return {
            'b1': tf.Variable(tf.random_normal([self.hidden_nodes])),
            'b2': tf.Variable(tf.random_normal([self.hidden_nodes])),
            'b3': tf.Variable(tf.random_normal([self.hidden_nodes])),
            'out': tf.Variable(tf.random_normal([self.output_width]))
        }


    def update_weights_and_biases(self):
        """
        Copies the weights and biases from Q_ to the Q network
        """
        functions = []
        for w_key, b_key in zip(self.weights.keys(), self.biases.keys()):
            functions.append(self.weights[w_key].assign(self.weights_[w_key]))
            functions.append(self.biases[b_key].assign(self.biases_[b_key]))
        
        return functions

    def get_action(self, obs, session):
        """
        Returns the action the network should take
        Inputs: 
            obs- observation of environment (environment inputs)
        """
        self.epsilon = max(self.epsilon, MIN_EPSILON)

        action = None
        if self.epsilon > random():
            action = self.act_space.sample()
        else:
            action_list = session.run(self.Q, feed_dict={self.x: np.array([obs])})[0]
            action = np.argmax(action_list)

        self.epsilon *= self.epsilon_decay

        return action


    def add_mem(self, mem, obs, action):
        """
        Add a memory to the experience replay
        Inputs:
            mem- return of env.step(action)
            obs- environment observation
            action- action taken
        """
        # deque automatically pops if len > MAX_REPLAYS
        curr_obs, reward, done, _ = mem
        self.replay_memory.append((obs, curr_obs, action, reward, done))
        self.num_steps += 1


def start_session():
    """
    Starts the tensorflow session and initializes the variables
    """
    session = tf.InteractiveSession()
    session.run(tf.initialize_all_variables())
    return session


if __name__ == '__main__':
    env_name = "CartPole-v0"
    hidden_nodes = 200
    env = gym.make(env_name)
    #env.monitor.start('/tmp/{}-run'.format(env_name), force=True)

    # Generate Q and Q_ networks
    agent = network(env.observation_space, env.action_space, hidden_nodes)
    agent.Q = agent.build_network(agent.weights, agent.biases)
    agent.Q_ = agent.build_network(agent.weights_, agent.biases_)

    # Training setup
    # Look for workaround of one_hot
    action_placeholder = tf.placeholder(tf.int32, [None], name="action_masks")
    action_mask = np.zeros([agent.output_width])
    #action_mask = tf.one_hot(action_placeholder, agent.output_width)
    QValue = tf.reduce_sum(tf.mul(agent.Q, action_mask), reduction_indices=1)
    Q_Value = tf.placeholder(tf.float32, [None,])

    loss = tf.reduce_mean(tf.square(QValue - Q_Value))
    training = tf.train.AdamOptimizer(0.0001).minimize(loss)
    update_functions = agent.update_weights_and_biases()
    
    sess = start_session()
       
    # Run and train network
    runs = 1501
    sequences = 250
    score = deque(maxlen=100)

    for run in range(runs):
        obs = env.reset()
        best_reward = reward = 0
        done = False

        for sequence in range(sequences):
            action = agent.get_action(obs, sess)
            mem = env.step(action)
            env.render()
            agent.add_mem(mem, obs, action)

            new_obs, reward, done, _ = mem
            obs = new_obs
            best_reward += reward
            
            # Create batch
            batch_size = min(len(agent.replay_memory), BATCH_SIZE)
            num_memories = len(agent.replay_memory)
            mem_indexes = np.random.choice(batch_size, num_memories, replace=True)
            memories_batch = []
            for index in mem_indexes:
                memories_batch.append(agent.replay_memory[index])
 
            # Test batch
            mem_next_observations, mem_observations, mem_actions, mem_rewards, mem_dones = zip(*memories_batch)
            next_Q_ = sess.run(agent.Q_, feed_dict={agent.x: mem_next_observations})

            y_ = []

            for index, _ in enumerate(mem_observations):
                if mem_dones[index]:
                    y_.append(mem_rewards[index])
                else:
                    curr_Q_ = next_Q_[index]
                    maxQ = max(curr_Q_)
                    y_.append(mem_rewards[index] + GAMMA*maxQ)
            
            feed = { agent.x: mem_observations,
                     Q_Value: y_,
                     action_placeholder: mem_actions
                   }
            sess.run([training], feed_dict=feed)

            training_step = run*sequences + sequence
            if training_step % UPDATE == 0:
                sess.run(update_functions)
            
            obs = new_obs
            if done:
                break
        if run % 30 == 0:
            print("Run: {}, Score: {}".format(run, best_reward))


#        score.append(best_reward)
#        print(average(score))

    #env.monitor.close()
    sess.close() 
