import gym
import tensorflow as tf
from collections import deque
from random import random, sample



MIN_EPSILON = 0.05
MAX_REPLAYS = 100


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
        self.gamme = 0.99
        self.num_steps = 0  
        self.input_width = obs_space.shape[0]
        self.output_width = act_space.n
        self.hidden_nodes = hidden_nodes

        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

        self.x = tf.placeholder(tf.float32, [None, self.input_width])
        self.y_ = self.y = tf.placeholder(tf.float32, [None, self.output_width])

        self.session = tf.InteractiveSession()
        self.session.run(tf.initialize_all_variables())

        
    def build_network(self): 
        """
        Constructs the network from the initialized dictionaries of weights & biases

        Returns: Final layer of the neural network
        """
        hidden1 = tf.nn.sigmoid(tf.matmul(self.x, self.weights['w1']) + self.biases['b1'])
        hidden2 = tf.nn.sigmoid(tf.matmul(hidden1, self.weights['w2']) + self.biases['b2'])
        hidden3 = tf.nn.sigmoid(tf.matmul(hidden2, self.weights['w3']) + self.biases['b3'])
        out = tf.nn.sigmoid(tf.matmul(hidden3, self.weights['out']) + self.biases['out'])

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
            'b1': tf.Variable(tf.random_normal([self.input_width])),
            'b2': tf.Variable(tf.random_normal([self.hidden_nodes])),
            'b3': tf.Variable(tf.random_normal([self.hidden_nodes])),
            'out': tf.Variable(tf.random_normal([self.output_width]))
        }

        
    def get_action(self, obs):
        """
        Returns the action the network should take
        Inputs: 
            obs- observation of environment (environment inputs)
        """
        self.epsilon *= self.epsilon_decay

        self.epsilon = max(self.epsilon, MIN_EPSILON)
        action = (self.act_space.sample() if self.epsilon > random()
            else tf.argmax(self.session.run(self.y, np.array([obs]))))

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


    def train(self):
        """
        Trains the neural network
        """
        # This network is a mess and I don't know what's going o
        current_batch_size = min(len(self.replay_memory), BATCH_SIZE)
        sample_batch = sample(self.replay_memory, current_batch_size)
        
        x = np.zeros((current_batch_size, self.input_width))
        y = np.zeros((current_batch_size, self.output_width))
        
        loss = 0
        for replay in range(current_batch_size):
            prev_state, curr_state, action, reward, done = sample_batch[replay]
            update = np.copy(self.model.predict(prev_state.reshape(1, self.input_width))[0])

        if done:
            update[action] = reward
        else:
            available_actions = self.model.predict(curr_state.reshape(1, self.input_width))[0]
            update[action] = reward + self.gamme*tf.argmax(available_actions)
        
        x[replay] = prev_state
        y[replay] = update

        loss += self.model.train_on_batch(x, y)


if __name__ == '__main__':
    env_name = "CartPole-v0"
    hidden_nodes = 200
    env = gym.make(env_name)
    #env.monitor.start('/tmp/{}-run'.format(env_name), force=True)

    agent = network(env.observation_space, env.action_space, hidden_nodes)
    # I think this is right?
    agent.y = tf.matmul(env.reset(), agent.weights['out']) + agent.biases['out']
    Q = tf.reduce_sum(agent.y, agent.y_, reduction_indices=1)
    # past here I'm not really sure what I'm doing    

    runs = 1501
    sequences = 249
    score = deque(maxlen=100)

    for run in range(runs):
        obs = env.reset()
        best_reward = reward = 0
        done = False

        for sequence in range(sequences):
            action = agent.get_action(obs)
            mem = env.step(action)
            agent.add_mem(mem, obs, action)

            new_obs, reward, _, _ = mem
            obs = new_obs
            best_reward += reward

            agent.train()

        score.append(best_reward)
        print(average(score))

    #env.monitor.close()
    sess.close() 
