import random
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras.optimizers import adam
from keras.activations import relu, linear
from sklearn.preprocessing import MinMaxScaler


class DQN:
    def __init__(self, dat, fgt, action_space, state_space, hidden_layer=(150, 120), batch_size=64, epsilon=1.0,
                 epsilon_decay=0.996, epsilon_min=0.1, gamma=0.99, alpha=0.001, func_approximation='NN'):
        self.dat = dat
        self.fgt = fgt
        self.act_space = action_space
        self.stat_space = state_space
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.hidden_layer = hidden_layer
        self.func_approximation = func_approximation
        self.exp_buffer = []
        self.target_network = []
        self.model = self.construct_model()
        np.random.seed(0)

    def construct_model(self):
        model = Sequential()
        if self.func_approximation == 'NN':  # control the function approximator we will use
            model.add(Dense(self.hidden_layer[0], input_dim=self.stat_space, activation=relu))
            for i in range(len(self.hidden_layer) - 1):
                model.add(Dense(self.hidden_layer[i + 1], activation=relu))
            model.add(Dense(self.act_space, activation=linear))
            model.compile(loss='mse', optimizer=adam(lr=self.alpha))
        if self.func_approximation == 'Linear':
            model.add(Dense(self.act_space, input_dim=self.stat_space, activation=linear))
            model.compile(loss='mse', optimizer=adam(lr=self.alpha))
        return model

    def greedy(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.act_space)
        else:
            # preprocessing
            normalization = MinMaxScaler()
            state = normalization.fit_transform(np.array(state).reshape(-1, 1)).reshape(1, -1)
            action = np.argmax(self.model.predict(state)[0])
        return action

    def replay(self):

        if len(self.exp_buffer) < self.batch_size: return
        minibatch = random.sample(self.exp_buffer, self.batch_size)
        states = np.squeeze(np.array([self.fgt.sum(i[0]) if self.dat.store_fea
                                      else self.fgt.sum(self.fgt.get_fea(i[0])) for i in minibatch]))
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.squeeze(np.array([self.fgt.sum(i[3]) if self.dat.store_fea
                                           else self.fgt.sum(self.fgt.get_fea(i[3])) for i in minibatch]))
        dones = np.array([i[4] for i in minibatch])

        # preprocessing
        normalization = MinMaxScaler()
        for i in range(len(states)):
            states[i] = normalization.fit_transform(states[i].reshape(-1, 1)).reshape(1, -1)
        for i in range(len(next_states)):
            next_states[i] = normalization.fit_transform(next_states[i].reshape(-1, 1)).reshape(1, -1)

        Qtargets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        Qvalue = self.model.predict_on_batch(states)
        Qvalue[[i for i in range(self.batch_size)], [actions]] = Qtargets
        self.model.fit(states, Qvalue, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def target_network_fit(self):
        if len(self.target_network) < self.batch_size: return
        states = np.squeeze(np.array([i[0] for i in self.target_network]))
        actions = np.array([i[1] for i in self.target_network])
        rewards = np.array([i[2] for i in self.target_network])
        next_states = np.squeeze(np.array([i[3] for i in self.target_network]))
        dones = np.array([i[4] for i in self.target_network])
        Qtargets = rewards + self.gamma * (np.amax(self.model.predict_on_batch(next_states), axis=1)) * (1 - dones)
        Qvalue = self.model.predict_on_batch(states)
        Qvalue[[i for i in range(self.batch_size)], [actions]] = Qtargets
        self.model.fit(states, Qvalue, epochs=1, verbose=0)
        self.target_network = []
        if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay

    def fit(self, state, action, reward, next_state, done, exp_replay=True, use_target=True):
        if exp_replay:  # control the experience replay on and off
            self.exp_buffer.append([state, action, reward, next_state, done])
            self.replay()
        else:
            if use_target:  # control the target network on and off
                self.target_network.append([state, action, reward, next_state, done])
                self.target_network_fit()
            else:
                Qtarget = reward + self.gamma * (np.amax(self.model.predict(next_state), axis=1)) * (1 - done)
                Qvalue = self.model.predict(state)
                Qvalue[0, action] = Qtarget
                self.model.fit(state, Qvalue, epochs=1, verbose=0)
                if self.epsilon > self.epsilon_min: self.epsilon *= self.epsilon_decay
