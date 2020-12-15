import os
import random
import itertools
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import MinMaxScaler


class FGT:

    def __init__(self, dat, constraints, batch_size):
        self.dat = dat
        self.constraints = constraints
        self.batch_size = batch_size
        self.valid_space = [[[] for i in range(self.dat.max_comp)], 0]
        self.valid_space_update()
        self.score_to_reward = lambda score: 1 / (1.001 - score)
        self.reward_to_score = lambda reward: 1.001 - 1 / reward
        self.normalization = lambda fea: MinMaxScaler().fit_transform(fea)

        self.sum = lambda fea: np.array([f.sum() for f in fea]).reshape(-1, 1)

        np.random.seed(0)

    def grow(self, state, action):

        fea = self.dat.fea[state] if self.dat.store_fea else self.get_fea(state)
        fea_name = self.dat.fea_name[state]

        # exp() defined as operation 'exp'
        if action == 'exp':
            if (fea_name[:3] == 'exp') | (fea_name[:3] == 'log'):
                reward = self.score_to_reward(0)
                next_state = state
                done = 2
            else:
                reward, next_state, done = self.operate('exp,' + str(state), lambda x: np.exp(x), fea, state)
        # log defined as operation 'log'
        elif action == 'log':
            if (fea_name[:3] == 'log') | (fea_name[:3] == 'exp') | \
                    (np.min(np.hstack(fea)) <= 0):
                reward = self.score_to_reward(0)
                next_state = state
                done = 2
            else:
                reward, next_state, done = self.operate('log,' + str(state), lambda x: np.log(x), fea, state)

        # ^-1 defined as operation 'p-1'
        elif action == 'p-1':
            if (fea_name[:3] == 'p-1') | (np.min(np.abs(np.hstack(fea))) == 0):
                reward = self.score_to_reward(0)
                next_state = state
                done = 2
            else:
                reward, next_state, done = self.operate('p-1,' + str(state), lambda x: np.power(x, -1), fea, state)


        # ^2 defined as operation 'p+2'
        elif action == 'p+2':
            if fea_name[:3] == 'sqr':
                reward = self.score_to_reward(0)
                next_state = state
                done = 2
            else:
                reward, next_state, done = self.operate('p+2,' + str(state), lambda x: np.power(x, 2), fea, state)

        # ^6 defined as operation 'p+3'
        elif action == 'p+3':
            if fea_name[:3] == 'cbr':
                reward = self.score_to_reward(0)
                next_state = state
                done = 2
            else:
                reward, next_state, done = self.operate('p+3,' + str(state), lambda x: np.power(x, 3), fea, state)

        # ^sqrt defined as operation 'sqr'
        elif action == 'sqr':
            if (fea_name[:3] == 'p+2') | (np.min(np.hstack(fea)) < 0):
                reward = self.score_to_reward(0)
                next_state = state
                done = 2
            else:
                reward, next_state, done = self.operate('sqr,' + str(state), lambda x: np.power(x, 1 / 2), fea, state)

        # ^cbrt defined as operation 'cbr'
        elif action == 'cbr':
            if fea_name[:3] == 'p+3':
                reward = self.score_to_reward(0)
                next_state = state
                done = 2
            else:
                reward, next_state, done = self.operate('cbr,' + str(state), lambda x: np.power(x, 1 / 3), fea, state)

        # sin defined as operation 'sin'
        elif action == 'sin':
            reward, next_state, done = self.operate('sin,' + str(state), lambda x: np.sin(x), fea, state)

        # cos defined as operation 'cos'
        elif action == 'cos':
            reward, next_state, done = self.operate('cos,' + str(state), lambda x: np.cos(x), fea, state)


        # - defined as 'neg'
        elif action == 'neg':
            if fea_name[:3] == 'neg':
                reward = self.score_to_reward(0)
                next_state = state
                done = 2
            else:
                reward, next_state, done = self.operate('neg,' + str(state), lambda x: -x, fea, state)


        # + defined as operation 'add'
        elif action == 'add':
            reward, next_state, done = self.binary_operation('add,' + str(state), lambda x: (x[0] + x[1]), fea, state)


        # - defined as operation 'sub'
        elif action == 'sub':
            reward, next_state, done = self.binary_operation('sub,' + str(state), lambda x: (x[0] - x[1]), fea, state)


        # * defined as operation 'mul'
        elif action == 'mul':
            reward, next_state, done = self.binary_operation('mul,' + str(state), lambda x: (x[0] * x[1]), fea, state)


        # / defined as operation 'div'
        elif action == 'div':
            if np.min(np.abs(np.hstack(fea))) == 0:
                reward = self.score_to_reward(0)
                next_state = state
                done = 2
            else:
                reward, next_state, done = self.binary_operation('div,' + str(state),
                                                                 lambda x: (x[1] / x[0]), fea, state)


        # stop growing as 'end'
        elif action == 'end':
            if self.dat.reward[state] != -1:
                reward = self.dat.reward[state]
            else:
                reward = self.compute_reward(fea)
                self.dat.reward[state] = reward
            next_state = state
            done = 1

        else:
            self.dat.printf(
                "[" + str(self.dat.local_time()) + "]" + " Warning: Invalid action detected: " + str(action),
                filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
            reward = self.score_to_reward(0)
            next_state = state
            done = 2
        return reward, next_state, done

    def operate(self, operator, func, fea, state, state_j=None):

        reward, next_state, done = self.check_redundant_operations(state, operator)
        if reward is not None:
            return reward, next_state, done

        if state_j is not None:
            var = (fea, self.dat.fea(state_j)) if self.dat.store_fea else (fea, self.get_fea(state_j))
        else:
            var = fea

        reward, next_state, done = self.generate_fea(operator, func, var, state, state_j)
        # self.dat.printf("[" + str(self.dat.local_time()) + "]" + '          reward calculated',
        #                 filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
        return reward, next_state, done

    def check_redundant_operations(self, state, operator):

        # if the node can't further grow
        if self.dat.done[state]:
            reward = self.score_to_reward(0)
            next_state = state
            done = self.dat.done[state]
            return reward, next_state, done

        # if the operation has been operated before
        if operator in self.dat.index:
            index = self.dat.index[operator]
            # If the operation has FloatingPointError
            if index == -1:
                reward = self.score_to_reward(0)
                next_state = state
                done = 2
            else:
                next_state = index
                if self.dat.reward[index] != -1:
                    reward = self.dat.reward[index]
                else:
                    ############## must start new! #################
                    reward = self.compute_reward(self.dat.fea[index] if self.dat.store_fea else self.get_fea(index))
                    self.dat.reward[index] = reward
                done = self.dat.done[index]
            return reward, next_state, done

        # Check equivalent operations
        if operator[:3] in ['add', 'mul']:
            new_operator = operator.split(',')
            new_operator = ','.join([new_operator[0], new_operator[2], new_operator[1]])
            if new_operator in self.dat.index:
                index = self.dat.index[new_operator]
                self.dat.index[operator] = index
                next_state = index
                if self.dat.reward[index] != -1:
                    reward = self.dat.reward[index]
                else:
                    ############## must start new! #################
                    reward = self.compute_reward(self.dat.fea[index] if self.dat.store_fea else self.get_fea(index))
                    self.dat.reward[index] = reward
                done = self.dat.done[index]
                return reward, next_state, done
        return None, None, None
    
    def generate_fea(self, operator, func, var, state, state_j):
        # Otherwise, generate a new feature

        new_fea = self.constraints.np_operate(func, var)
        comp = self.dat.comp[state] + 1 if state_j is None else self.dat.comp[state] + self.dat.comp[state_j] + 1
        # self.dat.printf("[" + str(self.dat.local_time()) + "]" + '          new fea generated',
        #                 filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
        if new_fea is not None:
            reward = self.compute_reward(new_fea)
            next_state = len(self.dat.reward)
            done = 1 if comp >= self.dat.max_comp else 0
            new_fea = self.constraints.insert_constants(new_fea) if self.dat.use_constants else new_fea
            if new_fea is not None:
                self.dat.index[operator] = len(self.dat.fea_name)
                if self.dat.store_fea:
                    self.dat.fea.append(new_fea)
                self.dat.fea_name.append(operator)
                self.dat.reward.append(reward)
                self.dat.comp.append(comp)
                self.dat.done.append(done)
                return reward, next_state, done

        reward = self.score_to_reward(0)
        next_state = state
        done = 2
        self.dat.index[operator] = -1
        return reward, next_state, done

    def binary_operation(self, operator, func, fea, state):
        reward_list = []
        next_state_list = []
        done_list = []
        batch_set = self.select_batch_set(state)

        # Apply binary operations on subset features
        for state_j in batch_set:
            reward, next_state, done = self.operate(operator + ',' + str(state_j), func, fea, state, state_j)
            reward_list.append(reward)
            next_state_list.append(next_state)
            done_list.append(done)
        reward = self.score_to_reward(0) if reward_list == [] else max(reward_list)

        # If the same operation has been operated before
        if operator in self.dat.index:
            stored_index = self.dat.index[operator]
            stored_reward = self.dat.reward[stored_index] if stored_index >= 0 else self.score_to_reward(0)
            stored_done = self.dat.done[stored_index] if stored_index >= 0 else 2
            if stored_reward >= reward:
                next_state = stored_index if stored_index >= 0 else state
                reward = stored_reward
                done = stored_done
            else:
                next_state = next_state_list[int(np.argmax(reward_list))]
                done = done_list[int(np.argmax(reward_list))]
                self.dat.index[operator] = next_state

        # If it is a new operation
        else:
            # If the reward list is empty
            if not reward_list:
                try:
                    reward, next_state, done = self.binary_operation(operator, func, fea, state)
                except Exception:
                    self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Warning: No available rewards",
                                    filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
                    next_state = state
                    done = 2
            else:
                next_state = next_state_list[int(np.argmax(reward_list))]
                if next_state == state:
                    self.dat.index[operator] = -1
                    done = 2
                else:
                    self.dat.index[operator] = next_state
                    done = done_list[int(np.argmax(reward_list))]
        return reward, next_state, done

    def valid_space_update(self):
        not_done_space = np.where(np.array(self.dat.done[self.valid_space[1]:]) == 0)[0] \
                         + self.valid_space[1]
        for i in range(min(np.max(self.dat.comp) + 1, self.dat.max_comp)):
            self.valid_space[0][i] += list(not_done_space[np.array(self.dat.comp)[not_done_space] == i])
        self.valid_space[1] = len(self.dat.done)

    def select_batch_set(self, state):
        # Select valid space
        valid_space = self.valid_space[0][0]
        for i in range(1, self.dat.max_comp - self.dat.comp[state]):
            valid_space = valid_space + self.valid_space[0][i]

        # Select subset for binary operation
        if self.batch_size < len(valid_space):
            batch_set = random.sample(valid_space, self.batch_size)
        else:
            batch_set = valid_space
        return batch_set

    def compute_reward(self, fea):
        clf = self.dat.clf
        if not self.dat.candidate_set[0]:
            # clf = clone(self.l_clf)
            if self.dat.cv is None:
                c_fea = self.combine_optimum_fea([], fea)
                # fea = self.normalization(fea)
                clf.fit(c_fea, self.dat.target)
                score = clf.score(c_fea, self.dat.target)
            else:
                c_fea = self.combine_optimum_fea([], fea)
                # fea = self.normalization(fea)
                score = np.mean(self.cross_val_score(clf, c_fea, self.dat.target, cv=self.dat.cv)[0])
        else:
            score = 0
            comb = itertools.product(*self.dat.candidate_set[0])
            if self.dat.cv is None:
                for i in comb:
                    c_fea = self.combine_optimum_fea(i, fea)
                    # fea = self.normalization(fea)
                    clf.fit(c_fea, self.dat.target)
                    score = max(score, clf.score(c_fea, self.dat.target))
            else:
                for i in comb:
                    c_fea = self.combine_optimum_fea(i, fea)
                    # fea = self.normalization(fea)
                    score = max(score, np.mean(self.cross_val_score(clf, c_fea, self.dat.target, cv=self.dat.cv)[0]))
        reward = self.score_to_reward(score)
        return reward

    def compute_candidate_scores(self, candidate_set):
        clf = self.dat.clf
        scores = []
        comb = itertools.product(*candidate_set)
        if self.dat.mode == 'c':
            if self.dat.cv is None:
                for i in comb:
                    c_fea = self.combine_optimum_fea(i)
                    clf.fit(c_fea, self.dat.target)
                    score = clf.score(c_fea, self.dat.target)
                    scores.append([i, score])
            else:
                for i in comb:
                    c_fea = self.combine_optimum_fea(i)
                    score = np.mean(self.cross_val_score(clf, c_fea, self.dat.target, cv=self.dat.cv)[0])
                    scores.append([i, score])
        else:
            if self.dat.cv is None:
                for i in comb:
                    c_fea = self.combine_optimum_fea(i)
                    clf.fit(c_fea, self.dat.target)
                    prediction = clf.predict(c_fea)
                    score = r2_score(self.dat.target, prediction)
                    rmse = np.sqrt(mean_squared_error(self.dat.target, prediction))
                    residue = self.dat.target - np.array(prediction)
                    scores.append([i, score, rmse, residue])
            else:
                for i in comb:
                    c_fea = self.combine_optimum_fea(i)
                    score, rmse = np.mean(self.cross_val_score(clf, c_fea, self.dat.target, cv=self.dat.cv), axis=1)
                    loo = LeaveOneOut()
                    prediction = []
                    for indices_train, index_test in loo.split(self.dat.target):
                        # clf = clone(self.clf)
                        clf.fit(c_fea[indices_train], self.dat.target[indices_train])
                        prediction.append(clf.predict(c_fea[index_test]))
                    residue = self.dat.target - np.array(prediction)
                    scores.append([i, score, rmse, residue])
        return scores

    def combine_optimum_fea(self, i, fea=None):
        if fea is not None:
            if self.dat.use_constants:
                fea = self.constraints.multiply_smoothing_function(fea)
            c_fea = self.sum(fea)
        else:
            c_fea = np.zeros((len(self.dat.fea[0]), 1))
        for j in i:
            optimum_fea = self.dat.fea[j] if self.dat.store_fea else self.get_fea(j)
            if self.dat.use_constants:
                optimum_fea = self.constraints.multiply_smoothing_function(optimum_fea)
            optimum_fea = self.sum(optimum_fea)
            c_fea = np.hstack((c_fea, optimum_fea))
        return c_fea if fea is not None else c_fea[:, 1:]

    def cross_val_score(self, clf, fea, target, cv):
        score = []
        if self.dat.mode == 'c':
            for i in range(cv):
                clf.fit(fea[self.dat.train[i]], target[self.dat.train[i]])
                score.append(clf.score(fea[self.dat.test[i]], target[self.dat.test[i]]))
            return [score]
        else:
            rmse = []
            for i in range(cv):
                clf.fit(fea[self.dat.train[i]], target[self.dat.train[i]])
                prediction = clf.predict(fea[self.dat.test[i]])
                score.append(r2_score(target[self.dat.test[i]], prediction))
                rmse.append(np.sqrt(mean_squared_error(target[self.dat.test[i]], prediction)))
            return [score, rmse]

    def get_fea(self, state):
        descriptor_indexes = self.trace_descriptor(state)
        fea = self.calculate_descriptor(descriptor_indexes)
        return fea

    def print_descriptor(self, descriptor_indexes, *weights):
        if type(descriptor_indexes) is not list:
            descriptor_indexes = self.trace_descriptor(int(descriptor_indexes))

        stored_indexes = {}
        i = 0
        for comp in range(len(descriptor_indexes)):
            for split_index in descriptor_indexes[comp]:
                if weights is not () and comp != 0:
                    stored_indexes[str(split_index[0])] = self.transform_operation(split_index, stored_indexes,
                                                                                   (weights[i], weights[i + 1]))
                    i += 2
                else:
                    stored_indexes[str(split_index[0])] = self.transform_operation(split_index, stored_indexes)
        descriptor = ''.join(stored_indexes[str(descriptor_indexes[-1][0][0])])
        return descriptor

    def calculate_descriptor(self, descriptor_indexes, *weights):
        if type(descriptor_indexes) is not list:
            descriptor_indexes = self.trace_descriptor(int(descriptor_indexes))
        stored_values = {}
        i = 0
        for comp in range(len(descriptor_indexes)):
            for split_index in descriptor_indexes[comp]:
                if weights is not () and comp != 0:
                    stored_values[str(split_index[0])] = self.calculate_operation(split_index, stored_values,
                                                                                  (weights[i], weights[i + 1]))
                    i += 2
                else:
                    stored_values[str(split_index[0])] = self.calculate_operation(split_index, stored_values)
        descriptor_value = stored_values[str(descriptor_indexes[-1][0][0])]
        # return descriptor_value
        return descriptor_value

    def trace_descriptor(self, index, descriptor_indexes=None):
        index = int(index)
        split_index = [index] + self.dat.fea_name[index].split(',')
        if descriptor_indexes is None:
            descriptor_indexes = [[] for i in range(self.dat.comp[index] + 1)]
        if split_index not in descriptor_indexes[self.dat.comp[index]]:
            descriptor_indexes[self.dat.comp[index]].append(split_index)
            if self.dat.comp[index] != 0 and split_index[1][:3] != 'can':
                for index in split_index[2:]:
                    self.trace_descriptor(index, descriptor_indexes)
        return descriptor_indexes

    def transform_operation(self, split_index, stored_indexes, w=None):
        if self.dat.use_constants:
            w = (self.dat.constants[split_index[0]][0], self.dat.constants[split_index[0]][1])

        w = ['', ''] if w is None else [str(w[0]) + '*', '+' if w[1] >= 0 else '' + str(w[1])]
        if split_index[1] == 'pri' or split_index[1][:3] == 'can':
            # index = ['(', w[0], '*'] + [split_index[2]] + [w[1], ')'] if w[0] is not '' else [split_index[2]]
            index = [split_index[2]]
        elif split_index[1] == 'exp':
            index = [w[0], 'exp('] + stored_indexes[split_index[2]] + [')', w[1]]
        elif split_index[1] == 'p-1':
            index = [w[0], '('] + stored_indexes[split_index[2]] + [')^-1', w[1]]
        elif split_index[1] == 'p+2':
            index = [w[0], '('] + stored_indexes[split_index[2]] + [')^2', w[1]]
        elif split_index[1] == 'p+3':
            index = [w[0], '('] + stored_indexes[split_index[2]] + [')^3', w[1]]
        elif split_index[1] == 'p+6':
            index = [w[0], '('] + stored_indexes[split_index[2]] + [')^6', w[1]]
        elif split_index[1] == 'sqr':
            index = [w[0], '('] + stored_indexes[split_index[2]] + [')^(1/2)', w[1]]
        elif split_index[1] == 'cbr':
            index = [w[0], '('] + stored_indexes[split_index[2]] + [')^(1/3)', w[1]]
        elif split_index[1] == 'log':
            index = [w[0], 'log('] + stored_indexes[split_index[2]] + [')', w[1]]
        elif split_index[1] == 'sin':
            index = [w[0], 'sin('] + stored_indexes[split_index[2]] + [')', w[1]]
        elif split_index[1] == 'cos':
            index = [w[0], 'cos('] + stored_indexes[split_index[2]] + [')', w[1]]
        elif split_index[1] == 'neg':
            index = [w[0], '(-'] + stored_indexes[split_index[2]] + [')', w[1]]
        elif split_index[1] == 'add':
            index = [w[0], '('] + stored_indexes[split_index[2]] + ['+'] \
                    + stored_indexes[split_index[3]] + [')', w[1]]
        elif split_index[1] == 'sub':
            index = [w[0], '('] + stored_indexes[split_index[2]] + ['-'] \
                    + stored_indexes[split_index[3]] + [')', w[1]]
        elif split_index[1] == 'mul':
            index = [w[0], '('] + stored_indexes[split_index[2]] + ['*'] \
                    + stored_indexes[split_index[3]] + [')', w[1]]
        elif split_index[1] == 'div':
            index = [w[0], '('] + stored_indexes[split_index[3]] + ['/'] \
                    + stored_indexes[split_index[2]] + [')', w[1]]
        else:
            self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Warning: Invalid operation \""
                            + str(split_index[1]) + "\"",
                            filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
            index = None
        return index

    def calculate_operation(self, split_index, stored_values, w=None):
        func = None
        if split_index[1] == 'pri' or split_index[1][:3] == 'can':
            value = self.dat.fea[int(split_index[0])]
            # if w is not None:
            #     func = lambda x: w[0]*x + w[1]
            #     value = self.constraints.np_operate(func, value)
            return value
        elif split_index[1] == 'exp':
            func = lambda x: np.exp(x)
        elif split_index[1] == 'p-1':
            func = lambda x: np.power(x, -1)
        elif split_index[1] == 'p+2':
            func = lambda x: np.power(x, 2)
        elif split_index[1] == 'p+3':
            func = lambda x: np.power(x, 3)
        elif split_index[1] == 'p+6':
            func = lambda x: np.power(x, 6)
        elif split_index[1] == 'sqr':
            func = lambda x: np.power(x, 1 / 2)
        elif split_index[1] == 'cbr':
            func = lambda x: np.power(x, 1 / 3)
        elif split_index[1] == 'log':
            func = lambda x: np.log(x)
        elif split_index[1] == 'sin':
            func = lambda x: np.sin(x)
        elif split_index[1] == 'cos':
            func = lambda x: np.cos(x)
        elif split_index[1] == 'neg':
            func = lambda x: -x
        if func is not None:
            variables = stored_values[split_index[2]]
            value = self.constraints.np_operate(func, variables)
            if w is not None:
                func = lambda x: w[0] * x + w[1]
                value = self.constraints.np_operate(func, value)
            return value

        if split_index[1] == 'add':
            func = lambda x: (x[0] + x[1])
        elif split_index[1] == 'sub':
            func = lambda x: (x[0] - x[1])
        elif split_index[1] == 'mul':
            func = lambda x: (x[0] * x[1])
        elif split_index[1] == 'div':
            func = lambda x: (x[1] / x[0])
        if func is not None:
            variables = (stored_values[split_index[2]], stored_values[split_index[3]])
            value = self.constraints.np_operate(func, variables)
            if w is not None:
                func = lambda x: w[0] * x + w[1]
                value = self.constraints.np_operate(func, value)
            return value

        self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Warning: Invalid operation \""
                        + str(split_index[1]) + "\"",
                        filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
        return None
