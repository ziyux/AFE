import os
import time
import numpy as np
import copy
from keras.models import load_model

from AFE_FGT import FGT
from AFE_DQN import DQN
from AFE_DAT import DAT
from AFE_Constraints import Constraints

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt


def rename_dir(directory, new_directory, i=1):
    try:
        new_directory = ''.join([new_directory[:-(2 + len(str(i - 1)))], '(', str(i), ')'])
        os.rename(directory, new_directory)
    except OSError:
        i += 1
        new_directory = rename_dir(directory, new_directory, i)
    return new_directory


def make_dir(directory, resume):
    try:
        os.mkdir(directory)
    except OSError:
        if not resume:
            new_directory = rename_dir(directory, ''.join([directory, '(1)']))
            print('Warning: Moving existing directory \"' + directory + '\" to \"' + new_directory + '\".')
            make_dir(directory, resume)
    return directory


class AFE(object):
    def __init__(self, filename, operation_set, mode, resume=False, episodes=None, max_comp=3,
                 batch_size=1000, threshold=2, clf=None, cv=None, use_constants=True, store_fea=True,
                 output_frequency=3, skip_read=False):
        dim = int(resume) if resume > 0 else 1
        self.operation_set = operation_set
        self.is_new_file = False if resume else True
        directory = make_dir(os.path.join(os.getcwd(), filename[:-4] + '_Results'), resume)
        make_dir(os.path.join(directory, str(dim) + 'D_feature_space'), resume)
        self.threshold = threshold
        self.output_frequency = output_frequency

        self.dat = DAT(filename, resume, dim, mode, operation_set, max_comp, clf, cv, use_constants, store_fea,
                       directory, skip_read)
        self.constraints = Constraints(self.dat)
        self.fgt = FGT(self.dat, self.constraints, batch_size=batch_size)
        self.policies = []
        self.setup_dqn(hidden_layer=(150, 120), batch_size=64, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05,
                       gamma=0.99, alpha=0.001, func_approximation='NN')
        self.episodes = self.dat.pri_num if not episodes else episodes
        self.initial_state = 0

    def setup_dqn(self, hidden_layer=(150, 120), batch_size=64, epsilon=1.0, epsilon_decay=0.99, epsilon_min=0.05,
                  gamma=0.99, alpha=0.001, func_approximation='NN', update_para=True):
        if update_para:
            self.dqn_para = {'dat': self.dat, 'fgt': self.fgt, 'action_space': len(self.operation_set),
                             'state_space': len(self.dat.fea[0]), 'hidden_layer': hidden_layer,
                             'batch_size': batch_size, 'epsilon': epsilon, 'epsilon_decay': epsilon_decay,
                             'epsilon_min': epsilon_min, 'gamma': gamma, 'alpha': alpha,
                             'func_approximation': func_approximation}
        dqn = DQN(dat=self.dqn_para['dat'], fgt=self.dqn_para['fgt'], action_space=self.dqn_para['action_space'],
                  state_space=self.dqn_para['state_space'], hidden_layer=self.dqn_para['hidden_layer'],
                  batch_size=self.dqn_para['batch_size'], epsilon=self.dqn_para['epsilon'],
                  epsilon_decay=self.dqn_para['epsilon_decay'], epsilon_min=self.dqn_para['epsilon_min'],
                  gamma=self.dqn_para['gamma'], alpha=self.dqn_para['alpha'],
                  func_approximation=self.dqn_para['func_approximation'])
        if update_para:
            if self.dat.resume:
                dqn.model = load_model(os.path.join(self.dat.dir, str(self.dat.dim) + 'D_NN' + '.h5'))
        if len(self.policies) < self.dat.dim:
            self.policies.append(dqn)
        else:
            self.policies[-1] = dqn

    def increase_dim(self, increase_dim=True, start_new=True, use_residue=False):
        if increase_dim is not True:
            return
        self.dat.candidate_set = copy.deepcopy(self.dat.candidate_set_temp)
        if use_residue and self.dat.candidate_set_size == 1:
            self.dat.target = self.fgt.compute_candidate_scores(self.dat.candidate_set[0])[0][3]

        # Delete the original feature space and create new one
        if start_new:
            fea = self.dat.fea[:self.dat.pri_num]
            fea_name = self.dat.fea_name[:self.dat.pri_num]
            comp = self.dat.comp[:self.dat.pri_num]
            done = self.dat.done[:self.dat.pri_num]
            constants = self.dat.constants[:self.dat.pri_num]
            self.dat.index = {}
            for i in range(len(fea_name)):
                self.dat.index[fea_name[i]] = i
            for i in range(len(self.dat.candidate_set[0])):
                for j in range(len(self.dat.candidate_set[0][i])):
                    fea.append(self.dat.candidate_set[1][i]['fea'][j])
                    fea_name.append(self.dat.candidate_set[1][i]['fea_name'][j])
                    comp.append(self.dat.candidate_set[1][i]['comp'][j])
                    done.append(self.dat.candidate_set[1][i]['done'][j])
                    if self.dat.use_constants:
                        constants.append(self.dat.candidate_set[1][i]['constants'][j])
                    self.dat.candidate_set[0][i][j] = len(fea) - 1
                    self.dat.index[fea_name[-1]] = self.dat.candidate_set[0][i][j]

            self.dat.fea = fea
            self.dat.fea_name = fea_name
            self.dat.comp = comp
            self.dat.done = done
            self.dat.constants = constants
            self.fgt.valid_space = [[[] for i in range(self.dat.max_comp)], 0]
            self.fgt.valid_space_update()

        self.dat.reward = [-1 for i in self.dat.fea_name]
        self.dat.dim += 1
        self.dat.score_list = []
        self.dat.max_score = 0
        self.is_new_file = True
        self.setup_dqn(update_para=False)
        make_dir(os.path.join(self.dat.dir, str(self.dat.dim) + 'D_feature_space'), False)
        self.dat.start_time = time.time()
        self.dat.save_start_tag = 0
        ##########################
        # self.dat.pri_num = len(self.dat.fea_name)
        #########################
        self.save_intermediate_data()

    def terminate(self, solved=False):
        if solved:
            optimum_fea = np.array(self.dat.reward).argsort()[::-1][:self.dat.candidate_set_size]
            self.dat.candidate_set[0].append(optimum_fea)

    def print_results(self, optimal_fea, candidate_scores):
        self.dat.printf("\nEpisodes: " + str(len(self.dat.score_list)) + "\nSamples: " + str(len(self.dat.fea[0])) +
                        "    Generated Features: " + str(len(self.dat.fea_name)) + "     Running Time: " +
                        str(time.time() - self.dat.start_time) + "\n",
                        filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_descriptors' + '.txt'),
                        is_new_file=self.is_new_file)
        self.is_new_file = False if self.is_new_file else False

        if self.dat.mode == 'c':
            for fea in optimal_fea:
                descriptors = ""
                for x in candidate_scores[fea][0]:
                    descriptors += "Descriptor[" + str(x) + "]: " + str(self.fgt.print_descriptor(int(x))) + '\n'
                self.dat.printf(descriptors + "Score: " + str(candidate_scores[fea][1]) + "\n",
                                filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_descriptors' + '.txt'),
                                is_new_file=self.is_new_file)
        else:
            for fea in optimal_fea:
                descriptors = ""
                for x in candidate_scores[fea][0]:
                    descriptors += "Descriptor[" + str(x) + "]: " + str(self.fgt.print_descriptor(int(x))) + '\n'
                self.dat.printf(descriptors + "Score: " + str(candidate_scores[fea][1])
                                + "    RMSE: " + str(candidate_scores[fea][2]) + "\n",
                                filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_descriptors' + '.txt'),
                                is_new_file=self.is_new_file)

    def plot_results(self):
        # Plot the reward for each episode
        plt.title('Scores of each episode')
        plt.xlabel('Episode')
        plt.plot(list(range(len(self.dat.score_list))), self.dat.score_list)
        plt.savefig(os.path.join(self.dat.dir, str(self.dat.dim) + 'D_scores_of_each_episode'))
        # plt.show()
        plt.clf()
        # Plot average rewards
        n = max(1, int(len(self.dat.score_list) / 10))
        cumsum, moving_aves = [0], []
        for i, x in enumerate(self.dat.score_list, 1):
            cumsum.append(cumsum[i - 1] + x)
            if i >= n:
                moving_ave = (cumsum[i] - cumsum[i - n]) / n
                # can do stuff with moving_ave here
                moving_aves.append(moving_ave)
        plt.title('Average scores of every ' + str(n) + ' episodes')
        plt.xlabel('Episode')
        plt.plot(range(len(moving_aves)), moving_aves)
        plt.savefig(os.path.join(self.dat.dir, str(self.dat.dim) + 'D_average_scores'))
        # plt.show()
        plt.clf()

    def save_intermediate_data(self):
        self.dat.save_stop_tag = len(self.dat.fea_name)
        if self.dat.save_start_tag != self.dat.save_stop_tag:
            self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Start saving data: " +
                            str(self.dat.save_start_tag) + "-" + str(self.dat.save_stop_tag),
                            filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
            self.dat.output_data(self.policies[-1])
            self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Data saving completed: " +
                            str(self.dat.save_start_tag) + "-" + str(self.dat.save_stop_tag) + "\n",
                            filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
            self.dat.save_start_tag = self.dat.save_stop_tag

    def update_candidate_set(self, optimum_fea):
        self.dat.candidate_set_temp[0] = self.dat.candidate_set[0] + [optimum_fea]
        self.dat.candidate_set_temp[1] = self.dat.candidate_set[1] + \
                                         [{'fea': [self.dat.fea[i] for i in optimum_fea]
                                          if self.dat.store_fea else [self.fgt.get_fea(i) for i in optimum_fea],
                                           'fea_name': ['can_' + str(
                                               len(self.dat.candidate_set[1]) + 1) + 'D,' +
                                                        self.fgt.print_descriptor(int(f)) for f in
                                                        optimum_fea],
                                           'comp': [self.dat.comp[i] for i in optimum_fea],
                                           'done': [self.dat.done[i] for i in optimum_fea],
                                           'constants': [self.dat.constants[i] for i in optimum_fea]
                                           if self.dat.use_constants else None,
                                           'descriptor_indexes': [self.fgt.trace_descriptor(int(f)) for f in
                                                                  optimum_fea]}]
        return self.dat.candidate_set_temp[0]

    def select_optimal_fea_set(self):
        optimum_fea = list(np.array(self.dat.reward).argsort()[::-1][:self.dat.candidate_set_size])
        candidate_set = self.update_candidate_set(optimum_fea)
        candidate_scores = self.fgt.compute_candidate_scores(candidate_set)
        optimal_fea = np.array([x[1] for x in candidate_scores]).argsort()[-self.dat.candidate_set_size:][::-1]
        return optimal_fea, candidate_scores

    def reset(self):
        if self.initial_state == len(list(range(0, self.dat.pri_num))):
            self.initial_state = 0
            self.fgt.valid_space_update()
        state = list(range(0, self.dat.pri_num))[self.initial_state]
        self.initial_state += 1
        return state

    # def reset(self):
    #     self.fgt.valid_space_update()
    #     return 0

    def check_exit_conditions(self, max_running_time):
        if (self.dat.max_score >= self.threshold) or \
                (max_running_time and (time.time() - self.dat.start_time >= max_running_time)):
            if self.dat.max_score >= self.threshold:
                self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Training has completed\n",
                                filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
            else:
                self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Attain run time limit\n",
                                filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
            optimal_fea, candidate_scores = self.select_optimal_fea_set()
            self.save_intermediate_data()
            self.print_results(optimal_fea, candidate_scores)
            self.plot_results()
            return True

    def explore(self, max_iter=1000, max_running_time=False):

        self.dat.printf("[" + str(self.dat.local_time()) + "]" +
                        ' Start exploring ' + str(self.dat.dim) + 'D descriptors\n',
                        filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))

        for iteration in range(max_iter):
            for episode in range(self.episodes):
                state = self.reset()
                scores = []

                while True:
                    action = self.policies[-1].greedy(self.fgt.sum(self.dat.fea[state]) if self.dat.store_fea
                                                      else self.fgt.sum(self.fgt.get_fea(state)))

                    # record FGT growing path
                    self.dat.printf("[" + str(self.dat.local_time()) + "]" + ' path ==> ' + ' '.join([
                        'state:', str(state), 'operation:', str(self.operation_set[action])]),
                                    filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))

                    reward, next_state, done = self.fgt.grow(state, self.operation_set[action])
                    done = 1 if done > 0 else 0
                    score = self.fgt.reward_to_score(reward)
                    scores.append(score)

                    # record FGT growing path
                    self.dat.printf("[" + str(self.dat.local_time()) + "]" + '      ==> ' + ' '.join([
                        'next state:', str(next_state), 'comp:', str(self.dat.comp[next_state]),
                        'done:', str(done), 'score:', str(score)]),
                                    filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))

                    self.policies[-1].fit(state, action, reward, next_state, done, self.dat)
                    state = next_state
                    if done:
                        break

                # record FGT score
                max_score = max(scores)
                self.dat.score_list.append(max_score)
                self.dat.max_score = max_score if max_score > self.dat.max_score else self.dat.max_score
                self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Episode: {}  Score: {}  Max score: {}\n"
                                .format(len(self.dat.score_list), max_score, self.dat.max_score),
                                filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))

                # # save intermediate data
                # if np.mod(len(self.dat.score_list), self.output_frequency) == 0:
                #     self.save_intermediate_data()

                if self.check_exit_conditions(max_running_time): return

            # print results
            optimal_fea, candidate_scores = self.select_optimal_fea_set()
            self.save_intermediate_data()
            self.print_results(optimal_fea, candidate_scores)
            self.plot_results()
