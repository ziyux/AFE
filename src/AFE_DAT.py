import csv
import os
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression


def read_csv_file(filename):
    try:
        with open(filename, mode='r') as csvfile:
            csv_reader = csv.reader(csvfile)
            data = []
            for row in csv_reader:
                data.append(row[:])
            csvfile.close()
            return np.array(data[:])
    except IOError:
        print('Problem reading: ' + filename)


def save_csv_file(filename, data):
    with open(filename, mode='w', newline='') as csvfile:
        csv_witer = csv.writer(csvfile)
        if type(data) == dict:
            for key, val in data.items():
                csv_witer.writerow([key, val])
        elif type(data) == list:
            csv_witer.writerow(data)
        else:
            for row in data:
                csv_witer.writerow(row)


def save_pickle(filename, obj):
    with open(filename, 'wb+') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def read_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


class DAT(object):
    def __init__(self, filename, resume, dim, mode, operation_set, max_comp, clf, cv, use_constants, store_fea,
                 directory, skip_read):
        self.resume = resume
        self.dim = dim
        self.mode = mode
        self.operation_set = operation_set
        self.candidate_set_size = 5
        self.max_comp = max_comp
        self.dir = directory
        self.cv = cv
        self.use_constants = use_constants
        self.store_fea = store_fea
        self.start_time = time.time()
        self.local_time = lambda: time.asctime(time.localtime(time.time()))
        self._last_string = ''
        self.fea_name, self.fea, self.target, self.pri_num, self.reward, self.comp, self.done, self.constants, \
            self.candidate_set, self.candidate_set_temp, self.index, self.score_list, self.max_score, self.train, \
            self.test = self.load_data(resume, filename, skip_read)
        if clf is None:
            self.clf = LinearRegression() if self.mode == 'r' \
                else SVC(kernel='linear', gamma='auto', max_iter=-1)
        else:
            self.clf = clf
        self.save_stop_tag = 0
        self.save_start_tag = 0 if not resume else len(self.fea[0])

    # save print statement
    def printf(self, string, filename='log.txt', is_new_file=False):
        if string != self._last_string:
            self._last_string = string
            mode = 'w+' if is_new_file else 'a+'
            with open(filename, mode) as f:
                print(string)
                print(string, file=f)
            f.close()

    def train_test_split(self, samples):
        index = list(range(samples))
        train, test = [], []
        for i in range(self.cv):
            train_index, test_index, y_train, y_test = train_test_split(
                index, index, test_size=1 / self.cv, random_state=i)
            train.append(train_index)
            test.append(test_index)
        return train, test

    def load_data(self, resume, filename, skip_read):
        self.printf("[" + str(self.local_time()) + "]" + " Start loading data",
                    filename=os.path.join(self.dir, str(self.dim) + 'D_log.txt'),
                    is_new_file=False if resume else True)

        if resume and skip_read is not True:
            # combine all files
            fea = []
            fea_name = []
            reward = []
            comp = []
            done = []
            constants = []
            for path, subdirs, files in os.walk(os.path.join(self.dir, str(self.dim) + 'D_feature_space')):
                # sort files in name order
                indices = [int(f[16 + len(str(self.dim)):-4].split('_')[0]) for f in files]
                files = [f for _, f in sorted(zip(indices, files))]
                for f in files:
                    data = read_pickle(os.path.join(path, f))
                    try:
                        fea = fea + data['fea']
                    except KeyError:
                        pass
                    fea_name = fea_name + data['fea_name']
                    reward = reward + data['reward']
                    comp = comp + data['comp']
                    done = done + data['done']
                    if self.use_constants:
                        constants = constants + data['constants']
                    del data

            pri_num = 0
            # find the number of primary features
            for i in range(len(fea_name)):
                ##################################
                if fea_name[i][:3] != 'pri':
                # if fea_name[i][:3] != 'pri' and fea_name[i][:3] != 'can':
                    ##################################
                    break
                pri_num += 1

            target, candidate_set_temp, index, score_list, train, test = \
                read_pickle(os.path.join(self.dir, str(self.dim) + 'D_data' + '.pkl'))
            max_score = max(score_list)
            candidate_set = [item[:-1] for item in candidate_set_temp]
        else:
            dataset = read_pickle(filename)
            target = np.array(dataset[0])
            fea = list(dataset[1])
            fea_name = list(dataset[2])

            # assign label name for classification
            if self.mode == 'c':
                name = [target[0]]
                for i in range(len(target)):
                    if target[i] in name:
                        target[i] = int(name.index(target[i]))
                    else:
                        name.append(target[i])
                        target[i] = int(name.index(target[i]))
                for i in range(len(fea)):
                    fea[i] = fea[i][np.argsort(target)]
                target = target[np.argsort(target)]

            # assign initial values if not resume
            fea_name = ['pri' + ',' + i for i in fea_name]
            pri_num = len(fea_name)
            reward = [1 / 1.001 for i in fea_name]
            comp = [0 for i in fea_name]
            done = [0 for i in fea_name]
            constants = [[1, 0] for i in fea_name]
            candidate_set = [[], []]
            candidate_set_temp = [[], []]
            index = {}
            for i in range(len(fea_name)):
                index['pri,' + fea_name[i]] = i
            score_list = []
            max_score = 0
            if self.cv is None:
                train, test = None, None
            else:
                train, test = self.train_test_split(len(fea[0]))

            if skip_read:
                target, candidate_set_temp, index, score_list, train, test = \
                    read_pickle(os.path.join(self.dir, str(self.dim) + 'D_data' + '.pkl'))
                max_score = max(score_list)
                candidate_set = [item[:-1] for item in candidate_set_temp]

        self.printf("[" + str(self.local_time()) + "]" + " Loading data completed\n",
                    filename=os.path.join(self.dir, str(self.dim) + 'D_log.txt'))

        return fea_name, fea, target, pri_num, reward, comp, done, constants, candidate_set, candidate_set_temp, \
            index, score_list, max_score, train, test

    def output_data(self, policy):
        output = {'fea_name': self.fea_name[self.save_start_tag:self.save_stop_tag],
                  'reward': self.reward[self.save_start_tag:self.save_stop_tag],
                  'comp': self.comp[self.save_start_tag:self.save_stop_tag],
                  'done': self.done[self.save_start_tag:self.save_stop_tag]}
        if self.save_start_tag <= len(self.fea):
            output['fea'] = self.fea[self.save_start_tag:min(self.save_stop_tag, len(self.fea))]

        if self.use_constants:
            output['constants'] = self.constants[self.save_start_tag:self.save_stop_tag]

        save_pickle(os.path.join(self.dir, str(self.dim) + 'D_feature_space', str(self.dim) + 'D_feature_space' +
                                 '_' + str(self.save_start_tag) + '_' + str(self.save_stop_tag - 1) + '.pkl'), output)
        save_pickle(os.path.join(self.dir, str(self.dim) + 'D_data' + '.pkl'), [
            self.target, self.candidate_set_temp, self.index, self.score_list, self.train, self.test])
        policy.model.save(os.path.join(self.dir, str(self.dim) + 'D_NN' + '.h5'))
