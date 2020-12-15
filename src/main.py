from AFE import AFE
# from AFE_NLR import NonlinearRegression
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':
    ###########  Configuration  ###################

    MODE = 'r'  # 'r' for regression and 'c' for classification
    FILENAME = 'DR_0.5train.pkl'  # dataset name
    # Operation set, which can include
    # ['exp', 'log', 'p-1', 'p+2', 'p+3', 'sqr', 'cbr', 'sin', 'cos', 'add', 'sub', 'mul', 'div', 'end']
    ACT = ['exp', 'p+2', 'p+3', 'neg', 'add', 'sub', 'mul', 'div', 'end']
    RESUME = False # the dimension of descriptors to continue exploring, False to start a new one
    INCREASE_DIM = False  # increase the dimension of descriptors at the beginning
    MAX_DIM = 2  # Max dimension of descriptors to explore
    MAX_COMP = 10  # Maximum complexity for descriptors
    MAX_ITER = 1000000  # Maximum number of iterations for one exploration
    TIME_BUDGET = 43200  # Maximum seconds of running time for one exploration, False for no limit
    EPISODE = 5  # Maximum number of episodes in each iteration, printing results frequency
    BATCH_SIZE = 1000  # The size of batch set for binary operation
    USE_CONSTANTS = False
    STORE_FEA = False

    ##############################################

    eng = AFE(FILENAME, ACT, MODE, resume=RESUME, episodes=EPISODE, batch_size=BATCH_SIZE, max_comp=MAX_COMP, cv=5,
              use_constants=USE_CONSTANTS, store_fea=STORE_FEA, threshold=1, skip_read=INCREASE_DIM)
    eng.setup_dqn(hidden_layer=(150, 120), batch_size=64, epsilon=1.0, epsilon_decay=0.99,
                  epsilon_min=0.05, gamma=0.99, alpha=0.001, func_approximation='NN')
    eng.increase_dim(INCREASE_DIM)

    for i in range(max(1, RESUME + 1 if INCREASE_DIM else RESUME), MAX_DIM + 1):
        eng.explore(MAX_ITER, TIME_BUDGET)
        if i < MAX_DIM:
            eng.increase_dim(True)

    # descriptor_indexes = eng.dat.candidate_set_temp[1][0][-1][1]
    # comp = (eng.dat.candidate_set_temp[1][0][2][1])
    # indexes = [[split_index[0] for split_index in group] for group in descriptor_indexes]
    # print(eng.fgt.print_descriptor(descriptor_indexes, *[1 for i in range(comp*2)]))
    # # print(eng.fgt.calculate_descriptor(descriptor_indexes, *[1 for i in range(comp*2)]))
    # nlr = NonlinearRegression(*[[-2, 2], [-2, 2]])
    # print(nlr.fit(eng.fgt.calculate_descriptor, descriptor_indexes, eng.dat.target, comp))
    # print(nlr.score())
    #
    # engt = AFE(FILENAME[:-9]+'test.pkl', ACT, MODE, resume=False, episodes=EPISODE, batch_size=BATCH_SIZE,
    #           max_comp=MAX_COMP, cv=5, threshold=1, skip_read=False)
    # engt.setup_dqn(hidden_layer=(150, 120), batch_size=64, epsilon=1.0, epsilon_decay=0.99,
    #               epsilon_min=0.05, gamma=0.99, alpha=0.001, func_approximation='NN')

    # descriptor_indexes = eng.dat.candidate_set_temp[1][0][-1][1]
    # print(eng.fgt.print_descriptor(descriptor_indexes))
    # fea = eng.fgt.calculate_descriptor(descriptor_indexes)
    # clf = LinearRegression()
    # clf.fit(fea, eng.dat.target)
    # print(clf.score(fea, eng.dat.target))
    # fea = engt.fgt.calculate_descriptor(descriptor_indexes)
    # print(clf.score(fea, engt.dat.target))