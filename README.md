# AFE

Please Configure main.py to run

# Parameter description:

MODE = 'r'  # 'r' for regression and 'c' for classification  
  
FILENAME = 'GAFUtrain.csv'  # dataset name  
  
\# Operation set, which can include  
\# ['exp', 'log', 'p-1', 'p+2', 'p+3', 'sqr', 'cbr', 'sin', 'cos', 'add', 'sub', 'mul', 'div', 'end']  
ACT = ['exp', 'log', 'sqr', 'p+2', 'p-1', 'add', 'div', 'end']  
  
DIM = 1  # Dimension of descriptors to explore  
  
RESUME = False  # True for continue a project, False for start a new one  
  
MAX_ITER = 10  # Maximum number of iterations for exploration  
  
EPISODE = 5  # Maximum nu mber of episodes in each iteration, printing results frequency  
  
GROUP_SIZE = 40  # The size of each group for group features. Use 1 if not a group feature  
  
MAX_COMP = 5  # Maximum complexity for descriptors  
  
BATCH_SIZE = 1000  # The size of batch set for binary operation  
