# AFE

This program aims at automatically extract features in fomulation form for predictive modeling. Our preprint paper can be viewed from: [Physics-constrained Automatic Feature Engineering for Predictive Modeling in Materials Science](https://www.aaai.org/AAAI21Papers/AAAI-8963.XiangZ.pdf)

# Usage

1. Prepare the dataset: The input dataset should be configured in .pkl files, which contains a list consists of a target list [], a three-layer feature list [] and a feature name list []. The first-layer feature list should contain each second-layer feature list [] for each feature, and each second-layer feature list [] should contain each third layer feature list [] for each sample. So group features with variant number of in-group features can be realized through variant number of in-group features in each third-layer feature list [] for each sample. The following line describes the data structure, and an example is given in /src/data_sample.
```bash
[[target_sample1, target_sample2, ...], [[[F1_sample1],[F1_sample2], ...],[[F2_sample1],[F2_sample2], ...], ...], [F1_name, F2_name, ...]]
```
2. Configure main.py to run

3. Results can be viewed from the "nD_descriptors.txt" file in the "Filename_results" folder.

# Parameter description:

MODE = 'r'  # 'r' for regression and 'c' for classification  
  
FILENAME = 'GAFUtrain.csv'  # dataset name  
  
\# Operation set, which can include  
\# ['exp', 'log', 'p-1', 'p+2', 'p+3', 'sqr', 'cbr', 'sin', 'cos', 'add', 'sub', 'mul', 'div', 'end']  
ACT = ['exp', 'log', 'sqr', 'p+2', 'p-1', 'add', 'div', 'end']  
  
DIM = 1  # Dimension of descriptors to explore  
  
RESUME = False  # Integer n lager than 0 for continuing a descriptor exploration with dimension n, False for start a new exploration
  
MAX_ITER = 10  # Maximum number of iterations for exploration  
  
EPISODE = 5  # Maximum number of episodes in each iteration, printing results frequency   
  
MAX_COMP = 5  # Maximum complexity for descriptors  
  
BATCH_SIZE = 1000  # The size of batch set for binary operation  

# Licence

AFE is released under the MIT Licence
