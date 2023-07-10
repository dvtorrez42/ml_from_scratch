import numpy as np 
from itertools import combinations

## Loss Functions
def gini_index(y):
    size = len(y)
    classes, counts = np.unique(y, return_counts=True)
    pmk = counts/size  # proportion of classes k in k=1 thru k in node m 
    return np.sum(pmk*(1-pmk))

def cross_entropy(y):
    size = len(y)
    classes, counts = np.unique(y, return_counts=True)
    pmk = counts/size
    return -np.sum(pmk*np.log2(pmk))

def split_loss(child1, child2, loss = cross_entropy):
    return (len(child1)*loss(child1) + len(child2)*loss(child2)) / (len(child1)+len(child2))


## Helper Functions
def all_rows_equal(X):
    """
    Checks if all of a bud's rows (observations) are equal accross all predictors. 
    If this is the case, this bud will not be split and instead becomes a terminal leaf.
    """
    return (X == X[0]).all()

def possible_splits(x):
    """
    Returns all possible ways to divide the classes in a categorical predictor into two.
    Specifically, it returns all possible sets of values which can be used to funnel observations
    into the "left" child node. 
    """
    L_values = []
    for i in range(1, int(np.floor(len(x)/2))+1):
        L_values.extend(list(combinations(x, i)))
    return L_values


## Helper Classes
class Node():
    
    def __init__(self, Xsub, ysub, ID, obs, depth=0, parent_ID=None, leaf=True):
        self.Xsub = Xsub
        self.ysub = ysub
        self.ID = ID
        self.obs = obs
        self.size = len(ysub)
        self.depth = depth
        self.parent_ID = parent_ID
        self.leaf = leaf
        
class Splitter:
    
    def __init__(self):
        self.loss = np.inf
        self.no_split = True
        
    def _replace_split(self, Xsub_d, loss, d, dtype='quant', t = None, L_values=None):
        self.loss = loss
        self.d = d
        self.dtype = dtype
        self.t = t
        self.L_values = L_values
        self.no_split = False
        if dtype == 'quant':
            self.L_obs = self.obs[Xsub_d <= t]
            self.R_obs = self.obs[Xsub_d > t]    
        else:
             self.L_obs[np.isin(Xsub_d, L_values)]
             self.R_obs[~np.isin(Xsub_d, L_values)]
             

## Main Classifier Class

class DecisionTreeClassifier:
    
    ## 1. Training ##
    
    ### Fit ###
    def fit(self, X, y, loss_func = cross_entropy, max_depth=100, min_size=2, C=None)
    
        ## Add data
        self.X = X
        self.y = y
        self.N, self.D = self.X.shape
        dtypes = [np.array(list(self.X[:,d])).dtype for d in range(self.D)]
        self.dtypes = ['quant' if (dtype == float or dtype == int) else 'cat' for dtype in dtypes]
        
        ## Add model parameters
        self.loss_func = loss_func
        self.max_depth = max_depth
        self.min_size = min_size
        self.C = C
        
        ## Initialize nodes
        self.nodes_dict = {}
        self.current_ID = 0
        initial_node = Node(Xsub=X, ysub=y, ID=self.current_ID, obs=np.arange(self.N), parent_ID=None)
        self.nodes_dict[self.current_ID] = initial_node
        self.curret_ID += 1
        
        #  Build
        self._build()
        
        ### Build Tree ###
        def _build(self):
            
            eligible_buds = self.nodes_dict
            for layer in range(self.max_depth):
                
                # Find eligible node for layer interation
                eligible_buds = {ID:node for (ID, node) in self.nodes_dict.items() if
                                    (node.leaf == True) &
                                    (node.size >= self.min_size) &
                                    (~all_rows_equal(node.Xsub)) &
                                    (len(np.unique(node.ysub)) > 1)}
                if len(eligible_buds) == 0:
                    break
                
                ## Split each eligible parent
                for ID, bud in eligible_buds.items():
                    
                    # Finde split
                    self._find_split(bud)
                    
                    # Make split
                    if not self.splitter.no_split:
                        self._make_split()