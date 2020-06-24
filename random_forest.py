from sklearn.tree import DecisionTreeClassifier
from scipy.stats import mode
import numpy as np

class RandomForestClassifier:
    '''
    A simple implementation of the Random Forest classifier which makes use
    of sklearn's DecisionTreeClassifier. Please see sklearn DecisionTreeClassifier documentation
    for how some of the attributes/arguments work.
    '''
    
    def __init__(
                 self,
                 n_estimators=20, 
                 max_features='sqrt',
                 max_depth=12,
                 min_samples_leaf=2,
                 min_samples_split=2,
                 bootstrap=0.95
                 ):
        '''
        Args:
            n_estimators      - int, the size of the forest
            max_features      - int, float, string or None, the number of features to consider when 
                                looking for the best split
            max_depth         - int or None, the maximum depth of the tree. If None, then nodes are 
                                expanded until all leaves are pure or until all leaves contain less 
                                than min_samples_split samples
            min_samples_leaf  - int or float, the minimum number of samples required to be at a leaf
                                node
            min_samples_split - int or float, the minimum number of samples required to split an internal 
                                node
            bootstrap         - float, 0 < x ≤ 1, the fraction of randomly choosen data on which to fit
                                each tree
        '''
        self.n_estimators      = n_estimators
        self.max_features      = max_features
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.bootstrap         = bootstrap
        self.min_samples_leaf  = min_samples_leaf

    def fit(self, X, y):
        '''
        Creates a forest of decision trees using a random subset of data and
        features
        
        Args:
            X - array-like, sample training data, shape[n_samples, n_features]
            y - array-like, target labels, shape[n_samples]
        
        '''
        self.forest   = []
        n_samples     = len(X)
        indx_choices  = np.arange(n_samples)
        n_sub_samples = round(n_samples * self.bootstrap)

        for i in range(self.n_estimators):
            
            # take a random sample of 
            idx = np.choice(indx_choices, n_sub_samples, replace=False)
            X_subset = X[idx]
            y_subset = y[idx]

            tree = DecisionTreeClassifier(
                                          max_features=self.max_features, 
                                          max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split
                                          )

            tree.fit(X_subset, y_subset)
            self.forest.append(tree)

    def predict(self, X):
        '''
        Predict the class of each sample in X
    
        Args:
            X           - array-like, sample training data, shape[n_samples, n_features]
        
        Returns:
            predictions - array-like, predicted labels, shape[n_samples]

        '''
        n_samples = len(X)
        predictions = np.empty([self.n_estimators, n_samples])

        for i in range(self.n_estimators):

            predictions[i] = self.forest[i].predict(X)

        return mode(predictions)[0][0].astype(int)