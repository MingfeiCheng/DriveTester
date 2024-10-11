import copy
import random
from .RBF import Model as RBF_Model
from .PR import Polynomial_Regression
from .Kriging import *

def train_test_spilt(cluster, train_percentage):
    random.shuffle(cluster)
    train = []
    test = []
    test_percentage = 100 - train_percentage
    for i in range(int(((len(cluster) * train_percentage) / 100))):
        train.append(cluster[i])

    total = 0
    for i in range(int(((len(cluster) * train_percentage) / 100)),
                   int(((len(cluster) * train_percentage) / 100)) + 1 + int(((len(cluster) * test_percentage) / 100))):

        if i <len(cluster):
            test.append(cluster[i])
    return train, test

class ensemble:
    def __init__(self,var_len, database, obj,deg=2):
        self.var_len = var_len
        self.objective = obj

        train, test = train_test_spilt(database,80)
        self.rbf = RBF_Model(var_len, 10, train)
        self.PR = Polynomial_Regression(var_len, degree=deg, cluster = train)
        self.KR = Kriging(var_len, train)

        self.rbf.test(test)
        self.PR.test(test)
        self.KR.test(test)

        total_mae = self.rbf.mae + self.PR.mae + self.KR.mae
        self.w_rbf = 0.5 * ((total_mae - self.rbf.mae)/total_mae)
        self.w_PR = 0.5 * ((total_mae - self.PR.mae) / total_mae)
        self.w_KR = 0.5 * ((total_mae - self.KR.mae) / total_mae)

    def predict (self, fv):
        fv = fv[:self.var_len]
        # fv = [0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 3, 1, 4, 1, 1, 1]
        # print(fv)
        y_rbf = self.rbf.predict(copy.deepcopy(fv))
        y_pr = self.PR.predict(copy.deepcopy(fv))
        y_kr = self.KR.predict(copy.deepcopy(fv))


        diff_rbf_pr = abs(y_rbf - y_pr)
        diff_rbf_kr = abs(y_rbf - y_kr)
        diff_pr_kr = abs(y_pr - y_kr)
        

        return (y_rbf*self.w_rbf) + (y_pr*self.w_PR) + (y_kr*self.w_KR), max([diff_rbf_pr,diff_rbf_kr,diff_pr_kr])
