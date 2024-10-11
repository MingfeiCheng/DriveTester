import numpy
import numpy as np
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from loguru import logger

class Kriging:

    def __init__(self, var_len, cluster = None):
        self.var_len = var_len
        self.scaler = preprocessing.StandardScaler()

        cluster = np.array(cluster)
        X = cluster[:, 0:self.var_len]  # features from 0 to 15th index
        y = cluster[:, self.var_len:self.var_len + 1]  # value at 16th index

        y[y < 0] = 0
        y[y > 1] = 1

        X = self.scaler.fit_transform(X)
        kernel = 1.0 * RBF(1.0)  # squared-exponential kernel
        self.model = GaussianProcessRegressor(kernel=kernel, random_state=0).fit(X, y)


    def test(self, cluster):
        mae = 0
        for i in range(len(cluster)):
            y_act = cluster[i][self.var_len]
            Y_pred = self.predict(cluster[i][:self.var_len])

            if y_act > 1:
                y_act =1
            if y_act < 0:
                y_act =0
            mae = mae + abs(y_act - Y_pred)
        self.mae = mae/len(cluster)
    
    def predict(self, value):
        value = numpy.array([value])
        B = np.reshape(value, (1, self.var_len))
        B= (self.scaler.transform(B))
        y_pred = self.model.predict(value)

        # logger.debug('y_pred: {}', y_pred)

        if y_pred[0] > 1:
            return 1
        if y_pred[0] < 0:
            return 0

        return y_pred[0]


