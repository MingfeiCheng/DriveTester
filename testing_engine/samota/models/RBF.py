import numpy as np
from loguru import logger
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.initializers import RandomUniform, Initializer, Constant
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import StandardScaler

class InitCentersRandom(Initializer):
    """ Initializer for initialization of centers of RBF network
        as random samples from the given data set.
    # Arguments
        X: matrix, dataset to choose the centers from (random rows
          are taken as centers)
    """

    def __init__(self, X):
        self.X = X

    def __call__(self, shape, dtype=None):
        assert shape[1] == self.X.shape[1]
        idx = np.random.randint(self.X.shape[0], size=shape[0])
        return self.X[idx, :]


class RBFLayer(Layer):
    """ Layer of Gaussian RBF units.
    # Example
    ```python
        model = Sequential()
        model.add(RBFLayer(10,
                           initializer=InitCentersRandom(X),
                           betas=1.0,
                           input_shape=(1,)))
        model.add(Dense(1))
    ```
    # Arguments
        output_dim: number of hidden units (i.e. number of outputs of the
                    layer)
        initializer: instance of initiliazer to initialize centers
        betas: float, initial value for betas
    """

    def __init__(self, var_len, output_dim, initializer=None, betas=1.0, **kwargs):
        self.var_len = var_len
        self.output_dim = output_dim
        self.init_betas = betas
        if not initializer:
            self.initializer = RandomUniform(0.0, 1.0)
        else:
            self.initializer = initializer
        super(RBFLayer, self).__init__(**kwargs)

    def build(self, input_shape):

        self.centers = self.add_weight(name='centers',
                                       shape=(self.output_dim, input_shape[1]),
                                       initializer=self.initializer,
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.output_dim,),
                                     initializer=Constant(
                                         value=self.init_betas),
                                     # initializer='ones',
                                     trainable=True)

        super(RBFLayer, self).build(input_shape)

    def call(self, x):
        C = K.expand_dims(self.centers)
        H = K.transpose(C-K.transpose(x))
        return K.exp(-self.betas * K.sum(H**2, axis=1))

    def gaussian(self, x, mu, sigma):
        return exp(- metrics(mu, x) ** 2 / (2 * sigma ** 2))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

    def get_config(self):
        # have to define get_config to be able to use model_from_json
        config = {
            'output_dim': self.output_dim
        }
        base_config = super(RBFLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

class Model:
    ss = StandardScaler()

    def __init__(self,var_len,no_of_neurons,cluster):
        self.var_len = var_len
        
        self.train(no_of_neurons,np.array(cluster))

    def train(self,no_of_neurons,cluster):
        # dataset = pd.read_csv(clean_file_name,header=None)
        # for x in cluster:
        #     print(x)
        # logger.debug(f'cluster: {cluster.shape}')
        X = cluster[:, 0:self.var_len]
        y = cluster[:, self.var_len:self.var_len+1]
        y[y < 0] = 0
        y[y > 1] = 1

        X = self.ss.fit_transform(X)
        self.model = Sequential()
        rbflayer = RBFLayer(self.var_len,
                            no_of_neurons,
                            initializer=InitCentersRandom(X),
                            betas=3.0,
                            input_shape=(self.var_len,))
        self.model.add(rbflayer)
        self.model.add(Dense(1))
        self.model.add(Activation('linear'))
        self.model.compile(loss='mean_absolute_error',
                          optimizer = 'adam')
        history = self.model.fit(X, y, epochs=1000, batch_size=8,verbose=0)

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
        self.mae = mae / len(cluster)

    def predict(self,val):
        value = np.array([val])
        B = self.ss.transform(value)
        y_pred = self.model.predict([B])

        if y_pred[0][0] > 1:
            return 1
        if y_pred[0][0] < 0:
            return 0
        return y_pred[0][0]

