import numpy as np
from loguru import logger
from tsfresh.feature_extraction import feature_calculators
from scipy import interpolate

from ..adapter.ApolloSim import ScenarioAdaptor


def static_calculator(X):
    X_feature = []
    attribution_size = X.shape[1]
    for attr_i in range(attribution_size):
        attribution_i = X[:, attr_i]
        mean = feature_calculators.mean(attribution_i)
        minimum = feature_calculators.minimum(attribution_i)
        maximum = feature_calculators.maximum(attribution_i)
        mean_change = feature_calculators.mean_change(attribution_i)
        mean_abs_change = feature_calculators.mean_abs_change(attribution_i)
        variance = feature_calculators.variance(attribution_i)
        c3 = feature_calculators.c3(attribution_i, 1)
        cid_ce = feature_calculators.cid_ce(attribution_i, True)

        attribution_i_feature = [mean, variance, minimum, maximum, mean_change, mean_abs_change, c3, cid_ce]

        X_feature += attribution_i_feature

    return X_feature

class FeatureNet(object):

    def __init__(self, window_size=1):
        self.resample_frequency = 0
        self.window_size = window_size # unit is second (s)
        self.local_feature_extractor = None

    @staticmethod
    def input_resample(xs, ts, resample='linear', sample_frequency=0.1):
        # logger.debug(f"ts: {ts}")
        # x: [t, m], t: [t]
        x = np.array(xs)
        resample_axis = np.arange(ts[0], ts[-1], sample_frequency)
        new_x = []
        for i in range(0, x.shape[1]):
            x_i = x[:, i] # [t]
            f_i = interpolate.interp1d(ts, x_i, kind=resample)
            new_x_i = f_i(resample_axis) # [t]
            new_x_i = np.append(new_x_i, x_i[-1])
            new_x.append(new_x_i)
        new_x = np.array(new_x)
        new_x = new_x.T
        # new_x: [t, m]
        return new_x

    def forward(self, scenario: ScenarioAdaptor, resample='linear'):
        # use attributes: heading, speed, acceleration
        scenario_record = scenario.recorder

        # obtain attributes and time
        x = []
        t = []
        for i in range(len(scenario_record)):
            frame_attribute = np.array(scenario_record[i])
            x_f = frame_attribute[1:]
            x.append(x_f)
            t.append(frame_attribute[0])

        # aims to assign the time stamp for feature extraction!!!
        x_behavior_vector = self.input_resample(x, t, resample)

        time_size = x_behavior_vector.shape[0]
        if time_size < self.window_size:
            last_element = x_behavior_vector[-1:, :]
            for _ in range(self.window_size - time_size):
                x_behavior_vector = np.concatenate([x_behavior_vector, last_element], axis=0)

        y = []
        for i in range(time_size - self.window_size + 1):
            x_segment = x_behavior_vector[i:i+self.window_size]
            x_feature = static_calculator(x_segment)
            y.append(x_feature)

        return np.array(y)

