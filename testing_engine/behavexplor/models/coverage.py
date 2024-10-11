import numpy as np

from sklearn.cluster import KMeans
from typing import TypeVar, List
from .feature import FeatureNet
from ..adapter.ApolloSim import ScenarioAdaptor

class ClusterModelBehavior(object):
    def __init__(self, cluster_num):
        """
        Initial cluster number
        """
        self.cluster_model = KMeans(cluster_num)
        self.cluster_center = []
        self.cluster_data = None

    def search(self, v):
        """
        @param: v is the query feature
        """
        # v represents the behaviors of a single case
        # @format is numpy with shape (n, 64)
        # @output is numpy with shape (n, )
        cls_labels = self.cluster_model.predict(v)
        # nearest_node = self.AI.get_nns_by_vector(v, 1, include_distances=True)
        # label(node id) & distance
        return cls_labels

    def update(self, v):
        """
        Need to change to load all corpus and re-cluster
        """
        v[np.isnan(v)] = 0
        # Step1: add new behavior data @format is numpy with shape (n, 64)
        if self.cluster_data is None:
            self.cluster_data = v
        else:
            self.cluster_data = np.concatenate([self.cluster_data, v], axis=0)
        # Step2: retrain kmeans model.
        y = self.cluster_model.fit_predict(self.cluster_data) # shape (n, )
        return y

    def get_centers(self):
        return self.cluster_model.cluster_centers_

class CoverageModel(object):

    def __init__(self, window_size, cluster_num, threshold_coverage):
        self.coverage_centers = []
        self.coverage_centers_index = []
        self.coverage_centers_pointer = 0

        self.window_size = window_size
        self.cluster_num = cluster_num
        self.threshold_coverage = threshold_coverage

        self.dynamic_threshold = np.inf

        self.feature_layer = FeatureNet(window_size)
        self.cluster_layer_behavior = ClusterModelBehavior(self.cluster_num)

    def _extract_feature(self, scenario: ScenarioAdaptor, resample='linear'):
        y_behavior = self.feature_layer.forward(scenario, resample)
        return y_behavior

    def initialize(self, scenarios: List[ScenarioAdaptor]):
        """
        X_behavior: list [item1, item2, ..., itemn]
            itemi : array [[x1...], [x2...]]
        X_trace: list [item1, item2, ..., itemn]
            itemi: list: [(x1, y1), (x2, y2), ..., (xn, yn)]
        """
        X_behavior = []
        for item in scenarios:
            X_behavior.append(self._extract_feature(item, 'linear'))

        # behavior model
        buffer_feature = None
        for i in range(len(X_behavior)):
            x = X_behavior[i] # shape (n, 64)
            if buffer_feature is None:
                buffer_feature = x
            else:
                buffer_feature = np.concatenate([buffer_feature, x], axis=0)

            self.coverage_centers_index.append([self.coverage_centers_pointer,
                                                self.coverage_centers_pointer + x.shape[0]])
            self.coverage_centers_pointer += x.shape[0]

        # initial train
        y = self.cluster_layer_behavior.update(buffer_feature) # n x 64
        self.update(y)

    def update(self, y):
        """
        y is the class labels of all cases
         shape is (n, ), the cluster label sequence.
        """
        self.coverage_centers = []
        for item in self.coverage_centers_index:
            start_index = item[0]
            end_index = item[1]
            y_i = y[start_index:end_index]
            self.coverage_centers.append(y_i)
        self._update_threshold()

    def _update_threshold(self):
        pattern_num = len(self.coverage_centers)
        distance_matrix = np.zeros((pattern_num, pattern_num))
        for i in range(pattern_num):
            distance_matrix[i][i] = 1000
            for j in range(i + 1, pattern_num):
                tmp_distance = self._compute_distance_behavior_states(self.coverage_centers[i], self.coverage_centers[j])
                distance_matrix[i][j] = tmp_distance
                distance_matrix[j][i] = tmp_distance

        pattern_min_distance = []
        for i in range(pattern_num):
            pattern_i_min = np.min(distance_matrix[i])
            pattern_min_distance.append(pattern_i_min)
        pattern_min_distance = np.array(pattern_min_distance)
        self.dynamic_threshold = np.mean(pattern_min_distance)

    def feedback_coverage_behavior(self, scenario: ScenarioAdaptor):

        x = self._extract_feature(scenario, 'linear')

        y_behavior = self.cluster_layer_behavior.search(x)
        find_new_coverage = False
        min_feedback = np.inf
        for i in range(len(self.coverage_centers)):
            cov_feedback = self._compute_distance_behavior_states(y_behavior, self.coverage_centers[i])
            if cov_feedback < min_feedback:
                min_feedback = cov_feedback
        # if min_feedback > min(self.dynamic_threshold, self.threshold_coverage):
        if min_feedback > self.threshold_coverage:

            find_new_coverage = True
            # if no_pass: # ignore whether the seed is fail or pass
            self.coverage_centers_index.append([self.coverage_centers_pointer,
                                                self.coverage_centers_pointer + x.shape[0]])
            self.coverage_centers_pointer += x.shape[0]
            # update behavior model (kmeans)
            y = self.cluster_layer_behavior.update(x)
            # update existing centers
            self.update(y)

        return find_new_coverage, min_feedback, y_behavior


    @staticmethod
    def _compute_distance_behavior_states(y1, y2):
        """
        y1 is a list
        """
        # y is numpy
        y1_length = len(y1)
        y2_length = len(y2)

        coverage_score = abs(y1_length - y2_length)

        common_length = min(y1_length, y2_length)
        y1_common = y1[:common_length]
        y2_common = y2[:common_length]
        for i in range(common_length):
            y1_e = y1_common[i]
            y2_e = y2_common[i]
            if y1_e == y2_e:
                continue
            else:
                coverage_score += 1

        coverage_score /= float(max(y1_length, y2_length))

        return coverage_score

    def get_centers(self):
        return self.coverage_centers