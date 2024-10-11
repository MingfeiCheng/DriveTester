import os
import sys
import copy
import time
import random
import numpy as np
import pandas as pd
from omegaconf import DictConfig
from datetime import datetime
from sklearn.cluster import DBSCAN
from loguru import logger
from sklearn import preprocessing
from pymoo.algorithms.nsga2 import calc_crowding_distance
from pymoo.algorithms.so_genetic_algorithm import GA
from pymoo.factory import get_crossover, get_mutation, get_sampling
from pymoo.model.problem import Problem
from pymoo.optimize import minimize as min_GA

from .core.candidate import Candidate
from .models.ensemble import ensemble
from .models.RBF import Model as RBF_Model

from common.logger_tools import get_instance_logger
from testing_engine.samota.adapter.ApolloSim import ApolloSimAdaptor
from common.data_provider import DataProvider
from ..common.result_recorder import ResultRecorder
from .. import register_tester

scaler = preprocessing.StandardScaler()

class Pylot_caseStudy(Problem):
    def __init__(self, i, n_var_in=16, xl_in=0, xu_in=1, number_of_neurons=10, index=16, percent=20, cluste=[],
                 clusters=-1):
        super().__init__(n_var=n_var_in, n_obj=1, xl=xl_in, xu=xu_in, type_var=int, elementwise_evaluation=True)

        self.model = RBF_Model(n_var_in, number_of_neurons, cluste)

    def _evaluate(self, x, out, *args, **kwargs):
        value = self.model.predict(x)
        out["F"] = value

@register_tester('tester.samota')
class SAMOTA(object):
    def __init__(
            self,
            scenario_runner,
            scenario_cfg: DictConfig,
            algorithm_cfg: DictConfig,
            oracle_cfg: DictConfig
    ):
        self.scenario_runner = scenario_runner
        self.map_name = scenario_cfg.map_name
        # Seed Scenario
        self.ego_start_lane_id = scenario_cfg.get("start_lane_id", None)
        self.ego_end_lane_id = scenario_cfg.get("end_lane_id", None)

        algorithm_parameters = algorithm_cfg.get("parameters", None)
        if algorithm_parameters is None:
            raise RuntimeError('Please config parameters')

        self.population_size = algorithm_parameters.get("population_size", 1)
        self.no_of_Objectives = algorithm_parameters.get("num_of_objectives", 2)
        self.run_hour = algorithm_parameters.get("run_hour", 4)

        self.save_root = DataProvider.output_folder()  # this is the fuzzing result folder to save the results
        self.result_recorder = ResultRecorder(
            self.save_root,
            self.run_hour
        )

        # create adapter
        self.adapter = ApolloSimAdaptor(
            self.map_name,
            self.ego_start_lane_id,
            self.ego_end_lane_id,
            oracle_cfg
        )

        self.current_index = self.result_recorder.current_index

        debug_folder = os.path.join(DataProvider.output_folder(), 'debug/algorithm')
        if not os.path.exists(debug_folder):
            os.makedirs(debug_folder)
        fitness_log_file = os.path.join(debug_folder, f"fitness.log")
        mutation_log_file = os.path.join(debug_folder, f"mutation.log")
        if os.path.isfile(fitness_log_file):
            os.remove(fitness_log_file)
        if os.path.isfile(mutation_log_file):
            os.remove(mutation_log_file)

        self.logger_fitness = get_instance_logger(f"fitness", fitness_log_file)
        self.logger_mutation = get_instance_logger("mutation", mutation_log_file)

        self.logger_fitness.info("Logger initialized for fitness")
        self.logger_mutation.info("Logger initialized for mutation")

         # collision, reaching destination
        self.database = []
        self.lb, self.ub = self.generate_input_interval()
        self.curr_iteration = 0
        self.last_update_time = None

    @staticmethod
    def generate_input_interval():
        # 0 Scenario ID
        # 1 Vehicle_in_front
        # 2 vehicle_in_adjcent_lane
        # 3 vehicle_in_opposite_lane
        # 4 vehicle_in_front_two_wheeled
        # 5 vehicle_in_adjacent_two_wheeled
        # 6 vehicle_in_opposite_two_wheeled
        # 7 Target Speed
        # 8 Trigger Time

        lb = [0, 0, 0, 0, 0, 0, 0, 0.3, 0]
        ub = [1, 2, 2, 2, 2, 2, 2, 25, 15]
        return lb, ub

    def evaulate_population_with_archive(self, pop, already_executed):
        # func: runner
        to_ret = []
        for candidate in pop:
            if isinstance(candidate, Candidate):
                if candidate.get_candidate_values() in already_executed:
                    continue

                start_time = time.time()
                candidate_value = candidate.get_candidate_values()
                mutated_scenario = self.adapter.mutation_adaptor(
                    candidate_value
                )
                mutated_scenario.idx = self.current_index
                mutated_scenario.export()
                self.current_index += 1
                scenario_recorder = self.scenario_runner.run(mutated_scenario)
                if scenario_recorder is None:
                    logger.warning(f'The scenario {mutated_scenario.idx} has bugs, continue to next')
                    continue
                self.result_recorder.update(
                    mutated_scenario.idx,
                    delta_time=time.time() - self.last_update_time,
                    result_details=scenario_recorder.to_json()
                )
                self.last_update_time = time.time()

                result, is_complete = self.adapter.feedback_adaptor(scenario_recorder)
                # convert results
                # result = [fitness_dest, fitness_collision]
                # result = self.result_converter(mutated_seed)
                logger.info('result: {}', result)
                candidate.set_objective_values(result)
                already_executed.append(candidate.get_candidate_values())
                to_ret.append(candidate)
        return to_ret


    def evaulate_population_with_archive_initial(self, pop, already_executed):
        # func: runner
        to_ret = []
        for candidate in pop:
            if isinstance(candidate, Candidate):
                if candidate.get_candidate_values() in already_executed:
                    continue

                start_time = time.time()
                candidate_value = candidate.get_candidate_values()
                mutated_scenario = self.adapter.mutation_adaptor(
                    candidate_value
                )
                mutated_scenario.idx = self.current_index
                mutated_scenario.export()
                self.current_index += 1
                scenario_recorder = self.scenario_runner.run(mutated_scenario)
                if scenario_recorder is None:
                    logger.warning(f'The scenario {mutated_scenario.idx} has bugs, continue to next')
                    continue
                self.result_recorder.update(
                    mutated_scenario.idx,
                    delta_time=time.time() - self.last_update_time,
                    result_details=scenario_recorder.to_json()
                )
                self.last_update_time = time.time()

                result, is_complete = self.adapter.feedback_adaptor(scenario_recorder)
                # convert results
                # result = [fitness_dest, fitness_collision]
                # result = self.result_converter(mutated_seed)
                logger.info('result: {}', result)

                if not is_complete:
                    continue

                # convert results
                # result = [fitness_dest, fitness_collision]
                candidate.set_objective_values(result)
                already_executed.append(candidate.get_candidate_values())
                to_ret.append(candidate)
        return to_ret

    @staticmethod
    def _terminal_check(start_time: time, run_hour: float) -> bool:
        curr_time = time.time()
        t_delta = curr_time - start_time
        if t_delta / 3600.0 > run_hour:
            logger.info(f'Finish fuzzer as reach the time limited: {t_delta / 3600.0}/{run_hour}')
            return True
        return False

    def run_search(self, start_time, criteria, archive, g_max):
        logger.info('Start run search ........')
        threshold_criteria = criteria
        already_executed = []
        objective_uncovered = []

        size = self.population_size
        lb = self.lb
        ub = self.ub
        no_of_Objectives = self.no_of_Objectives

        for obj in range(no_of_Objectives):
            objective_uncovered.append(obj)  # initialising number of uncovered objective

        random_population = []
        while len(random_population) < size:
            random_population_cfgs = self.generate_adaptive_random_population(size, lb, ub)  # Generating random population
            # runner
            random_population += self.evaulate_population_with_archive_initial(random_population_cfgs,already_executed)  # evaluating whole generation and storing results
        random_population = random_population[:size]

        self.database.extend(copy.deepcopy(random_population))
        self.update_archive(random_population, objective_uncovered, archive, no_of_Objectives, threshold_criteria)  # updateing archive
        iteration = 0

        while True:
            if self._terminal_check(start_time, self.run_hour):
                return

            logger.info('Search iteration: ' + str(iteration))
            iteration += 1

            if len(objective_uncovered) == 0:  # checking if all objectives are covered
                for obj in range(no_of_Objectives):
                    objective_uncovered.append(obj)  # initialising number of uncovered objective

            m_start_time = datetime.now()
            T_g = self.GS(start_time, self.database, objective_uncovered, size, g_max, criteria, lb, ub)
            m_end_time = datetime.now()
            logger.info('Mutation GS Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())
            if self._terminal_check(start_time, self.run_hour):
                return
            T_g = self.remove_covered_objs(T_g)
            T_g = self.evaulate_population_with_archive(T_g, already_executed)
            self.update_archive(T_g, objective_uncovered, archive, no_of_Objectives, threshold_criteria)
            self.database.extend(T_g)

            m_start_time = datetime.now()
            T_l = self.LS(start_time, self.database, objective_uncovered, size, g_max, criteria, lb, ub)
            if self._terminal_check(start_time, self.run_hour):
                return
            m_end_time = datetime.now()
            logger.info('Mutation LS Spend Time: [=]{}[=]', (m_end_time - m_start_time).total_seconds())
            T_l = self.evaulate_population_with_archive(T_l, already_executed)
            self.update_archive(T_l, objective_uncovered, archive, no_of_Objectives, threshold_criteria)
            self.database.extend(T_l)
            iteration += 1

    @staticmethod
    def evaulate_population_using_ensemble(M_gs, pop):
        for candidate in pop:
            result = [1, 1]
            uncertaininty = [0, 0]
            for M_g in M_gs:
                res, unc = M_g.predict(candidate.get_candidate_values())
                result[M_g.objective] = res
                uncertaininty[M_g.objective] = unc

            candidate.set_objective_values(result)
            candidate.set_uncertainity_values(uncertaininty)

    @staticmethod
    def dominates(value_from_pop, value_from_archive, objective_uncovered):
        dominates_f1 = False
        dominates_f2 = False
        for each_objective in objective_uncovered:
            f1 = value_from_pop[each_objective]
            f2 = value_from_archive[each_objective]
            if f1 < f2:
                dominates_f1 = True
            if f2 < f1:
                dominates_f2 = True
            if dominates_f1 and dominates_f2:
                break
        if dominates_f1 == dominates_f2:
            return False
        elif dominates_f1:
            return True
        return False

    def select_best(self, tournament_candidates, objective_uncovered):
        best = tournament_candidates[0]  # in case none is dominating other
        for i in range(len(tournament_candidates)):
            candidate1 = tournament_candidates[i]
            for j in range(len(tournament_candidates)):
                candidate2 = tournament_candidates[j]
                if self.dominates(candidate1.get_objective_values(), candidate2.get_objective_values(), objective_uncovered):
                    best = candidate1
        return best

    def tournament_selection(self, pop, size, objective_uncovered):
        tournament_candidates = []
        for i in range(size):
            indx = random.randint(0, len(pop) - 1)
            random_candidate = pop[indx]
            tournament_candidates.append(random_candidate)

        best = self.select_best(tournament_candidates, objective_uncovered)
        return best

    @staticmethod
    def do_uniform_mutation(parent1, parent2, lb, ub, threshold):
        child1 = []
        child2 = []

        parent1 = parent1.get_candidate_values()
        parent2 = parent2.get_candidate_values()

        for parent1_index in range(len(parent1)):
            probability_mutation = random.uniform(0, 1)
            if probability_mutation <= threshold:
                random_value = random.uniform(lb[parent1_index], ub[parent1_index])
                if parent1_index % 7 == 0:
                    child1.append(random_value)
                else:
                    child1.append(round(random_value))
            else:
                child1.append(parent1[parent1_index])

        for parent2_index in range(len(parent2)):
            probability_mutation = random.uniform(0, 1)
            if probability_mutation <=threshold:  # 1/4         25% probability
                random_value = random.uniform(lb[parent2_index], ub[parent2_index])
                if parent2_index % 7 == 0:
                    child2.append(random_value)
                else:
                    child2.append(round(random_value))
            else:
                child2.append(parent2[parent2_index])

        return Candidate(child1), Candidate(child2)

    @staticmethod
    def do_single_point_crossover(parent1, parent2):
        parent1 = parent1.get_candidate_values()
        parent2 = parent2.get_candidate_values()
        crossover_point = random.randint(1, len(parent1) - 1)
        t_parent1 = parent1[0:crossover_point]
        t_parent2 = parent2[0:crossover_point]
        for i in range(crossover_point, len(parent1)):
            t_parent1.append(parent2[i])
            t_parent2.append(parent1[i])

        return Candidate(t_parent1), Candidate(t_parent2)

    def generate_off_spring(self, pop, objective_uncovered, lb,ub):
        size = len(pop)
        population_to_return = []
        while len(population_to_return) < size:
            parent1 = self.tournament_selection(pop, 10, objective_uncovered)  # tournament selection same size as paper
            parent2 = self.tournament_selection(pop, 10, objective_uncovered)
            probability_crossover = random.uniform(0, 1)
            if probability_crossover <= 0.75:  # 75% probability
                parent1, parent2 = self.do_single_point_crossover(parent1, parent2)
            child1, child2 = self.do_uniform_mutation(parent1, parent2, lb, ub, (1 / len(parent1.get_candidate_values())))
            population_to_return.append(child1)
            population_to_return.append(child2)
        return population_to_return

    @staticmethod
    def update_iteration_bests(R_T,iteartion_b,iteration_n,objective_uncovered):
        for objective in objective_uncovered:
            best_b = R_T[0]
            best_n = R_T[0]
            for candidate in R_T:
                if candidate.get_objective_value(objective) < best_b.get_objective_value(objective):
                    best_b = candidate
                if candidate.get_uncertainity_value(objective) > best_n.get_uncertainity_value(objective):
                    best_n = candidate

            if len(iteartion_b[objective])==0:
                iteartion_b[objective] = best_b
            else:
                if best_b.get_objective_value(objective) < iteartion_b[objective].get_objective_value(objective):
                    iteartion_b[objective] = best_b

            if len(iteration_n[objective]) == 0:
                iteration_n[objective] = best_n
            else:
                if best_n.get_uncertainity_value(objective) > iteration_n[objective].get_uncertainity_value(objective):
                    iteration_n[objective] = best_n


        return iteartion_b,iteration_n, R_T

    @staticmethod
    def update_global_bests(T_b,iteration_b):
        imp = False
        for index in range(len(T_b)):
            if isinstance(T_b[index], Candidate):
                if iteration_b[index].get_objective_value(index) < T_b[index].get_objective_value(index):
                    imp = True
                    T_b[index] = iteration_b[index]
            else:
                if isinstance(iteration_b[index], Candidate):
                    imp = True
                    T_b[index]= iteration_b[index]
        return T_b,imp

    @staticmethod
    def update_global_bests_uncertainity(T_b, iteration_b):
        for index in range(len(T_b)):
            if isinstance(T_b[index], Candidate):
                if iteration_b[index].get_uncertainity_value(index) > T_b[index].get_uncertainity_value(index):
                    T_b[index] = iteration_b[index]
            else:
                if isinstance(iteration_b[index], Candidate):
                    T_b[index] = iteration_b[index]
        return T_b

    def fast_dominating_sort(self, R_T, objective_uncovered):
        to_return = []
        front = []
        count = 0
        while len(R_T) > 1:
            count = 0
            for outer_loop in range(len(R_T)):
                best = R_T[outer_loop]
                add = True
                for inner_loop in range(len(R_T)):
                    against = R_T[inner_loop]
                    if best == against:
                        continue
                    if self.dominates(best.get_objective_values(), against.get_objective_values(), objective_uncovered):
                        continue
                    else:
                        add = False
                        break
                if add:
                    if best not in front:
                        front.append(best)
                    count = count + 1
            if len(front) > 0:
                to_return.append(front)
                for i in range(len(front)):
                    R_T.remove(front[i])
                    front = []

            if (len(to_return) == 0) or (count == 0):  # to check if no one dominates no one
                to_return.append(R_T)
                break

        return to_return

    def preference_sort(self, R_T, size, objective_uncovered):
        to_return = []
        for objective_index in objective_uncovered:
            min = 100
            best = R_T[0]
            for index in range(len(R_T)):
                objective_values = R_T[index].get_objective_values()
                if objective_values[objective_index] < min:
                    min = objective_values[objective_index]
                    best = R_T[index]
            to_return.append(best)
            R_T.remove(best)

        if len(to_return) >= size:
            F1 = R_T
            for i in range(len(F1)):
                to_return.append(F1[i])
        else:
            E = self.fast_dominating_sort(R_T, objective_uncovered)
            for i in range(len(E)):
                to_return.append(E[i])
        return to_return

    @staticmethod
    def sort_based_on(e):
        values = e.get_objective_values()
        return values[0]

    def sort_worse(self, pop):
        pop.sort(key=self.sort_based_on, reverse=True)
        return pop

    @staticmethod
    def get_array_for_crowding_distance(sorted_front):
        list_ = []
        for value in sorted_front:
            objective_values = value.get_objective_values()
            # logger.debug('objective_values: {}, {}', objective_values, type(objective_values))
            np_array = [] #objective_values # np.array([objective_values[0], objective_values[1]])
            for oi in range(len(objective_values)):
                if isinstance(objective_values[oi], float) or isinstance(objective_values[oi], int):
                    np_array.append(np.array([objective_values[oi]]))
                elif isinstance(objective_values[oi], list):
                    np_array.append(np.array(objective_values[oi]))
                else:
                    np_array.append(objective_values[oi])
            list_.append(np_array)

        np_list = np.array(list_)
        if len(np_list.shape) > 2:
            np_list = np_list.reshape(np_list.shape[0], np_list.shape[1])
        # logger.debug('np_list: {} {}', np_list.shape, type(np_list))
        cd = calc_crowding_distance(np_list)
        return cd

    @staticmethod
    def assign_crowding_distance_to_each_value(sorted_front, crowding_distance):
        for candidate_index in range(len(sorted_front)):
            objective_values = sorted_front[candidate_index]
            objective_values.set_crowding_distance(crowding_distance[candidate_index])

    @staticmethod
    def sort_based_on_crowding_distance(e):
        values = e.get_crowding_distance()
        return values

    @staticmethod
    def preprocess_data(database2,percent,index):
        all_combined = []
        database = copy.deepcopy(database2)
        for c in database:
            to_add = c.get_candidate_values()
            to_add.extend(c.get_objective_values())
            all_combined.append(to_add)

        sorted_data = sorted(all_combined, key=lambda x: x[index])
        if not os.path.exists('lib'):
            os.makedirs('lib')
        file_writer = open("lib/clean_data.csv", 'w')
        to_write_count =(int((len(all_combined) * percent) / 100))
        if to_write_count<5:
            to_write_count =5
        for i in range(to_write_count):
            if i >= len(sorted_data):
                break
            to_write = str(sorted_data[i]).replace('[', '').replace(']', '') + "\n"
            file_writer.write(to_write)

    @staticmethod
    def values_with_label(X, label, labels, values):
        # X[9, 1]
        # logger.info('values_with_label: ' + str(X))
        to_ret = []
        # logger.debug('X shape:{}', X.shape)
        for val in range(len(labels)):
            if labels[val] == label:
                arr = np.array([X[val]])
                arr = scaler.inverse_transform(arr)
                new_arr = []
                for x_index in range(len(arr)):
                    if x_index % 7 == 0:
                        new_arr.append(arr[x_index])
                    else:
                        new_arr.append(round(arr[x_index]))
                arr = new_arr
                if values[val] < 0:
                    arr = np.append(arr,0)
                else:
                    arr = np.append(arr, values[val])

                to_ret.append(arr)
        return to_ret

    @staticmethod
    def calculate_distance(list_x, list_y):

        assert len(list_x) == len(list_y)
        # assert len(types) == len(list_x)
        distance_sum = 0
        for i in range(len(list_y)):
            # attr_dist = 0  # the distance for each attribute; normalized between 0 and 1
            if i % 4 == 0 or i % 4 == 1:
                if list_x[i] == list_y[i]:
                    attr_dist = 0
                else:
                    attr_dist = 1
            else:
                attr_dist = abs(list_x[i] - list_y[i]) / 12.0

            distance_sum += attr_dist
        distance = distance_sum / len(list_x)
        return distance

    def generate_clusters_using_database(self, database,percent,index):
        index = index + len(self.ub)
        self.preprocess_data(database,percent,index)
        if not os.path.exists('lib'):
            os.makedirs('lib')
        dataset = pd.read_csv("lib/clean_data.csv",header=None)
        X = dataset.iloc[:, 0:len(self.ub)].values
        Y = dataset.iloc[:, index:index + 1].values
        X = scaler.fit_transform(X)

        clusterer = DBSCAN(metric=self.calculate_distance)
        clusterer.fit(X)
        all_data_clusters = []
        for label in range(0, clusterer.labels_.max() + 1):
            array = self.values_with_label(X, label, clusterer.labels_, Y)
            all_data_clusters.append(array)
        if clusterer.labels_.max() == -1:
            array = self.values_with_label(X, -1, clusterer.labels_, Y)
            all_data_clusters.append(array)
            return all_data_clusters, -1

        return all_data_clusters,clusterer.labels_.max()

    def find_bounds(self, cluster):
        # print(cluster)
        ub = self.ub
        lb = self.lb

        for feature_vector in cluster:
            fv = feature_vector[0:len(ub)]

            for i in range(len(fv)):
                if fv[i] < lb[i]:
                    lb[i] = fv[i]
                if fv[i] > ub[i]:
                    ub[i] = fv[i]
        return lb, ub

    @staticmethod
    def run_local(i, objective, no_of_neurons, sed, cluster, cluster_id, lb, ub):
        ind = objective + len(ub)
        cs = Pylot_caseStudy(i, n_var_in=len(ub), xl_in=lb, xu_in=ub, number_of_neurons=no_of_neurons, index=ind,
                             percent=20, cluste=cluster, clusters=cluster_id)
        algorithm = GA(
            pop_size=6,
            sampling=get_sampling("int_random"),
            crossover=get_crossover("int_sbx"),
            mutation=get_mutation("int_pm"),
            eliminate_duplicates=True)
        res = min_GA(cs,
                     algorithm,
                     seed=sed,
                     termination=('n_gen', 200),
                     verbose=False)
        logger.info("Best solution found: \nX = %s\nF = %s" % (res.X, res.F))
        toc = time.time()
        X = res.X
        return X, res.F[0]

    def GS(self, start_time, database, obj_uncovered, size, g_max, criteria, lb, ub):
        # start_time, self.database, objective_uncovered, size, g_max, criteria, lb, ub
        logger.info('[GS] Start GS .....')
        objective_uncovered = copy.deepcopy(obj_uncovered)
        M_g = self.train_globals(database, objective_uncovered)
        T_b,T_n = [[],[],[],[],[],[]],[[],[],[],[],[],[]]
        iteration = 0
        P_T = self.generate_random_population(size, lb, ub)  # Generating random population, P_T -> population
        self.evaulate_population_using_ensemble(M_g, P_T)

        while iteration < g_max:
            if self._terminal_check(start_time, self.run_hour):
                return

            iteartion_b, iteration_n = [[], [], [], [], [], []], [[], [], [], [], [], []]
            iteration = iteration + 1  # iteration count
            R_T = []
            Q_T = self.generate_off_spring(P_T, objective_uncovered, lb, ub)  # generating off spring uning ONE point crossover and uniform mutation
            self.evaulate_population_using_ensemble(M_g, Q_T)  # evaluating offspring
            # update_archive(Q_T, objective_uncovered, archive, no_of_Objectives, threshold_criteria)  # updating archive
            R_T = copy.deepcopy(P_T)  # R_T = P_T union Q_T
            R_T.extend(Q_T)

            iteartion_b, iteration_n, R_T = self.update_iteration_bests(R_T, iteartion_b, iteration_n, objective_uncovered)
            T_b, _ = self.update_global_bests(T_b, iteartion_b)
            T_n = self.update_global_bests_uncertainity(T_n, iteration_n)

            F = self.preference_sort(R_T, size, objective_uncovered)  # Reference sorts and getting fronts
            if len(objective_uncovered) == 0:  # checking if all objectives are covered
                logger.info("All Objectives Covered")
                return
            P_T_1 = []  # creating next generatint PT+1
            index = 0
            while len(P_T_1) <= size:  # if length of current generation is less that size of front at top then add it
                if not isinstance(F[index], Candidate):
                    if len(P_T_1) + len(F[index]) > size:
                        break
                else:
                    if len(P_T_1) + 1 > size:
                        break
                front = F[index]
                if isinstance(F[index], Candidate):  # if front contains only one item
                    P_T_1.append(F[index])
                    F.remove(F[index])
                else:
                    for ind in range(len(F[index])):  # if front have multiple items
                        val = F[index][ind]
                        P_T_1.append(val)
                    F.remove(F[index])

            while (len(P_T_1)) < size:  # crowding distance
                copyFront = copy.deepcopy(F[index])
                sorted_front = self.sort_worse(copyFront)  # sort before crowding distance

                crowding_distance = self.get_array_for_crowding_distance(sorted_front)  # coverting to libaray compaitble array
                self.assign_crowding_distance_to_each_value(sorted_front, crowding_distance)  # assinging each solution its crowding distance
                sorted_front.sort(key=self.sort_based_on_crowding_distance, reverse=True)  # sorting based on crowding distance

                if (len(sorted_front) + len(
                        P_T_1)) > size:  # maintaining length and adding solutions with most crowding distances
                    for sorted_front_indx in range(len(sorted_front)):
                        candidate = sorted_front[sorted_front_indx]
                        P_T_1.append(candidate)
                        if len(P_T_1) >= size:
                            break

                index = index + 1

            P_T_1 = P_T_1[0:size]
            P_T = P_T_1  # assigning PT+1 to PT

        T_b.extend(T_n)
        return T_b

    def train_globals(self, database, objective_uncovered):
        ensemble_models = []
        for obj in objective_uncovered:
            db = self.get_data_for_objective(database, obj)
            # logger.debug(db)
            ensemble_model = ensemble(len(self.ub), db, obj)
            ensemble_models.append(ensemble_model)
        return ensemble_models

    @staticmethod
    def get_data_for_objective(database_2, index):
        to_ret = []
        database = copy.deepcopy(database_2)
        for data in database:
            d = data.get_candidate_values()
            d.append(data.get_objective_value(index))
            to_ret.append(d)
        return to_ret

    def LS(self, start_time, database, objective_uncovered, size, l_max, criteria, lb, ub):
        # start_time, self.database, objective_uncovered, size, g_max, criteria, lb, ub

        seed = 0
        T_l = []

        for obj in objective_uncovered:
            if self._terminal_check(start_time, self.run_hour):
                return
            clusters, cluster_id = self.generate_clusters_using_database(database, 20, obj)
            cluster_best_fv = None
            cluster_best_ov = 1
            for cluster in clusters:
                if self._terminal_check(start_time, self.run_hour):
                    return
                seed = seed + 1
                i = seed
                cluster_id = cluster_id + 1
                lb, ub = self.find_bounds(cluster)
                X, R = self.run_local(i, obj, 10, seed, cluster, cluster_id, lb, ub)
                # if R < cluster_best_ov:
                cluster_best_fv = X
                cluster_best_ov = R
                T_l.append(Candidate(cluster_best_fv))
                # cluster is a cluster of data
        return T_l

    @staticmethod
    def calculate_minimum_distance(candidate, random_pop):
        distance = 1000
        for each_candidate in random_pop:
            vals = each_candidate.get_candidate_values()
            candidate_vals = candidate.get_candidate_values()
            dist = np.linalg.norm(np.array(vals) - np.array(candidate_vals))
            if dist < distance:
                distance = dist
        return distance

    @staticmethod
    def generate_random_population(size, lb, ub):
        random_pop = []

        for i in range(size):
            candidate_vals = []
            for index in range(len(lb)):
                candidate_vals.append(int(random.uniform(lb[index], ub[index])))

            random_pop.append(Candidate(candidate_vals))
        return random_pop

    def generate_adaptive_random_population(self, size, lb, ub, i=0):
        random_pop = []

        random_pop.append(self.generate_random_population(1, lb, ub)[0])

        while len(random_pop) < size:
            D = 0
            selected_candidate = None
            rp = self.generate_random_population(size, lb, ub)
            for each_candidate in rp:
                min_dis = self.calculate_minimum_distance(each_candidate, random_pop)
                if min_dis > D:
                    D = min_dis
                    selected_candidate = each_candidate
            random_pop.append(selected_candidate)

        return random_pop

    @staticmethod
    def exists_in_archive(archive, index):
        for candidate in archive:
            if candidate.exists_in_satisfied(index):
                return True
        return False

    @staticmethod
    def get_from_archive(obj_index, archive):
        for candIndx in range(len(archive)):
            candidate = archive[candIndx]
            if candidate.exists_in_satisfied(obj_index):
                return candidate, candIndx
        return None

    def update_archive(self, pop, objective_uncovered, archive, no_of_Objectives, threshold_criteria):
        for objective_index in range(no_of_Objectives):
            for pop_index in range(len(pop)):
                objective_values = pop[pop_index].get_objective_values()
                if objective_values[objective_index] <= threshold_criteria[objective_index]:
                    if self.exists_in_archive(archive, objective_index):
                        archive_value, cand_indx = self.get_from_archive(objective_index, archive)
                        obj_archive_values = archive_value.get_objective_values()
                        if obj_archive_values[objective_index] > objective_values[objective_index]:
                            value_to_add = pop[pop_index]
                            value_to_add.add_objectives_covered(objective_index)
                            # archive.append(value_to_add)
                            archive[cand_indx] = value_to_add
                            if objective_index in objective_uncovered:
                                objective_uncovered.remove(objective_index)
                            # archive.remove(archive_value)
                    else:
                        value_to_add = pop[pop_index]
                        value_to_add.add_objectives_covered(objective_index)
                        archive.append(value_to_add)
                        if objective_index in objective_uncovered:
                            objective_uncovered.remove(objective_index)

    def minimize(self, start_time, criteria, archive, g_max):
        self.run_search(start_time, criteria, archive, g_max)

    def run(self):
        self.last_update_time = time.time()
        logger.info('===== Start Fuzzer (SAMOTA) =====')
        archive = []
        threshold_criteria = [0, 0] # collision 0.0, reach_destination 0.0
        g_max = 200
        self.minimize(self.last_update_time, threshold_criteria, archive, g_max)

    @staticmethod
    def remove_covered_objs(T_g):
        to_ret = []
        for g in T_g:
            if isinstance(g, Candidate):
                to_ret.append(g)
        return to_ret


