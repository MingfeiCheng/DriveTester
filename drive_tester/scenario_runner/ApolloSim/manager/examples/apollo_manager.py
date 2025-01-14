import copy

from typing import List
from loguru import logger

from apollo_sim.agents.vehicle.apollo_vehicle import ApolloConfig, ApolloAgent
from apollo_sim.sim_env import SimEnv
from apollo_sim.registry import ORACLE_REGISTRY, ACTOR_REGISTRY, FITNESS_REGISTRY
from drive_tester.scenario_runner.ApolloSim.manager import SubScenarioManager

class ApolloManager(SubScenarioManager):

    def __init__(
            self,
            scenario_idx: str,
            config_pool: List[ApolloConfig],
            sim_env: SimEnv,
            output_folder: str,
            # other configs
            debug: bool = False
    ):
        super(ApolloManager, self).__init__(
            scenario_idx=scenario_idx,
            config_pool=config_pool,
            sim_env=sim_env,
            output_folder=output_folder,
            debug=debug
        )

    def _initialize(self):
        for i, config in enumerate(self.config_pool):
            actor_class = ACTOR_REGISTRY.get(config.category)
            actor_location = copy.deepcopy(config.location)
            actor = actor_class(
                id=config.idx,
                location=actor_location,
                role=config.role
            )
            self.sim_env.register_actor(actor)

            # register oracle for each apollo
            actor_oracle_lst = self._create_oracle(actor, config.route)
            for actor_oracle in actor_oracle_lst:
                self.sim_env.register_oracle(actor_oracle)

            # register fitness for each apollo
            actor_fitness_lst = self._create_fitness(actor)
            for actor_fitness in actor_fitness_lst:
                self.sim_env.register_fitness(actor_fitness)

            self.agent_list.append(
                ApolloAgent(
                    actor=actor,
                    sim_env=self.sim_env,
                    trigger_time=config.trigger_time,
                    route=config.route,
                    output_folder=self.output_folder,
                    scenario_idx=self.scenario_idx,
                    debug=self.debug,
                )
            )

    def _create_oracle(self, actor, actor_route) -> List:
        oracle_list = []

        # collision
        oracle_collision_class = ORACLE_REGISTRY.get('oracle.collision')
        oracle_collision = oracle_collision_class(
            idx=f"{actor.id}_collision",
            actor=actor,
            sim_env=self.sim_env,
            terminate_on_failure=True,
            threshold=0.01
        )

        # destination
        oracle_destination_class = ORACLE_REGISTRY.get('oracle.destination')
        oracle_destination = oracle_destination_class(
            idx=f"{actor.id}_destination",
            actor=actor,
            destination=copy.deepcopy(actor_route[-1]),
            sim_env=self.sim_env,
            threshold=3.0,
            terminate_on_failure=False
        )

        # stuck
        oracle_stuck_class = ORACLE_REGISTRY.get('oracle.stuck')
        oracle_stuck = oracle_stuck_class(
            idx=f"{actor.id}_stuck",
            actor=actor,
            sim_env=self.sim_env,
            speed_threshold=0.3,
            max_stuck_time=60,
            terminate_on_failure=False
        )

        # timeout
        oracle_timeout_class = ORACLE_REGISTRY.get('oracle.timeout')
        oracle_timeout = oracle_timeout_class(
            idx=f"{actor.id}_timeout",
            sim_env=self.sim_env,
            time_limit=300,
            terminate_on_failure=True
        )

        oracle_list.append(oracle_collision)
        oracle_list.append(oracle_destination)
        oracle_list.append(oracle_stuck)
        oracle_list.append(oracle_timeout)

        return oracle_list

    def _create_fitness(self, actor) -> List:
        fitness_list = []

        # collision
        fitness_collision_class = FITNESS_REGISTRY.get('fitness.collision')
        fitness_collision = fitness_collision_class(
            idx=f"{actor.id}_collision",
            actor=actor,
            sim_env=self.sim_env
        )

        # destination
        fitness_destination_class = FITNESS_REGISTRY.get('fitness.destination')
        fitness_destination = fitness_destination_class(
            idx=f"{actor.id}_destination",
            actor=actor,
            sim_env=self.sim_env,
            threshold_speed=0.3
        )

        fitness_list.append(fitness_collision)
        fitness_list.append(fitness_destination)
        return fitness_list