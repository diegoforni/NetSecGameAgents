import sys
from os import path
# This is used so the agent can see the environment and game components
sys.path.append(path.dirname(path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) )))
basePath = path.dirname(path.dirname(path.dirname( path.dirname( path.abspath(__file__) ) ) ))

import logging
import os
from random import choice
import argparse
import numpy as np
import math
import yaml

import pandas as pd 
import copy
import json
import csv
import time
import random
from env.network_security_game import NetworkSecurityEnvironment
from os import path

import mlflow

# This is used so the agent can see the environment and game components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))
# with the path fixed, we can import now
from env.game_components import Action, Observation, GameState, ActionType
from base_agent import BaseAgent
from agent_utils import generate_valid_actions
from datetime import datetime


class GeneticAgent(BaseAgent):

    def __init__(self, host, port,role, seed) -> None:
        super().__init__(host, port, role)
        np.set_printoptions(suppress=True, precision=6)
        self.parsed_population = []

    def append_to_parsed_population(self, individual):
        self.parsed_population.append(individual)

    def parse_action(action):
        return {
            "type": action.action_type.name,
            "params": action.params
        }
    



    def generate_valid_actions_separated(self,state: GameState)->list:
        """Function that generates a list of all valid actions in a given state"""
        valid_scan_network = set()
        valid_find_services = set()
        valid_exploit_service = set()
        valid_find_data = set()
        valid_exfiltrate_data = set()


        for src_host in state.controlled_hosts:
            #Network Scans
            for network in state.known_networks:
                # TODO ADD neighbouring networks
                valid_scan_network.add(Action(ActionType.ScanNetwork, params={"target_network": network, "source_host": src_host,}))
            # Service Scans
            for host in state.known_hosts:
                valid_find_services.add(Action(ActionType.FindServices, params={"target_host": host, "source_host": src_host,}))
            # Service Exploits
            for host, service_list in state.known_services.items():
                for service in service_list:
                    valid_exploit_service.add(Action(ActionType.ExploitService, params={"target_host": host,"target_service": service,"source_host": src_host,}))
        # Data Scans
        for host in state.controlled_hosts:
            valid_find_data.add(Action(ActionType.FindData, params={"target_host": host, "source_host": host}))

        # Data Exfiltration
        for src_host, data_list in state.known_data.items():
            for data in data_list:
                for trg_host in state.controlled_hosts:
                    if trg_host != src_host:
                        valid_exfiltrate_data.add(Action(ActionType.ExfiltrateData, params={"target_host": trg_host, "source_host": src_host, "data": data}))


        return list(valid_scan_network), list(valid_find_services), list(valid_exploit_service), list(valid_find_data), list(valid_exfiltrate_data), []
  
  
    def select_action_random_agent(self, observation: Observation, lastActionType) -> Action:
        # Generate valid actions as a tuple of lists
        valid_actions = self.generate_valid_actions_separated(observation.state)



        
        
        # Transition probabilities

        def normalize_probabilities(transitions):
            """Function to normalize transition probabilities to ensure they sum up to 1."""
            normalized_transitions = {}
            
            for action_type, probs in transitions.items():
                total = sum(probs)
                if total != 1:
                    # Normalize the probabilities by dividing each by the total sum
                    normalized_transitions[action_type] = [prob / total for prob in probs]
                else:
                    normalized_transitions[action_type] = probs  # Already normalized

            return normalized_transitions
        
        

        transitions = {
            # Action type: [ScanNetwork, FindServices, ExploitService, FindData, ExfiltrateData, InvalidAction]
            "Initial": [1.0, 0, 0, 0, 0, 0],
            ActionType.ScanNetwork: [0.05, 0.93, 0.02, 0, 0, 0],
            ActionType.FindServices: [0.02, 0.45, 0.0, 0.52,0,0],
            ActionType.ExploitService: [0.0, 0.1, 0.99, 0.0, 0,0],
            ActionType.FindData: [0.39, 0.12, 0.21, 0.01, 0.28, 0],
            ActionType.ExfiltrateData: [0.12, 0.11, 0.0, 0, 0.65,0.12]
        }


        transitions = normalize_probabilities(transitions)

        # Set default lastActionType
        if lastActionType is None:
            lastActionType = "Initial"
        
        # Ensure the length of transitions matches the number of action categories
        if len(valid_actions) != len(transitions[lastActionType]):
            raise ValueError(f"Mismatch between valid action lists and transition probabilities: {len(valid_actions)} vs {len(transitions[lastActionType])}")

        # Step 1: Select which action type to pick from, based on probabilities
        actions_to_pick_from = 0

        while actions_to_pick_from == 0:
            selected_action_type_index = np.random.choice(len(valid_actions), p=transitions[lastActionType])
            if selected_action_type_index == 5:
                all_valid_actions = valid_actions[0] + valid_actions[1] + valid_actions[2] + valid_actions[3] + valid_actions[4]
                invalid_actions = []
                env = NetworkSecurityEnvironment(path.join(basePath, 'env', 'netsecenv_conf.yaml'))
                all_actions = env.get_all_actions()
                for action in all_actions:
                    if action not in all_valid_actions:
                        invalid_actions.append(action)
                
            selected_action_list = valid_actions[selected_action_type_index]
            actions_to_pick_from = len(selected_action_list)
        if selected_action_type_index == 5:
            selected_action = random.choice(invalid_actions)
            print("Invalid action selected")

        else:
            selected_action = np.random.choice(selected_action_list)
        return selected_action
    
    def play_game_random_agent(self, observation):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        num_steps = 0
        actions = []
        taken_actions = {}
        episodic_returns = []
        lastActionType = None
        while observation and not observation.end:
            num_steps += 1
            # Store returns in the episode
            episodic_returns.append(observation.reward)
            # Select the action randomly
            action = agent.select_action_random_agent(observation, lastActionType)
            #Arreglar posible falla
            lastActionType = action.type
            taken_actions[action] = True
            actions.append(action)
            
            observation = agent.make_step(action)
            # To return
        # actions needs to be of length max_steps
        # select random actions to fill the rest of the list
        #Arreglar 100 hardcoded

        while len(actions) < 100:
            valid_action = agent.select_action_random_agent(observation, lastActionType)
            actions.append(valid_action)

        
        return actions



    def play_game(self, observation, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """

        DEFAULT_POPULATION_SIZE = 2500
        DEFAULT_NUM_GENERATIONS = -1
        DEFAULT_REPLACEMENT = True
        DEFAULT_NUM_PER_TOURNAMENT = 5
        DEFAULT_N_POINTS = True
        DEFAULT_NUM_POINTS = 6
        DEFAULT_P_VALUE = 0.5
        DEFAULT_CROSS_PROB = 1.0 
        DEFAULT_PARAMETER_MUTATION = False # if False, mutation is by action
        DEFAULT_MUTATION_PROB = 0.0333 # This was calculated for an individual of 30 actions
        DEFAULT_NUM_REPLACE = 50 
        DEFAULT_PATH_GENETIC = "./genetic"
        DEFAULT_PATH_RESULTS = "./results"

        
        

        def fitness_eval_v02(individual, observation, final, num_steps = 0):
            #This function rewards when a changing state is observed, it does not care if the action is valid or not (e.g. FindServices on a host before doing the corresponding ScanNetwork is not valid, but it is possible and the state will probably change, so it is rewarded).
            #Furthermore, if the state does not change but the action is valid, it does not contribute to the reward.
            #Finally, actions that do not change the state and are not valid are penalized.
            #A "good action" is an action that changes the state (not necessarily a "valid" action).
            i = 0
            num_good_actions = 0
            num_boring_actions = 0
            num_bad_actions = 0
            reward = 0
            reward_goal = 0

            individual_result = [[0,0] for _ in range(len(individual))]

            if observation is None:
                observation = agent.request_game_reset()

            current_state = observation.state
            while i < len(individual) and not observation.end:
                valid_actions = generate_valid_actions(current_state)

                individual_result[i][0] = individual[i]
                
                if individual[i] in valid_actions:
                    observation = agent.make_step(individual[i])
                if num_steps is None:
                    num_steps = 0
                num_steps += 1 
                new_state = observation.state

                
                if current_state != new_state:
                    num_good_actions += 1
                    individual_result[i][1] = 1
                    action_type = individual[i].type
                    if action_type == ActionType.ScanNetwork:
                        reward = 10
                    elif action_type == ActionType.FindServices:
                        reward = 20
                    elif action_type == ActionType.ExploitService:
                        reward = 50
                    elif action_type == ActionType.FindData:
                        reward = 75
                    elif action_type == ActionType.ExfiltrateData:
                        reward = 75
                else:
                    if individual[i] in valid_actions:
                        reward += -10
                        num_boring_actions += 1
                        individual_result[i][1] = 0

                    else:
                        reward += -100
                        num_bad_actions += 1
                        individual_result[i][1] = -1
                current_state = observation.state
                i += 1
                #print(reward)
            

            if "end_reason" in observation.info and observation.info["end_reason"] == "goal_reached":
                individual_result[i - 1][1] = 9
                won = 1
            else:
                won = 0

            final_reward = reward + reward_goal
            div_aux = num_steps - num_good_actions + num_bad_actions


          
                
            #print(reward,reward_goal,num_steps,div_aux)
            if div_aux == 0:
                # i.e. when num_steps == num_good_actions and num_bad_actions == 0
                # if num_bad_actions > 0, then num_steps + num_bad_actions != num_good_actions because num_steps > num_good_actions
                div = num_steps
            else:
                div = div_aux

            if final_reward >= 0:
                return_reward = final_reward / div
            else:
                return_reward = final_reward 

            if won == 1:
                return_reward = 7500 + 100000/(num_good_actions + 1.5 * num_boring_actions + 2 * num_bad_actions )
            #print(return_reward, num_good_actions, num_boring_actions, num_bad_actions, num_steps)
            if final is True:
                parsed_individual_result = []
                i = 0
                while i < len(individual):
                    parsed_individual_result.append([individual[i], individual_result[i][1]])
                    i += 1
                    if individual_result[i - 1][1] == 9:
                        break
                parsed_individual_result.append(return_reward)
                agent.append_to_parsed_population(parsed_individual_result)

            return return_reward, num_good_actions, num_boring_actions, num_bad_actions, num_steps, won
        

        def get_all_actions_by_type(all_actions):
            all_actions_by_type = {}
            ScanNetwork_list=[]
            FindServices_list=[]
            ExploitService_list=[]
            FindData_list=[]
            ExfiltrateData_list=[]
            for i in range(len(all_actions)):
                if ActionType.ScanNetwork==all_actions[i].type:
                    ScanNetwork_list.append(all_actions[i])
                elif ActionType.FindServices==all_actions[i].type:
                    FindServices_list.append(all_actions[i])
                elif ActionType.ExploitService==all_actions[i].type:
                    ExploitService_list.append(all_actions[i])
                elif ActionType.FindData==all_actions[i].type:
                    FindData_list.append(all_actions[i])
                else:
                    ExfiltrateData_list.append(all_actions[i])
            all_actions_by_type["ActionType.ScanNetwork"] = ScanNetwork_list
            all_actions_by_type["ActionType.FindServices"] = FindServices_list
            all_actions_by_type["ActionType.ExploitService"] = ExploitService_list
            all_actions_by_type["ActionType.FindData"] = FindData_list
            all_actions_by_type["ActionType.ExfiltrateData"] = ExfiltrateData_list
            return all_actions_by_type
        
        
        

        env = NetworkSecurityEnvironment(path.join(basePath, 'env', 'netsecenv_conf.yaml'))
        all_actions = env.get_all_actions()
        max_number_steps = env._max_steps

        all_actions_by_type = get_all_actions_by_type(all_actions)

        # GA parameters
        population_size = DEFAULT_POPULATION_SIZE
        num_generations = DEFAULT_NUM_GENERATIONS

        # parents selection (tournament) parameters
        select_parents_with_replacement = DEFAULT_REPLACEMENT
        num_per_tournament = DEFAULT_NUM_PER_TOURNAMENT

        # crossover parameters
        Npoints = DEFAULT_N_POINTS
        if DEFAULT_N_POINTS:
            num_points = DEFAULT_NUM_POINTS
        else:
            p_value = DEFAULT_P_VALUE
        cross_prob = DEFAULT_CROSS_PROB

        # mutation parameters
        parameter_mutation = DEFAULT_PARAMETER_MUTATION
        mutation_prob = DEFAULT_MUTATION_PROB

        # survivor selection parameters
        num_replace = DEFAULT_NUM_REPLACE

        # path to save results
        PATH_GENETIC = DEFAULT_PATH_GENETIC
        PATH_RESULTS = DEFAULT_PATH_RESULTS


        # Initialize population
        # Arreglar, sacar la inicializacion y remplazo
        population = [[random.choice(all_actions) for _ in range(max_number_steps)] for _ in range(population_size)]

        # Replace 10% of the population with greedy agents
        for i in range(int(population_size)):
            population[i] = agent.play_game_random_agent(agent.request_game_reset())
            print(" Individual ", i, " done")

        #print("Best initial fitness: ", max([fitness_eval_v02(individual, agent.request_game_reset(),False, 0)[0] for individual in population]))

        # Generations
        print("Population done")

        generation = 0


       



        # calculate scores for last generation, and update files:

        last_generation_scores = np.array([fitness_eval_v02(individual,  agent.request_game_reset(),True, 0) for individual in population])
        index_best_score = np.argmax(last_generation_scores[:,0])
        best_score_complete = last_generation_scores[index_best_score, :]
        metrics_mean = np.mean(last_generation_scores, axis=0)
        metrics_std = np.std(last_generation_scores, axis=0)
        # save best, mean and std scores from last generation
        with open(path.join(PATH_RESULTS, 'best_scores.csv'), 'a', newline='') as partial_file:
            writer_csv = csv.writer(partial_file)
            writer_csv.writerow(best_score_complete)
        with open(path.join(PATH_RESULTS, 'metrics_mean.csv'), 'a', newline='') as partial_file:
            writer_csv = csv.writer(partial_file)
            writer_csv.writerow(metrics_mean)
        with open(path.join(PATH_RESULTS, 'metrics_std.csv'), 'a', newline='') as partial_file:
            writer_csv = csv.writer(partial_file)
            writer_csv.writerow(metrics_std)


        # the best sequence
        best_sequence = population[index_best_score]

        print("\nGeneration = ", generation)

        print("\nBest sequence: \n")
        for i in range(max_number_steps):
            print(best_sequence[i])

        print("\nBest sequence score: ", best_score_complete)
        # Serializar la población completa
        parsed_population_json = []

        population = agent.parsed_population

        # Save the parsed population in a json file
        for individual in population:
            individual_str = []
            for i in range(len(individual) - 1):
                # add the action and the result of the action as string, transform action to string, not as_json
                string = ""
                string += str(individual[i]) 
                individual_str.append(string)


            #individual_str.append(individual[-1])
            parsed_population_json.append(individual_str)
              
        with open(path.join(PATH_RESULTS, 'parsed_population.json'), "a") as f:
            json.dump(parsed_population_json, f, indent=2)
        
        parsed_best_sequence = []
        individual = population[index_best_score]
        i = 0
        while i < len(individual):
            parsed_best_sequence.append(str([individual[i]]))
            i += 1
            if individual[i - 1][1] == 9:
                break
        
        with open(path.join(PATH_RESULTS, 'best_sequence.json'), "a") as f:
            json.dump(parsed_best_sequence, f, indent=2)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9005, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to play or evaluate", default=100, type=int) 
    args = parser.parse_args()

 
    # Create agent 
    # read seed from /Users/diegoforni/Documents/labsin/NetSecGame/env/netsecenv_conf.yaml
    config_path = "/Users/diegoforni/Documents/labsin/NetSecGame/env/netsecenv_conf.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    seedYaml = config['env']['random_seed']
    
    print("Starting genetic agent with seed: ", seedYaml)
    agent = GeneticAgent(args.host, args.port,"Attacker", seed=seedYaml)



    observation = agent.register()
    agent.play_game(observation, args.episodes)
    agent._logger.info("Terminating interaction")
    agent.terminate_connection()
    