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

import json

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
       
       
    #Fix, this should be outside the function, only executed once
    # Load JSON data from a file
    with open("transition_matrix.json", "r") as file:
        transitions_data = json.load(file)

    # Initialize an empty dictionary to store the transitions
    transitions = {}

    # Mapping from string keys to ActionType enum members
    action_mapping = {
        "ScanNetwork": ActionType.ScanNetwork,
        "FindServices": ActionType.FindServices,
        "ExploitService": ActionType.ExploitService,
        "FindData": ActionType.FindData,
        "ExfiltrateData": ActionType.ExfiltrateData,
    }

    # Loop through each action's transition probabilities
    for action_data in transitions_data["transition_probabilities"]:
        action = action_data["Action"]

        # Extract the probabilities and store them in the correct order
        probabilities = [
            action_data["ScanNetwork"],
            action_data["FindServices"],
            action_data["ExploitService"],
            action_data["FindData"],
            action_data["ExfiltrateData"]
        ]

        # Assign the probabilities list to the correct action key in the transitions dictionary
        if action == "Initial Action":
            transitions["Initial"] = probabilities
        else:
            # Use the ActionType mapping for other actions
            transitions[action_mapping[action]] = probabilities



    transitions = normalize_probabilities(transitions)


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


        return list(valid_scan_network), list(valid_find_services), list(valid_exploit_service), list(valid_find_data), list(valid_exfiltrate_data), 
  
    
    def select_action_markov_chain_agent(self, observation: Observation, lastActionType) -> Action:
        # Generate valid actions as a tuple of lists
        valid_actions = self.generate_valid_actions_separated(observation.state)
        # Transition probabilities

        

        # Set default lastActionType
        if lastActionType is None:
            lastActionType = "Initial"
        
        # Ensure the length of transitions matches the number of action categories
        if len(valid_actions) != len(self.transitions[lastActionType]):
            raise ValueError(f"Mismatch between valid action lists and transition probabilities: {len(valid_actions)} vs {len(self.transitions[lastActionType])}")

        # Step 1: Select which action type to pick from, based on probabilities
        actions_to_pick_from = 0
        selected_action_type_index = None
        selected_action = None
        while selected_action is None:
            selected_action_type_index = np.random.choice(len(valid_actions), p=self.transitions[lastActionType])
                
            selected_action_list = valid_actions[selected_action_type_index]
            actions_to_pick_from = len(selected_action_list)
            if actions_to_pick_from > 0:
                selected_action = np.random.choice(selected_action_list)
        return selected_action
    
    def play_game_markov_chain_agent(self, observation):
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
            action = agent.select_action_markov_chain_agent(observation, lastActionType)
            lastActionType = action.type
            taken_actions[action] = True
            actions.append(action)
            
            observation = agent.make_step(action)

        
        return actions



    def play_game(self, observation, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        # Fix, this should be episodes like in random agent
        DEFAULT_POPULATION_SIZE = 100
        DEFAULT_PATH_RESULTS = "./results"


        def fitness_eval_v02(individual, observation):
            #This function rewards when a changing state is observed, it does not care if the action is valid or not (e.g. FindServices on a host before doing the corresponding ScanNetwork is not valid, but it is possible and the state will probably change, so it is rewarded).
            #Furthermore, if the state does not change but the action is valid, it does not contribute to the reward.
            #Finally, actions that do not change the state and are not valid are penalized.
            #A "good action" is an action that changes the state (not necessarily a "valid" action).
            i = 0
            num_good_actions = 0
            num_boring_actions = 0
            num_bad_actions = 0


            individual_result = [[0,0] for _ in range(len(individual))]

            if observation is None:
                observation = agent.request_game_reset()

            current_state = observation.state
            while i < len(individual) and not observation.end:
                valid_actions = generate_valid_actions(current_state)

                individual_result[i][0] = individual[i]
                
                if individual[i] in valid_actions:
                    observation = agent.make_step(individual[i])

                new_state = observation.state

                
                if current_state != new_state:
                    num_good_actions += 1
                    individual_result[i][1] = 1
                else:
                    if individual[i] in valid_actions:
                        num_boring_actions += 1
                        individual_result[i][1] = 0

                    else:
                        num_bad_actions += 1
                        individual_result[i][1] = -1
                current_state = observation.state
                i += 1
            
            if "end_reason" in observation.info and observation.info["end_reason"] == "goal_reached":
                individual_result[i - 1][1] = 9

            #print(return_reward, num_good_actions, num_boring_actions, num_bad_actions, num_steps)
            parsed_individual_result = []
            i = 0
            while i < len(individual):
                parsed_individual_result.append([individual[i], individual_result[i][1]])
                i += 1
                if individual_result[i - 1][1] == 9:
                    break
            # Fix, 0 came from reward on GA, it should not exist, but it can not be deleted
            parsed_individual_result.append(0)
            agent.append_to_parsed_population(parsed_individual_result)

            return 
        

        # Fix, this should be episodes like in random agent
        population_size = DEFAULT_POPULATION_SIZE

        PATH_RESULTS = DEFAULT_PATH_RESULTS

        # Initialize population
        # Fix, there is no population
        population = [None] * int(population_size)
        for i in range(int(population_size)):
            population[i] = agent.play_game_markov_chain_agent(agent.request_game_reset())
            print(" Solution ", i, " done")


        #Fix, rename, this saves the solutions
        last_generation_scores = np.array([fitness_eval_v02(individual,  agent.request_game_reset()) for individual in population])


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
    