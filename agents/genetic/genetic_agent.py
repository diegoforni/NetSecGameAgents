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

  
    def select_action_random_agent(self, observation: Observation) -> Action:
        valid_actions = generate_valid_actions(observation.state)
        action = choice(valid_actions)
        return action
    
    def play_game_random_agent(self, observation):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        num_steps = 0
        actions = []
        taken_actions = {}
        episodic_returns = []
        while observation and not observation.end:
            num_steps += 1
            # Store returns in the episode
            episodic_returns.append(observation.reward)
            # Select the action randomly
            action = agent.select_action_random_agent(observation)
            taken_actions[action] = True
            actions.append(action)
            
            observation = agent.make_step(action)
            # To return
        # actions needs to be of length max_steps
        # select random actions to fill the rest of the list
        #Arreglar 100 hardcoded

        while len(actions) < 100:
            valid_action = agent.select_action_random_agent(observation)
            actions.append(valid_action)
        
        return actions



    def play_game(self, observation, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """

        DEFAULT_POPULATION_SIZE = 2500
        DEFAULT_NUM_GENERATIONS = 55
        DEFAULT_REPLACEMENT = True
        DEFAULT_NUM_PER_TOURNAMENT = 5
        DEFAULT_N_POINTS = True
        DEFAULT_NUM_POINTS = 6
        DEFAULT_P_VALUE = 0.5
        DEFAULT_CROSS_PROB = 1.0
        DEFAULT_PARAMETER_MUTATION = False
        DEFAULT_MUTATION_PROB = 0.0333
        DEFAULT_NUM_REPLACE = 50
        DEFAULT_PATH_GENETIC = "./genetic"
        DEFAULT_PATH_RESULTS = "./results"

        def choose_parents_tournament(population, goal, fitness_func, num_per_tournament=2, parents_should_differ=True):
            """ Tournament selection """
            from_population = population.copy()
            chosen = []
            for i in range(2):
                options = []
                for _ in range(num_per_tournament):
                    options.append(random.choice(from_population))
                chosen.append(max(options, key=lambda x:fitness_func(x,agent.request_game_reset(),goal)[0])) # add [0] because fitness_eval_v3 returns a tuple
                #chosen.append(max(options, key=lambda x:fitness_eval_v2(x,env.reset(),goal)))
                if i==0 and parents_should_differ:
                    from_population.remove(chosen[0])
            return chosen[0], chosen[1]
        
        def mutation_operator_by_parameter(individual, all_actions_by_type, mutation_prob):
            new_individual = []
            for i in range(len(individual)):
                if random.random() < mutation_prob:
                    action_type = individual[i].type
                    new_individual.append(random.choice(all_actions_by_type[str(action_type)]))
                else:
                    new_individual.append(individual[i])
            return new_individual


        def mutation_operator_by_action(individual, all_actions, mutation_prob):
            new_individual = []
            for i in range(len(individual)):
                if random.random() < mutation_prob:
                    new_individual.append(random.choice(all_actions))
                else: 
                    new_individual.append(individual[i])
            return new_individual


        def crossover_operator_Npoints(parent1, parent2, num_points, cross_prob):
            if random.random() < cross_prob:
                len_ind = len(parent1)
                cross_points = np.sort(np.random.choice(len_ind, num_points, replace=False))
                child1 = []
                child2 = []
                current_parent1 = parent1
                current_parent2 = parent2
                for i in range(len_ind):
                    child1.append(current_parent1[i])
                    child2.append(current_parent2[i])
                    if i in cross_points:
                        current_parent1 = parent2 if current_parent1 is parent1 else parent1
                        current_parent2 = parent1 if current_parent2 is parent2 else parent2
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            return child1, child2


        def crossover_operator_uniform(parent1, parent2, p_value, cross_prob):
            if random.random() < cross_prob:
                len_ind = len(parent1)
                child1 = []
                child2 = []
                for i in range(len_ind):
                    if random.random() < p_value:
                        child1.append(parent1[i])
                        child2.append(parent2[i])
                    else:
                        child1.append(parent2[i])
                        child2.append(parent1[i])
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()
            return child1, child2
        

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

            if won == 1:
                div = div * 0.75
                
                def sigmoid_good_steps(good_steps):
                    return 1 / (1 + math.exp(-((1 / 5) * (10 - good_steps))))
                def sigmoid_step_ratio(ratio):
                    return 1 / (1 + math.exp(-((5) * (ratio - 0.5))))

                final_reward = 7500 * sigmoid_good_steps(num_good_actions) * sigmoid_step_ratio(num_good_actions / num_steps)

            if final_reward >= 0:
                return_reward = final_reward / div
            else:
                return_reward = final_reward 

            if won == 1:
                return_reward += 7500
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
        

        def steady_state_selection(parents, parents_scores, offspring, offspring_scores, num_replace):
            # parents
            best_indices_parents = np.argsort(parents_scores, axis=0)[:,0] # min to max fitness (higher is better)
            parents_sort = [parents[i] for i in best_indices_parents]
            # offspring
            best_indices_offspring = np.argsort(offspring_scores, axis=0)[:,0] # min to max fitness (higher is better)
            offspring_sort = [offspring[i] for i in best_indices_offspring]
            # new generation
            #Arreglar (que no sea el default)
            new_generation = parents_sort[num_replace:] + offspring_sort[population_size-num_replace:]
            return new_generation


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
        population = [[random.choice(all_actions) for _ in range(max_number_steps)] for _ in range(population_size)]

        # Replace 10% of the population with greedy agents
        for i in range(int(population_size/10)):
            population[i] = agent.play_game_random_agent(agent.request_game_reset())

        print("Best initial fitness: ", max([fitness_eval_v02(individual, agent.request_game_reset(),False, 0)[0] for individual in population]))

        # Generations

        generation = 0
        best_score = 0


        try:
            while (generation < num_generations) and (best_score < 12500):
                print("Generation: ", generation)
                #print(generation)
                offspring = []
                #print("inic offspring")
                popu_crossover = population.copy()
                #print("copy population")
                parents_scores = np.array([fitness_eval_v02(individual, agent.request_game_reset(),False, 0) for individual in population])
                #print("parents_scores")
                index_best_score = np.argmax(parents_scores[:, 0])
                best_score_complete = parents_scores[index_best_score, :]
                best_score = best_score_complete[0]

                index_worst_score = np.argmin(parents_scores[:, 0])
                worst_score_complete = parents_scores[index_worst_score, :]
                worst_score = worst_score_complete[0]

                print("Amount of individuals: ", len(parents_scores))
                print("Total good actions: ", np.sum(parents_scores[:, 1]))
                print("Best score complete: ", best_score_complete)
                print("Worst score complete: ", worst_score_complete)
                print("Average score complete: ", np.mean(parents_scores, axis=0))
                metrics_mean = np.mean(parents_scores, axis=0)
                metrics_std = np.std(parents_scores, axis=0)
                print("Standard deviation: ", metrics_std)

                if generation % 10 == 0:  # Check every 10 generations
                    
                    if metrics_std[0] < 200:  # Define a threshold for diversity
                        mutation_prob = min(0.1, mutation_prob * 2)  # Increase mutation rate
                        print("Mutation rate increased to: ", mutation_prob)

                #print(best_score,metrics_mean,metrics_std)
                # save best, mean and std scores
                with open(path.join(PATH_RESULTS, 'best_scores.csv'), 'a', newline='') as partial_file:
                    writer_csv = csv.writer(partial_file)
                    writer_csv.writerow(best_score_complete)
                with open(path.join(PATH_RESULTS, 'metrics_mean.csv'), 'a', newline='') as partial_file:
                    writer_csv = csv.writer(partial_file)
                    writer_csv.writerow(metrics_mean)
                with open(path.join(PATH_RESULTS, 'metrics_std.csv'), 'a', newline='') as partial_file:
                    writer_csv = csv.writer(partial_file)
                    writer_csv.writerow(metrics_std)
                for j in range(int(population_size/2)):
                    if j == 0 or select_parents_with_replacement:
                        pass
                    else:
                        popu_crossover.remove(parent1)
                        popu_crossover.remove(parent2)
                    # parents selection

                    parent1, parent2 = choose_parents_tournament(popu_crossover, None, fitness_eval_v02, num_per_tournament, True)
                    #print("parets_selection")
                    # cross-over
                    if Npoints:
                        child1, child2 = crossover_operator_Npoints(parent1, parent2, num_points, cross_prob)
                    else:
                        child1, child2 = crossover_operator_uniform(parent1, parent2, p_value, cross_prob)
                    #print("crossover")
                    # mutation
                    if parameter_mutation:
                        child1 = mutation_operator_by_parameter(child1, all_actions_by_type, mutation_prob)
                        child2 = mutation_operator_by_parameter(child2, all_actions_by_type, mutation_prob)
                    else:
                        child1 = mutation_operator_by_action(child1, all_actions, mutation_prob)
                        child2 = mutation_operator_by_action(child2, all_actions, mutation_prob)
                    #print("mutation")
                    offspring.append(child1)
                    offspring.append(child2)

                offspring_scores = np.array([fitness_eval_v02(individual, agent.request_game_reset(),False, 0) for individual in offspring])
                # survivor selection
                new_generation = steady_state_selection(population, parents_scores, offspring, offspring_scores, num_replace)
                population = new_generation
                generation += 1
                #print("survivor")
                #print("\n")

        except Exception as e:
                print(f"Error: {e}")



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
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to play or evaluate", default=100, type=int) 
    parser.add_argument("--test_each", help="Evaluate performance during testing every this number of episodes.", default=10, type=int)
    parser.add_argument("--logdir", help="Folder to store logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--evaluate", help="Evaluate the agent and report, instead of playing the game only once.", default=True)
    args = parser.parse_args()

 
    # Create agent
    agent = GeneticAgent(args.host, args.port,"Attacker", seed=42)


    observation = agent.register()
    agent.play_game(observation, args.episodes)
    agent._logger.info("Terminating interaction")
    agent.terminate_connection()
    