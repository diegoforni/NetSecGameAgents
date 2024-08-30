import sys
from os import path
# script:
sys.path.append( path.dirname(path.dirname( path.dirname(path.dirname( path.abspath(__file__) ) ) )))
# terminal:
#sys.path.append(path.dirname(path.dirname(path.abspath(''))))

import time
import random
import numpy as np
import pandas as pd
import copy
import json
import csv
#import argparse
from env.network_security_game import NetworkSecurityEnvironment
#from agents.random.random_agent import RandomAgent
from env.game_components import Action, ActionType #, IP, Data, Network, Service


POPULATION_SIZE = eval(sys.argv[1])
print("POPULATION_SIZE = ", POPULATION_SIZE)

NUM_GENERATIONS = eval(sys.argv[2])
print("NUM_GENERATIONS = ", NUM_GENERATIONS)

# parents selection (tournament) parameters
REPLACEMENT = sys.argv[3].lower()=="true"
print("REPLACEMENT = ", REPLACEMENT)

NUM_PER_TOURNAMENT = eval(sys.argv[4])
print("NUM_PER_TOURNAMENT = ", NUM_PER_TOURNAMENT)

# crossover parameters
N_POINTS = sys.argv[5].lower()=="true"
if N_POINTS:
    print("CROSSOVER = N-points")
    NUM_POINTS = eval(sys.argv[6])
    print("NUM_POINTS = ", NUM_POINTS)
else:
    print("CROSSOVER = uniform")
    P_VALUE = eval(sys.argv[7])
    print("P_VALUE = ", P_VALUE)

CROSS_PROB = eval(sys.argv[8])
print("CROSS_PROB = ", CROSS_PROB)

# mutation parameters
PARAMETER_MUTATION = sys.argv[9].lower()=="true"
if PARAMETER_MUTATION:
    print("MUTATION = by parameter")
else:
    print("MUTATION = by action")

MUTATION_PROB = eval(sys.argv[10])
print("MUTATION_PROB = ", MUTATION_PROB)

# survivor selection parameters
NUM_REPLACE = eval(sys.argv[11])
print("NUM_REPLACE = ", NUM_REPLACE)

# paths
PATH_GENETIC = str(sys.argv[12])
print("PATH_GENETIC ", PATH_GENETIC)

PATH_RESULTS = str(sys.argv[13])
print("PATH_RESULTS ", PATH_RESULTS)


def generate_valid_actions(state):
    # (function copied from RandomAgent)
    valid_actions = set()
    #Network Scans
    for network in state.known_networks:
        # TODO ADD neighbouring networks
        valid_actions.add(Action(ActionType.ScanNetwork, params={"target_network": network}))
    # Service Scans
    for host in state.known_hosts:
        valid_actions.add(Action(ActionType.FindServices, params={"target_host": host}))
    # Service Exploits
    for host, service_list in state.known_services.items():
        for service in service_list:
            valid_actions.add(Action(ActionType.ExploitService, params={"target_host": host , "target_service": service}))
    # Data Scans
    for host in state.controlled_hosts:
        valid_actions.add(Action(ActionType.FindData, params={"target_host": host}))
    # Data Exfiltration
    for src_host, data_list in state.known_data.items():
        for data in data_list:
            for trg_host in state.controlled_hosts:
                if trg_host != src_host:
                    valid_actions.add(Action(ActionType.ExfiltrateData, params={"target_host": trg_host, "source_host": src_host, "data": data}))
    return list(valid_actions)


def choose_parents_tournament(population, goal, fitness_func, num_per_tournament=2, parents_should_differ=True):
    """ Tournament selection """
    from_population = population.copy()
    chosen = []
    for i in range(2):
        options = []
        for _ in range(num_per_tournament):
            options.append(random.choice(from_population))
        chosen.append(max(options, key=lambda x:fitness_func(x,env.reset(),goal)[0])) # add [0] because fitness_eval_v3 returns a tuple
        #chosen.append(max(options, key=lambda x:fitness_eval_v2(x,env.reset(),goal)))
        if i==0 and parents_should_differ:
            from_population.remove(chosen[0])
    return chosen[0], chosen[1]



#def choose_parents_rouletteWheel(population, population_scores, parents_should_differ=True):
#    # Roulette wheel  selection 
#    prob = population_scores / np.sum(population_scores)
#    wheel = np.cumsum(prob)
#    chosen = []
#    run = True
#    while run:
#        random_num = random.random()
#        index_wheel = np.searchsorted(wheel, random_num)
#        chosen.append(population[index_wheel])
#        if len(chosen)==2:
#            if (parents_should_differ and chosen[0] != chosen[1]) or not parents_should_differ:
#                run = False
#            else:
#                _ = chosen.pop(1)
#    return chosen[0], chosen[1]


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



#def fitness_eval_v1(individual, observation, goal): 
#    # This function rewards when a valid action is performed and a changing state is observed. If the state does not change, it does not contribute to the reward.
#    # Actions that are not valid are penalized.
#    # A 'good action' is an action that is valid and changes the state.
#    num_good_actions = 0
#    reward = 0
#    reward_goal = 0
#    current_state = observation.state
#    for i in range(len(individual)):
#        valid_actions = generate_valid_actions(current_state)
#        observation = env.step(individual[i])
#        new_state = observation.state
#        if individual[i] in valid_actions:
#            if current_state != new_state:
#                num_good_actions += 1
#                reward += 10
#            else: 
#                reward += 0
#        else:
#            reward += -100
#        current_state = observation.state
#        if observation.info != {} and observation.info["end_reason"]=="goal_reached":
#            reward_goal = 10000
#            break
#    final_reward = reward + reward_goal
#    if final_reward >= 0:
#        return final_reward / num_good_actions
#    else:
#        return final_reward


#def fitness_eval_v2(individual, observation, goal): 
#    # This function rewards when a changing state is observed, it does not care if the action is valid or not (e.g. FindServices on a host before doing the corresponding ScanNetwork is not valid, but it is possible and the state will probably change, so it is rewarded).
#    # Furthermore, if the state does not change but the action is valid, it does not contribute to the reward.
#    # Finally, actions that do not change the state and are not valid are penalized.
#    # A "good action" is an action that changes the state (not necessarily a "valid" action).
#    #num_good_actions = 0
#    #num_bad_actions = 0
#    reward = 0
#    reward_goal = 0
#    current_state = observation.state
#    for i in range(len(individual)):
#        valid_actions = generate_valid_actions(current_state)
#        observation = env.step(individual[i])
#        new_state = observation.state
#        if current_state != new_state:
#            reward += 10
#            #num_good_actions += 1
#        else:
#            if individual[i] in valid_actions:
#                reward += 0
#            else:
#                reward += -100
#                #num_bad_actions += 1
#        current_state = observation.state
#        if observation.info != {} and observation.info["end_reason"] == "goal_reached":
#            reward_goal = 10000
#            break
#    final_reward = reward + reward_goal
#    num_steps = env.timestamp
#    #div_aux = num_steps - num_good_actions + num_bad_actions
#    #if div_aux == 0:
#        # i.e. when num_steps == num_good_actions and num_bad_actions == 0
#        # if num_bad_actions > 0, then num_steps + num_bad_actions != num_good_actions because num_steps > num_good_actions
#    #    div = num_steps
#    #else:
#    #    div = div_aux
#    if final_reward >= 0:
#        return final_reward / num_steps #div
#    else:
#        return final_reward

    
#def fitness_eval_v01(individual, observation, goal): 
#    # This function rewards when a changing state is observed, it does not care if the action is valid or not (e.g. FindServices on a host before doing the corresponding ScanNetwork is not valid, but it is possible and the state will probably change, so it is rewarded).
#    # Furthermore, if the state does not change but the action is valid, it does not contribute to the reward.
#    # Finally, actions that do not change the state and are not valid are penalized.
#    # A "good action" is an action that changes the state (not necessarily a "valid" action).
#    i = 0
#    num_good_actions = 0
#    num_boring_actions = 0
#    num_bad_actions = 0
#    reward = 0
#    reward_goal = 0
#    current_state = observation.state
#    while i < len(individual) and not observation.done:
#        valid_actions = generate_valid_actions(current_state)
#        observation = env.step(individual[i])
#        new_state = observation.state
#        if current_state != new_state:
#            reward += 10
#            num_good_actions += 1
#        else:
#            if individual[i] in valid_actions:
#                reward += 0
#                num_boring_actions += 1
#            else:
#                reward += -100
#                num_bad_actions += 1
#        current_state = observation.state
#        i += 1
#    num_steps = env.timestamp
#    if observation.info != {} and observation.info["end_reason"] == "goal_reached":
#        reward_goal = 10000
#    final_reward = reward + reward_goal
#    if final_reward >= 0:
#        return_reward = final_reward / num_good_actions
#    else:
#        return_reward = final_reward 
#    return return_reward, num_good_actions, num_boring_actions, num_bad_actions, num_steps
    

def fitness_eval_v02(individual, observation, goal):
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
    current_state = observation.state
    while i < len(individual) and not observation.done:
        valid_actions = generate_valid_actions(current_state)
        observation = env.step(individual[i])
        new_state = observation.state
        if current_state != new_state:
            reward += 10
            num_good_actions += 1
        else:
            if individual[i] in valid_actions:
                reward += 0
                num_boring_actions += 1
            else:
                reward += -100
                num_bad_actions += 1
        current_state = observation.state
        i += 1
        #print(reward)
    num_steps = env.timestamp
    if observation.info != {} and observation.info["end_reason"] == "goal_reached":
        reward_goal = 10000
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
    #print(return_reward, num_good_actions, num_boring_actions, num_bad_actions, num_steps)
    return return_reward, num_good_actions, num_boring_actions, num_bad_actions, num_steps
   


#def fitness_eval_v03(individual, observation, goal): 
#    # This function rewards when a changing state is observed, it does not care if the action is valid or not (e.g. FindServices on a host before doing the corresponding ScanNetwork is not valid, but it is possible and the state will probably change, so it is rewarded).
#    # Furthermore, if the state does not change but the action is valid, it does not contribute to the reward.
#    # Finally, actions that do not change the state and are not valid are penalized.
#    # A "good action" is an action that changes the state (not necessarily a "valid" action).
#    i = 0
#    num_good_actions = 0
#    num_boring_actions = 0
#    num_bad_actions = 0
#    reward = 0
#    reward_goal = 0
#    current_state = observation.state
#    while i < len(individual) and not observation.done:
#        valid_actions = generate_valid_actions(current_state)
#        observation = env.step(individual[i])
#        new_state = observation.state
#        if current_state != new_state:
#            reward += 10
#            num_good_actions += 1
#        else:
#            if individual[i] in valid_actions:
#                reward += -20
#                num_boring_actions += 1
#            else:
#                reward += -100
#                num_bad_actions += 1
#        current_state = observation.state
#        i += 1
#    num_steps = env.timestamp
#    if observation.info != {} and observation.info["end_reason"] == "goal_reached":
#        reward_goal = 10000
#    final_reward = reward + reward_goal
#    if final_reward >= 0:
#        return_reward = final_reward / num_good_actions
#    else:
#        return_reward = final_reward 
#    return return_reward, num_good_actions, num_boring_actions, num_bad_actions, num_steps
    


#def is_fitness_equal(best_scores):
#    return all(fitness == best_scores[0] for fitness in best_scores)


def steady_state_selection(parents, parents_scores, offspring, offspring_scores, num_replace):
    # parents
    best_indices_parents = np.argsort(parents_scores, axis=0)[:,0] # min to max fitness (higher is better)
    parents_sort = [parents[i] for i in best_indices_parents]
    # offspring
    best_indices_offspring = np.argsort(offspring_scores, axis=0)[:,0] # min to max fitness (higher is better)
    offspring_sort = [offspring[i] for i in best_indices_offspring]
    # new generation
    new_generation = parents_sort[num_replace:] + offspring_sort[population_size-num_replace:]
    return new_generation



#def random_selection(parents, offspring, num_replace):
#    len_parents = len(parents)
#    indices = np.sort(np.random.choice(len_parents, num_replace, replace=False))
#    new_generation = []
#    for i in range(len_parents): 
#        if i not in indices:
#            new_generation.append(parents[i])
#        else:
#            new_generation.append(offspring[i])
#    return new_generation


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


#def evaluate(self, observation:Observation) -> tuple: #(cumulative_reward, goal?, detected?, num_steps)
#    """
#    Evaluate the agent so far for one episode
#    Do without learning
#    """
#    return_value = 0
#    while not observation.done:
#        action = self.move(observation, testing=True)
#        next_observation = self.env.step(action)
#        return_value += next_observation.reward
#        observation = next_observation
#    # Has to return
#    # 1. returns
#    # 2. if it is a win
#    # 3. if it is was detected
#    # 4. amount of steps when finished
#    wins = next_observation.reward > 0
#    detected = self.env.detected
#    steps = self.env.timestamp
#    return return_value, wins, detected, steps
#

################################################################################################################

start_time = time.time()

## The environment is initialized
env = NetworkSecurityEnvironment(path.join(PATH_GENETIC,"netsecenv-task.yaml"))

## Info from environment
goal = copy.deepcopy(env._goal_conditions)
max_number_steps = env._max_steps

## Possible actions
all_actions = env.get_all_actions()
all_actions_by_type = get_all_actions_by_type(all_actions)


## GA parameters
population_size = POPULATION_SIZE
num_generations = NUM_GENERATIONS

# parents selection (tournament) parameters
select_parents_with_replacement = REPLACEMENT
num_per_tournament = NUM_PER_TOURNAMENT

# crossover parameters
Npoints = N_POINTS
if N_POINTS:
    num_points = NUM_POINTS
else:
    p_value = P_VALUE
cross_prob = CROSS_PROB

# mutation parameters
parameter_mutation = PARAMETER_MUTATION
mutation_prob = MUTATION_PROB

# survivor selection parameters
num_replace = NUM_REPLACE


# Initialize population
population = [[random.choice(all_actions) for _ in range(max_number_steps)] for _ in range(population_size)]


# Generations

generation = 0
best_score = 0

try:
    while (generation < num_generations) and (best_score < 2500):
        #print(generation)
        offspring = []
        #print("inic offspring")
        popu_crossover = population.copy()
        #print("copy population")
        parents_scores = np.array([fitness_eval_v02(individual, env.reset(), goal) for individual in population])
        #print("parents_scores")
        index_best_score = np.argmax(parents_scores[:,0])
        #print(index_best_score)
        best_score_complete = parents_scores[index_best_score, :]
        #print(best_score_complete)
        best_score = best_score_complete[0]
        metrics_mean = np.mean(parents_scores, axis=0)
        metrics_std = np.std(parents_scores, axis=0)
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
            parent1, parent2 = choose_parents_tournament(popu_crossover, goal, fitness_eval_v02, num_per_tournament, True)
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
        offspring_scores = np.array([fitness_eval_v02(individual, env.reset(), goal) for individual in offspring])
        # survivor selection
        new_generation = steady_state_selection(population, parents_scores, offspring, offspring_scores, num_replace)
        population = new_generation
        generation += 1
        #print("survivor")
        #print("\n")

except Exception as e:
        print(f"Error: {e}")



# calculate scores for last generation, and update files:
last_generation_scores = np.array([fitness_eval_v02(individual, env.reset(), goal) for individual in population])
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
# save the best sequence in file
best_sequence_json = []
for j in range(max_number_steps):
    best_sequence_json.append(best_sequence[j].as_json())
with open(path.join(PATH_RESULTS,'best_sequence.json'), "a") as f:
    json.dump(best_sequence_json, f, indent=2)


print("\nGeneration = ", generation)

print("\nBest sequence: \n")
for i in range(max_number_steps):
    print(best_sequence[i])

print("\nBest sequence score: ", best_score_complete)

end_time = time.time()
print("\ntime: ", end_time - start_time, "\n")


