# Authors:  Ondrej Lukas - ondrej.lukas@aic.fel.cvut.cz
#           Arti
#           Sebastian Garcia. sebastian.garcia@agents.fel.cvut.cz
import sys
import os
import numpy as np
import random
import pickle
import argparse
import logging
# This is used so the agent can see the environment and game component
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))

# This is used so the agent can see the environment and game component
# with the path fixed, we can import now
from env.game_components import Action, Observation, GameState
from base_agent import BaseAgent
from agent_utils import generate_valid_actions, state_as_ordered_string
import mlflow
import subprocess


class QAgent(BaseAgent):

    def __init__(self, host, port, role="Attacker", alpha=0.1, gamma=0.6, epsilon=0.1) -> None:
        super().__init__(host, port, role)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_values = {}
        self._str_to_id = {}

    def store_q_table(self,filename):
        with open(filename, "wb") as f:
            data = {"q_table":self.q_values, "state_mapping": self._str_to_id}
            pickle.dump(data, f)

    def load_q_table(self,filename):
        with open(filename, "rb") as f:
            data = pickle.load(f)
            self.q_values = data["q_table"]
            self._str_to_id = data["state_mapping"]

    def get_state_id(self, state:GameState) -> int:
        state_str = state_as_ordered_string(state)
        if state_str not in self._str_to_id:
            self._str_to_id[state_str] = len(self._str_to_id)
        return self._str_to_id[state_str]
    
    def max_action_q(self, observation:Observation) -> Action:
        state = observation.state
        actions = generate_valid_actions(state)
        state_id = self.get_state_id(state)
        tmp = dict(((state_id, a), self.q_values.get((state_id, a), 0)) for a in actions)
        return tmp[max(tmp,key=tmp.get)] #return maximum Q_value for a given state (out of available actions)
   
    def select_action(self, observation:Observation, testing=False) -> Action:
        state = observation.state
        actions = generate_valid_actions(state)
        state_id = self.get_state_id(state)
        
        # E-greedy play. If the random number is less than the e, then choose random to explore.
        # But do not do it if we are testing a model. 
        if random.uniform(0, 1) <= self.epsilon and not testing:
            # We are training
            # Random choose an ation from the list of actions?
            action = random.choice(list(actions))
            if (state_id, action) not in self.q_values:
                self.q_values[state_id, action] = 0
            return action, state_id
        else: 
            # We are training
            # Select the action with highest q_value
            # The default initial q-value for a (state, action) pair is 0.
            initial_q_value = 0
            tmp = dict(((state_id, action), self.q_values.get((state_id, action), initial_q_value)) for action in actions)
            ((state_id, action), value) = max(tmp.items(), key=lambda x: (x[1], random.random()))
            #if max_q_key not in self.q_values:
            try:
                self.q_values[state_id, action]
            except KeyError:
                self.q_values[state_id, action] = 0
            return action, state_id
        
    def play_game(self, observation, num_episodes=1, testing=False):
        """
        The main function for the gameplay. Handles the main interaction loop.
        """
        returns = []
        num_steps = 0
        for episode in range(num_episodes):
            episodic_rewards = []
            while observation and not observation.end:
                self._logger.debug(f'Observation received:{observation}')
                # Store steps so far
                num_steps += 1
                # Get next_action. If we are not training, selection is different, so pass it
                action, state_id = self.select_action(observation, testing)
                # Perform the action and observe next observation
                observation = self.make_step(action)
                # Store the reward of the next observation
                episodic_rewards.append(observation.reward)
                if not testing:
                    # If we are training update the Q-table
                    self.q_values[state_id, action] += self.alpha * (observation.reward + self.gamma * self.max_action_q(observation)) - self.q_values[state_id, action]
                # Copy the last observation so we can return it and avoid the empty observation after the reset
                last_observation = observation
            # Sum all episodic returns 
            returns.append(np.sum(episodic_rewards))
            # Reset the episode
            observation = self.request_game_reset()
        #agent._logger.info(f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}")
        # This will be the last observation played before the reset
        return (last_observation, num_steps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('You can train the agent, or test it. \n Test is also to use the agent. \n During training and testing the performance is logged.')
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to run.", default=1000, type=int)
    parser.add_argument("--test_each", help="Evaluate the performance every this number of episodes. During training and testing.", default=100, type=int)
    parser.add_argument("--epsilon", help="Sets epsilon for exploration during training.", default=0.2, type=float)
    parser.add_argument("--gamma", help="Sets gamma discount for Q-learing during training.", default=0.9, type=float)
    parser.add_argument("--alpha", help="Sets alpha for learning rate during training.", default=0.1, type=float)
    parser.add_argument("--logdir", help="Folder to store logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--previous_model", help="Load the previous model. If training, it will start from here. If testing, will use to test.", default='./q_agent_marl.pickle', type=str)
    parser.add_argument("--testing", help="Test the agent. No train.", default=False)
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "q_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.INFO)

    # Create agent
    agent = QAgent(args.host, args.port, alpha=args.alpha, gamma=args.gamma, epsilon=args.epsilon)

    # If there is a previous model passed. Always use it for both training and testing.
    if args.previous_model:
        # Load table
        agent._logger.info(f'Loading the previous model in file {args.previous_model}')
        try:
            agent.load_q_table(args.previous_model)
        except:
            message = f'Problem loading the file: {args.previous_model}'
            agent._logger.info(message)
            print(message)


    if not args.testing:
        # Mlflow experiment name        
        experiment_name = "Training and testing of Q-learning Agent"
        mlflow.set_experiment(experiment_name)
    elif args.testing:
        # Evaluate the agent performance

        # Mlflow experiment name        
        experiment_name = "Testing of Q-learning Agent"
        mlflow.set_experiment(experiment_name)


    # This code runs for both training and testing. The difference is in the args.testing variable that is passed along
    # How it works:
    # - Evaluate for several 'episodes' (parameter)
    # - Each episode finishes with: steps played, return, win/lose. Store all
    # - Each episode compute the avg and std of all.
    # - Every X episodes (parameter), report in log and mlflow
    # - At the end, report in log and mlflow and console

    # Register the agent
    observation = agent.register()

    with mlflow.start_run(run_name=experiment_name) as run:
        # To keep statistics of each episode
        wins = 0
        detected = 0
        max_steps = 0
        num_win_steps = []
        num_detected_steps = []
        num_max_steps_steps = []
        num_detected_returns = []
        num_win_returns = []
        num_max_steps_returns = []

        # Log more things in Mlflow
        mlflow.set_tag("experiment_name", experiment_name)
        # Log notes or additional information
        mlflow.set_tag("notes", "This is an evaluation")
        mlflow.set_tag("Previous q-learning model loaded", str(args.previous_model))
        mlflow.log_param("alpha", args.alpha)
        mlflow.log_param("epsilon", args.epsilon)
        mlflow.log_param("gamma", args.gamma)
        mlflow.set_tag("Experiment ID", '')
        # Use subprocess.run to get the commit hash
        netsecenv_command = "git rev-parse HEAD"
        netsecenv_git_result = subprocess.run(netsecenv_command, shell=True, capture_output=True, text=True).stdout
        agents_command = "cd NetSecGameAgents; git rev-parse HEAD"
        agents_git_result = subprocess.run(agents_command, shell=True, capture_output=True, text=True).stdout
        agent._logger.info(f'Using commits. NetSecEnv: {netsecenv_git_result}. Agents: {agents_git_result}')
        mlflow.set_tag("NetSecEnv commit", netsecenv_git_result)
        mlflow.set_tag("Agents commit", agents_git_result)

        for episode in range(1, args.episodes + 1):
            #agent.logger.info(f'Starting the testing for episode {episode}')
            #print(f'Starting the testing for episode {episode}')

            # Play 1 episode
            observation, num_steps = agent.play_game(observation, 1, testing=args.testing)       

            state = observation.state
            reward = observation.reward
            end = observation.end
            info = observation.info

            if observation.info and observation.info['end_reason'] == 'detected':
                detected +=1
                num_detected_steps += [num_steps]
                num_detected_returns += [reward]
            elif observation.info and observation.info['end_reason'] == 'goal_reached':
                wins += 1
                num_win_steps += [num_steps]
                num_win_returns += [reward]
            elif observation.info and observation.info['end_reason'] == 'max_steps':
                max_steps += 1
                num_max_steps_steps += [num_steps]
                num_max_steps_returns += [reward]

            agent._logger.info(f"This episode: Steps={num_steps}. Reward {reward}. States in Q_table = {len(agent.q_values)}")

            # Reset the game
            observation = agent.request_game_reset()

            eval_win_rate = (wins/episode) * 100
            eval_detection_rate = (detected/episode) * 100
            eval_average_returns = np.mean(num_detected_returns+num_win_returns+num_max_steps_returns)
            eval_std_returns = np.std(num_detected_returns+num_win_returns+num_max_steps_returns)
            eval_average_episode_steps = np.mean(num_win_steps+num_detected_steps+num_max_steps_steps)
            eval_std_episode_steps = np.std(num_win_steps+num_detected_steps+num_max_steps_steps)
            eval_average_win_steps = np.mean(num_win_steps)
            eval_std_win_steps = np.std(num_win_steps)
            eval_average_detected_steps = np.mean(num_detected_steps)
            eval_std_detected_steps = np.std(num_detected_steps)
            eval_average_max_steps_steps = np.mean(num_max_steps_steps)
            eval_std_max_steps_steps = np.std(num_max_steps_steps)

            # Log and report every X episodes
            if episode % args.test_each == 0 and episode != 0:
                text = f'''Tested after {episode} episodes.
                    Wins={wins},
                    Detections={detected},
                    winrate={eval_win_rate:.3f}%,
                    detection_rate={eval_detection_rate:.3f}%,
                    average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
                    average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
                    average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
                    average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
                    average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
                    '''
                agent.logger.info(text)
                # Store in mlflow
                mlflow.log_metric("eval_avg_win_rate", eval_win_rate, step=episode)
                mlflow.log_metric("eval_avg_detection_rate", eval_detection_rate, step=episode)
                mlflow.log_metric("eval_avg_returns", eval_average_returns, step=episode)
                mlflow.log_metric("eval_std_returns", eval_std_returns, step=episode)
                mlflow.log_metric("eval_avg_episode_steps", eval_average_episode_steps, step=episode)
                mlflow.log_metric("eval_std_episode_steps", eval_std_episode_steps, step=episode)
                mlflow.log_metric("eval_avg_win_steps", eval_average_win_steps, step=episode)
                mlflow.log_metric("eval_std_win_steps", eval_std_win_steps, step=episode)
                mlflow.log_metric("eval_avg_detected_steps", eval_average_detected_steps, step=episode)
                mlflow.log_metric("eval_std_detected_steps", eval_std_detected_steps, step=episode)
                mlflow.log_metric("eval_avg_max_steps_steps", eval_average_max_steps_steps, step=episode)
                mlflow.log_metric("eval_std_max_steps_steps", eval_std_max_steps_steps, step=episode)

        
        # Log the last final episode when it ends
        text = f'''Episode {episode}. Final eval after {episode} episodes, for {args.episodes} steps.
            Wins={wins},
            Detections={detected},
            winrate={eval_win_rate:.3f}%,
            detection_rate={eval_detection_rate:.3f}%,
            average_returns={eval_average_returns:.3f} +- {eval_std_returns:.3f},
            average_episode_steps={eval_average_episode_steps:.3f} +- {eval_std_episode_steps:.3f},
            average_win_steps={eval_average_win_steps:.3f} +- {eval_std_win_steps:.3f},
            average_detected_steps={eval_average_detected_steps:.3f} +- {eval_std_detected_steps:.3f}
            average_max_steps_steps={eval_std_max_steps_steps:.3f} +- {eval_std_max_steps_steps:.3f},
            '''

        agent.logger.info(text)
        print(text)
        agent._logger.info("Terminating interaction")
        agent.terminate_connection()

        # Store the q-table
        agent.store_q_table(args.previous_model)