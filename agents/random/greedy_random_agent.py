#Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# This agents just randomnly picks actions. No learning
import sys
import logging
import os
from random import choice
import argparse
import numpy as np
import mlflow

# This is used so the agent can see the environment and game components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))
# with the path fixed, we can import now
from env.game_components import Action, Observation, GameState, ActionType
from base_agent import BaseAgent
from datetime import datetime

from collections import defaultdict

class RandomAgent(BaseAgent):

    def __init__(self, host, port,role, seed) -> None:
        super().__init__(host, port, role)

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
        return list(valid_scan_network), list(valid_find_services), list(valid_exploit_service), list(valid_find_data), list(valid_exfiltrate_data)

    def select_action_greedy(self, observation: Observation, taken_actions) -> Action:
        
        valid_actions = self.generate_valid_actions_separated(observation.state)

        i = len(valid_actions) - 1
        action = None

        while action is None:
            if valid_actions[i] != []:
                action = choice(valid_actions[i])
                if action in taken_actions:
                    valid_actions[i].remove(action)
                    action = None
            else:
                i -= 1
        return action

    def play_game(self, observation, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        returns = []
        num_steps = 0
        for episode in range(num_episodes):
            taken_actions = {}
            self._logger.info(f"Playing episode {episode}")
            episodic_returns = []
            while observation and not observation.end:
                num_steps += 1
                self._logger.debug(f'Observation received:{observation}')
                # Store returns in the episode
                episodic_returns.append(observation.reward)
                # Select the action randomly
                action = self.select_action_greedy(observation, taken_actions)
                taken_actions[action] = 1
                
                observation = self.make_step(action)
                # To return
                last_observation = observation
            self._logger.debug(f'Observation received:{observation}')
            returns.append(np.sum(episodic_returns))
            self._logger.info(f"Episode {episode} ended with return{np.sum(episodic_returns)}. Mean returns={np.mean(returns)}±{np.std(returns)}")
            # Reset the episode
            observation = self.request_game_reset()
        self._logger.info(f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}")
        # This will be the last observation played before the reset
        return (last_observation, num_steps)
    
   

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of episodes to play or evaluate", default=100, type=int) 
    parser.add_argument("--test_each", help="Evaluate performance during testing every this number of episodes.", default=10, type=int)
    parser.add_argument("--logdir", help="Folder to store logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--evaluate", help="Evaluate the agent and report, instead of playing the game only once.", default=True)
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "random_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.INFO)

    # Create agent
    agent = RandomAgent(args.host, args.port,"Attacker", seed=42)

    if not args.evaluate:
        # Play the normal game
        observation = agent.register()
        agent.play_game(observation, args.episodes)
        agent._logger.info("Terminating interaction")
        agent.terminate_connection()
    else:
        # Evaluate the agent performance

        # How it works:
        # - Evaluate for several 'episodes' (parameter)
        # - Each episode finishes with: steps played, return, win/lose. Store all
        # - Each episode compute the avg and std of all.
        # - Every X episodes (parameter), report in log and mlflow
        # - At the end, report in log and mlflow and console

        # Mlflow experiment name        
        experiment_name = "Evaluation of Random Agent"
        mlflow.set_experiment(experiment_name)
        # Register in the game
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
            #mlflow.log_param("learning_rate", learning_rate)

            for episode in range(1, 100):
                agent.logger.info(f'Starting the testing for episode {episode}')
                print(f'Starting the testing for episode {episode}')

                # Play the game for one episode
                observation, num_steps = agent.play_game(observation, 1)

                state = observation.state
                reward = observation.reward
                end = observation.end
                info = observation.info

                print(observation)

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