#Author: Ondrej Lukas, ondrej.lukas@aic.cvut.cz
# This agents just randomnly picks actions. No learning
import sys
import logging
import os
from random import choice
import argparse
from random import choice
import numpy as np
#from torch.utils.tensorboard import SummaryWriter

# This is used so the agent can see the environment and game components
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__) ) ) )))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__) )))
# with the path fixed, we can import now
from env.game_components import Action, Observation, ActionType
from base_agent import BaseAgent
from agent_utils import generate_valid_actions


class RandomBenignAgent(BaseAgent):

    def __init__(self, host:str, port:int,role:str, allowed_actions:list) -> None:
        super().__init__(host, port, role)
        self._allowed_actions = allowed_actions
    

    def play_game(self, num_episodes=1):
        """
        The main function for the gameplay. Handles agent registration and the main interaction loop.
        """
        
        observation = self.register()
        returns = []
        for episode in range(num_episodes):
            episodic_returns = []
            while observation and not observation.end:
                self._logger.debug(f'Observation received:{observation}')
                # select the action randomly
                action = self.select_action(observation)
                episodic_returns.append(observation.reward)
                observation = self.make_step(action)
            self._logger.debug(f'Observation received:{observation}')
            returns.append(np.sum(episodic_returns))
            self._logger.info(f"Episode {episode} ended with return{np.sum(episodic_returns)}. Mean returns={np.mean(returns)}±{np.std(returns)}")
            # Reset the episode
            observation = self.request_game_reset()
        self._logger.info(f"Final results for {self.__class__.__name__} after {num_episodes} episodes: {np.mean(returns)}±{np.std(returns)}")
        self._logger.info("Terminating interaction")
        self.terminate_connection()
    
    def select_action(self, observation:Observation)->Action:
        valid_actions = generate_valid_actions(observation.state)
        # filter actions based on the allowed action types
        allowed_actions = filter(lambda action: action.type in self._allowed_actions, valid_actions)
        action = choice(allowed_actions)
        return action

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", help="Host where the game server is", default="127.0.0.1", action='store', required=False)
    parser.add_argument("--port", help="Port where the game server is", default=9000, type=int, action='store', required=False)
    parser.add_argument("--episodes", help="Sets number of testing episodes", default=10, type=int)
    parser.add_argument("--logdir", help="Folder to store logs", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    args = parser.parse_args()

    if not os.path.exists(args.logdir):
        os.makedirs(args.logdir)
    logging.basicConfig(filename=os.path.join(args.logdir, "benign_random_agent.log"), filemode='w', format='%(asctime)s %(name)s %(levelname)s %(message)s', datefmt='%H:%M:%S',level=logging.INFO)

    # Create agent
    agent = RandomBenignAgent(args.host, args.port,"Human",allowed_actions=[ActionType.FindData, ActionType.ExfiltrateData, ActionType.FindServices])
    agent.play_game(args.episodes)
