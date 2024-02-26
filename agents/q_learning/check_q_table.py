import argparse
import pickle

q_values = {}
states = {}

def load_q_table(filename):
    global q_values
    global states
    print(f'Loading file {filename}')
    with open(filename, "rb") as f:
        data = pickle.load(f)
        q_values = data["q_table"]
        states = data["state_mapping"]

def show_q_table():
    """
    Show details about a state in the qtable
    """
    print(f'State: {list(states.items())[args.state_id]}')
    for ((id, action), value) in list(q_values.items()):
        if int(id)==args.state_id:
            print(f'{id}: {action} -> {value}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser('You can train the agent, or test it. \n Test is also to use the agent. \n During training and testing the performance is logged.')
    parser.add_argument("--file", help="Q-table file to load", default="q_agent_marl.pickle", required=False, type=str)
    parser.add_argument("--state_id", help="ID of the state to print", default=0, required=False, type=int)
    args = parser.parse_args()

    load_q_table(args.file)
    show_q_table()
