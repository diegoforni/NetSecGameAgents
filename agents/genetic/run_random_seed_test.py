import subprocess
import random
import yaml

# Number of times to run genetic_agent
num_runs = 20

# Path to the configuration file
path_config = "/Users/diegoforni/Documents/labsin/NetSecGame/env/netsecenv_conf.yaml"

# Function to run genetic_agent
def run_genetic_agent():
    # Run genetic_agent
    subprocess.run(['python3', 'genetic_agent.py'])

# Function to update the seed in the YAML config file
def update_seed_in_config(new_seed, config_path):
    # Load the YAML file
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Update the random_seed value
    config['env']['random_seed'] = new_seed

    # Overwrite the original YAML file with the updated seed
    with open(config_path, 'w') as file:
        yaml.dump(config, file)

# Run genetic_agent num_runs times
for i in range(num_runs):
    # Generate a new 5-digit random seed
    new_seed = random.randint(10000, 99999)

    # Update the seed in the YAML config file
    update_seed_in_config(new_seed, path_config)
    
    print(f'Run {i+1}/{num_runs} with seed {new_seed}')
    
    # Run the genetic agent with the new seed
    run_genetic_agent()
