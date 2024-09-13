
import subprocess

# Number of times to run genetic_agent
num_runs = 2

# use excec to run genetic_agent
def run_genetic_agent():
    # Run genetic_agent
    subprocess.run(['python3', 'genetic_agent.py'])

# Run genetic_agent num_runs times
for i in range(num_runs):
    print(f'Run {i+1}/{num_runs}')
    run_genetic_agent()
    