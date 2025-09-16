"""
This module implements an agent that is using LLM as a planning agent 
Authors:  Maria Rigaki - maria.rigaki@aic.fel.cvut.cz
          Harpo MAxx - harpomaxx@gmail.com
"""
import logging
import argparse
import numpy as np
import pandas as pd
import mlflow
import os
import sys
import json
from llm_action_planner import LLMActionPlanner
from os import path
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from dotenv import load_dotenv, find_dotenv
import urllib.request
import urllib.error
import signal

sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
)

from AIDojoCoordinator.game_components import AgentStatus, Action, ActionType
from NetSecGameAgents.agents.base_agent import BaseAgent

#mlflow.set_tracking_uri("http://147.32.83.60")
#mlflow.set_experiment("LLM_QA_netsecgame_dec2024")


if __name__ == "__main__":
    # Load environment defaults (supports both local .env and inherited env)
    try:
        _THIS_DIR = path.dirname(path.abspath(__file__))
        # Load local .env in agent folder (non‑overriding)
        load_dotenv(path.join(_THIS_DIR, ".env"), override=False)
        # Also search upwards for a .env if present
        load_dotenv(find_dotenv(), override=False)
    except Exception:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--llm",
        type=str,
        # choices=[
        #     "gpt-4",
        #     "gpt-4-turbo-preview",
        #     "gpt-3.5-turbo",
        #     "gpt-3.5-turbo-16k",
        #     "HuggingFaceH4/zephyr-7b-beta",
        # ],
        default="gpt-3.5-turbo",
        help="LLM used with OpenAI API",
    )
    parser.add_argument(
        "--test_episodes",
        help="Number of test episodes to run",
        default=30,
        action="store",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--memory_buffer",
        help="Number of actions to remember and pass to the LLM",
        default=5,
        action="store",
        required=False,
        type=int,
    )
    parser.add_argument(
        "--host",
        help="Host where the game server is",
        default="127.0.0.1",
        action="store",
        required=False,
    )
    parser.add_argument(
        "--port",
        help="Port where the game server is",
        default=9000,
        type=int,
        action="store",
        required=False,
    )
    
    parser.add_argument(
        "--api_url",
        type=str, 
        default="http://127.0.0.1:11434/v1/"
        )

    parser.add_argument(
        "--use_reasoning",
        action="store_true",
        help="Required for models that output reasoning using <think>...</think>."
    )

    parser.add_argument(
        "--use_reflection",
        action="store_true",
        help="To use reflection prompting technique in the LLM calls."
    )

    parser.add_argument(
        "--use_self_consistency",
        action="store_true",
        help="To use self-consistency prompting technique in the LLM calls."
    )

    parser.add_argument(
        "--max_tokens_limit",
        type=int,
        default=0,
        help="If cumulative tokens across all episodes exceed this limit, terminate the run. 0 disables the limit.",
    )

    parser.add_argument(
        "--mlflow_tracking_uri",
        type=str,
        default=os.getenv("MLFLOW_TRACKING_URI", "databricks"),
        help="MLflow tracking server URI (default reads MLFLOW_TRACKING_URI or 'databricks')",
    )

    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default=os.getenv("MLFLOW_EXPERIMENT", "/Shared/NetSecGame/UTEP_S1/Train1"),
        help="MLflow experiment name or path (default reads MLFLOW_EXPERIMENT)",
    )

    parser.add_argument(
        "--mlflow_description",
        type=str,
        default=os.getenv("MLFLOW_DESCRIPTION", None),
        help="Optional description for MLflow run (default reads MLFLOW_DESCRIPTION or is generated)",
    )

    parser.add_argument(
        "--disable_mlflow",
        action="store_true",
        help="Disable mlflow logging",
    )

    
    args = parser.parse_args()

    logging.basicConfig(
        filename="llm_react.log",
        filemode="w",
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO,
    )

    logger = logging.getLogger("llm_react")
    logger.info("Start")
    agent = BaseAgent(args.host, args.port, "Attacker")
    
    def _ensure_databricks_workspace_dir(dir_path: str) -> bool:
        """Ensures a Databricks workspace directory exists using the Workspace API.
        Returns True if the directory exists or was created, False otherwise.
        """
        host = os.getenv("DATABRICKS_HOST")
        token = os.getenv("DATABRICKS_TOKEN")
        if not host or not token or not dir_path:
            return False
        try:
            url = f"{host.rstrip('/')}/api/2.0/workspace/mkdirs"
            payload = json.dumps({"path": dir_path}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                # 200/201 indicate success; 404 means base (/Shared) missing (unlikely)
                return 200 <= resp.status < 300
        except urllib.error.HTTPError as he:  # Already exists or other conditions
            # If the directory already exists, Databricks may still respond 200.
            # For other HTTP errors, we just return False to allow fallback logic.
            logging.getLogger("llm_react").warning(
                f"Databricks mkdirs for '{dir_path}' returned HTTP {he.code}: {he.reason}"
            )
            return False
        except Exception as e:
            logging.getLogger("llm_react").warning(
                f"Failed ensuring Databricks dir '{dir_path}': {e}"
            )
            return False

    if not args.disable_mlflow:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)

        experiment_path = args.mlflow_experiment

        # If using Databricks with a workspace path, ensure parent directory exists
        if (args.mlflow_tracking_uri == "databricks") and experiment_path.startswith("/"):
            parent_dir = os.path.dirname(experiment_path.rstrip("/"))
            if parent_dir and parent_dir != "/":
                _ensure_databricks_workspace_dir(parent_dir)

        try:
            mlflow.set_experiment(experiment_path)
        except Exception as e:
            msg = str(e)
            logger.warning(f"mlflow.set_experiment failed for '{experiment_path}': {msg}")
            # Fallback: if creating nested path under /Shared fails, flatten under /Shared
            if experiment_path.startswith("/Shared/") and "Parent directory" in msg:
                flattened = experiment_path[len("/Shared/"):].replace("/", "_")
                fallback_path = f"/Shared/{flattened}"
                logger.warning(
                    f"Falling back to Databricks experiment path '{fallback_path}'"
                )
                mlflow.set_experiment(fallback_path)
                experiment_path = fallback_path
            else:
                raise

        # Use custom description if given, otherwise build a default
        experiment_description = args.mlflow_description or (
            f"{experiment_path} | Model: {args.llm}"
        )

        mlflow.start_run(description=experiment_description)

        params = {
            "model": args.llm,
            "memory_len": args.memory_buffer,
            "episodes": args.test_episodes,
            "host": args.host,
            "port": args.port,
            "api_url": args.api_url,
            "max_tokens_limit": args.max_tokens_limit,
        }
        mlflow.log_params(params)
        mlflow.set_tag("agent_role", "Attacker")

    # Run multiple episodes to compute statistics
    wins = 0
    detected = 0
    reach_max_steps = 0
    returns = []
    num_steps = []
    num_win_steps = []
    num_detected_steps = []
    num_actions_repeated = []
    reward_memory = ""
    total_tokens_used = 0

 
    # Create an empty DataFrame for storing prompts and responses, and evaluations
    #prompt_table = pd.DataFrame(columns=["state", "prompt", "response", "evaluation"])
    prompt_table = []

    def _register_interrupt_saver(get_data_fn, filename: str = "episode_data.json") -> None:
        """Register Ctrl-C handler to persist current episodes JSON before exiting."""
        def _handler(signum, frame):
            try:
                data = get_data_fn()
                with open(filename, "w") as json_file:
                    json.dump(data, json_file, indent=4)
                print(f"\nInterrupted (Ctrl-C). Saved {len(data)} episodes to {filename}.")
            except Exception as e:
                print(f"\nInterrupted (Ctrl-C). Failed to save data: {e}")
            finally:
                try:
                    if not args.disable_mlflow:
                        mlflow.set_tag("termination_reason", "keyboard_interrupt")
                        mlflow.log_metric("total_tokens_at_termination", total_tokens_used)
                        mlflow.end_run("KILLED")
                except Exception:
                    pass
                os._exit(130)
        signal.signal(signal.SIGINT, _handler)
        try:
            signal.signal(signal.SIGTERM, _handler)
        except Exception:
            pass

    _register_interrupt_saver(lambda: prompt_table)
    
    # We are still not using this, but we keep track
    is_detected = False

    # Initialize the game
    print("Registering")
    agent.register()
    print("Done")
    # -------------------------------------------------------------
    # Optional: On-demand topology randomization (new IPs once)
    # To switch tasks to a new set of IPs, uncomment ONE or more of
    # the following lines. Each uncommented line will request a reset
    # with randomize_topology=True exactly once before episodes start.
    # Requires the coordinator config to have `use_dynamic_addresses: True`.
    agent.make_step(Action(ActionType.ResetGame, parameters={"randomize_topology": True}))  # 1) Randomize once before run
    agent.make_step(Action(ActionType.ResetGame, parameters={"randomize_topology": True}))  # 2) Randomize once before run
    agent.make_step(Action(ActionType.ResetGame, parameters={"randomize_topology": True}))  # 3) Randomize once before run
    #agent.make_step(Action(ActionType.ResetGame, parameters={"randomize_topology": True}))  # 4) Randomize once before run
    # agent.make_step(Action(ActionType.ResetGame, parameters={"randomize_topology": True}))  # 5) Randomize once before run
    # -------------------------------------------------------------
    for episode in range(1, args.test_episodes + 1):
        actions_took_in_episode = []
        evaluations = [] # used for prompt table storage.
        logger.info(f"Running episode {episode}")
        print(f"Running episode {episode}")

        # Reset the game at every episode and store the goal that changes
        observation = agent.request_game_reset()
        num_iterations = observation.info["max_steps"]
        current_state = observation.state

        
        taken_action = None
        memories = []
        total_reward = 0
        num_actions = 0
        repeated_actions = 0

        if args.llm is not None:
            llm_query = LLMActionPlanner(
            model_name=args.llm,
            goal=observation.info["goal_description"],
            memory_len=args.memory_buffer,
            api_url=args.api_url,
            use_reasoning=args.use_reasoning,
            use_reflection=args.use_reflection,
            use_self_consistency=args.use_self_consistency
        )
        print(observation)
        for i in range(num_iterations):
            good_action = False
            #is_json_ok = True
            is_valid, response_dict, action, tokens_used = llm_query.get_action_from_obs_react(observation, memories)

            # Update and enforce token limit if configured
            try:
                total_tokens_used += int(tokens_used or 0)
            except Exception:
                pass
            if args.max_tokens_limit > 0 and total_tokens_used > args.max_tokens_limit:
                termination_message = (
                    f"CRITICAL: Total token limit of {args.max_tokens_limit} reached "
                    f"(cumulative total: {total_tokens_used}). Terminating run."
                )
                print(termination_message)
                logger.critical(termination_message)

                # Persist data collected so far
                try:
                    with open("episode_data.json", "w") as json_file:
                        json.dump(prompt_table, json_file, indent=4)
                except Exception:
                    pass

                if not args.disable_mlflow:
                    try:
                        mlflow.set_tag("termination_reason", "max_tokens_limit_exceeded")
                        mlflow.log_param("termination_episode", episode)
                        mlflow.log_metric("total_tokens_at_termination", total_tokens_used)
                        mlflow.end_run("FAILED")
                    except Exception:
                        pass

                sys.exit(1)
            if is_valid and action is not None:
                observation = agent.make_step(action)
                logger.info(f"Observation received: {observation}")
                taken_action = action
                total_reward += observation.reward

                if observation.state != current_state:
                    good_action = True
                    current_state = observation.state
                    evaluations.append(8)
                else:
                    evaluations.append(3)
            else:
                print("Invalid action: ")
                evaluations.append(0)

            try:
                if not is_valid:
                    memories.append(
                        (
                            (response_dict["action"],
                            response_dict["parameters"]),
                            "not valid based on your status."
                        )
                    )
                    print("not valid based on your status.")

                else:
                    # This is based on the assumption that more valid actions in the state are better/more helpful.
                    # But we could a manual evaluation based on the prior knowledge and weight the different components.
                    # For example: finding new data is better than discovering hosts (?)
                    if good_action:
                        memories.append(
                            (
                                (response_dict["action"],
                                response_dict["parameters"]),
                                "helpful."
                            )
                        )
                        print("Helpful")
                    else:
                        memories.append(
                            (
                                (response_dict["action"],
                                response_dict["parameters"]),
                                "not helpful."
                            )
                        )
                        print("Not Helpful")
                    # If the action was repeated count it
                    if action in actions_took_in_episode:
                        repeated_actions += 1

                    # Store action in memory of all actions so far
                    actions_took_in_episode.append(action)
            except:
                # if the LLM sends a response that is not properly formatted.
                memories.append(
                                (response_dict["action"],
                                response_dict["parameters"]),
                                "badly formated."
                )
                print("badly formated")
            if len(memories) > args.memory_buffer:
                # If the memory is full, remove the oldest memory
                memories.pop(0)
            # logger.info(f"Iteration: {i} JSON: {is_json_ok} Valid: {is_valid} Good: {good_action}")
            logger.info(f"Iteration: {i} Valid: {is_valid} Good: {good_action}")
            
            if observation.end or i == (
                num_iterations - 1
            ):  # if it is the last iteration gather statistics
                if i < (num_iterations - 1):
                    # TODO: Fix this
                    reason = observation.info
                else:
                    reason = {"end_reason": AgentStatus.TimeoutReached }

                win = 0
                # is_detected if boolean
                # is_detected = observation.info.detected
                # TODO: Fix this
                steps = i
                epi_last_reward = observation.reward
                num_actions_repeated += [repeated_actions]
                if AgentStatus.Success == reason["end_reason"]:
                    wins += 1
                    num_win_steps += [steps]
                    type_of_end = "win"
                    evaluations[-1] = 10
                elif AgentStatus.Fail == reason["end_reason"]:
                    detected += 1
                    num_detected_steps += [steps]
                    type_of_end = "detection"
                elif AgentStatus.TimeoutReached == reason["end_reason"]:
                    # TODO: Fix this
                    reach_max_steps += 1
                    type_of_end = "max_iterations"
                    total_reward = -100
                    #steps = observation.info["max_steps"] #this fails
                    steps = num_iterations
                else:
                    reach_max_steps += 1
                    type_of_end = "max_steps"
                returns += [total_reward]
                num_steps += [steps]

                if not args.disable_mlflow:
                    # Episodic value
                    mlflow.log_metric("wins", wins, step=episode)
                    mlflow.log_metric("num_steps", steps, step=episode)
                    mlflow.log_metric("return", total_reward, step=episode)

                    # Running metrics
                    mlflow.log_metric("wins", wins, step=episode)
                    mlflow.log_metric("reached_max_steps", reach_max_steps, step=episode)
                    mlflow.log_metric("detected", detected, step=episode)
                    mlflow.log_metric("total_tokens_used", total_tokens_used, step=episode)

                    # Running averages
                    mlflow.log_metric("win_rate", (wins / (episode)) * 100, step=episode)
                    mlflow.log_metric("avg_returns", np.mean(returns), step=episode)
                    mlflow.log_metric("avg_steps", np.mean(num_steps), step=episode)

                logger.info(
                    f"\tEpisode {episode} of game ended after {steps} steps. Reason: {reason}. Last reward: {epi_last_reward}"
                )
                print(
                    f"\tEpisode {episode} of game ended after {steps} steps. Reason: {reason}. Last reward: {epi_last_reward}"
                )
                break

        episode_prompt_table = {
            "episode": episode,
            "state": llm_query.get_states(),
            "prompt": llm_query.get_prompts(),
            "response": llm_query.get_responses(),
            "evaluation": evaluations,
            "end_reason": str(reason["end_reason"])
        }
        prompt_table.append(episode_prompt_table)
        #episode_prompt_table = pd.DataFrame(episode_prompt_table)
        #prompt_table = pd.concat([prompt_table,episode_prompt_table],axis=0,ignore_index=True)
        
    #prompt_table.to_csv("states_prompts_responses_new.csv", index=False)
    # Save the JSON file
    with open("episode_data.json", "w") as json_file:
        json.dump(prompt_table, json_file, indent=4)

    # After all episodes are done. Compute statistics
    test_win_rate = (wins / (args.test_episodes)) * 100
    test_detection_rate = (detected / (args.test_episodes)) * 100
    test_max_steps_rate = (reach_max_steps / (args.test_episodes)) * 100
    test_average_returns = np.mean(returns)
    test_std_returns = np.std(returns)
    test_average_episode_steps = np.mean(num_steps)
    test_std_episode_steps = np.std(num_steps)
    test_average_win_steps = np.mean(num_win_steps)
    test_std_win_steps = np.std(num_win_steps)
    test_average_detected_steps = np.mean(num_detected_steps)
    test_std_detected_steps = np.std(num_detected_steps)
    test_average_repeated_steps = np.mean(num_actions_repeated)
    test_std_repeated_steps = np.std(num_actions_repeated)
    # Store in tensorboard
    tensorboard_dict = {
        "test_avg_win_rate": test_win_rate,
        "test_avg_detection_rate": test_detection_rate,
        "test_avg_max_steps_rate": test_max_steps_rate,
        "test_avg_returns": test_average_returns,
        "test_std_returns": test_std_returns,
        "test_avg_episode_steps": test_average_episode_steps,
        "test_std_episode_steps": test_std_episode_steps,
        "test_avg_win_steps": test_average_win_steps,
        "test_std_win_steps": test_std_win_steps,
        "test_avg_detected_steps": test_average_detected_steps,
        "test_std_detected_steps": test_std_detected_steps,
        "test_avg_repeated_steps": test_average_repeated_steps,
        "test_std_repeated_steps": test_std_repeated_steps,
        "final_total_tokens_used": total_tokens_used,
    }

    if not args.disable_mlflow:
        mlflow.log_metrics(tensorboard_dict)

    text = f"""Final test after {args.test_episodes} episodes
        Wins={wins},
        Detections={detected},
        winrate={test_win_rate:.3f}%,
        detection_rate={test_detection_rate:.3f}%,
        max_steps_rate={test_max_steps_rate:.3f}%,
        average_returns={test_average_returns:.3f} +- {test_std_returns:.3f},
        average_episode_steps={test_average_episode_steps:.3f} +- {test_std_episode_steps:.3f},
        average_win_steps={test_average_win_steps:.3f} +- {test_std_win_steps:.3f},
        average_detected_steps={test_average_detected_steps:.3f} +- {test_std_detected_steps:.3f}
        average_repeated_steps={test_average_repeated_steps:.3f} += {test_std_repeated_steps:.3f}"""

    print(text)
    logger.info(text)
    
    # Ensure resources are closed so the process can exit cleanly
    try:
        agent.terminate_connection()
    except Exception:
        pass

    if not args.disable_mlflow:
        try:
            mlflow.end_run("FINISHED")
        except Exception as e:
            logger.warning(f"Failed to end MLflow run cleanly: {e}")

    # Explicitly exit to avoid any lingering atexit handlers/threads from blocking
    sys.exit(0)
