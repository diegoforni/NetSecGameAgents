"""
@file llm_action_planner.py

@brief Implementation of an LLM-based action planner for reactive agent systems.

This script defines classes and methods to facilitate interaction with language models (LLMs),
manage configuration files, and parse responses from LLM queries. The core functionality includes
planning actions based on observations and memory, parsing LLM responses, and dynamically loading
configuration files using YAML. Most of the code is adapted from the original implementation in 
the `assistan.py` from the `interactive_tui` agent.

@author Maria Rigaki - maria.rigaki@aic.fel.cvut.cz
@author Harpo Maxx - harpomaxx@gmail.com

@date [Date]
"""

import sys
from os import path
import yaml
import logging
import json
from dotenv import dotenv_values
from openai import OpenAI
from typing import Tuple
from tenacity import retry, stop_after_attempt
import jinja2

import re
from collections import Counter
from . import validate_responses

# Add parent directories dynamically
sys.path.append(
    path.dirname(path.dirname(path.dirname(path.dirname(path.dirname(path.abspath(__file__))))))
)
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

from AIDojoCoordinator.game_components import ActionType, Observation
from NetSecGameAgents.agents.llm_utils import create_action_from_response, create_status_from_state


class ConfigLoader:
    """Class to handle loading YAML configurations."""
    
    @staticmethod
    def load_config(file_name: str = 'prompts.yaml') -> dict:
        possible_paths = [
            path.join(path.dirname(__file__), file_name),
            path.join(path.dirname(path.dirname(__file__)), file_name),
            path.join(path.dirname(path.dirname(path.dirname(__file__))), file_name),
        ]
        for yaml_file in possible_paths:
            if path.exists(yaml_file):
                with open(yaml_file, 'r') as file:
                    return yaml.safe_load(file)
        raise FileNotFoundError(f"{file_name} not found in expected directories.")


ACTION_MAPPER = {
    "ScanNetwork": ActionType.ScanNetwork,
    "ScanServices": ActionType.FindServices,
    "FindData": ActionType.FindData,
    "ExfiltrateData": ActionType.ExfiltrateData,
    "ExploitService": ActionType.ExploitService,
}


class LLMActionPlanner:
    def __init__(self, model_name: str, goal: str, memory_len: int = 10, api_url=None, config: dict = None, use_reasoning: bool = False, use_reflection: bool = False, use_self_consistency: bool = False):
        self.model = model_name
        self.config = config or ConfigLoader.load_config()
        self.use_reasoning = use_reasoning
        self.use_reflection = use_reflection
        self.use_self_consistency = use_self_consistency

        if "gpt" in self.model:
            env_config = dotenv_values(".env")
            api_key = env_config.get("OPENAI_API_KEY")
            self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        else:
            self.client = OpenAI(base_url=api_url, api_key="ollama")

        self.memory_len = memory_len
        self.logger = logging.getLogger("REACT-agent")
        self.update_instructions(goal.lower())
        self.prompts = []
        self.states = []
        self.responses = []

    def get_prompts(self) -> list:
        """
        Returns the list of prompts sent to the LLM."""
        return self.prompts

    def get_responses(self) -> list:
        """
        Returns the list of responses received from the LLM. Only Stage 2 responses are included.
        """
        return self.responses
    
    def get_states(self) -> list:
        """
        Returns the list of states received from the LLM. In JSON format.
        """
        return self.states
    
    def update_instructions(self, new_goal: str) -> None:
        template = jinja2.Environment().from_string(self.config['prompts']['INSTRUCTIONS_TEMPLATE'])
        self.instructions = template.render(goal=new_goal)

    def create_mem_prompt(self, memory_list: list) -> str:
        prompt = ""
        for memory, goodness in memory_list:
            prompt += f"You have taken action {memory} in the past. This action was {goodness}.\n"
        return prompt

    @retry(stop=stop_after_attempt(3))
    def openai_query(self, msg_list: list, max_tokens: int = 60, model: str = None, fmt=None, temperature: float = 0.0) -> Tuple[str, int]:
        """Calls the chat completion API and returns (content, tokens_used).

        tokens_used is 0 when usage information is unavailable (e.g., some local backends).
        """
        llm_response = self.client.chat.completions.create(
            model=model or self.model,
            messages=msg_list,
            max_tokens=max_tokens,
            temperature=temperature,
            response_format=fmt or {"type": "text"},
        )
        content = llm_response.choices[0].message.content
        try:
            usage = getattr(llm_response, "usage", None)
            tokens_used = getattr(usage, "total_tokens", 0) if usage is not None else 0
        except Exception:
            tokens_used = 0
        return content, tokens_used

    def parse_response_deprecated(self, llm_response: str, state: Observation.state):
        try:
            response = json.loads(llm_response)
        except json.JSONDecodeError:
            self.logger.error("Failed to parse LLM response as JSON.")
            return False,llm_response, None

        try:
            action_str = response["action"]
            action_params = response["parameters"]
            valid, action = create_action_from_response(response, state)
            #return valid,f"You can take action {action_str} with parameters {action_params}", action
            return valid, {action_str:action_str,action_params:action_params}, action
       
        except KeyError:
            return False, llm_response, None

    def parse_response(self, llm_response: str, state: Observation.state):
        response_dict = {"action": None, "parameters": None}
        valid = False
        action = None

        try:
            response = json.loads(llm_response)
            action_str = response.get("action", None)
            action_params = response.get("parameters", None)
            
            if action_str and action_params:
                valid, action = create_action_from_response(response, state)
                response_dict["action"] = action_str
                response_dict["parameters"] = action_params
            else:
                self.logger.warning("Missing action or parameters in LLM response.")
        except json.JSONDecodeError:
            self.logger.error("Failed to parse LLM response as JSON.")
            response_dict["action"] = "InvalidJSON"
            response_dict["parameters"] = llm_response  # Return raw response for debugging
        except KeyError:
            self.logger.error("Missing keys in LLM response.")
        
        # Extra guard: if we couldn't construct an Action object, treat as invalid
        if valid and action is None:
            valid = False
        return valid, response_dict, action

    def remove_reasoning(self, text):
        match = re.search(r'</think>(.*)', text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return text

    def check_repetition(self, memory_list):
        repetitions = 0
        past_memories = []
        for memory, goodness in memory_list:
            if memory in past_memories:
                repetitions += 1
            past_memories.append(memory)
        return repetitions

    def get_self_consistent_response(self, messages, temp=0.4, max_tokens=1024, n=3):
        """Returns (best_content, tokens_used_sum) using simple majority among n samples."""
        candidates = []
        tokens_acc = 0
        for _ in range(n):
            response, used = self.openai_query(messages, temperature=temp, max_tokens=max_tokens)
            tokens_acc += used
            candidates.append(response.strip())

        counts = Counter(candidates)
        most_common = counts.most_common(1)
        if most_common:
            self.logger.info(f"Self-consistency candidates: {counts}")
            return most_common[0][0], tokens_acc
        return candidates[0], tokens_acc

    def get_action_from_obs_react(self, observation: Observation, memory_buf: list) -> tuple:
        self.states.append(observation.state.as_json())
        status_prompt = create_status_from_state(observation.state)
        q1 = self.config['questions'][0]['text']
        q4 = self.config['questions'][3]['text']
        cot_prompt = self.config['prompts']['COT_PROMPT']
        memory_prompt = self.create_mem_prompt(memory_buf)

        repetitions = self.check_repetition(memory_buf)
        tokens_used_total = 0
        messages = [
            {"role": "user", "content": self.instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": memory_prompt},
            {"role": "user", "content": q1},
        ]
        self.logger.info(f"Text sent to the LLM: {messages}")

        if self.use_self_consistency:
            response, used = self.get_self_consistent_response(messages, temp=repetitions/9, max_tokens=1024)
            tokens_used_total += used
        else:
            response, used = self.openai_query(messages, max_tokens=1024)
            tokens_used_total += used

        if self.use_reflection:
            reflection_prompt = [
                {
                    "role": "user",
                    "content": f"""
                    Instructions: {self.instructions}
                    Task: {q1}

                    Status: {status_prompt}
                    Memory: {memory_prompt}

                    Reasoning:
                    {response}

                    Is this reasoning valid given the Instructions, Status, and Memory?
                    - If YES, repeat it exactly.
                    - If NO, output the corrected reasoning only (no commentary).
                    """
                }
            ]
            response, used = self.openai_query(reflection_prompt, max_tokens=1024)
            tokens_used_total += used
        #print("response after reflection: ",response)

        # Optional: parse response if reasoning is expected and outputs <think> ... </think>
        if self.use_reasoning:
            response = self.remove_reasoning(response)
        self.logger.info(f"(Stage 1) Response from LLM: {response}")

        #memory_prompt = self.create_mem_prompt(memory_buf)

        messages = [
            {"role": "user", "content": self.instructions},
            {"role": "user", "content": status_prompt},
            {"role": "user", "content": cot_prompt},
            {"role": "user", "content": response},
            {"role": "user", "content": memory_prompt},
            {"role": "user", "content": q4},
        ]
        self.prompts.append(messages)
        response, used = self.openai_query(messages, max_tokens=80, fmt={"type": "json_object"})
        tokens_used_total += used

        # Build minimal context for validator
        # - known_data_map: known data by source host
        # - known_services_map: discovered service names by host
        ctx = {"known_data_map": {}, "known_services_map": {}}
        try:
            for ip_data, data_set in observation.state.known_data.items():
                ctx["known_data_map"][str(ip_data)] = [
                    {"owner": getattr(d, "owner", None), "id": getattr(d, "id", None)}
                    for d in list(data_set)
                ]
            for ip_srv, srv_set in observation.state.known_services.items():
                ctx["known_services_map"][str(ip_srv)] = [getattr(s, "name", None) for s in list(srv_set)]
        except Exception:
            pass

        validated, error_msg = validate_responses.validate_agent_response(response, context=ctx)
        if validated is None:
            self.logger.info(f"Invalid response format: {response} - Error: {error_msg}")
            try:
                parsed_response = json.loads(response)
            except json.JSONDecodeError:
                parsed_response = response
            response = json.dumps(
                {
                    "action": "InvalidResponse",
                    "parameters": {
                        "error": error_msg,
                        "original": parsed_response,
                    },
                },
                indent=2,
            )
        else:
            # Use normalized/validated structure going forward
            response = json.dumps(validated)

        if self.use_reasoning:
            response = self.remove_reasoning(response)

        self.responses.append(response)
        self.logger.info(f"(Stage 2) Response from LLM: {response}")
        print(f"(Stage 2) Response from LLM: {response}")
        valid, response_dict, action = self.parse_response(response, observation.state)
        return valid, response_dict, action, tokens_used_total
