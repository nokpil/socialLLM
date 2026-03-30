import os
import json
import time
import re
import sys
from collections import OrderedDict
from argparse import ArgumentParser


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from collections import Counter

import openai
from openai import OpenAI


def parse_output_json(response: str) -> str:
    """
    simple setting용 파서.
    LLM 응답 전체 텍스트에서 마지막으로 등장하는 "grid":"..."를 뽑아서
    grid 문자열을 반환한다.

    - "grid":"...여러줄..." 꼴도 허용 (DOTALL)
    - 중간에 {5}, {8,0} 같은 중괄호들이 있어도 전혀 신경 쓰지 않음
    - 만약 "grid"를 아예 못 찾으면, 전체 텍스트를 그대로 grid로 반환 (fallback)
    """
    if response is None:
        return ""

    text = str(response).strip()

    # -------------------------------
    # 1) 정상 패턴: "grid":"...여러줄..."
    #    여러 번 나올 수 있으니 "마지막" 매치를 사용
    # -------------------------------
    grid_matches = list(re.finditer(r'"grid"\s*:\s*"(.*?)"', text, flags=re.S))
    if grid_matches:
        grid = grid_matches[-1].group(1).strip()
        return grid.replace("\\n", "\n")

    # -------------------------------
    # 2) 불완전한 "grid" (따옴표가 안 닫히거나, 끝이 잘린 경우) 처리
    #    -> 마지막 "grid" 이후를 통으로 보고, 뒷부분 정리해서 사용
    # -------------------------------
    idx = text.rfind('"grid"')
    if idx != -1:
        sub = text[idx:]
        # '"grid": " ...' 이후를 통으로 잡는다.
        m = re.search(r'"grid"\s*:\s*"(.*)', sub, flags=re.S)
        if m:
            g = m.group(1)
            # 뒤에 ``` 코드펜스가 있으면 그 앞까지만 사용
            g = re.split(r'```', g, 1)[0]
            g = g.rstrip()
            # 맨 끝에 붙어 있을 수 있는 "나 }를 제거
            g = re.sub(r'["}]+$', "", g).rstrip()
            if g:
                return g.replace("\\n", "\n")

    # -------------------------------
    # 3) 여기까지 왔으면 "grid"를 못 찾은 것 → 전체 텍스트를 grid로
    # -------------------------------
    return text


# --------------------------
# response에서 rule, output 부분을 파싱
# --------------------------
def parse_rule_and_output_json(response: str):
    """
    LLM 응답 문자열에서 {"rule": "...", "grid": "..."} 형태를 최대한 robust하게 뽑아낸다.
    - JSON이 제대로 되어 있으면 json.loads로 처리
    - grid 안에 실제 개행이 있는 "가짜 JSON", 불완전 JSON, ```json{...``` 같은 것도
      정규식으로 rule / grid를 직접 뽑는다.
    - 실패하면 (rule="", grid=전체 텍스트)로 fallback
    """
    if response is None:
        return "", ""

    text = str(response).strip()
    rule = ""
    grid = text  # 최종 fallback: 전체 텍스트를 grid로 본다

    # -------------------------------
    # 1) 마지막 { ... } 덩어리를 JSON으로 한 번 시도
    # -------------------------------
    try:
        start = text.rfind("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and start < end:
            candidate = text[start:end+1].strip()
        else:
            candidate = text

        obj = json.loads(candidate)
        rule = str(obj.get("rule", "")).strip()
        grid = str(obj.get("grid", grid)).strip()
        return rule, grid.replace("\\n", "\n")  # 여기까지 왔으면 정상 JSON
    except Exception:
        # 정상 JSON이 아니면 정규식으로 수동 파싱
        pass

    # -------------------------------
    # 2) 정규식으로 "rule" 뽑기 (가장 마지막 것 사용)
    # -------------------------------
    rule_matches = list(re.finditer(r'"rule"\s*:\s*"(.*?)"', text, flags=re.S))
    if rule_matches:
        rule = rule_matches[-1].group(1).strip()
    else:
        rule = ""

    # -------------------------------
    # 3) 정규식으로 "grid" 뽑기 (가장 마지막 것 사용)
    #    3-1) 먼저 정상 패턴: "grid":"...여러줄..."
    # -------------------------------
    grid_matches = list(re.finditer(r'"grid"\s*:\s*"(.*?)"', text, flags=re.S))
    if grid_matches:
        grid = grid_matches[-1].group(1).strip()
        return rule, grid.replace("\\n", "\n")

    # -------------------------------
    # 3-2) 불완전한 "grid" (끝이 잘린 경우) 처리
    # -------------------------------
    idx = text.rfind('"grid"')
    if idx != -1:
        sub = text[idx:]
        m = re.search(r'"grid"\s*:\s*"(.*)', sub, flags=re.S)
        if m:
            g = m.group(1)
            # 뒤에 코드펜스가 있으면 그 앞까지만
            g = re.split(r'```', g, 1)[0]
            g = g.rstrip()
            # 맨 끝에 붙은 "나 }를 제거
            g = re.sub(r'["}]+$', "", g).rstrip()
            if g:
                grid = g.replace("\\n", "\n")

    return rule, grid


def extract_answer_array(response, label_shape=None):
        
    # if response contains other than '\n\, ' ', ',', '[', ']', and numerals (0~9), return False
    
    response = response.replace("\\n", "\n") # final cleanup
    
    if re.search(r'[^\d\s\n]', response):
        return None
    
    arr = response.strip().split('\n')
    arr = [list(map(int, sub_arr.split())) for sub_arr in arr]
    
    # Check if all sublists have the same length
    max_length = max(len(sublist) for sublist in arr)
    if all(len(sublist) == max_length for sublist in arr):
        array = np.array(arr)
    else:
        # Pad shorter sublists with -1
        padded_arr = [sublist + [-1] * (max_length - len(sublist)) for sublist in arr]
        array = np.array(padded_arr)
    
    if label_shape is not None and array.shape != label_shape:
        array = label_padding(array, label_shape)
    
    return array

def label_padding(array, label_shape):
    
    # cropping
    if array.shape[0] > label_shape[0]:
        array = array[:label_shape[0], :]
    if array.shape[1] > label_shape[1]:
        array = array[:, :label_shape[1]]
    # padding
    if array.shape[0] < label_shape[0]:
        array = np.pad(array, ((0, label_shape[0] - array.shape[0]), (0, 0)), mode='constant', constant_values=-1)
    if array.shape[1] < label_shape[1]:
        array = np.pad(array, ((0, 0), (0, label_shape[1] - array.shape[1])), mode='constant', constant_values=-1)

    return array

class LLMCaller:
    def __init__(self, model):
        self.model = model
        if 'gpt' in model:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif 'deepseek' in model:
            self.client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com/")
        elif 'llama' in model:
            self.client = OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1")
        else:
            raise ValueError("Unsupported model type")

    def call_llm(self, messages, temperature: float=1.0):

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            reasoning_effort='low',
            temperature=temperature,
        )

        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens

class LLMAgent:
    def __init__(self, name, model='gpt-4o-mini', system_prompt="", extract_answer=lambda x: x, temperature=1.0):
        self.name = name
        self.temperature = temperature
        self.llm = LLMCaller(model)
        self.answer = None
        self.system_prompt = system_prompt
        if self.system_prompt == "":
            self.chat_history = []
        else:
            self.chat_history = [self.construct_system_message(self.system_prompt)]
        self.extract_answer = extract_answer
        
    def generate_answer(self, messages):
        response, prompt_tokens, completion_tokens = self.llm.call_llm(messages, self.temperature)
        self.answer = self.extract_answer(response)
        self.chat_history.append(self.construct_assistant_message(response))
        return response, self.answer, prompt_tokens, completion_tokens
    
    def construct_system_message(self, content):
        return {"role": "system", "content": content}
    
    def construct_user_message(self, content):
        return {"role": "user", "content": content}
    
    def construct_assistant_message(self, content):
        return {"role": "assistant", "content": content}
    
    def clear_chat_history(self):
        self.chat_history = [self.construct_system_message(self.system_prompt)]
        
class Coordinator:
    def __init__(self, exp_name, problem_name, problem, problem_prompt, label, model, agent_list, log_dir='log_social', prompter=None, max_turn=3, trial_num=1, verbose=False):
        self.exp_name = exp_name
        self.problem_name = problem_name
        self.problem = problem
        self.problem_prompt = problem_prompt
        self.label = label
        self.model = model
        self.agent_list = agent_list
        self.prompter = prompter
        self.max_turn = max_turn
        self.trial_num = trial_num
        self.max_rate = max_rate_dict.get(model, 300000)  # Default to 300000 if model not found
        
        self.answers = [[] for _ in range(max_turn+1)]
        self.accuracies = np.zeros((max_turn+1, len(agent_list)))
        self.distances = np.zeros((max_turn+1, len(agent_list)))
        self.answer_type = np.zeros((max_turn+1, len(agent_list)))
        self.input_token_usage = 0
        self.output_token_usage = 0
        
        # Generate a timestamped log filename
        timestamp = time.strftime('%Y-%m-%d_%H-%M-%S')
        self.save_path = f"{log_dir}/{self.model}/log_{self.exp_name}_{self.model}_A{len(self.agent_list)}_T{self.max_turn}_{self.problem_name}_{self.trial_num}.txt"
        
        if verbose:
            with open(self.save_path, "w") as f:
                f.write("###### PROBLEM SOLVIONG LOG ######\n")
                f.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Number of agents: {len(agent_list)}\n")
                f.write(f"Temperature list: {[agent.temperature for agent in agent_list]}\n")

                f.write("=" * 80 + "\n\n")

    def log_interaction(self, text):
        with open(self.save_path, "a") as f:
            f.write(text + "\n")

    def start_solving(self):
        if verbose:
            self.log_interaction(f"###### STARTING SOLVING ######\nTimestamp: {self.problem_name}, {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.log_interaction(f"Problem\n\"{self.problem_prompt}\"\n")

        # Initial guess from all agents
        for agent in self.agent_list:
            problem = agent.construct_user_message(self.problem_prompt)
            agent.chat_history.append(problem)
            response, answer, prompt_tokens, completion_tokens = agent.generate_answer(agent.chat_history)
            self.input_token_usage += prompt_tokens
            self.output_token_usage += completion_tokens
            self._enforce_rate_limit(prompt_tokens + completion_tokens)
            if verbose:
                self.log_interaction(f"###### {agent.name} INITIAL RESPONSE ######\n{response}\n") 
                self.log_interaction(f"###### {agent.name} INITIAL ANSWER ######\n{answer}\n")
            self.answers[0].append(answer)
            
        # Conduct the social interaction
        for turn in range(self.max_turn):
            if verbose:
                self.log_interaction(f"###### TURN {turn + 1} ######\n")  # Updated to include turn number
            for agent_id, agent in enumerate(self.agent_list):
                prompt = self.prompter(turn, self.max_turn, agent_id, self.answers[turn])
                user_message = agent.construct_user_message(prompt)
                agent.chat_history.append(user_message)
                response, answer, prompt_tokens, completion_tokens = agent.generate_answer(agent.chat_history)
                self.input_token_usage += prompt_tokens
                self.output_token_usage += completion_tokens
                self._enforce_rate_limit(prompt_tokens + completion_tokens)
                if verbose:
                    self.log_interaction(f"###### {agent.name} RESPONSE ######\n{response}\n")
                    self.log_interaction(f"###### {agent.name} ANSWER ######\n{answer}\n")
                self.answers[turn+1].append(answer)

        if verbose:
            self.log_interaction("Solving completed.\n")

    def _enforce_rate_limit(self, tokens_used):
        # Calculate the minimum time required to process the tokens
        time_per_token = 60 / self.max_rate  # seconds per token
        required_time = tokens_used * time_per_token

        # Check the elapsed time since the last token usage
        current_time = time.time()
        if not hasattr(self, '_last_token_time'):
            self._last_token_time = current_time

        elapsed_time = current_time - self._last_token_time
        if elapsed_time < required_time:
            time.sleep(required_time - elapsed_time)

        # Update the last token time
        self._last_token_time = time.time()
        
    def summary_stats(self):
        pass
        
# ConceptARC specific functions

def replace_all(text, dic):
    for i, j in dic.items():
        text = text.replace(i, j)
    return text

class ConceptARC_Coordinator(Coordinator):
    
    def parse_output_json(response: str):
        """
        LLM이 반환한 문자열에서
        {"grid":"..."}
        형태의 JSON을 파싱해서 grid_str를 반환한다.
        - code fence(````json`)가 붙어 있어도 웬만하면 처리
        - 실패하면 전체 텍스트를 반환
        """
        s = response.strip()

        # 1) ```json ... ``` 형태일 수도 있으니 제거 시도
        if s.startswith("```"):
            # 맨 앞의 ```json / ``` 제거
            s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
            # 맨 뒤의 ``` 제거
            s = re.sub(r"\s*```$", "", s).strip()

        # 2) 혹시 앞뒤에 다른 텍스트가 붙어 있으면
        #    첫 '{'와 마지막 '}' 사이만 떼어냄
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and start < end:
            candidate = s[start:end+1]
        else:
            candidate = s

        grid = candidate  # fallback으로는 전체 텍스트를 반환

        # 3) JSON 파싱 시도
        try:
            obj = json.loads(candidate)
            grid = str(obj.get("grid", "")).strip()
            grid = grid.replace("\\n", "\n") # 개행문자 복원
        except Exception:
            # 실패하면 위에서 잡아둔 fallback 유지
            pass

        return "", grid # rule은 빈 문자열로 반환


    # --------------------------
    # response에서 rule, output 부분을 파싱
    # --------------------------
    def parse_rule_and_output_json(response: str):
        """
        LLM이 반환한 문자열에서
        {"rule":"...","grid":"..."}
        형태의 JSON을 파싱해서 (rule, grid_str)를 반환한다.
        - code fence(````json`)가 붙어 있어도 웬만하면 처리
        - 실패하면 (\"\", 전체 텍스트)로 fallback
        """
        s = response.strip()

        # 1) ```json ... ``` 형태일 수도 있으니 제거 시도
        if s.startswith("```"):
            # 맨 앞의 ```json / ``` 제거
            s = re.sub(r"^```[a-zA-Z]*\s*", "", s)
            # 맨 뒤의 ``` 제거
            s = re.sub(r"\s*```$", "", s).strip()

        # 2) 혹시 앞뒤에 다른 텍스트가 붙어 있으면
        #    첫 '{'와 마지막 '}' 사이만 떼어냄
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and start < end:
            candidate = s[start:end+1]
        else:
            candidate = s

        rule = ""
        grid = candidate  # fallback으로는 전체 텍스트를 grid로 본다

        # 3) JSON 파싱 시도
        try:
            obj = json.loads(candidate)
            rule = str(obj.get("rule", "")).strip()
            grid = str(obj.get("grid", "")).strip()
        except Exception:
            # 실패하면 위에서 잡아둔 fallback 유지
            pass

        return rule, grid
    
    def extract_answer_array(self, response, label_shape=None):
        
        # if response contains other than '\n\, ' ', ',', '[', ']', and numerals (0~9), return False
        if re.search(r'[^\d\s\n]', response):
            return None
        
        arr = response.strip().split('\n')
        arr = [list(map(int, sub_arr.split())) for sub_arr in arr]
        
        # Check if all sublists have the same length
        max_length = max(len(sublist) for sublist in arr)
        if all(len(sublist) == max_length for sublist in arr):
            array = np.array(arr)
        else:
            # Pad shorter sublists with -1
            padded_arr = [sublist + [-1] * (max_length - len(sublist)) for sublist in arr]
            array = np.array(padded_arr)
        
        if label_shape is not None and array.shape != label_shape:
            # cropping
            if array.shape[0] > label_shape[0]:
                array = array[:label_shape[0], :]
            if array.shape[1] > label_shape[1]:
                array = array[:, :label_shape[1]]
            # padding
            if array.shape[0] < label_shape[0]:
                array = np.pad(array, ((0, label_shape[0] - array.shape[0]), (0, 0)), mode='constant', constant_values=-1)
            if array.shape[1] < label_shape[1]:
                array = np.pad(array, ((0, 0), (0, label_shape[1] - array.shape[1])), mode='constant', constant_values=-1)
        return array

    
    def summary_stats(self):
        # Convert answers to numpy array for easier manipulation

        label = self.extract_answer_array(self.label)
        answers_array = np.array([np.array([self.extract_answer_array(answer, label.shape) for answer in turn_answers]) for turn_answers in self.answers])
        
        total_turns = answers_array.shape[0]
        num_agents = answers_array.shape[1]
        d1 = answers_array.shape[2]
        d2 = answers_array.shape[3]
    
        unique_answers = np.unique(answers_array.reshape(-1, d1, d2), axis=0)
        
        self.accuracies = np.array([np.mean(np.array_equal(label, answer)) for answer in answers_array.reshape(-1, d1, d2)]).reshape(total_turns, num_agents)
        self.distances =  np.array([1-np.mean(label == answer) for answer in answers_array.reshape(-1, d1, d2)]).reshape(total_turns, num_agents)
        self.answer_type = np.array([int(np.where(np.all(unique_answers == answers_array[i][j], axis=(1, 2)))[0][0]) for i in range(total_turns) for j in range(num_agents)]).reshape(total_turns, num_agents)
        
        if verbose:
            # Log the summary statistics
            with open(self.save_path, "a") as f:
                f.write("=" * 80 + "\n")
                f.write("###### SUMMARY STATISTICS ######\n")
                f.write(f"Total turns: {total_turns}\n")
                f.write(f"Number of agents: {num_agents}\n")
                f.write(f"Input token usage: {self.input_token_usage}\n")
                f.write(f"Output token usage: {self.output_token_usage}\n")
                f.write(f"Accuracies:\n{self.accuracies}\n")
                f.write(f"Distances:\n{self.distances}\n")
                f.write(f"Answer types:\n{self.answer_type}\n")
                f.write("=" * 80 + "\n")
            
        # save the summary statistics to a JSON file
        summary_stats = {
            "total_turns": total_turns,
            "num_agents": num_agents,
            "input_token_usage": self.input_token_usage,
            "output_token_usage": self.output_token_usage,
            "accuracies": self.accuracies.tolist(),
            "distances": self.distances.tolist(),
            "answer_types": self.answer_type.tolist(),
            "unique_answers": unique_answers.tolist()
        }
        with open(self.save_path.replace('.txt', f'summary.json'), 'w') as f:
            json.dump(summary_stats, f, indent=4)

def extract_answer(response):
    # Extract the answer from the response
    if "Output:" in response:
        return response.split("Output:")[-1].strip()
    return response.strip()

def array_formatter(arr):
    return ''.join([' '.join(map(str, sub_arr))+'\n' for sub_arr in arr])

def problem_generator(prefix="", suffix="", number_range=[0], case_range=[0]):
    dataset_folder = "dataset/ConceptARC"
    folders = [f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))]
    folders.sort()
    
    # Iterate through each folder and read the specified number of files
    for folder in folders:
        for i in number_range:
            with open(f'dataset/ConceptARC/{folder}/{folder}{i+1}.json') as f:
                data = [json.loads(line) for line in f]
            
            problem = []
            
            problem_prompt = prefix
            len_train = len(data[0]['train'])
            
            for j in range(len_train):
                problem.append([data[0]['train'][j]['input'], data[0]['train'][j]['output']])
                formatted_train_input = array_formatter(data[0]['train'][j]['input'])
                formatted_train_output = array_formatter(data[0]['train'][j]['output'])

                problem_prompt += f'Example {j+1}:\nInput:\n{formatted_train_input}\nOutput:\n{formatted_train_output}\n'
            
            problem_prompt += suffix
            
            for k in case_range:
                # pad k when it is a single digit
                problem_name = f"{folder}{i+1}_{k+1}"
                formatted_test_input = array_formatter(data[0]['test'][k]['input'])
                formatted_test_output = array_formatter(data[0]['test'][k]['output'])

                yield problem_name, problem + [[data[0]['test'][k]['input'], data[0]['test'][k]['output']]], problem_prompt + f'\nTest Input:\n{formatted_test_input}', formatted_test_output

def indv_simple_prompter(turn, max_turn, agent_id, previous_answers):
    prompt = f"[Turn {turn}/{max_turn}]\n"
    prompt += "Check again that your previous answer really does follow the common rule between given inputs and outputs. "
    prompt += "You may keep your answer if you think your previous answer is correct, or revise it if you think it is wrong. "
    prompt += "As before, your answer should be 'Output:' and the predicted output grid. Only provide the final answer, no other text or explanation. "
    return prompt

def social_simple_prompter(turn, max_turn, agent_id, previous_answers):
    prompt = f"[Turn {turn}/{max_turn}]\n"
    prompt += "These are other agents' current answers :\n"
    for i, answer in enumerate(previous_answers):
        if i != agent_id:
            prompt += f"Agent {i} : {answer}\n"
    prompt += "First, compare your answers with others. "
    prompt += "Now, if any of those agents have different answers than you, guess the possible rule between inputs and outputs that each other agents have found based on their answers, and compare it to yours. "
    prompt += "Finally, review your rules and answer again. "
    prompt += "Keep your answer if you think your previous answer is correct, or revise it if you think it's wrong. "
    prompt += "As before, your answer should be 'Output:' and the predicted output grid. Only provide the final answer, no other text or explanation. "
    return prompt

def indv_rule_prompter(turn, max_turn, agent_id, previous_answers):
    prompt = f"[Turn {turn}/{max_turn}]\n"
    prompt += "Now, you have a chance to revise your answer. "
    prompt += "Keep your answer if you think your previous answer is correct, or revise it if you think it's wrong. "
    prompt += "As before, your answer should end with 'Rule:' and the simple description of the rule you found, followed by 'Output:' and the predicted output grid itself."
    return prompt

def social_rule_prompter(turn, max_turn, agent_id, previous_answers):
    prompt = f"[Turn {turn}/{max_turn}]\n"
    prompt += "These are other agents' current answers :\n"
    for i, answer in enumerate(previous_answers):
        if i != agent_id:
            prompt += f"Agent {i} : {answer}\n"
    prompt += "First, compare your answers with others. "
    prompt += "Now, if any of those agents have different answers than you, guess the possible rule between inputs and outputs that each other agents have found based on their answers, and compare it to yours. "
    prompt += "Finally, review your rules and answer again. "
    prompt += "Keep your answer if you think your previous answer is correct, or revise it if you think it's wrong. "
    prompt += "As before, your answer should end with 'Rule:' and the simple description of the rule you found, followed by 'Output:' and the predicted output grid itself."
    return prompt        
           
prompter_dict = {'indv_simple': indv_simple_prompter, 'indv_rule': indv_rule_prompter, 'social_simple': social_simple_prompter, 'social_rule': social_rule_prompter}
max_rate_dict = {'gpt-4o': 30000, 'gpt-4o-mini': 200000, 'gpt-5-mini': 1000000, 'deepseek-chat': 1000000, 'deepseek-reasoner': 1000000, 'meta-llama/llama-4-maverick-17b-128e-instruct':300000}  # tokens per minute for different models
temperature_dict = {'t1':[0.2, 0.2, 0.2, 0.2, 0.2], 't2':[1.4, 1.4, 1.4, 1.4, 1.4], 't3':[0.2, 0.5, 0.8, 1.1, 1.4]}
          

# if main

if __name__ == "__main__":
           
    # Predefined constants
    max_turn = 3
                
    print('main')
                
    # Argument parsing   
    parser = ArgumentParser()
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--rule_name", choices=["indv_simple", "indv_rule", "social_simple", "social_rule"])
    parser.add_argument("--temperature_type", choices=["t1", "t2", "t3"], default="t1")
    args = vars(parser.parse_args(sys.argv[1:]))
    print(args)

    # Get the arguments
    model = args["model"]
    rule_name = args["rule_name"]
    temperature_type = args["temperature_type"]
    prompter = prompter_dict[rule_name]

    # Specific ranges for the experiment

    number_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    case_range = [0]
    temperature_list = temperature_dict[temperature_type]
    trial_range = range(1)

    if 'indv' in rule_name:
        system_prompt_list = [f"You are Agent {i}, an expert in pattern findings. Your task is completing the given pattern of inputs and outputs by finding a common rule between them. After your initial guess, you will engage in {max_turn} turn of revision. Always consider the possibility that you are wrong, but be careful not to change your answer if it is truly correct. " for i in range(len(temperature_list))]

    elif 'social' in rule_name:
        system_prompt_list = [f"You are Agent {i}, an expert in pattern findings. Your task is completing the given pattern of inputs and outputs by finding a common rule between them. There are {len(temperature_list)-1} other agents who will try to solve the same problem. After your initial guess, you and other agents will engage in {max_turn} turn of revision, where you're presented with the answers submitted by others. If other people's answers differ from yours, always consider the possibility that you are wrong and try to learn from others. But at the same time, be careful not to change your answer if it is truly correct, and don't go with the majority simply because your answer is different from the majority." for i in range(len(temperature_list))]

    else:
        raise ValueError("Invalid rule name")
        
    if 'simple' in rule_name:
        prefix = "Find the common rule that maps an input grid to an output grid, given the examples below.\n"
        suffix = "Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Your answer should be 'Output:' and the predicted output grid. Only provide the final answer, no other text or explanation. \n"

    elif 'rule' in rule_name:
        prefix = "Find the common rule that maps an input grid to an output grid, given the examples below.\n"
        suffix = "Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Your answer should end with 'Rule:' and the simple description of the rule you found, followed by 'Output:' and the predicted output grid.\n"

    else:
        raise ValueError("Invalid rule name")


    # Main loop
    for trial_num in trial_range:
        agent_list = [
            LLMAgent(name=f"Agent {i}", model=model, temperature=temperature_list[i], system_prompt=system_prompt_list[i], extract_answer=extract_answer) for i in range(len(temperature_list))
        ]

        generator = problem_generator(prefix=prefix, suffix=suffix, number_range=number_range, case_range=case_range)

        num_problem = 16 * len(number_range) * len(case_range) # 16 folders, 10 files per folder, 3 problems per file

        for i in range(num_problem):
            problem_name, problem, problem_prompt, label = next(generator)
            print(f"\nSolving problem: {problem_name}")
            log_file_path = f"log/{model}/log_{rule_name}_{temperature_type}_{model}_A{len(agent_list)}_T{max_turn}_{problem_name}_{trial_num}summary.json"
            
            # Check if the log file already exists
            if os.path.exists(log_file_path):
                print(f"Skipping problem {problem_name} as log file already exists.")
                continue
            
            # Initialize the coordinator with the problem, problem_prompt, label, agent list, and prompter
            coordinator = ConceptARC_Coordinator(
                exp_name=f'{rule_name}_{temperature_type}',
                problem_name=problem_name,
                problem=problem,
                problem_prompt=problem_prompt,
                label=label,
                model=model,
                agent_list=agent_list,
                prompter=prompter,
                max_turn=max_turn,
                trial_num=trial_num
            )
            coordinator.start_solving()
            print(f"\nSummary statistics for problem {problem_name}")
            coordinator.summary_stats()
            for agent in agent_list:
                agent.clear_chat_history()


    # nohup python socialLLM_ConceptARC.py --model deepseek-chat --rule_name social_simple > transform_atlantic.out &