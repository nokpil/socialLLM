import io, os, json, glob, re, time
from collections import OrderedDict, Counter

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import openai
from openai import OpenAI

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from socialLLM_ConceptARC import *


# --------------------------
# 설정
# --------------------------
MODEL = "gpt-5-mini"
TEMPERATURE = 1.0
NUM_REPEATS = 10   # 같은 문제에 대해 10번 샷
RULE_NAME = "indv_simple_parallel"

LOG_DIR = os.path.join("log_temp_chk", MODEL)
os.makedirs(LOG_DIR, exist_ok=True)

# 🔹 진행 상황 로그 파일
PROGRESS_LOG_DIR = "log_progress"
PROGRESS_LOG_PATH = os.path.join(PROGRESS_LOG_DIR, f"log_{MODEL}_{RULE_NAME}2.txt")

# 🔹 여러 thread가 같이 쓰니까 lock
progress_lock = threading.Lock()

def log_progress(message: str):
    """공용 progress log 파일에 thread-safe하게 한 줄씩 append."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
    with progress_lock:
        with open(PROGRESS_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line)

# --------------------------
# 문제 목록 가져오기
# --------------------------
prefix = (
    "Find the common rule that maps an input grid to an output grid, given the examples below.\n"
)

suffix = (
    "Below is a test input grid. Predict the corresponding output grid by applying the rule you found. Return only this minified JSON (no markdown, no extra keys): {\"grid\":\"<final grid>\"}"
)

number_range = list(range(10))   # 각 폴더마다 10 문제
#number_range = [0, 5, 9]
case_range = [0, 1, 2]                 # test case 3개

problems = list(problem_generator(
    prefix=prefix,
    suffix=suffix,
    number_range=number_range,
    case_range=case_range
))
print("Total problems:", len(problems))

TOTAL_PROBLEMS = len(problems)
# 전체 문제 수를 progress log에 남김
log_progress(f"Experiment {RULE_NAME} started. TOTAL_PROBLEMS={TOTAL_PROBLEMS}, NUM_REPEATS={NUM_REPEATS}")

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
        return "", ""

    text = str(response).strip()

    # -------------------------------
    # 1) 정상 패턴: "grid":"...여러줄..."
    #    여러 번 나올 수 있으니 "마지막" 매치를 사용
    # -------------------------------
    grid_matches = list(re.finditer(r'"grid"\s*:\s*"(.*?)"', text, flags=re.S))
    if grid_matches:
        grid = grid_matches[-1].group(1).strip()
        return "", grid

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
                return "", g

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
        return rule, grid  # 여기까지 왔으면 정상 JSON
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
        return rule, grid

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
                grid = g

    return rule, grid

# --------------------------
# worker 함수
# --------------------------
def worker_run(worker_id: int, rule_name: str):

    agent = LLMAgent(
        name=f"Worker{worker_id}",
        model=MODEL,
        temperature=TEMPERATURE,
        extract_answer=extract_answer,
    )

    log_progress(f"Worker {worker_id} started. TOTAL_PROBLEMS={TOTAL_PROBLEMS}")
    
    parse = lambda x: parse_rule_and_output_json(x) if "rule" in rule_name else parse_output_json(x)

    out = []
    for idx, (pname, _, prompt, label) in enumerate(problems):

        # 🔹 여기가 진행상황 로그 포인트
        log_progress(
            f"Worker {worker_id} processing problem {idx+1}/{TOTAL_PROBLEMS} ({pname})"
        )

        messages = [
            agent.construct_user_message(prompt),
        ]

        try:
            raw, ans, ptok, ctok = agent.generate_answer(messages)
        except Exception as e:
            log_progress(f"Worker {worker_id} ERROR at {pname}: {e}")
            raw = ""
            ans = ""
            ptok = ctok = 0

        rule_text, grid_text = parse(raw)
        
        out.append({
            "worker": worker_id,
            "problem_name": pname,
            "problem_index": idx,
            "prompt": prompt,
            "label": label,
            "raw_response": raw,
            "rule": rule_text,
            "grid": grid_text,
            "prompt_tokens": ptok,
            "completion_tokens": ctok,
            "timestamp": time.time()
        })

    path = os.path.join(LOG_DIR, f"{RULE_NAME}_worker{worker_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    log_progress(f"Worker {worker_id} finished. Saved -> {path}")
    return path

if __name__ == "__main__":
    
    os.environ["PYTHONUNBUFFERED"] = "1"
    # --------------------------
    # 병렬 실행
    # --------------------------
    with ThreadPoolExecutor(max_workers=NUM_REPEATS) as ex:
        futures = [ex.submit(worker_run, w, rule_name=RULE_NAME) for w in range(NUM_REPEATS)]
        for f in as_completed(futures):
            f.result()

    log_progress(f"Experiment {RULE_NAME} finished for all workers.")
    print("Done.")