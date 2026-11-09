from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple, Union

ROOT = Path(__file__).resolve().parent

API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL_NAME = 'qwen3-max'
API_KEY = ''
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 300
MAX_ATTEMPTS = 3
RETRY_DELAYS = (2, 4)
EXTRA_BODY = {}
STREAMING = False
MAX_TOKENS = None
RUN_ID = 'run01'
DATA_PATH = ROOT / 'data/task/detect_methods.json'

class InvalidResponse(ValueError):
    pass

class ConfigurationError(RuntimeError):
    pass

def serializable(value):
    if isinstance(value, BaseException):
        return {'error_type': type(value).__name__}
    if hasattr(value, 'model_dump'):
        return serializable(value.model_dump())
    if hasattr(value, 'dict') and not isinstance(value, dict):
        return serializable(value.dict())
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()
                if str(k).lower() not in {'api_key', 'authorization', 'headers', 'api-key'}}
    if isinstance(value, (list, tuple, set)):
        return [serializable(x) for x in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported persisted type: {type(value).__name__}')

def digest(value):
    return hashlib.sha256(json.dumps(serializable(value), ensure_ascii=False,
                                     sort_keys=True, allow_nan=False).encode('utf-8')).hexdigest()

def response_payload(raw):
    if isinstance(raw, dict) and 'raw' in raw and 'parsed' in raw:
        return response_payload(raw['raw'])
    if isinstance(raw, dict) and ('content' in raw or 'tool_calls' in raw):
        calls = raw.get('tool_calls') or []
        if calls:
            if len(calls) != 1:
                raise InvalidResponse('Expected one structured answer')
            return calls[0].get('args', calls[0].get('function', {}).get('arguments', {}))
        additional = raw.get('additional_kwargs') or {}
        calls = additional.get('tool_calls') or []
        if calls:
            if len(calls) != 1:
                raise InvalidResponse('Expected one structured answer')
            return calls[0].get('function', {}).get('arguments', '')
        content = raw.get('content', '')
        if isinstance(content, list):
            content = ''.join(x.get('text', '') for x in content if isinstance(x, dict))
        return content
    return raw

def response_text(raw):
    value = response_payload(raw)
    if not isinstance(value, str):
        raise InvalidResponse('Expected text answer')
    return answer_text(value)

def answer_text(value):
    if not isinstance(value, str) or not value.strip():
        raise InvalidResponse('Empty text answer')
    value = value.strip()
    if re.match(r"^(?:(?:很抱歉|抱歉|对不起)[，,。\s]|我(?:无法|不能)(?:回答|提供|协助|完成)|"
                r"无法(?:回答|提供)|I(?:'m| am) sorry\b|I (?:cannot|can't|am unable to)\b)", value, re.I):
        raise InvalidResponse('Explicit refusal')
    return value


def error_info(exc):
    status = getattr(exc, 'status_code', None)
    if status is None:
        status = getattr(getattr(exc, 'response', None), 'status_code', None)
    return {'type': type(exc).__name__, 'http_status': status}

def error_kind(exc):
    status = error_info(exc)['http_status']
    if status in (408, 429) or isinstance(status, int) and 500 <= status <= 599:
        return 'api_error'
    if isinstance(status, int) and 400 <= status <= 499:
        return 'config_error'
    names = {cls.__name__ for cls in type(exc).__mro__}
    if isinstance(exc, (TimeoutError, ConnectionError)) or names & {
        'APITimeoutError', 'APIConnectionError', 'TimeoutException', 'TransportError',
        'ReadTimeout', 'ConnectTimeout', 'RemoteProtocolError', 'IncompleteRead',
    }:
        return 'api_error'
    return 'execution_error'


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(serializable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)

class APIExhausted(RuntimeError):
    pass


def model_directory(model):
    name = re.sub(r'[\\/:*?"<>|\s]+', '_', model).strip('._') or 'model'
    if name != model:
        name += '-' + digest(model)[:8]
    return name


def new_output_directory(task, output_root=None, run_id=None):
    run_id = RUN_ID if run_id is None else run_id
    if not re.fullmatch(r'[\w.-]+', run_id) or run_id in {'.', '..'}:
        raise ValueError('RUN_ID 必须是单个目录名')
    parent = Path(output_root or ROOT / 'outputs') / task
    target = parent / run_id / model_directory(MODEL_NAME)
    suffix = 2
    while target.exists():
        target = parent / f'{run_id}_{suffix}' / model_directory(MODEL_NAME)
        suffix += 1
    target.mkdir(parents=True)
    return target


def print_summary(summary):
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary.get('score_pct') is None:
        print('本次尚无正式分数，请查看失败数量或运行说明。')
    else:
        print(f"最终得分：{summary['score_pct']:.2f}%")

def validate_api():
    if not API_KEY.strip():
        raise RuntimeError('请在本脚本顶部填写 API_KEY；尚未发起模型请求。')
    if not API_URL.strip() or not MODEL_NAME.strip():
        raise ValueError('请配置 API_URL 和 MODEL_NAME')


def create_llm():
    validate_api()
    from langchain_openai import ChatOpenAI
    options = dict(api_key=API_KEY, base_url=API_URL, model=MODEL_NAME,
                   temperature=TEMPERATURE, request_timeout=REQUEST_TIMEOUT, max_retries=0,
                   streaming=STREAMING)
    if STREAMING:
        options['stream_usage'] = True
    if EXTRA_BODY:
        options['extra_body'] = EXTRA_BODY
    if MAX_TOKENS is not None:
        options['max_tokens'] = MAX_TOKENS
    return ChatOpenAI(**options)


def request_with_retry(invoke):
    for attempt in range(MAX_ATTEMPTS):
        try:
            return invoke()
        except Exception as exc:
            kind = error_kind(exc)
            if kind == 'config_error':
                raise ConfigurationError(str(error_info(exc))) from exc
            if kind != 'api_error':
                raise
            if attempt == MAX_ATTEMPTS - 1:
                raise APIExhausted(str(error_info(exc))) from exc
            time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])

def evaluate_items(groups, invoke, parse, score, primary, keys, task, output_root=None, run_id=None):
    if not groups or any(not data for data in groups.values()):
        raise ValueError('题库不能为空')
    destination = new_output_directory(task, output_root, run_id)
    stats = {name: {'sample_count': len(data), 'status_counts': Counter(),
                    'totals': dict.fromkeys(keys, 0.0)} for name, data in groups.items()}
    def summarize():
        per_file = {}
        all_counts = Counter()
        totals = dict.fromkeys(keys, 0.0)
        total = sum(x['sample_count'] for x in stats.values())
        for name, values in stats.items():
            counts = values['status_counts'].copy()
            counts['unprocessed'] += values['sample_count'] - sum(counts.values())
            counts = +counts
            complete = counts['ok'] + counts['invalid_response'] == values['sample_count']
            metrics = {key: values['totals'][key] / values['sample_count'] if complete else None for key in keys}
            per_file[name] = {'sample_count': values['sample_count'], 'status_counts': dict(counts),
                              'metrics': metrics, 'score_pct': 100 * metrics[primary] if complete else None}
            all_counts.update(counts)
            for key in keys:
                totals[key] += values['totals'][key]
        complete = all_counts['ok'] + all_counts['invalid_response'] == total
        metrics = {key: totals[key] / total if complete else None for key in keys}
        return {'task': task, 'model': MODEL_NAME, 'status': 'complete' if complete else 'incomplete',
                'sample_count': total, 'status_counts': dict(all_counts), 'primary_metric': primary,
                'metrics': metrics, 'score_pct': 100 * metrics[primary] if complete else None, 'per_file': per_file}
    try:
        for name, data in groups.items():
            for index, item in enumerate(data, 1):
                print(f'[{name}] {index}/{len(data)}')
                try:
                    raw = serializable(request_with_retry(lambda: invoke(item)))
                except APIExhausted:
                    stats[name]['status_counts']['api_error'] += 1
                    continue
                except Exception as exc:
                    stats[name]['status_counts']['config_error' if isinstance(exc, ConfigurationError) else 'execution_error'] += 1
                    raise
                try:
                    answer = parse(raw)
                except InvalidResponse:
                    stats[name]['status_counts']['invalid_response'] += 1
                else:
                    metrics = score(answer, item)
                    for key in keys:
                        value = metrics[key]
                        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                            raise ValueError('逐题指标必须位于 [0, 1]')
                        stats[name]['totals'][key] += value
                    stats[name]['status_counts']['ok'] += 1
    finally:
        result = summarize()
        _write(destination / 'summary.json', result)
        for name, values in result['per_file'].items():
            value = values['score_pct']
            print(f"{name}：{value:.2f}%" if value is not None else f'{name}：不完整，暂无正式成绩')
        print_summary(result)
        print(f'汇总文件：{destination / "summary.json"}')
    return result
VALID_DETECT_METHODS: Tuple[str, ...] = (
    "环绕探测",
    "飞越探测",
    "伴飞探测",
    "采样返回",
    "巡视探测",
    "着陆探测",
)
REQUIRED_FIELDS = ["id", "任务", "时间", "探测对象", "科学目标", "探测方式"]
def load_detect_method_dataset(input_file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    input_path = Path(input_file_path)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{input_path} should contain a JSON list.")

    valid_method_set = set(VALID_DETECT_METHODS)
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} should be a JSON object.")

        missing_fields = [field for field in REQUIRED_FIELDS if field not in item]
        if missing_fields:
            raise ValueError(f"Item {index} is missing fields: {missing_fields}")

        if not isinstance(item["探测方式"], list):
            raise ValueError(f"Item {index} field '探测方式' should be a list.")

        unknown_methods = sorted(set(item["探测方式"]) - valid_method_set)
        if unknown_methods:
            raise ValueError(f"Item {index} contains unknown detect methods: {unknown_methods}")

    return data

def parse_detect_methods_output(model_output: Any) -> Set[str]:
    text = str(model_output or "").strip()
    if not text:
        return set()
    return {method for method in VALID_DETECT_METHODS if method in text}

def format_method_set(methods: Sequence[str]) -> str:
    method_set = set(methods)
    return "、".join(method for method in VALID_DETECT_METHODS if method in method_set)

def build_detect_method_prompt(item: Dict[str, Any]) -> str:
    return f"""
这是一项发射时间为{item['时间']}的深空探测任务，探测对象是{item["探测对象"]}， 该任务要实现的科学目标为: {item["科学目标"]}。

输出要求：
1. 只能从可选探测方式中选择，可多选。
2. 如果有多个探测方式，用中文逗号分隔。
3. 只输出探测方式名称，不要输出解释、推理过程、编号、JSON 或 Markdown。"""

def score_detect_methods(gold_methods: Sequence[str], predicted_methods: Set[str]) -> Dict[str, float]:
    gold_set = set(gold_methods)
    true_positive = len(gold_set & predicted_methods)
    false_positive = len(predicted_methods - gold_set)
    false_negative = len(gold_set - predicted_methods)

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def is_exact_match(gold_methods: Sequence[str], predicted_methods: Set[str]) -> bool:
    return set(gold_methods) == predicted_methods
def create_detect_method_chain(llm_url: str, api_key: str, model_name: str) -> Any:
    try:
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError('Missing LangChain dependencies. Install langchain-core and langchain-openai before running detect method evaluation.') from exc
    llm = create_llm()
    prompt = ChatPromptTemplate.from_messages([('system', f'你是一个深空探测领域的任务助手，用户会给你提供任务的探测对象、大致任务时间和任务的科学目标相关信息，请你确认该任务的探测方式。在确认的过程中，你需要充分考虑该任务时间下的技术背景，以及探测对象的相关性质，并确保你所选择的探测方式足以实现科学目标。不要从历史深空探测任务中检索相关信息，而是利用你的推理能力给出你的答案。可选探测方式：{format_method_set(VALID_DETECT_METHODS)}必须只从用户给出的候选探测方式中选择答案，不要输出解释、推理过程或任何额外内容。'), ('human', '{input}')])
    return prompt | llm

def evaluate(invoke=None, data_path=None, output_root=None, run_id=None):
    if invoke is None:
        validate_api()
    data = load_detect_method_dataset(data_path or DATA_PATH)
    if invoke is None:
        chain = create_detect_method_chain(API_URL, API_KEY, MODEL_NAME)
        invoke = lambda item: chain.invoke({'input': build_detect_method_prompt(item)})
    def parse(raw):
        methods = parse_detect_methods_output(response_text(raw))
        if not methods:
            raise InvalidResponse('没有识别到探测方式')
        return methods
    def score(answer, item):
        return {**score_detect_methods(item['探测方式'], answer),
                'strict_accuracy': float(is_exact_match(item['探测方式'], answer))}
    return evaluate_items({'detect_methods': data}, invoke, parse, score, 'f1',
                          ['precision', 'recall', 'f1', 'strict_accuracy'], 'detect_methods', output_root, run_id)


def main():
    evaluate()


if __name__ == '__main__':
    main()
