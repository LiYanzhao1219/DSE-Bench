from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path

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
MAX_TOKENS = 5000
RUN_ID = 'run01'
TASK = 'program'
MULTI_CHOICE = True
INSTRUCTION = '请阅读下面的多项选择题，直接给出所有正确选项字母；如果答案是 A、B、C，请输出 ABC。'
DATA_PATHS = [
    ROOT / 'data/knowledge/program/control.json',
    ROOT / 'data/knowledge/program/launch.json',
    ROOT / 'data/knowledge/program/make.json',
    ROOT / 'data/knowledge/program/test.json',
]

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


def answer_text(value):
    if not isinstance(value, str) or not value.strip():
        raise InvalidResponse('Empty text answer')
    value = value.strip()
    if re.match(r"^(?:(?:很抱歉|抱歉|对不起)[，,。\s]|我(?:无法|不能)(?:回答|提供|协助|完成)|"
                r"无法(?:回答|提供)|I(?:'m| am) sorry\b|I (?:cannot|can't|am unable to)\b)", value, re.I):
        raise InvalidResponse('Explicit refusal')
    return value

def json_answer(raw):
    value = response_payload(raw)
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str) or not value.strip():
        raise InvalidResponse('Empty or non-JSON answer')
    value = answer_text(value)
    decoder = json.JSONDecoder()
    for index, char in enumerate(value):
        if char in '{[':
            try:
                result, _ = decoder.raw_decode(value[index:])
                return result
            except json.JSONDecodeError:
                continue
    raise InvalidResponse('No complete JSON value')

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

def load_groups(paths):
    groups = {}
    for path in paths:
        path = Path(path)
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(data, list) or not data:
            raise ValueError(f'题库必须是非空列表：{path}')
        for row in data:
            if not isinstance(row, dict) or not row.get('question') or not row.get('options') or not row.get('answer'):
                raise ValueError(f'题目缺少 question/options/answer：{path}')
        groups[path.name] = data
    return groups


def create_chain():
    from langchain_core.output_parsers import JsonOutputParser
    from langchain_core.prompts import PromptTemplate
    from pydantic import BaseModel, Field
    class ChoiceAnswer(BaseModel):
        answer: str = Field(description='Correct option letters, such as A or ABC.')
    parser = JsonOutputParser(pydantic_object=ChoiceAnswer)
    llm = create_llm().bind(response_format={'type': 'json_object'})
    prompt = PromptTemplate(template='{instruction}\n问题：{question}\n选项：{options}\n{format_instructions}\n'
        '硬性规则：只输出 JSON 对象，不要输出解释、推理过程或多余文本。',
        input_variables=['instruction', 'question', 'options'],
        partial_variables={'format_instructions': parser.get_format_instructions()})
    return prompt | llm


def parse_answer(raw):
    result = json_answer(raw)
    if not isinstance(result, dict) or not isinstance(result.get('answer'), str) or not result['answer'].strip():
        raise InvalidResponse('缺少选项答案')
    text = result['answer'].strip().upper()
    letters = re.findall(r'[A-Z]', text)
    return ''.join(letters) if letters else text


def score_answer(prediction, item):
    gold = str(item['answer']).strip().upper()
    if MULTI_CHOICE:
        prediction, gold = ''.join(sorted(set(prediction))), ''.join(sorted(set(gold)))
    return {'accuracy': float(prediction == gold)}


def evaluate(invoke=None, data_paths=None, output_root=None, run_id=None):
    if invoke is None:
        validate_api()
    groups = load_groups(DATA_PATHS if data_paths is None else data_paths)
    if invoke is None:
        chain = create_chain()
        invoke = lambda item: chain.invoke({'instruction': INSTRUCTION, 'question': item['question'],
                                           'options': '\n'.join(str(x) for x in item['options'])})
    return evaluate_items(groups, invoke, parse_answer, score_answer, 'accuracy', ['accuracy'],
                          TASK, output_root, run_id)


def main():
    evaluate()


if __name__ == '__main__':
    main()
