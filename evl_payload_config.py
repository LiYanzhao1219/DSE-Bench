from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

from pydantic import BaseModel, Field

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
RESUME = True
DATA_PATH = ROOT / 'data/task/payload_set.json'
OUTPUT_MODE = 'structured'

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

def _read(path, default):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else default

def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(serializable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def model_directory(model):
    name = re.sub(r'[\\/:*?"<>|\s]+', '_', model).strip('._') or 'model'
    if name != model:
        name += '-' + digest(model)[:8]
    return name


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


def clean(value):
    value = serializable(value)
    if isinstance(value, str):
        return value.replace(API_KEY, '[REDACTED]') if API_KEY else value
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean(v) for v in value]
    return value


def open_run(task, samples, config, output_root=None, run_id=None, resume=None, path=None):
    run_id = RUN_ID if run_id is None else run_id
    resume = RESUME if resume is None else resume
    if not re.fullmatch(r'[\w.-]+', run_id) or run_id in {'.', '..'}:
        raise ValueError('RUN_ID 必须是单个目录名')
    ids = [item['sample_id'] for item in samples]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('题目不能为空，且行标识必须唯一')
    path = Path(path) if path is not None else Path(output_root or ROOT / 'outputs') / task / run_id / model_directory(MODEL_NAME)
    versions = {}
    for package in ['langchain-openai', 'langchain-core', 'openai', 'pydantic']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = clean({'protocol_version': 2, 'task': task, 'model': MODEL_NAME, 'samples': samples,
        'config': config, 'package_versions': versions,
        'api': {'url': API_URL, 'temperature': TEMPERATURE, 'extra_body': EXTRA_BODY,
                'streaming': STREAMING, 'max_tokens': MAX_TOKENS},
        'policy': {'timeout': REQUEST_TIMEOUT, 'attempts': MAX_ATTEMPTS, 'delays': RETRY_DELAYS},
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    manifest['fingerprint'] = digest(manifest)
    existing = _read(path / 'run.json', None)
    if existing is not None and (not resume or existing != manifest):
        raise ValueError('输入、模型参数或代码与原运行不同，请使用新的 RUN_ID')
    if existing is None:
        if path.exists() and any(path.iterdir()):
            raise ValueError('输出目录已有文件但缺少运行清单，请使用新的 RUN_ID')
        _write(path / 'run.json', manifest)
    ctx = {'path': path, 'samples': samples, 'model': MODEL_NAME, 'task': task,
           'records': _read(path / 'samples.json', {}), 'requests': _read(path / 'requests.json', [])}
    if not set(ctx['records']) <= set(ids):
        raise ValueError('续跑记录中存在未知样本')
    return ctx


def sample(path, row, original_id=None, **fields):
    path = Path(path).resolve()
    try:
        source = path.relative_to(ROOT).as_posix()
    except ValueError:
        source = path.as_posix()
    return {'sample_id': f'{source}#row={row}', 'source': source, 'row': row,
            'original_id': original_id, **fields}


def done(ctx, item):
    return ctx['records'].get(item['sample_id'], {}).get('status') in {'ok', 'invalid_response'}


def save_record(ctx, item, status, *, parsed=None, metrics=None, request_id=None, error=None):
    ctx['records'][item['sample_id']] = clean({**item, 'status': status, 'parsed': parsed,
        'metrics': metrics or {}, 'request_id': request_id, 'error': error})
    _write(ctx['path'] / 'samples.json', ctx['records'])


def request_saved(ctx, request_id, invoke):
    previous = [x for x in ctx['requests'] if x['request_id'] == request_id]
    for entry in previous:
        if entry['status'] == 'received':
            return True, entry['raw']
    for attempt in range(MAX_ATTEMPTS):
        entry = {'request_id': request_id, 'attempt': len(previous) + attempt + 1}
        try:
            raw = invoke()
        except Exception as exc:
            kind = error_kind(exc)
            entry.update(status=kind, error=error_info(exc))
            ctx['requests'].append(entry)
            _write(ctx['path'] / 'requests.json', clean(ctx['requests']))
            if kind == 'config_error':
                raise ConfigurationError(str(error_info(exc))) from exc
            if kind != 'api_error':
                raise
            if attempt == MAX_ATTEMPTS - 1:
                return False, None
            time.sleep(RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)])
        else:
            raw = clean(raw)
            entry.update(status='received', raw=raw)
            ctx['requests'].append(entry)
            _write(ctx['path'] / 'requests.json', ctx['requests'])
            return True, raw


def process(ctx, item, invoke, parse):
    if done(ctx, item):
        return ctx['records'][item['sample_id']]
    request_id = item['sample_id']
    try:
        received, raw = request_saved(ctx, request_id, invoke)
        if not received:
            save_record(ctx, item, 'api_error', request_id=request_id)
        else:
            try:
                parsed = parse(raw)
            except InvalidResponse as exc:
                save_record(ctx, item, 'invalid_response', request_id=request_id, error=error_info(exc))
            else:
                save_record(ctx, item, 'ok', parsed=parsed, request_id=request_id)
    except Exception as exc:
        save_record(ctx, item, 'config_error' if isinstance(exc, ConfigurationError) else 'execution_error',
                    request_id=request_id, error=error_info(exc))
        raise
    return ctx['records'][item['sample_id']]


def generation_summary(ctx):
    counts = Counter(ctx['records'].get(x['sample_id'], {}).get('status', 'unprocessed') for x in ctx['samples'])
    summary = {'protocol_version': 2, 'task': ctx['task'], 'model': ctx['model'],
        'sample_count': len(ctx['samples']), 'status_counts': dict(counts),
        'status': 'complete' if counts['ok'] + counts['invalid_response'] == len(ctx['samples']) else 'incomplete',
        'generation_only': True, 'score_pct': None}
    _write(ctx['path'] / 'summary.json', summary)
    return summary


def export_answers(ctx, headers, base_rows, answer_for):
    from openpyxl import Workbook
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'payload_config' if ctx['task'] == 'payload_config' else 'answers'
    sheet.append([*headers, ctx['model']])
    for item, base in zip(ctx['samples'], base_rows):
        rec = ctx['records'].get(item['sample_id'], {})
        answer = answer_for(rec['parsed']) if rec.get('status') == 'ok' else (
            'ERROR: invalid_response' if rec.get('status') == 'invalid_response' else None)
        sheet.append([*base, answer])
    temporary = ctx['path'] / 'answers.tmp.xlsx'
    workbook.save(temporary)
    temporary.replace(ctx['path'] / 'answers.xlsx')


def parsed_value(call):
    try:
        return serializable(call())
    except (ValueError, TypeError) as exc:
        raise InvalidResponse('本地格式校验失败') from exc
REQUIRED_FIELDS = ["id", "任务", "时间", "探测对象", "科学目标", "探测方式", "载荷"]
class PayloadConfigResult(BaseModel):
    payloads: List[str] = Field(
        default_factory=list,
        description="为该深空探测任务配置的载荷名称列表，只包含载荷名称。",
    )

def load_payload_dataset(data_path: Union[str, Path]) -> List[Dict[str, Any]]:
    data_path = Path(data_path)
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"{data_path} should contain a JSON list.")

    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item {index} should be a JSON object.")
        missing_fields = [field for field in REQUIRED_FIELDS if field not in item]
        if missing_fields:
            raise ValueError(f"Item {index} is missing fields: {missing_fields}")
        if not isinstance(item["载荷"], list):
            raise ValueError(f"Item {index} field '载荷' should be a list.")

    return data

def normalize_payload_name(text: Any) -> str:
    normalized = str(text).strip()
    normalized = re.sub(r"[\u00a0\u3000]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    normalized = normalized.strip(" \t\r\n-•·,，;；、")
    normalized = re.sub(r"^\d+[\.\)、)]\s*", "", normalized)
    return normalized.strip()

def join_payloads(payloads: Sequence[Any]) -> str:
    return "\n".join(
        payload for payload in (normalize_payload_name(item) for item in payloads) if payload
    )

def build_user_prompt(item: Dict[str, Any]) -> str:
    return (
        f"这是一项任务时间设定为{item['时间']}的深空探测任务，"
        f"探测对象是{item['探测对象']}。"
        f"任务的科学目标是{item['科学目标']}，"
        f"其探测方式是{item['探测方式']}，"
        f"请帮我进行该探测任务的载荷配置。"
    )

def build_plain_text_prompt(item: Dict[str, Any], retry: bool = False) -> str:
    retry_instruction = ""
    if retry:
        retry_instruction = "上一次输出不符合要求。请重新输出，每行只写一个载荷名称。\n\n"

    return f"""{retry_instruction}{build_user_prompt(item)}

输出要求：
1. 每行只输出一个载荷名称。
2. 不要输出 JSON、Markdown、编号、解释、用途说明或推理过程。
3. 不要输出“以下是”“建议载荷”等引导语。
"""

def extract_first_json_object(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"JSON root should be an object, got {type(parsed).__name__}.")
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in model output: {stripped[:200]!r}")

    depth = 0
    in_string = False
    escape = False
    for pos in range(start, len(stripped)):
        char = stripped[pos]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = stripped[start : pos + 1]
                parsed = json.loads(candidate)
                if not isinstance(parsed, dict):
                    raise ValueError(f"JSON root should be an object, got {type(parsed).__name__}.")
                return parsed

    raise ValueError(f"No complete JSON object found in model output: {stripped[:200]!r}")

def coerce_payload_result(raw_result: Any) -> PayloadConfigResult:
    if isinstance(raw_result, PayloadConfigResult):
        return raw_result

    if isinstance(raw_result, dict):
        payloads = raw_result.get("payloads")
        if payloads is None:
            payloads = raw_result.get("载荷")
        if isinstance(payloads, str):
            payloads = parse_plain_text_payloads(payloads)
        if not isinstance(payloads, list):
            raise ValueError("Result does not contain a payloads list.")

        data = {"payloads": [normalize_payload_name(item) for item in payloads]}
        data["payloads"] = [item for item in data["payloads"] if item]
        if hasattr(PayloadConfigResult, "model_validate"):
            return PayloadConfigResult.model_validate(data)
        return PayloadConfigResult.parse_obj(data)

    if isinstance(raw_result, str):
        try:
            return coerce_payload_result(extract_first_json_object(raw_result))
        except Exception:
            return PayloadConfigResult(payloads=parse_plain_text_payloads(raw_result))

    raise TypeError(f"Unsupported LLM result type: {type(raw_result)!r}")

def parse_plain_text_payloads(text: str) -> List[str]:
    cleaned_text = text.strip()
    cleaned_text = re.sub(r"^```(?:json|text)?", "", cleaned_text, flags=re.IGNORECASE).strip()
    cleaned_text = re.sub(r"```$", "", cleaned_text).strip()

    payloads: List[str] = []
    for raw_line in re.split(r"[\r\n]+", cleaned_text):
        line = normalize_payload_name(raw_line)
        if not line:
            continue
        if line in {"载荷", "载荷名称", "推荐载荷", "配置载荷"}:
            continue
        if "：" in line or ":" in line:
            prefix, suffix = re.split(r"[:：]", line, maxsplit=1)
            if prefix.strip() in {"载荷", "载荷名称", "推荐载荷", "配置载荷"}:
                line = normalize_payload_name(suffix)
        if line:
            payloads.append(line)

    return payloads
def create_payload_chain_bundle(api_url: str, api_key: str, api_model_name: str) -> Dict[str, Any]:
    try:
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError as exc:
        raise RuntimeError('Missing LangChain dependencies. Install langchain-core and langchain-openai in the current PyCharm interpreter before running payload configuration evaluation.') from exc
    llm = create_llm()
    system_prompt = '你是一个深空探测领域的任务论证专家。下面我将给你提供探测对象（深空天体）、科学目标以及探测任务的大致进行时间（可能是过去的时间），请你根据这些信息配置探测器应该搭载的载荷。输出结果只能包含载荷名称，不要输出解释、编号、推理过程、用途说明或其他信息。'
    if OUTPUT_MODE == 'plain_text':
        plain_prompt = ChatPromptTemplate.from_messages([('system', system_prompt), ('human', '{input}')])
        return {'primary': plain_prompt | llm, 'mode': 'plain_text'}
    parser = JsonOutputParser(pydantic_object=PayloadConfigResult)
    json_prompt = ChatPromptTemplate.from_messages([('system', system_prompt + ' 必须只输出合法 JSON 对象。'), ('human', '{input}\n\n输出格式说明：\n{format_instructions}')]).partial(format_instructions=parser.get_format_instructions())
    json_chain = json_prompt | llm.bind(response_format={'type': 'json_object'})
    plain_prompt = ChatPromptTemplate.from_messages([('system', system_prompt), ('human', '{input}')])
    plain_chain = plain_prompt | llm
    if OUTPUT_MODE == 'json':
        return {'primary': json_chain, 'mode': 'json_parser'}
    try:
        structured_chain = ChatPromptTemplate.from_messages([('system', system_prompt + ' 必须按指定结构输出。'), ('human', '{input}')]) | llm.with_structured_output(PayloadConfigResult, include_raw=True)
        return {'primary': structured_chain, 'mode': 'with_structured_output'}
    except Exception:
        return {'primary': json_chain, 'mode': 'json_parser'}

def evaluate(invoke=None, data_path=None, output_root=None, run_id=None, resume=None):
    if invoke is None:
        validate_api()
    path = Path(data_path or DATA_PATH)
    data = load_payload_dataset(path)
    items = [sample(path, i, x['id'], input=x) for i, x in enumerate(data, 1)]
    ctx = open_run('payload_config', items, {'output_mode': OUTPUT_MODE}, output_root, run_id, resume)
    mode = OUTPUT_MODE
    if invoke is None:
        bundle = create_payload_chain_bundle(API_URL, API_KEY, MODEL_NAME)
        mode = bundle['mode']
        invoke = lambda item: bundle['primary'].invoke({'input': build_plain_text_prompt(item['input'])
                   if mode == 'plain_text' else build_user_prompt(item['input'])})
    def parse(raw):
        value = response_payload(raw) if mode == 'plain_text' else json_answer(raw)
        if isinstance(value, str):
            value = answer_text(value)
        result = parsed_value(lambda: coerce_payload_result(value))
        if not result['payloads']:
            raise InvalidResponse('载荷回答为空')
        return result
    try:
        for item in items:
            process(ctx, item, lambda: invoke(item), parse)
    finally:
        export_answers(ctx, ['序号', '任务', '科学目标', '真实载荷'],
            [[x['input']['id'], x['input']['任务'], x['input']['科学目标'], join_payloads(x['input']['载荷'])] for x in items],
            lambda p: join_payloads(p['payloads']))
        result = generation_summary(ctx)
        print_summary(result)
    return result


def main():
    evaluate()


if __name__ == '__main__':
    main()
