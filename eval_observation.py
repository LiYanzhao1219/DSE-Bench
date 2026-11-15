from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping

from datetime import datetime
import unicodedata

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
LEVEL = 'l1'
DATASET_PATH = ROOT / 'data/task/dataset'
SCENARIOS = tuple(f'{i:02d}' for i in range(1, 21))
DEFAULT_DATASET_ROOT = DATASET_PATH

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


def model_directory(model_name):
    return re.sub(r'[\\/:*?"<>|\s]+', '_', model_name).strip('_') or 'unknown_model'


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


def read_max_power(l2_dir):
    path = Path(l2_dir) / 'max_power.txt'
    try:
        text = path.read_text(encoding='utf-8-sig').strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'Missing power limit file: {path}') from exc
    except UnicodeError as exc:
        raise ValueError(f'Invalid power limit encoding: {path}') from exc
    if not re.fullmatch(r'\+?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?', text):
        raise ValueError(f'Expected one finite positive power value in watts: {path}')
    power = float(text)
    if not math.isfinite(power) or power <= 0:
        raise ValueError(f'Power limit must be finite and positive: {path}')
    return power

def canonical_payload_name(name):
    return re.sub(r'[\s\u200b\u200c\u200d\ufeff]+', '', unicodedata.normalize('NFKC', str(name)))

def parse_timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', value):
        raise ValueError('Expected YYYY-MM-DD HH:MM:SS timestamp')
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')

def parse_plan(value):
    import copy
    plan = copy.deepcopy(json_answer(value))
    if not isinstance(plan, dict) or not isinstance(plan.get('observation_plan'), list) or not plan['observation_plan']:
        raise InvalidResponse('Missing or empty observation plan')
    plan.setdefault('adjustment_plan', [])
    plan.setdefault('achieved_objectives', [])
    if not isinstance(plan['adjustment_plan'], list) or not isinstance(plan['achieved_objectives'], list):
        raise InvalidResponse('Invalid adjustment/objective list')
    for group in ('observation_plan', 'adjustment_plan'):
        for index, entry in enumerate(plan[group], 1):
            if not isinstance(entry, dict):
                raise InvalidResponse('Invalid plan entry')
            if group == 'observation_plan' and (not isinstance(entry.get('name'), str) or not entry['name'].strip()):
                raise InvalidResponse('Missing payload name')
            try:
                parse_timestamp(entry['start_time'])
                parse_timestamp(entry['end_time'])
            except (KeyError, ValueError, TypeError) as exc:
                raise InvalidResponse(f'{group} record {index}: invalid timestamp') from exc
    return plan

def _objective_name(objective: Mapping[str, Any]) -> str:
    return str(objective["name"])

def _objective_requirements(objective: Mapping[str, Any]) -> List[Dict[str, Any]]:
    load_data = objective.get("loadData", [])
    if not isinstance(load_data, list):
        raise ValueError(f"Objective {_objective_name(objective)!r} has invalid loadData.")
    return load_data

def _required_seconds(requirement: Mapping[str, Any]) -> float:
    for key, scale in [('require_time/s', 1), ('require_time/sec', 1),
                       ('require_time/min', 60), ('require_time', 1)]:
        if key in requirement:
            seconds = float(requirement[key]) * scale
            if not math.isfinite(seconds) or seconds < 0:
                raise ValueError('Invalid required duration')
            return seconds
    raise ValueError(f'Requirement {requirement!r} has no supported required-time field')

def _merge_objective_requirements(objective: Mapping[str, Any]) -> Dict[str, float]:
    merged: Dict[str, float] = {}
    for requirement in _objective_requirements(objective):
        payload_name = canonical_payload_name(requirement['pay_load'])
        merged[payload_name] = merged.get(payload_name, 0.0) + _required_seconds(requirement)
    return merged

def load_scenario_inputs(dataset_root=DEFAULT_DATASET_ROOT, levels=None):
    inputs = {}
    for level in (levels or ('l1', 'l2')):
        for number in SCENARIOS:
            folder = Path(dataset_root) / number / level
            objective_path = folder / f'object_{number}.json'
            window_path = folder / ('detect_windows_no_plane.json' if level == 'l1' else 'detect_windows.json')
            objectives = json.loads(objective_path.read_text(encoding='utf-8'))
            windows = json.loads(window_path.read_text(encoding='utf-8'))
            power_limit = read_max_power(folder) if level == 'l2' else None
            try:
                if not isinstance(objectives, list) or not objectives or not isinstance(windows, dict) or not windows:
                    raise ValueError('Empty or invalid scientific input')
                required = {}
                for objective in objectives:
                    if not isinstance(objective.get('name'), str) or not objective['name'].strip():
                        raise ValueError('Missing scientific objective name')
                    requirements = _merge_objective_requirements(objective)
                    if not requirements or '' in requirements:
                        raise ValueError('Missing payload requirements')
                    required.update(requirements)
                known = set()
                for name, info in windows.items():
                    key = canonical_payload_name(name)
                    if not key or key in known or not isinstance(info, dict):
                        raise ValueError('Invalid or ambiguous payload definition')
                    known.add(key)
                    spans = info.get('observation_windows')
                    if not isinstance(spans, list):
                        raise ValueError('Missing observation windows')
                    for span in spans:
                        if not isinstance(span, (list, tuple)) or len(span) != 2 or parse_timestamp(span[0]) >= parse_timestamp(span[1]):
                            raise ValueError('Invalid observation window')
                    if level == 'l2' and (not isinstance(info.get('plane'), str) or not info['plane'].strip()
                                          or info['plane'].strip() == 'Unknown' or info.get('power_consumption') is None):
                        raise ValueError('Missing L2 plane/power')
                    if info.get('power_consumption') is not None:
                        power = float(info['power_consumption'])
                        if not math.isfinite(power) or power < 0:
                            raise ValueError('Invalid power consumption')
                if not set(required) <= known:
                    raise ValueError('Required payload has no window definition')
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f'Invalid scientific input: {folder}: {exc}') from exc
            inputs[(level, number)] = objectives, windows, power_limit
    return inputs
def build_l1_prompt(
    detect_windows: Dict[str, Any],
    science_obj: List[Dict[str, Any]],
) -> str:
    return (
        "请帮我规划探测器各个载荷的探测窗口。"
        "我会给你提供各个载荷的可探测窗口和科学目标的相关信息。\n"
        f"各个载荷的可探测窗口是：{detect_windows},  "
        f"科学目标相关信息为：{science_obj}，"
        "请对满足实现条件的科学目标进行规划，也就是说，如果某科学目标的载荷探测需求时长超出了可探测窗口，"
        "那么可以选择不实现这一科学目标。"
        "可实现科学目标、探测载荷规划结果和调姿结果用JSON格式进行结果输出,具体格式为："
        '{"achieved_objectives" : ['
        '\n "可实现科学目标","可实现科学目标","可实现科学目标","..."'
        "],  "
        '"observation_plan": [ '
        '{"name":"载荷名称",\n '
        '"start_time":"开始探测时间",\n '
        '"end_time":"结束探测时间",\n '
        '"duration":"探测持续时间（end_time与start_time的时间差）"}, \n'
        '{"name":"载荷名称",\n '
        '"start_time":"开始探测时间",\n '
        '"end_time":"结束探测时间",\n '
        '"duration":"探测持续时间（end_time与start_time的时间差）"}],'
        '"adjustment_plan": ['
        '{"name":"第1/2/3次调姿",\n '
        '"start_time":"开始调姿时间",\n '
        '"end_time":"结束调姿时间",\n '
        '"duration":"调姿持续时间（end_time与start_time的时间差）"}, \n'
        "]  }"
        "其中持续时间的单位为秒。\n "
        "请注意：1.除了我需要的规划结果json文件以外，其他内容的输出都不要使用json格式。\n"
        "2.如果载荷的探测窗口是连续的，那么请把连续的探测窗口进行合并。\n"
        "3.仅仅在两个载荷不处于同一平面的情况下，切换载荷前需要调姿，如果探测窗口中没有载荷的平面信息，"
        "则不需要规划调姿窗口。"
        "4.我能提供给你的信息只有这些，就基于我提供给你的信息来进行规划，不要再找我提出其他的需求！"
    )
def build_l2_prompt(
    detect_windows: Dict[str, Any],
    science_obj: List[Dict[str, Any]],
    max_power: float,
) -> str:
    return (
        "请帮我规划探测器各个载荷的探测窗口。"
        "我会给你提供各个载荷的可探测窗口和科学目标的相关信息。\n"
        f"各个载荷的可探测窗口是：{detect_windows},其中plane表示载荷所处平面，"
        "power_consumption表示载荷的功耗。"
        f"科学目标相关信息为：{science_obj}，"
        "请对满足实现条件的科学目标进行规划，也就是说，如果某科学目标的载荷探测需求时长超出了可探测窗口，"
        "那么可以选择不实现这一科学目标。"
        "可实现科学目标、探测载荷规划结果和调姿结果用JSON格式进行结果输出,具体格式为："
        '{"achieved_objectives" : ['
        '\n "可实现科学目标","可实现科学目标","可实现科学目标","..."'
        "],  "
        '"observation_plan": [ '
        '{"name":"载荷名称",\n '
        '"start_time":"开始探测时间",\n '
        '"end_time":"结束探测时间",\n '
        '"duration":"探测持续时间（end_time与start_time的时间差）"}, \n'
        '{"name":"载荷名称",\n '
        '"start_time":"开始探测时间",\n '
        '"end_time":"结束探测时间",\n '
        '"duration":"探测持续时间（end_time与start_time的时间差）"}],'
        '"adjustment_plan": ['
        '{"name":"第1/2/3次调姿",\n '
        '"start_time":"开始调姿时间",\n '
        '"end_time":"结束调姿时间",\n '
        '"duration":"调姿持续时间（end_time与start_time的时间差）"}, \n'
        "]  }"
        "其中持续时间的单位为秒。"
        "请注意：1.除了我需要的规划结果json文件以外，其他内容的输出都不要使用json格式。\n"
        f"2.载荷探测的总功耗约束是{max_power}w，也就是说总功耗大于{max_power}w的载荷不能同时进行观测！\n"
        "3.如果载荷的探测窗口是连续的，那么请把连续的探测窗口进行合并。\n"
        "4.仅仅在两个载荷不处于同一平面的情况下，切换载荷前需要调姿，如果探测窗口中没有载荷的平面信息，"
        "则不需要规划调姿窗口。"
        "5.我能提供给你的信息只有这些，就基于我提供给你的信息来进行规划，不要再找我提出其他的需求！"
    )

SYSTEM_PROMPT = '你是一个专用于小天体深空探测任务观测方案制定的助手，负责帮助用户规划深空探测任务的科学观测流程，并协助解决观测方案制定过程中的相关问题。\n在深空探测器对目标天体开展探测时，不同探测载荷具有各自的可观测时间窗口。为实现多个科学目标，需要使用指定载荷采集满足时长要求的探测数据。因此，你需要根据用户提供的载荷可观测窗口、科学目标需求、载荷安装平面、功率约束及其他任务约束，制定合理的观测方案。\n用户可能提供以下信息：\n1. 各探测载荷的可观测时间窗口；\n2. 各探测载荷的功率；\n3. 最大允许工作功率上限；\n4. 各载荷的安装平面信息；\n5. 调姿时间；\n6. 科学目标信息，包括科学目标名称、所需载荷，以及各载荷对应的所需观测时长。\n制定观测方案时，必须遵守以下规则：\n1. 在多科学目标探测任务中，同一探测载荷获取的数据可被多个科学目标共享使用。\n例如：\n科学目标A需要“中视场相机”观测300分钟；\n科学目标B需要“中视场相机”观测500分钟。\n若实际安排“中视场相机”观测500分钟，则科学目标A和科学目标B均可实现，因为500分钟的观测数据已覆盖科学目标A的300分钟需求。\n若该载荷的可观测窗口仅有400分钟，则科学目标B无法实现，但科学目标A仍可实现。\n2. 对同一载荷进行观测时，应尽量安排连续观测，避免将观测时间分散到多个不连续窗口中。\n3. 调姿约束\n若相邻观测使用的两个载荷不位于同一安装平面，例如“中视场相机”位于-Z面，“探测雷达”位于-X面，则必须在两段观测之间插入调姿时间窗口。\n调姿窗口须完整位于切换前后的观测之间，持续时间大于零，不设最小时长。\n4. 如果用户未提供各载荷的安装平面信息，则无需考虑调姿时间，也不需要在观测方案中插入调姿窗口。\n5. 在调姿时间窗口内，所有载荷均不可进行科学观测。\n6. 用户会提供每个载荷的工作功率。\n如果用户提供了最大功率上限，则任意时刻同时工作的所有载荷功率之和不得超过该上限。\n若多个载荷可同时观测，只有在其总功率不超过最大功率上限时，才允许并行安排观测；否则应调整为分时观测或重新规划。\n若用户未提供最大功率上限，则无需进行功率约束校验。\n7. 为载荷分配观测时间时，应严格匹配对应科学目标所需的观测时长，不得超出需求时长。\n例如，若某项观测需求为3600秒，则应从观测起始时刻开始，精确安排至满足3600秒为止，不得额外延长。\n需要注意：若同一载荷的数据被多个科学目标共享，则该载荷只需安排满足最高需求的观测时长，不需要将多个目标的需求时长累加。\n8. 所有输入和输出中的时间信息均应统一使用以下格式：%Y-%m-%d %H:%M:%S。\n示例：2026-05-26 00:21:00。\n输出方案中不得出现“秒”“分钟”“小时”等时间单位描述，所有时间均应以完整时间戳表示。\n'

def evaluate(invoke=None, dataset_path=None, output_root=None, run_id=None, resume=None, level=None):
    if invoke is None:
        validate_api()
    level = LEVEL if level is None else level
    if level not in {'l1', 'l2'}:
        raise ValueError('LEVEL 必须为 l1 或 l2')
    inputs = load_scenario_inputs(dataset_path or DATASET_PATH, [level])
    items = []
    for index, number in enumerate(SCENARIOS, 1):
        objectives, windows, power = inputs[(level, number)]
        items.append(sample(Path(dataset_path or DATASET_PATH) / number / level / f'object_{number}.json',
            index, number, objectives=objectives, windows=windows, power_limit=power))
    path = Path(output_root or ROOT / 'outputs') / 'observation' / (run_id or RUN_ID) / level / model_directory(MODEL_NAME)
    ctx = open_run('observation_' + level, items, {'level': level, 'system_prompt': SYSTEM_PROMPT},
                   output_root, run_id, resume, path=path)
    if invoke is None:
        llm = create_llm()
        def invoke(item):
            prompt = build_l1_prompt(item['windows'], item['objectives']) if level == 'l1' else build_l2_prompt(
                item['windows'], item['objectives'], item['power_limit'])
            return llm.invoke([('system', SYSTEM_PROMPT), ('human', prompt)])
    try:
        for item in items:
            target = ctx['path'] / f'{item["original_id"]}.json'
            completed_before = done(ctx, item)
            try:
                process(ctx, item, lambda: invoke(item), parse_plan)
            finally:
                rec = ctx['records'].get(item['sample_id'])
                if rec is not None and not (completed_before and target.exists()):
                    received = [x for x in ctx['requests'] if x['request_id'] == item['sample_id'] and x['status'] == 'received']
                    raw = received[-1]['raw'] if received else None
                    plan = rec['parsed']
                    if not isinstance(plan, dict):
                        try:
                            value = json_answer(raw)
                            plan = value if isinstance(value, dict) else {}
                        except InvalidResponse:
                            plan = {}
                    _write(target, {'level': level, 'model_name': MODEL_NAME, 'data_num': item['original_id'],
                        'generated': rec['status'] == 'ok', 'reason': rec['status'], 'plan': plan,
                        'metrics': {}, 'raw_response': raw, 'power_limit_w': item['power_limit'], 'error': rec.get('error')})
    finally:
        result = generation_summary(ctx)
        print_summary(result)
    return result


def main():
    evaluate()


if __name__ == '__main__':
    main()
