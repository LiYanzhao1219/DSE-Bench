from __future__ import annotations

import hashlib
import importlib.metadata
import json
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
MAX_TOKENS = None
RUN_ID = 'run01'
RESUME = True
TASK = 'concept_words'
IS_ABBREVIATION = True
DATA_FILES = [
    (ROOT / 'data/knowledge/concept/words.xlsx', True),
]
SYSTEM_PROMPT = '你是一个航天领域专家，下面来进行航天领域的缩略词解释。我会给你缩略词的细分领域，请你给出供该缩略词的英文全称和中文全称。例如，测控领域的VLBI，你应该解释为：“Very Long Baseline Interferometry 甚长基线干涉测量”严格要求：你的回答只能包含缩略词的解释本身，绝不能包含任何打招呼的话语、前缀（如‘解释：’、‘这个词的意思是’）或多余的标点符号。'
USER_PROMPT = '请解释{field}相关缩略词：{noun}'

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


def load_samples(paths=None):
    from openpyxl import load_workbook
    samples = []
    for path, has_header in (DATA_FILES if paths is None else paths):
        path = Path(path)
        book = load_workbook(path, read_only=True, data_only=True)
        try:
            rows = book.active.iter_rows(min_row=2 if has_header else 1, values_only=True)
            for row_number, row in enumerate(rows, 2 if has_header else 1):
                width = 3 if IS_ABBREVIATION else 2
                values = list(row[:width]) + [None] * max(0, width - len(row))
                if all(x is None or not str(x).strip() for x in values):
                    continue
                if any(x is None or not str(x).strip() for x in values):
                    raise ValueError(f'{path} 第 {row_number} 行缺少必要字段')
                if IS_ABBREVIATION:
                    field, noun, reference = map(lambda x: str(x).strip(), values)
                else:
                    noun, reference = map(lambda x: str(x).strip(), values)
                    field = path.stem
                samples.append(sample(path, row_number, input=noun, reference=reference, field=field))
        finally:
            book.close()
    return samples


def create_chain():
    from langchain_core.prompts import ChatPromptTemplate
    return ChatPromptTemplate.from_messages([('system', SYSTEM_PROMPT), ('user', USER_PROMPT)]) | create_llm()


def evaluate(invoke=None, paths=None, output_root=None, run_id=None, resume=None):
    if invoke is None:
        validate_api()
    items = load_samples(paths)
    ctx = open_run(TASK, items, {'system_prompt': SYSTEM_PROMPT, 'user_prompt': USER_PROMPT},
                   output_root, run_id, resume)
    if invoke is None:
        chain = create_chain()
        invoke = lambda item: chain.invoke({'noun': item['input'], 'field': item['field']})
    try:
        for index, item in enumerate(items, 1):
            print(f'[{TASK}] {index}/{len(items)}')
            process(ctx, item, lambda: invoke(item), response_text)
    finally:
        if IS_ABBREVIATION:
            headers = ['领域', '缩写', '解释']
            rows = [[x['field'], x['input'], x['reference']] for x in items]
        else:
            headers = ['名词', '解释', '来源文件']
            rows = [[x['input'], x['reference'], x['source']] for x in items]
        export_answers(ctx, headers, rows, lambda x: x)
        result = generation_summary(ctx)
        print_summary(result)
    return result


def main():
    evaluate()


if __name__ == '__main__':
    main()
