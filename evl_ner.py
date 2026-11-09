from __future__ import annotations

import hashlib
import json
import math
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
DATA_PATH = ROOT / 'data/nlp/ner.json'
MATCH_MODE = 'exact'
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
NEW_ENTITY_LABEL = "新实体"
VALID_MATCH_MODES = {"exact", "restricted_containment", "any_containment"}
OVER_GENERAL_ENTITIES = {"水", "氧", "氢", "氮", "风", "冰", "核", "质量", "形成", "自转", "采样"}
class EntityMention(BaseModel):
    text: str = Field(description="原文中连续出现的实体文本，不包含实体类别。")

class NERResult(BaseModel):
    entities: List[EntityMention] = Field(
        default_factory=list,
        description="实体名称列表；同一实体名称即使在原文中出现多次，也只需输出一次。",
    )

def load_label_studio_data(data_path: Union[str, Path]) -> List[Dict[str, Any]]:
    data_path = Path(data_path)
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{data_path} should contain a JSON list.")
    return data

def extract_reference_labels(data: Sequence[Dict[str, Any]]) -> List[str]:
    labels = set()
    for item in data:
        for ann in item.get("label", []):
            for label in ann.get("labels", []):
                if label != NEW_ENTITY_LABEL:
                    labels.add(label)
    return sorted(labels)

def extract_gold_mentions(item: Dict[str, Any]) -> List[str]:
    mentions = []
    for ann in item.get("label", []):
        text = ann.get("text", "")
        if text:
            mentions.append(text)
    return mentions

def normalize_entity_text(text: str) -> str:
    text = str(text).strip()
    text = re.sub(r"[\u00a0\u3000]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    translation = str.maketrans(
        {
            "（": "(",
            "）": ")",
            "【": "[",
            "】": "]",
            "［": "[",
            "］": "]",
            "｛": "{",
            "｝": "}",
        }
    )
    return text.translate(translation)

def set_from_mentions(mentions: Sequence[str]) -> set:
    return {
        normalized for normalized in (normalize_entity_text(text) for text in mentions) if normalized
    }

def is_containment_match(gold_entity: str, pred_entity: str, match_mode: str) -> bool:
    if gold_entity == pred_entity:
        return True
    if gold_entity not in pred_entity and pred_entity not in gold_entity:
        return False
    if match_mode == "any_containment":
        return True
    if match_mode == "restricted_containment":
        shorter = gold_entity if len(gold_entity) <= len(pred_entity) else pred_entity
        return len(shorter) >= 2 and shorter not in OVER_GENERAL_ENTITIES
    return False

def count_matched_entities(gold_set: set, pred_set: set, match_mode: str) -> int:
    if match_mode not in VALID_MATCH_MODES:
        raise ValueError(f"Unsupported match_mode: {match_mode}")

    exact_matched = gold_set & pred_set
    matched_count = len(exact_matched)
    if match_mode == "exact":
        return matched_count

    remaining_gold = sorted(gold_set - exact_matched, key=lambda entity: (-len(entity), entity))
    remaining_pred = set(pred_set - exact_matched)

    for gold_entity in remaining_gold:
        candidates = [
            pred_entity
            for pred_entity in remaining_pred
            if is_containment_match(gold_entity, pred_entity, match_mode)
        ]
        if not candidates:
            continue
        best_pred = min(candidates, key=lambda entity: (abs(len(entity) - len(gold_entity)), len(entity), entity))
        remaining_pred.remove(best_pred)
        matched_count += 1

    return matched_count

def score_mentions(
    gold_mentions: Sequence[str],
    pred_mentions: Sequence[str],
    match_mode: str = MATCH_MODE,
) -> Dict[str, Any]:
    gold_set = set_from_mentions(gold_mentions)
    pred_set = set_from_mentions(pred_mentions)

    tp = count_matched_entities(gold_set, pred_set, match_mode)
    fp = len(pred_set) - tp
    fn = len(gold_set) - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def build_prompt(text: str, reference_labels: Sequence[str]) -> str:
    label_text = "、".join(reference_labels) if reference_labels else "无"
    return f"""请完成命名实体识别任务。

你可以参考以下实体类别范围，但输出中不要包含类别：{label_text}

硬性要求：
1. 只抽取输入原文中连续出现的实体文本。
2. 同一实体名称即使在原文中出现多次，也只需输出一次。
3. 不要输出标签、解释、推理过程或原文中不存在的文本。
4. 返回结构必须只包含 entities，每个元素只包含 text。

输入文本：
{text}
"""

def build_plain_text_json_prompt(text: str, reference_labels: Sequence[str], retry: bool = False) -> str:
    label_text = "、".join(reference_labels) if reference_labels else "无"
    retry_instruction = ""
    if retry:
        retry_instruction = "上一次输出不是合法 JSON 对象。请修正格式，只输出 JSON 对象。\n\n"
    return f"""{retry_instruction}请完成命名实体识别任务。

你可以参考以下实体类别范围，但输出中不要包含类别：{label_text}

只输出一个 JSON 对象，格式如下：
{{"entities":[{{"text":"实体1"}},{{"text":"实体2"}}]}}

硬性要求：
1. 只输出 JSON 对象，不要输出 Markdown，不要输出解释，不要输出推理过程。
2. 只抽取输入原文中连续出现的实体文本。
3. 同一实体名称即使在原文中出现多次，也只需输出一次。
4. 不要输出标签或原文中不存在的文本。
5. 如果没有实体，输出 {{"entities":[]}}。

输入文本：
{text}
"""

def coerce_ner_result(raw_result: Any) -> NERResult:
    if isinstance(raw_result, NERResult):
        return raw_result
    if isinstance(raw_result, dict):
        entities = raw_result.get("entities")
        if isinstance(entities, list):
            normalized_entities = []
            for entity in entities:
                if isinstance(entity, str):
                    normalized_entities.append({"text": entity})
                else:
                    normalized_entities.append(entity)
            raw_result = {**raw_result, "entities": normalized_entities}
        if hasattr(NERResult, "model_validate"):
            return NERResult.model_validate(raw_result)
        return NERResult.parse_obj(raw_result)
    raise TypeError(f"Unsupported LLM result type: {type(raw_result)!r}")


def create_chain(labels):
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import JsonOutputParser
    llm = create_llm()
    prompt = ChatPromptTemplate.from_messages([('system', '你是一个严格的命名实体识别系统。必须按指定结构输出，不要输出解释。'), ('human', '{input}')])
    if OUTPUT_MODE == 'plain_text_json':
        return prompt | llm
    if OUTPUT_MODE == 'structured':
        try:
            return prompt | llm.with_structured_output(NERResult, include_raw=True)
        except NotImplementedError:
            pass
    parser = JsonOutputParser(pydantic_object=NERResult)
    prompt = ChatPromptTemplate.from_messages([('system', '你是一个严格的命名实体识别系统。必须只输出合法 JSON，不要输出解释。'),
        ('human', '{input}\n\n输出格式说明：\n{format_instructions}')]).partial(format_instructions=parser.get_format_instructions())
    return prompt | llm.bind(response_format={'type': 'json_object'})


def parse_answer(raw):
    value = json_answer(raw)
    if not isinstance(value, dict) or 'entities' not in value:
        raise InvalidResponse('缺少 entities 字段')
    try:
        return coerce_ner_result(value)
    except (TypeError, ValueError) as exc:
        raise InvalidResponse('实体结构不合法') from exc


def evaluate(invoke=None, data_path=None, output_root=None, run_id=None):
    if invoke is None:
        validate_api()
    data = load_label_studio_data(data_path or DATA_PATH)
    labels = extract_reference_labels(data)
    for item in data:
        if not isinstance(item.get('text'), str) or not isinstance(item.get('label'), list):
            raise ValueError('NER 输入必须包含 text 和 label 列表')
        if any(not isinstance(x, dict) or not isinstance(x.get('text'), str) or not isinstance(x.get('labels'), list) for x in item['label']):
            raise ValueError('NER 参考标注不合法')
    if invoke is None:
        chain = create_chain(labels)
        builder = build_plain_text_json_prompt if OUTPUT_MODE == 'plain_text_json' else build_prompt
        invoke = lambda item: chain.invoke({'input': builder(item['text'], labels)})
    def score(answer, item):
        return score_mentions(extract_gold_mentions(item), [x.text for x in answer.entities], MATCH_MODE)
    return evaluate_items({'ner': data}, invoke, parse_answer, score, 'f1', ['precision', 'recall', 'f1'],
                          'ner', output_root, run_id)


def main():
    evaluate()


if __name__ == '__main__':
    main()
