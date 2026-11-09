from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

ROOT = Path(__file__).resolve().parent

from datetime import datetime
import unicodedata
RUN_ID = 'run01'
SCORING_VERSION = 'paper_planning_fixed20_powerfile_v3'
METRIC_KEYS = ('gen_success', 'constraint_ratio_first', 'objective_ratio_first', 'payload_avg_ratio_first', 'weighted_score')
SCORE_KEYS = METRIC_KEYS[1:]
SCENARIOS = tuple(f'{i:02d}' for i in range(1, 21))
INPUT_PATH = ROOT / 'outputs/observation' / RUN_ID
DATASET_PATH = ROOT / 'data/task/dataset'
OUTPUT_PATH = ROOT / 'outputs/observation_score' / RUN_ID
MODEL_NAMES = None
DEFAULT_PLAN_ROOT = INPUT_PATH
DEFAULT_DATASET_ROOT = DATASET_PATH
class InvalidResponse(ValueError):
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


def _write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(serializable(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def archive_model_directory(model):
    name = re.sub(r'[\\/:*?"<>|\s]+', '_', model).strip('._') or 'model'
    if name != model:
        name += '-' + digest(model)[:8]
    return name


def _result(errors):
    return {'valid': not errors, 'errors': errors}

def _entries(plan, group, errors):
    result = []
    for index, entry in enumerate(plan.get(group, []), 1):
        try:
            start, end = parse_timestamp(entry['start_time']), parse_timestamp(entry['end_time'])
            name = canonical_payload_name(entry.get('name', ''))
        except (KeyError, TypeError, ValueError, AttributeError):
            errors.append(f'{group} record {index}: invalid timestamp/entry')
            continue
        if start >= end:
            errors.append(f'{group} record {index}: start must precede end')
            continue
        result.append((name, start, end, index))
    return result

def _payloads(windows):
    result = {}
    for name, info in windows.items():
        key = canonical_payload_name(name)
        if key in result:
            raise ValueError('Ambiguous payload names in input windows')
        result[key] = info
    return result

def check_content_constraints(plan, science_obj, detect_windows):
    errors = []
    observations = _entries(plan, 'observation_plan', errors)
    adjustments = _entries(plan, 'adjustment_plan', errors)
    if not observations:
        errors.append('No positive-duration observation')
    payloads = _payloads(detect_windows)
    seen = {}
    for name, start, end, index in observations:
        if name not in payloads:
            errors.append(f'observation_plan record {index}: undefined payload {name}')
        else:
            windows = merged_intervals((parse_timestamp(a), parse_timestamp(b))
                                       for a, b in payloads[name].get('observation_windows', []))
            if not any(a <= start and end <= b for a, b in windows):
                errors.append(f'observation_plan record {index}: outside windows for {name}')
        for a, b, previous in seen.setdefault(name, []):
            if max(a, start) < min(b, end):
                errors.append(f'observation_plan records {previous},{index}: overlapping payload {name}')
        seen[name].append((start, end, index))
        for _, a, b, adj_index in adjustments:
            if max(a, start) < min(b, end):
                errors.append(f'observation_plan record {index}: overlaps adjustment record {adj_index}')
    return _result(errors)

def check_attitude_constraints(plan, detect_windows):
    errors = []
    observations = _entries(plan, 'observation_plan', errors)
    adjustments = _entries(plan, 'adjustment_plan', errors)
    payloads = _payloads(detect_windows)
    for name, _, _, index in observations:
        if name not in payloads:
            errors.append(f'observation_plan record {index}: undefined payload {name}')
    for _, a, b, adj_index in adjustments:
        for _, start, end, index in observations:
            if max(a, start) < min(b, end):
                errors.append(f'adjustment record {adj_index}: overlaps observation record {index}')
    points = sorted({t for _, a, b, _ in observations for t in (a, b)})
    previous_plane = previous_end = None
    for start, end in zip(points, points[1:]):
        active = [(name, index) for name, a, b, index in observations if a < end and b > start]
        planes = {str(payloads.get(name, {}).get('plane') or 'Unknown').strip() for name, _ in active}
        known_planes = planes - {'Unknown', ''}
        if len(known_planes) > 1:
            errors.append(f'different mounting planes overlap at {start}: records {[i for _, i in active]}')
            continue
        if not active:
            continue
        if not known_planes:
            previous_plane = previous_end = None
            continue
        current_plane = next(iter(known_planes))
        if previous_plane is not None and current_plane != previous_plane:
            if not any(previous_end <= a < b <= start for _, a, b, _ in adjustments):
                errors.append(f'missing adjustment between {previous_plane} and {current_plane} before {start}')
        previous_plane, previous_end = current_plane, end
    return _result(errors)

def check_power_consumption(plan, detect_windows, power_limit=None):
    errors = []
    observations = _entries(plan, 'observation_plan', errors)
    payloads = _payloads(detect_windows)
    powers = {}
    for name, info in payloads.items():
        raw = info.get('power_consumption')
        if raw is None:
            continue
        power = float(raw)
        if not math.isfinite(power) or power < 0:
            raise ValueError(f'Invalid input power for {name}')
        powers[name] = power
    if not powers:
        return _result(errors)
    if power_limit is None:
        raise ValueError('Explicit power limit required for payload power data')
    limit = float(power_limit)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError('Invalid input power limit')
    for name, _, _, index in observations:
        if name not in powers:
            errors.append(f'observation_plan record {index}: missing power for {name}')
    points = sorted({t for _, a, b, _ in observations for t in (a, b)})
    for start, end in zip(points, points[1:]):
        active = {name for name, a, b, _ in observations if a < end and b > start}
        power = sum(powers.get(name, 0) for name in active)
        if power > limit + 1e-9:
            errors.append(f'power {power} exceeds {limit} at {start}: {sorted(active)}')
    return _result(errors)
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

def archive_model_directory(model_name):
    return re.sub(r'[\\/:*?"<>|\s]+', '_', model_name).strip('_') or 'unknown_model'

def canonical_payload_name(name):
    return re.sub(r'[\s\u200b\u200c\u200d\ufeff]+', '', unicodedata.normalize('NFKC', str(name)))

def parse_timestamp(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', value):
        raise ValueError('Expected YYYY-MM-DD HH:MM:SS timestamp')
    return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')

def merged_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged

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

def empty_metrics():
    return dict.fromkeys(METRIC_KEYS, 0.0)

def score_plan(plan, science_obj, detect_windows, power_limit=None, *, engineering_required=False):
    if power_limit is None and (engineering_required or any(
            info.get('power_consumption') is not None for info in detect_windows.values())):
        raise ValueError('An explicit power limit is required when power data are supplied')
    plan = parse_plan(plan)
    canonical_windows = {}
    for name, info in detect_windows.items():
        key = canonical_payload_name(name)
        if key in canonical_windows:
            raise ValueError('Ambiguous payload names in input windows')
        if engineering_required and (not info.get('plane') or info.get('power_consumption') is None):
            raise ValueError('L2 payload requires mounting plane and power')
        canonical_windows[key] = info
    for entry in plan['observation_plan']:
        entry['name'] = canonical_payload_name(entry['name'])
    diagnostics = []
    for group in ('observation_plan', 'adjustment_plan'):
        for index, entry in enumerate(plan[group], 1):
            actual = (parse_timestamp(entry['end_time']) - parse_timestamp(entry['start_time'])).total_seconds()
            try:
                declared = float(entry['duration'])
                mismatch = not math.isfinite(declared) or abs(declared - actual) > 1e-9
            except (KeyError, ValueError, TypeError, OverflowError):
                mismatch = True
            if mismatch:
                diagnostics.append({'kind': 'duration_mismatch', 'group': group, 'record': index,
                                    'payload': entry.get('name'), 'declared': entry.get('duration'),
                                    'timestamp_seconds': actual})
    checks = {
        'content': check_content_constraints(plan, science_obj, canonical_windows),
        'attitude': check_attitude_constraints(plan, canonical_windows),
        'power': check_power_consumption(plan, canonical_windows, power_limit),
    }
    constraint = float(all(result['valid'] for result in checks.values()))
    objective = calculate_objective_completion_ratio(plan, science_obj)
    payload = calculate_average_payload_requirement_completion_ratio(plan, science_obj)
    metrics = dict(gen_success=1.0, constraint_ratio_first=constraint,
                   objective_ratio_first=objective, payload_avg_ratio_first=payload,
                   weighted_score=(constraint + objective + payload) / 3)
    return metrics, {'checks': checks, 'duration_diagnostics': diagnostics}

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

def observation_duration_by_payload(schedule: Mapping[str, Any]) -> Dict[str, float]:
    intervals = {}
    observation_plan = schedule.get("observation_plan", [])
    if not isinstance(observation_plan, list):
        raise ValueError("Schedule field 'observation_plan' must be a list.")

    for item in observation_plan:
        payload_name = canonical_payload_name(item['name'])
        intervals.setdefault(payload_name, []).append((parse_timestamp(item['start_time']),
                                                       parse_timestamp(item['end_time'])))
    return {name: sum((end-start).total_seconds() for start, end in merged_intervals(spans))
            for name, spans in intervals.items()}

def evaluate_schedule_metrics(
    objectives: Sequence[Mapping[str, Any]],
    schedule: Mapping[str, Any],
) -> Dict[str, Any]:
    observed_by_payload = observation_duration_by_payload(schedule)
    objective_details = []

    completed_objective_count = 0
    total_required_payload_items = 0
    covered_required_payload_items = 0
    total_required_seconds = 0.0
    covered_required_seconds = 0.0

    for objective in objectives:
        name = _objective_name(objective)
        requirements = _merge_objective_requirements(objective)

        requirement_details = []
        objective_required_seconds = 0.0
        objective_covered_seconds = 0.0
        objective_payload_count = len(requirements)
        objective_covered_payload_count = 0

        for payload_name, required_seconds in requirements.items():
            observed_seconds = observed_by_payload.get(payload_name, 0.0)
            covered_seconds = min(observed_seconds, required_seconds)
            payload_covered = observed_seconds >= required_seconds

            total_required_payload_items += 1
            covered_required_payload_items += int(payload_covered)
            total_required_seconds += required_seconds
            covered_required_seconds += covered_seconds

            objective_required_seconds += required_seconds
            objective_covered_seconds += covered_seconds
            objective_covered_payload_count += int(payload_covered)

            requirement_details.append(
                {
                    "payload": payload_name,
                    "required_duration": required_seconds,
                    "observed_duration": observed_seconds,
                    "covered_duration": covered_seconds,
                    "payload_covered": payload_covered,
                    "duration_coverage_ratio": (
                        1.0 if required_seconds <= 0 else covered_seconds / required_seconds
                    ),
                }
            )

        objective_completed = (
            objective_payload_count > 0
            and objective_covered_payload_count == objective_payload_count
        )
        completed_objective_count += int(objective_completed)

        objective_details.append(
            {
                "name": name,
                "completed": objective_completed,
                "required_payload_count": objective_payload_count,
                "covered_payload_count": objective_covered_payload_count,
                "payload_coverage_ratio": (
                    1.0
                    if objective_payload_count == 0
                    else objective_covered_payload_count / objective_payload_count
                ),
                "required_duration": objective_required_seconds,
                "covered_duration": objective_covered_seconds,
                "duration_coverage_ratio": (
                    1.0
                    if objective_required_seconds <= 0
                    else objective_covered_seconds / objective_required_seconds
                ),
                "requirements": requirement_details,
            }
        )

    objective_count = len(objectives)
    return {
        "summary": {
            "objective_count": objective_count,
            "completed_objective_count": completed_objective_count,
            "objective_completion_ratio": (
                1.0 if objective_count == 0 else completed_objective_count / objective_count
            ),
            "required_payload_item_count": total_required_payload_items,
            "covered_payload_item_count": covered_required_payload_items,
            "payload_coverage_ratio": (
                1.0
                if total_required_payload_items == 0
                else covered_required_payload_items / total_required_payload_items
            ),
            "required_duration": total_required_seconds,
            "covered_duration": covered_required_seconds,
            "duration_coverage_ratio": (
                1.0
                if total_required_seconds <= 0
                else covered_required_seconds / total_required_seconds
            ),
        },
        "objectives": objective_details,
        "observed_by_payload": observed_by_payload,
    }

def calculate_objective_completion_ratio(
    result_plan: Mapping[str, Any],
    science_obj: Sequence[Mapping[str, Any]],
) -> float:
    metrics = evaluate_schedule_metrics(science_obj, result_plan)
    return float(metrics["summary"]["objective_completion_ratio"])

def _payload_required_seconds_for_shared_data(
    science_obj: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    required_by_payload: Dict[str, float] = {}
    for objective in science_obj:
        for payload_name, required_seconds in _merge_objective_requirements(objective).items():
            current_required = required_by_payload.get(payload_name, 0.0)
            required_by_payload[payload_name] = max(current_required, required_seconds)
    return required_by_payload

def calculate_payload_requirement_completion_ratios(
    result_plan: Mapping[str, Any],
    science_obj: Sequence[Mapping[str, Any]],
) -> Dict[str, float]:
    observed_by_payload = observation_duration_by_payload(result_plan)
    required_by_payload = _payload_required_seconds_for_shared_data(science_obj)

    completion_ratios: Dict[str, float] = {}
    for payload_name, required_seconds in required_by_payload.items():
        observed_seconds = observed_by_payload.get(payload_name, 0.0)
        if required_seconds <= 0:
            completion_ratios[payload_name] = 1.0
            continue
        completion_ratios[payload_name] = min(observed_seconds / required_seconds, 1.0)

    return completion_ratios

def calculate_average_payload_requirement_completion_ratio(
    result_plan: Mapping[str, Any],
    science_obj: Sequence[Mapping[str, Any]],
) -> float:
    completion_ratios = calculate_payload_requirement_completion_ratios(result_plan, science_obj)
    if not completion_ratios:
        return 1.0
    return float(sum(completion_ratios.values()) / len(completion_ratios))

def load_scenario_inputs(dataset_root=DEFAULT_DATASET_ROOT):
    inputs = {}
    for level in ('l1', 'l2'):
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

def calculate_saved_metrics(plan_root=None, dataset_root=None, models=None):
    plan_root = Path(plan_root or DEFAULT_PLAN_ROOT)
    models = tuple(discover_models(plan_root) if models is None else models)
    if not models or any(not isinstance(model, str) or not model for model in models):
        raise ValueError('Expected nonempty model names')
    if len(set(models)) != len(models) or len({archive_model_directory(m) for m in models}) != len(models):
        raise ValueError('Duplicate model names or colliding archive directories')
    inputs = load_scenario_inputs(dataset_root or DEFAULT_DATASET_ROOT)
    results = {'scoring_version': SCORING_VERSION, 'missing_answer_policy': 'zero',
               'scenarios_per_level': len(SCENARIOS), 'levels': {}, 'details': []}
    for level in ('l1', 'l2'):
        level_results = []
        for model in models:
            sample_metrics = []
            for number in SCENARIOS:
                path = plan_root / level / archive_model_directory(model) / f'{number}.json'
                metrics, diagnostics, status = empty_metrics(), {}, 'missing_file'
                if path.is_file():
                    try:
                        answer = json.loads(path.read_text(encoding='utf-8'))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        status, diagnostics = 'invalid_json', {'error': str(exc)}
                    else:
                        if not isinstance(answer, dict):
                            status = 'invalid_answer'
                        else:
                            for field, expected in [('level', level), ('model_name', model), ('data_num', number)]:
                                if field in answer and answer[field] != expected:
                                    raise ValueError(f'Archive identity mismatch: {path}: {field}')
                            plan = answer.get('plan')
                            if not plan:
                                plan = answer.get('raw_response')
                            if plan is None or plan == {} or isinstance(plan, str) and not plan.strip():
                                status = 'missing_answer'
                            else:
                                objectives, windows, power_limit = inputs[(level, number)]
                                try:
                                    metrics, diagnostics = score_plan(plan, objectives, windows, power_limit,
                                                                       engineering_required=level == 'l2')
                                    status = 'ok'
                                except InvalidResponse as exc:
                                    status, diagnostics = 'invalid_answer', {'error': str(exc)}
                sample_metrics.append(metrics)
                results['details'].append({'level': level, 'model_name': model, 'data_num': number,
                                           'power_limit_w': inputs[(level, number)][2],
                                           'status': status, 'metrics': {key: metrics[key] for key in SCORE_KEYS},
                                           'diagnostics': diagnostics})
            means = {key: math.fsum(row[key] for row in sample_metrics) / len(SCENARIOS) for key in SCORE_KEYS[:3]}
            means['weighted_score'] = math.fsum(means.values()) / 3
            level_results.append({'model_name': model, 'result': means})
        results['levels'][level] = level_results
    return results

def discover_models(plan_root):
    models = set()
    for level in ('l1', 'l2'):
        folder = Path(plan_root) / level
        for path in sorted(folder.glob('*/*.json')):
            if path.stem not in SCENARIOS:
                continue
            try:
                record = json.loads(path.read_text(encoding='utf-8'))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            if isinstance(record, dict) and isinstance(record.get('model_name'), str):
                models.add(record['model_name'])
    if not models:
        raise ValueError('未发现模型名单，请设置 MODEL_NAMES 或检查 INPUT_PATH')
    return sorted(models)


def export_analysis(analysis, output_path):
    from openpyxl import Workbook
    output_path = Path(output_path)
    if output_path.exists() and any(output_path.iterdir()):
        raise ValueError('评分输出目录已有文件，请选择新的 RUN_ID 或 OUTPUT_PATH')
    output_path.mkdir(parents=True, exist_ok=True)
    book = Workbook()
    by_level = {}
    for index, level in enumerate(('l1', 'l2')):
        by_level[level] = {x['model_name']: x for x in analysis['levels'][level]}
        text = ''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in analysis['levels'][level])
        (output_path / f'{level}_model_metrics.txt').write_text(text, encoding='utf-8')
        sheet = book.active if index == 0 else book.create_sheet()
        sheet.title = level
        sheet.append(['model_name', *SCORE_KEYS])
        for item in analysis['levels'][level]:
            sheet.append([item['model_name'], *[item['result'][k] for k in SCORE_KEYS]])
    sheet = book.create_sheet('combined')
    sheet.append(['model', 'sample_count', *SCORE_KEYS, 'L1_score_pct', 'L2_score_pct', 'score_pct'])
    summaries = []
    for model in by_level['l1']:
        l1, l2 = by_level['l1'][model]['result'], by_level['l2'][model]['result']
        metrics = {key: (l1[key] + l2[key]) / 2 for key in SCORE_KEYS}
        row = {'model': model, 'sample_count': 2 * len(SCENARIOS), 'metrics': metrics,
               'score_pct': 100 * metrics['weighted_score'], 'L1_score': 100 * l1['weighted_score'],
               'L2_score': 100 * l2['weighted_score'], 'scoring_version': SCORING_VERSION, 'missing_answer_policy': 'zero'}
        summaries.append(row)
        sheet.append([model, 2 * len(SCENARIOS), *[metrics[k] for k in SCORE_KEYS], row['L1_score'], row['L2_score'], row['score_pct']])
    book.save(output_path / 'scores.xlsx')
    _write(output_path / 'summary.json', summaries)
    _write(output_path / 'details.json', analysis['details'])
    return summaries


def main():
    analysis = calculate_saved_metrics(INPUT_PATH, DATASET_PATH, MODEL_NAMES)
    result = export_analysis(analysis, OUTPUT_PATH)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
