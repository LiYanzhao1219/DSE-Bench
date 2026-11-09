from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parent
import unicodedata
KIND = 'abbreviation'
RUN_ID = 'run01'
RESUME = True
INPUT_PATH = ROOT / 'outputs/concept_words/run01/qwen3-max/answers.xlsx'
STATUS_PATH = None


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


ERROR_PREFIXES = (
    "error:",
    "exception:",
    "traceback",
    "api error",
    "request error",
    "request failed",
    "failed:",
    "timeout",
    "time out",
    "connectionerror",
    "rate limit",
    "报错",
    "错误:",
    "api请求错误",
)
WORD_RE = re.compile(r"[A-Za-z0-9]+")
CJK_RE = re.compile(r"[\u3400-\u9fff]+")
SPLIT_CANDIDATE_RE = re.compile(r"[\r\n;；|]+|\s+/\s+")
ASCII_KEEP_RE = re.compile(r"[^A-Za-z0-9\s:：,\.\-\_'\(\)\[\]/&]+")
CHINESE_PUNCT_RE = re.compile(r"[\s\u3000,，.。;；:：!！?？'\"“”‘’`、/\\|\-_\(\)（）\[\]【】<>《》]+")
def as_text(value: Any) -> Tuple[str, str]:
    if value is None:
        return "", "missing_answer"
    if not isinstance(value, str):
        value = str(value)
    text = value.strip()
    if not text:
        return "", "empty_answer"
    if is_error_text(text):
        return text, "error_answer"
    return text, "ok"

def is_error_text(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith(ERROR_PREFIXES)

def normalize_unicode(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return (
        text.replace("–", "-")
        .replace("—", "-")
        .replace("−", "-")
        .replace("‐", "-")
        .replace("‑", "-")
    )

def normalize_key(text: str) -> str:
    return "".join(token.lower() for token in WORD_RE.findall(normalize_unicode(text)))

def normalize_chinese_key(text: str) -> str:
    return CHINESE_PUNCT_RE.sub("", normalize_unicode(text)).strip()

def strip_repeated_abbr(tokens: List[str], abbr: str) -> List[str]:
    abbr_key = normalize_key(abbr)
    while tokens and normalize_key(tokens[0]) == abbr_key:
        tokens = tokens[1:]
    while tokens and normalize_key(tokens[-1]) == abbr_key:
        tokens = tokens[:-1]
    return tokens

def strip_explanatory_prefix(tokens: List[str], abbr: str) -> List[str]:
    if not tokens:
        return tokens

    lowered = [token.lower() for token in tokens]
    abbr_key = normalize_key(abbr)

    if lowered[:2] == ["stands", "for"]:
        return tokens[2:]
    if lowered[:1] in (["means"], ["is"]):
        return tokens[1:]
    if len(tokens) >= 3 and normalize_key(tokens[0]) == abbr_key and lowered[1:3] == ["stands", "for"]:
        return tokens[3:]
    if len(tokens) >= 2 and normalize_key(tokens[0]) == abbr_key and lowered[1] in {"means", "is"}:
        return tokens[2:]
    if len(tokens) >= 3 and lowered[:3] == ["the", "full", "name"]:
        rest = tokens[3:]
        if rest and rest[0].lower() in {"is", "of"}:
            rest = rest[1:]
        return rest
    return tokens

def tokens_to_candidate(tokens: Sequence[str]) -> str:
    return " ".join(tokens).strip()

def extract_english_candidates(value: Any, abbr: str) -> Tuple[List[str], str]:
    text, status = as_text(value)
    if status != "ok":
        return [], status

    normalized_text = normalize_unicode(text)
    candidates: List[str] = []

    for raw_part in SPLIT_CANDIDATE_RE.split(normalized_text):
        ascii_part = ASCII_KEEP_RE.sub(" ", raw_part)
        tokens = WORD_RE.findall(ascii_part)
        tokens = strip_repeated_abbr(tokens, abbr)
        tokens = strip_explanatory_prefix(tokens, abbr)
        tokens = strip_repeated_abbr(tokens, abbr)

        candidate = tokens_to_candidate(tokens)
        if candidate:
            candidates.append(candidate)

    unique_candidates: List[str] = []
    seen = set()
    for candidate in candidates:
        key = normalize_key(candidate)
        if key and key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    if not unique_candidates:
        return [], "no_english_candidate"
    return unique_candidates, "ok"

def extract_chinese_candidates(value: Any) -> Tuple[List[str], str]:
    text, status = as_text(value)
    if status != "ok":
        return [], status

    normalized_text = normalize_unicode(text)
    candidates: List[str] = []

    for raw_part in SPLIT_CANDIDATE_RE.split(normalized_text):
        candidate = "".join(CJK_RE.findall(raw_part))
        candidate = normalize_chinese_key(candidate)
        if candidate:
            candidates.append(candidate)

    unique_candidates: List[str] = []
    seen = set()
    for candidate in candidates:
        if candidate not in seen:
            seen.add(candidate)
            unique_candidates.append(candidate)

    if not unique_candidates:
        return [], "no_chinese_candidate"
    return unique_candidates, "ok"

def compare_answer(reference_value: Any, answer_value: Any, abbr: str) -> Dict[str, Any]:
    reference_candidates, reference_status = extract_english_candidates(reference_value, abbr)
    answer_candidates, answer_status = extract_english_candidates(answer_value, abbr)
    reference_chinese_candidates, reference_chinese_status = extract_chinese_candidates(reference_value)
    answer_chinese_candidates, answer_chinese_status = extract_chinese_candidates(answer_value)

    reference_english = reference_candidates[0] if reference_candidates else ""
    reference_key = normalize_key(reference_english)
    answer_keys = [normalize_key(candidate) for candidate in answer_candidates]
    reference_chinese = reference_chinese_candidates[0] if reference_chinese_candidates else ""
    reference_chinese_key = normalize_chinese_key(reference_chinese)
    answer_chinese_keys = [normalize_chinese_key(candidate) for candidate in answer_chinese_candidates]

    english_correct = (
        reference_status == "ok"
        and bool(reference_key)
        and answer_status == "ok"
        and len(answer_keys) == 1
        and answer_keys[0] == reference_key
    )
    chinese_correct = (
        reference_chinese_status == "ok"
        and bool(reference_chinese_key)
        and answer_chinese_status == "ok"
        and len(answer_chinese_keys) == 1
        and answer_chinese_keys[0] == reference_chinese_key
    )

    if english_correct:
        is_correct = True
        reason = "english_correct"
        match_part = "english"
    elif chinese_correct:
        is_correct = True
        reason = "chinese_correct"
        match_part = "chinese"
    else:
        is_correct = False
        match_part = ""
        if reference_status != "ok" or not reference_key:
            reason = "missing_reference_english"
        elif answer_status != "ok":
            reason = answer_status
        elif len(answer_keys) != 1:
            reason = "multiple_english_candidates"
        else:
            reason = "english_and_chinese_mismatch"

    return {
        "is_correct": is_correct,
        "reason": reason,
        "match_part": match_part,
        "reference_english": reference_english,
        "answer_english_candidates": answer_candidates,
        "candidate_count": len(answer_candidates),
        "normalized_answer": answer_keys[0] if len(answer_keys) == 1 else "",
        "reference_chinese": reference_chinese,
        "answer_chinese_candidates": answer_chinese_candidates,
        "chinese_candidate_count": len(answer_chinese_candidates),
        "normalized_chinese_answer": answer_chinese_keys[0] if len(answer_chinese_keys) == 1 else "",
        "english_correct": english_correct,
        "chinese_correct": chinese_correct,
    }

def load_answer_statuses(input_path, status_path, rows):
    statuses = json.loads(Path(status_path).read_text(encoding='utf-8'))
    workbook_hash = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    if statuses.get('protocol_version') == 2:
        if statuses.get('input_sha256') != workbook_hash:
            raise ValueError('回答工作簿与状态侧文件的 SHA256 不一致')
        return statuses['models']
    manifest = json.loads((Path(status_path).parent / 'run.json').read_text(encoding='utf-8'))
    expected_task = 'concept_words'
    if manifest.get('protocol_version') != 2 or manifest.get('task') != expected_task:
        raise ValueError('状态记录不是对应任务的生成记录')
    items = manifest['samples']
    if len(items) != len(rows) - 1 or len(statuses) != len(items):
        raise ValueError('回答与生成记录的题数不一致，尚未完成导出或记录缺失')
    by_row = {}
    for excel_row, (item, row) in enumerate(zip(items, rows[1:]), 2):
        rec = statuses.get(item['sample_id'])
        if rec is None:
            raise ValueError('生成记录缺少样本')
        expected = [item['field'], item['input'], item['reference']]
        if any(str(a) != str(b) for a, b in zip(row[:3], expected)):
            raise ValueError('回答表的题目或参考答案与生成记录不一致')
        if rec['status'] == 'ok' and row[3] != rec['parsed']:
            raise ValueError('回答内容与生成记录不一致')
        by_row[str(excel_row)] = rec['status']
    if list(rows[0][3:]) != [manifest['model']]:
        raise ValueError('模型列与生成记录不一致')
    return {manifest['model']: by_row}


def evaluate(input_path=None, status_path=None, output_root=None, run_id=None, resume=None):
    from openpyxl import load_workbook, Workbook
    input_path = Path(input_path or INPUT_PATH)
    status_path = Path(status_path) if status_path is not None else (Path(STATUS_PATH) if STATUS_PATH else input_path.parent / 'samples.json')
    if not status_path.exists():
        raise ValueError('评分需要明确的回答状态记录。请设置 STATUS_PATH；不从历史 ERROR 字符串推断失败类型。')
    book = load_workbook(input_path, read_only=True, data_only=True)
    try:
        rows = list(book.active.iter_rows(values_only=True))
    finally:
        book.close()
    if len(rows) < 2 or len(rows[0]) < 4:
        raise ValueError('回答表至少四列：序号/领域、题目、参考答案、模型回答')
    models = [str(x) for x in rows[0][3:]]
    statuses = load_answer_statuses(input_path, status_path, rows)
    if len(models) != len(set(models)) or set(models) != set(statuses):
        raise ValueError('模型列与状态记录不一致')
    run_id = RUN_ID if run_id is None else run_id
    if not re.fullmatch(r'[\w.-]+', run_id) or run_id in {'.', '..'}:
        raise ValueError('RUN_ID 必须是单个目录名')
    output = Path(output_root or ROOT / 'outputs') / (KIND + '_score') / run_id / 'all_models'
    metadata = {'input_path': str(input_path.resolve()), 'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
        'status_sha256': hashlib.sha256(status_path.read_bytes()).hexdigest(),
        'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    previous = _read(output / 'run.json', None)
    if previous is not None and (not (RESUME if resume is None else resume) or previous != metadata):
        raise ValueError('评分输入或代码改变，请使用新的 RUN_ID')
    if previous is None and output.exists() and any(output.iterdir()):
        raise ValueError('输出目录已存在且缺少清单，请使用新的 RUN_ID')
    _write(output / 'run.json', metadata)
    results, details = {}, []
    for column, model in enumerate(models, 3):
        counts, values = Counter(), []
        for excel_row, row in enumerate(rows[1:], 2):
            reference, term = row[2], row[1]
            if not term or not reference:
                raise ValueError(f'第 {excel_row} 行缺少缩略词或参考答案')
            status = statuses[model].get(str(excel_row), 'unprocessed')
            if status not in {'ok', 'invalid_response', 'api_error', 'config_error', 'execution_error', 'unprocessed'}:
                raise ValueError('未知回答状态')
            answer = row[column] if column < len(row) else None
            if status == 'ok' and (answer is None or not str(answer).strip() or str(answer).startswith('ERROR')):
                raise ValueError('ok 状态与空白或 ERROR 回答矛盾')
            counts[status] += 1
            correct = None
            if status in {'ok', 'invalid_response'}:
                correct = False if status == 'invalid_response' else compare_answer(reference, answer, str(term))['is_correct']
                values.append(float(correct))
            details.append({'excel_row': excel_row, 'model': model, 'term': term, 'reference': reference,
                            'answer': answer, 'status': status, 'accuracy': float(correct) if correct is not None else None})
        complete = len(values) == len(rows) - 1
        accuracy = sum(values) / (len(rows) - 1) if complete else None
        results[model] = {'status': 'complete' if complete else 'incomplete', 'sample_count': len(rows) - 1,
            'scored_count': len(values), 'status_counts': dict(counts), 'accuracy': accuracy,
            'score_pct': 100 * accuracy if complete else None}
    _write(output / 'summary.json', results)
    _write(output / 'detail_scores.json', details)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = 'summary'
    sheet.append(['model', 'status', 'sample_count', 'scored_count', 'accuracy', 'score_pct'])
    for model, result in results.items():
        sheet.append([model, *[result[key] for key in ['status', 'sample_count', 'scored_count', 'accuracy', 'score_pct']]])
    sheet = workbook.create_sheet('detail_scores')
    headers = ['excel_row', 'model', 'term', 'reference', 'answer', 'status', 'accuracy']
    sheet.append(headers)
    for item in details:
        sheet.append([item[key] for key in headers])
    temporary = output / 'scores.tmp.xlsx'
    workbook.save(temporary)
    temporary.replace(output / 'scores.xlsx')
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def main():
    evaluate()


if __name__ == '__main__':
    main()
