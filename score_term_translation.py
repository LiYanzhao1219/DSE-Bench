from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set

ROOT = Path(__file__).resolve().parent
import unicodedata
KIND = 'translation'
RUN_ID = 'run01'
RESUME = True
INPUT_PATH = ROOT / 'outputs/term_translation/run01/qwen3-max/answers.xlsx'
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


HYPHEN_VARIANTS = [
    "-",
    "\u2010",
    "\u2011",
    "\u2012",
    "\u2013",
    "\u2014",
    "\u2212",
]
STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}
ARTICLES = {"a", "an", "the"}
ACRONYM_ALIASES: Dict[str, Set[str]] = {
    "arrayoflowenergyxrayimagingsensors": {"alexis"},
    "interplanetarykitecraftacceleratedbyradiationofthesun": {"ikaros"},
    "mercurysurfacespaceenvironmentgeochemistryandranging": {"messenger"},
    "mobileasteroidsurfacescout": {"mascot"},
    "outstandingmoonexplorationtechnologiesdemonstratedbynanosemihardimpactor": {
        "omotenashi"
    },
}
def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = unicodedata.normalize("NFKC", text).lower()

    for hyphen in HYPHEN_VARIANTS:
        text = text.replace(hyphen, " ")

    text = text.replace("&", " and ")
    text = re.sub(r"[''`]", "", text)
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def compact_text(value: Any) -> str:
    return re.sub(r"[^0-9a-z]+", "", normalize_text(value))

def split_references(value: Any) -> List[str]:
    if value is None:
        return []
    return [part.strip() for part in str(value).splitlines() if part.strip()]

def split_answers(value: Any) -> List[str]:
    if value is None:
        return []

    answers = []
    for part in str(value).splitlines():
        stripped = part.strip()
        if stripped:
            answers.append(stripped)
    return answers


def singularize_token(token: str) -> str:
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith(
        ("ss", "us", "is", "ics")
    ):
        return token[:-1]
    return token

def content_tokens(value: Any, *, remove_stopwords: bool = True) -> List[str]:
    tokens = normalize_text(value).split()
    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]
    return [singularize_token(token) for token in tokens]

def content_token_set(value: Any) -> Set[str]:
    return set(content_tokens(value))

def acronym_candidates(reference: str) -> Set[str]:
    tokens = normalize_text(reference).split()
    candidates = set()

    if tokens:
        candidates.add("".join(token[0] for token in tokens))

    without_articles = [token for token in tokens if token not in ARTICLES]
    if without_articles:
        candidates.add("".join(token[0] for token in without_articles))

    without_stopwords = [token for token in tokens if token not in STOPWORDS]
    if without_stopwords:
        candidates.add("".join(token[0] for token in without_stopwords))

    compact_reference = compact_text(reference)
    candidates.update(ACRONYM_ALIASES.get(compact_reference, set()))
    return {candidate for candidate in candidates if candidate}

def is_short_acronym_like(value: str) -> bool:
    compact = compact_text(value)
    if len(compact) < 2 or len(compact) > 16:
        return False

    stripped = unicodedata.normalize("NFKC", value).strip()
    letters = [char for char in stripped if char.isalpha()]
    return bool(letters) and all(char.upper() == char for char in letters)

def reference_matches_answer(reference: str, answer: str) -> bool:
    normalized_reference = normalize_text(reference)
    normalized_answer = normalize_text(answer)

    if not normalized_reference or not normalized_answer:
        return False

    if normalized_reference == normalized_answer:
        return True

    if compact_text(reference) == compact_text(answer):
        return True

    reference_tokens = content_token_set(reference)
    answer_tokens = content_token_set(answer)
    if reference_tokens and reference_tokens == answer_tokens:
        return True

    answer_compact = compact_text(answer)
    if is_short_acronym_like(answer) and answer_compact in acronym_candidates(reference):
        return True

    return False

def is_correct_answer(references: Sequence[str], answer_value: Any) -> bool:
    for answer in split_answers(answer_value):
        for reference in references:
            if reference_matches_answer(reference, answer):
                return True
    return False

def load_answer_statuses(input_path, status_path, rows):
    statuses = json.loads(Path(status_path).read_text(encoding='utf-8'))
    workbook_hash = hashlib.sha256(Path(input_path).read_bytes()).hexdigest()
    if statuses.get('protocol_version') == 2:
        if statuses.get('input_sha256') != workbook_hash:
            raise ValueError('回答工作簿与状态侧文件的 SHA256 不一致')
        return statuses['models']
    manifest = json.loads((Path(status_path).parent / 'run.json').read_text(encoding='utf-8'))
    expected_task = 'term_translation'
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
        expected = [item['original_id'], item['input']['zh'], '\n'.join(item['input']['en'])]
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
            if not split_references(reference):
                raise ValueError(f'第 {excel_row} 行缺少翻译参考答案')
            status = statuses[model].get(str(excel_row), 'unprocessed')
            if status not in {'ok', 'invalid_response', 'api_error', 'config_error', 'execution_error', 'unprocessed'}:
                raise ValueError('未知回答状态')
            answer = row[column] if column < len(row) else None
            if status == 'ok' and (answer is None or not str(answer).strip() or str(answer).startswith('ERROR')):
                raise ValueError('ok 状态与空白或 ERROR 回答矛盾')
            counts[status] += 1
            correct = None
            if status in {'ok', 'invalid_response'}:
                correct = False if status == 'invalid_response' else is_correct_answer(split_references(reference), answer)
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
