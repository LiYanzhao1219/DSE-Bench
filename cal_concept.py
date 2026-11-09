from __future__ import annotations
import hashlib
import json
import math


import re

import time

from pathlib import Path

from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parent
RUN_ID = 'run01'
INPUT_PATH = ROOT / 'outputs/concept/run01/qwen3-max/answers.xlsx'

OUTPUT_PATH = ROOT / 'outputs/concept_score' / RUN_ID / 'all_models/scores.xlsx'

API_KEY = ''

API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

MODEL_NAME = 'qwen3.8-flash'
REQUEST_TIMEOUT = 300

TEMPERATURE = 0.0

REQUEST_INTERVAL_SECONDS = 0.5

MAX_RETRIES = 3

RESUME_FROM_OUTPUT = True

CHECKPOINT_EVERY_N = 50

EXTRA_BODY = None

METADATA_COLUMNS = {"来源文件", "领域"}

TERM_COLUMN = "名词"

REFERENCE_COLUMN = "解释"

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
    "connection error",
    "rate limit",
    "报错",
    "错误:",
    "api请求错误",
)

SCORING_STANDARD = """4：核心含义准确，关键专业信息基本完整，无实质性错误
3：整体正确，核心含义准确，但存在少量次要信息遗漏或轻微不严谨
2：部分正确，但存在重要信息遗漏，或包含一定程度的专业错误
1：与术语相关，但核心理解存在明显偏差或回答严重不完整
0：核心解释错误、与术语基本无关或无法形成有效解释"""

def is_error_text(text: str) -> bool:
    lowered = text.strip().lower()
    return lowered.startswith(ERROR_PREFIXES)

def answer_text(value: Any) -> Tuple[str, str]:
    if pd.isna(value):
        return "", "missing_answer"
    if not isinstance(value, str):
        return "", "non_text_answer"
    text = value.strip()
    if not text:
        return "", "empty_answer"
    if is_error_text(text):
        return text, "error_answer"
    return text, "ok"

def safe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()

def read_input_excel(input_path: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(input_path)
    except ImportError as exc:
        raise RuntimeError(
            "读取 Excel 失败，通常是 openpyxl 未安装或版本过低。请先安装或升级："
            "pip install --upgrade openpyxl"
        ) from exc

def resolve_columns(df: pd.DataFrame) -> Tuple[str, str, List[str]]:
    columns = list(df.columns)
    if TERM_COLUMN in columns and REFERENCE_COLUMN in columns:
        term_col = TERM_COLUMN
        reference_col = REFERENCE_COLUMN
    elif len(columns) >= 2:
        term_col = columns[0]
        reference_col = columns[1]
    else:
        raise ValueError("输入表格至少需要包含术语列和参考解释列。")

    model_cols = [
        col
        for col in columns
        if col not in {term_col, reference_col} and col not in METADATA_COLUMNS
    ]
    if not model_cols:
        raise ValueError("输入表格中没有发现模型答案列。")
    return term_col, reference_col, model_cols

def load_langchain_components():
    try:
        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "缺少 LangChain 相关依赖。请先安装："
            "pip install langchain-openai langchain-core openpyxl"
        ) from exc
    return ChatOpenAI, ChatPromptTemplate, StrOutputParser

def create_judge_chain():
    if not API_KEY:
        raise RuntimeError("API_KEY 为空。请在本脚本顶部填写 API_KEY。")

    ChatOpenAI, ChatPromptTemplate, StrOutputParser = load_langchain_components()
    llm_kwargs: Dict[str, Any] = {
        "api_key": API_KEY,
        "base_url": API_URL,
        "model": MODEL_NAME,
        "temperature": TEMPERATURE,
        "request_timeout": REQUEST_TIMEOUT,
    }
    if EXTRA_BODY is not None:
        llm_kwargs["extra_body"] = EXTRA_BODY
    llm = ChatOpenAI(**llm_kwargs)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是航天领域术语解释任务的严格评判专家。"
                "你只能根据给定的术语、参考解释和待评价答案进行评分。"
                "你不知道、也不应推测待评价答案来自哪个模型。"
                "只输出一个整数：0、1、2、3 或 4。不要输出理由、标点、JSON、Markdown 或任何额外文字。",
            ),
            (
                "user",
                "评分标准：\n{scoring_standard}\n\n"
                "评分要求：\n"
                "1. 只根据正确性、完整性、专业错误和有效解释程度评分。\n"
                "2. 不因答案长短本身扣分；只有答案过短导致信息缺失，或答案扩展中包含专业错误时才降分。\n"
                "3. 若答案包含正确核心解释但有轻微相关扩展，可给高分。\n"
                "4. 若答案包含明显专业错误，即使部分内容正确，也应根据错误严重程度降分。\n"
                "5. 最终只能输出一个整数等级。\n\n"
                "术语：{term}\n"
                "参考解释：{reference}\n"
                "待评价答案：{answer}",
            ),
        ]
    )
    return prompt | llm | StrOutputParser()

def parse_judge_score(output: str) -> Optional[int]:
    text = output.strip()
    if re.fullmatch(r"[0-4]", text):
        return int(text)
    match = re.search(r"(?<!\d)([0-4])(?!\d)", text)
    if match:
        return int(match.group(1))
    return None

def percent_score(raw_score: Optional[int]) -> float:
    if raw_score is None:
        return math.nan
    return raw_score / 4 * 100

def load_completed_scores(output_path: Path) -> Dict[Tuple[int, str], Dict[str, Any]]:
    if not RESUME_FROM_OUTPUT or not output_path.exists():
        return {}
    try:
        detail_df = pd.read_excel(output_path, sheet_name="detail_scores")
    except Exception:
        return {}

    required = {"行号", "模型列名", "答案状态", "原始等级分", "百分制分数"}
    if not required.issubset(set(detail_df.columns)):
        return {}

    completed: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for _, row in detail_df.iterrows():
        row_index = row.get("行号")
        model_col = row.get("模型列名")
        status = row.get("答案状态")
        raw_score = row.get("原始等级分")
        score = row.get("百分制分数")
        if pd.isna(row_index) or not isinstance(model_col, str):
            continue
        if status == "ok" and not pd.isna(raw_score) and not pd.isna(score):
            completed[(int(row_index), model_col)] = {
                "raw_score": int(raw_score),
                "percent_score": float(score),
                "status": "ok",
                "judge_output": row.get("评判模型原始输出", ""),
            }
    return completed

def merge_detail_rows(
    existing_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for row in existing_rows + new_rows:
        row_index = row.get("行号")
        model_col = row.get("模型列名")
        if pd.isna(row_index) or not isinstance(model_col, str):
            continue
        merged[(int(row_index), model_col)] = row
    return [
        merged[key]
        for key in sorted(merged.keys(), key=lambda item: (item[1], item[0]))
    ]

def load_existing_detail_rows(output_path: Path) -> List[Dict[str, Any]]:
    if not RESUME_FROM_OUTPUT or not output_path.exists():
        return []
    try:
        detail_df = pd.read_excel(output_path, sheet_name="detail_scores")
    except Exception:
        return []
    return detail_df.to_dict(orient="records")

def existing_detail_keys(detail_rows: List[Dict[str, Any]]) -> set[Tuple[int, str]]:
    keys = set()
    for row in detail_rows:
        row_index = row.get("行号")
        model_col = row.get("模型列名")
        if pd.isna(row_index) or not isinstance(model_col, str):
            continue
        keys.add((int(row_index), model_col))
    return keys

def save_progress(detail_rows: List[Dict[str, Any]], input_path: Path, output_path: Path) -> pd.DataFrame:
    detail_df = pd.DataFrame(detail_rows)
    summary_df = build_summary(detail_df)
    parameters_df = build_parameters_df(input_path, output_path)
    write_scores(detail_df, summary_df, parameters_df, output_path)
    return summary_df

def print_model_status(summary_df: pd.DataFrame, model_col: str) -> None:
    matched = summary_df[summary_df["模型"] == model_col]
    print(f"\n已完成模型：{model_col}")
    if matched.empty or pd.isna(matched.iloc[0]["平均分"]):
        print("平均分：无有效评分")
        print("计分题数：0")
    else:
        row = matched.iloc[0]
        print(f"平均分：{row['平均分']:.2f}")
        print(f"计分题数：{int(row['计分题数'])}")
    print_model_ranking(summary_df, title="当前模型排名：")

def invoke_with_retry(chain: Any, term: str, reference: str, answer: str) -> str:
    last_error: Optional[Exception] = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return chain.invoke(
                {
                    "scoring_standard": SCORING_STANDARD,
                    "term": term,
                    "reference": reference,
                    "answer": answer,
                }
            ).strip()
        except Exception as exc:
            last_error = exc
            if attempt < MAX_RETRIES:
                time.sleep(REQUEST_INTERVAL_SECONDS * attempt)
    raise RuntimeError(str(last_error))

def build_detail_scores(df: pd.DataFrame) -> pd.DataFrame:
    validate_resume(INPUT_PATH, OUTPUT_PATH, {'model': MODEL_NAME, 'url': API_URL, 'temperature': TEMPERATURE,
        'extra_body': EXTRA_BODY, 'timeout': REQUEST_TIMEOUT, 'retries': MAX_RETRIES}, RESUME_FROM_OUTPUT)
    term_col, reference_col, model_cols = resolve_columns(df)
    completed_scores = load_completed_scores(OUTPUT_PATH)
    detail_rows: List[Dict[str, Any]] = load_existing_detail_rows(OUTPUT_PATH)
    chain = None

    total_items = len(df) * len(model_cols)
    current_item = len(completed_scores)
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)

    for model_col in model_cols:
        model_rows: List[Dict[str, Any]] = []
        expected_keys = {(int(row_idx), model_col) for row_idx in df.index}
        existing_keys = existing_detail_keys(detail_rows)
        completed_model_keys = expected_keys.intersection(completed_scores.keys())
        existing_model_keys = expected_keys.intersection(existing_keys)
        if len(completed_model_keys) == len(expected_keys):
            print(f"模型已完成，跳过：{model_col}")
            continue

        print(f"\n开始评估模型列：{model_col}")
        for row_idx, row in df.iterrows():
            current_item += 1
            term = safe_text(row[term_col])
            reference = safe_text(row[reference_col])
            answer, status = answer_text(row[model_col])
            raw_score: Any = math.nan
            normalized_score = math.nan
            judge_output = ""
            answer_status = status if reference else "missing_reference"

            if status == "ok" and reference:
                cached = completed_scores.get((int(row_idx), model_col))
                if cached:
                    raw_score = cached["raw_score"]
                    normalized_score = cached["percent_score"]
                    judge_output = cached.get("judge_output", "")
                    answer_status = "ok"
                    print(f"[{current_item}/{total_items}] 复用已完成评分：{model_col} 行 {row_idx + 1}")
                else:
                    print(f"[{current_item}/{total_items}] 正在盲评：{model_col} 行 {row_idx + 1}")
                    try:
                        if chain is None:
                            chain = create_judge_chain()
                        judge_output = invoke_with_retry(chain, term, reference, answer)
                        parsed_score = parse_judge_score(judge_output)
                        if parsed_score is None:
                            answer_status = "parse_error"
                        else:
                            raw_score = parsed_score
                            normalized_score = percent_score(parsed_score)
                            answer_status = "ok"
                    except Exception as exc:
                        judge_output = str(exc)
                        answer_status = "judge_error"
                    time.sleep(REQUEST_INTERVAL_SECONDS)

            detail_rows.append(
                {
                    "行号": int(row_idx),
                    "术语": term,
                    "参考解释": reference,
                    "模型列名": model_col,
                    "模型答案": answer,
                    "答案状态": answer_status,
                    "原始等级分": raw_score,
                    "百分制分数": normalized_score,
                    "评判模型原始输出": judge_output,
                }
            )
            model_rows.append(detail_rows[-1])

            if CHECKPOINT_EVERY_N and current_item % CHECKPOINT_EVERY_N == 0:
                detail_rows = merge_detail_rows(detail_rows, [])
                save_progress(detail_rows, input_path, output_path)
                print(f"已保存阶段性结果：{current_item}/{total_items}")

        detail_rows = merge_detail_rows(detail_rows, model_rows)
        summary_df = save_progress(detail_rows, input_path, output_path)
        print_model_status(summary_df, model_col)
    return pd.DataFrame(detail_rows)

def build_summary(detail_df: pd.DataFrame) -> pd.DataFrame:
    summary_df = (
        detail_df.groupby("模型列名", sort=False)
        .agg(
            平均分=("百分制分数", "mean"),
            标准差=("百分制分数", "std"),
            最高分=("百分制分数", "max"),
            最低分=("百分制分数", "min"),
            平均原始等级分=("原始等级分", "mean"),
            有效答案数=("答案状态", lambda values: int((values == "ok").sum())),
            计分题数=("百分制分数", "count"),
            总题数=("答案状态", "count"),
        )
        .reset_index()
        .rename(columns={"模型列名": "模型"})
    )
    summary_df["标准差"] = summary_df["标准差"].fillna(0.0)
    summary_df["排名"] = summary_df["平均分"].rank(ascending=False, method="min")
    return summary_df.sort_values(["排名", "模型"]).reset_index(drop=True)

def build_parameters_df(input_path: Path, output_path: Path) -> pd.DataFrame:
    rows = [
        ("input_file", str(input_path)),
        ("output_file", str(output_path)),
        ("judge_model", MODEL_NAME),
        ("base_url", API_URL),
        ("temperature", TEMPERATURE),
        ("request_interval_seconds", REQUEST_INTERVAL_SECONDS),
        ("max_retries", MAX_RETRIES),
        ("resume_from_output", RESUME_FROM_OUTPUT),
        ("checkpoint_every_n", CHECKPOINT_EVERY_N),
        ("normalization", "percent_score = raw_score / 4 * 100"),
        ("blind_judging", "模型列名、来源文件、领域不进入评判 prompt"),
        ("scoring_standard", SCORING_STANDARD),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value"])

def write_scores(
    detail_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    parameters_df: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            detail_df.to_excel(writer, sheet_name="detail_scores", index=False)
            summary_df.to_excel(writer, sheet_name="model_summary", index=False)
            parameters_df.to_excel(writer, sheet_name="parameters", index=False)

            for sheet_name in ("detail_scores", "model_summary", "parameters"):
                worksheet = writer.sheets[sheet_name]
                worksheet.freeze_panes = "A2"
                for column_cells in worksheet.columns:
                    header = str(column_cells[0].value or "")
                    max_len = max(len(str(cell.value or "")) for cell in column_cells[:80])
                    worksheet.column_dimensions[column_cells[0].column_letter].width = min(
                        max(max_len + 2, len(header) + 2), 60
                    )
    except ImportError as exc:
        raise RuntimeError(
            "写入 Excel 失败，通常是 openpyxl 未安装或版本过低。请先安装或升级："
            "pip install --upgrade openpyxl"
        ) from exc

def print_model_ranking(summary_df: pd.DataFrame, title: str = "模型排名：") -> None:
    print(f"\n{title}")
    for _, row in summary_df[["排名", "模型", "平均分", "计分题数"]].iterrows():
        rank = "未计分" if pd.isna(row["排名"]) else str(int(row["排名"]))
        score = "无有效评分" if pd.isna(row["平均分"]) else f"{row['平均分']:.2f}"
        print(f"{rank}. {row['模型']}: {score}，计分题数 {int(row['计分题数'])}")

def main() -> None:
    if not API_KEY.strip():
        raise RuntimeError('请在本脚本顶部填写 API_KEY；尚未发起模型请求。')
    input_path = Path(INPUT_PATH)
    output_path = Path(OUTPUT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在：{input_path}")

    df = read_input_excel(input_path)
    detail_df = build_detail_scores(df)
    summary_df = build_summary(detail_df)
    parameters_df = build_parameters_df(input_path, output_path)
    write_scores(detail_df, summary_df, parameters_df, output_path)

    print(f"\n评分完成：{output_path}")
    print_model_ranking(summary_df, title="最终模型排名：")
    print("overover")

def validate_resume(input_path, output_path, settings, resume):
    input_path, output_path = Path(input_path), Path(output_path)
    metadata = {'input_sha256': hashlib.sha256(input_path.read_bytes()).hexdigest(),
                'input_path': str(input_path.resolve()), 'settings': settings,
                'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    metadata = json.loads(json.dumps(metadata, sort_keys=True))
    manifest_path = output_path.with_suffix('.run.json')
    if output_path.exists() or manifest_path.exists():
        if not resume or not manifest_path.exists():
            raise ValueError('输出已存在或缺少续跑清单，请选择新的输出目录 / RUN_ID')
        if json.loads(manifest_path.read_text(encoding='utf-8')) != metadata:
            raise ValueError('输入、裁判参数或评分代码改变，请选择新的输出目录 / RUN_ID')
    else:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
