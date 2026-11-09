from __future__ import annotations
import hashlib


import json

import math

import re


from pathlib import Path

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from openpyxl import load_workbook

from openpyxl.styles import Alignment, Font

from pydantic import BaseModel, Field, field_validator

ROOT = Path(__file__).resolve().parent
RUN_ID = 'run01'
RESUME = True
API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL_NAME = 'qwen3.8-flash'
API_KEY = ''
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 12000
EXTRA_BODY = {}

INPUT_PATH = ROOT / 'outputs/payload_config/run01/qwen3-max/answers.xlsx'

OUTPUT_PATH = ROOT / 'outputs/payload_config_score' / RUN_ID / 'all_models/scores.xlsx'

TASK_DETAIL_COLUMNS = [
    "序号",
    "任务",
    "科学目标",
    "model",
    "参考载荷",
    "模型输出载荷",
    "judge_output",
    "M_count",
    "P_count",
    "C_count",
    "U_count",
    "precision",
    "recall",
    "f1",
    "status",
    "error",
]

JUDGE_PAIR_COLUMNS = [
    "序号",
    "任务",
    "model",
    "ref_id",
    "reference_payload",
    "candidate_id",
    "candidate_payload",
    "result",
    "certainty",
]

MODEL_SUMMARY_COLUMNS = [
    "model",
    "valid_task_count",
    "failed_task_count",
    "precision",
    "recall",
    "f1",
]

class CoverageItem(BaseModel):
    ref_id: int
    candidate_ids: List[int]
    result: str
    certainty: str

    @field_validator("ref_id")
    @classmethod
    def validate_ref_id(cls, value: Any) -> int:
        if isinstance(value, bool):
            raise ValueError("ref_id must be an integer.")
        try:
            normalized = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("ref_id must be an integer.") from exc
        return normalized

    @field_validator("candidate_ids")
    @classmethod
    def validate_candidate_ids(cls, value: Any) -> List[int]:
        if not isinstance(value, list):
            raise ValueError("candidate_ids must be a list.")
        normalized_ids: List[int] = []
        for item in value:
            if isinstance(item, bool):
                raise ValueError("candidate_id must be an integer.")
            try:
                normalized_ids.append(int(item))
            except (TypeError, ValueError) as exc:
                raise ValueError("candidate_id must be an integer.") from exc
        if len(normalized_ids) != len(set(normalized_ids)):
            raise ValueError("candidate_ids must not contain duplicates.")
        return normalized_ids

    @field_validator("result")
    @classmethod
    def validate_result(cls, value: str) -> str:
        value = str(value).strip().upper()
        if value not in {"M", "P"}:
            raise ValueError("result must be M or P")
        return value

    @field_validator("certainty")
    @classmethod
    def validate_certainty(cls, value: str) -> str:
        value = str(value).strip().upper()
        if value not in {"C", "U"}:
            raise ValueError("certainty must be C or U")
        return value

class CoverageOutput(BaseModel):
    coverage: List[List[Any]] = Field(default_factory=list)

    @field_validator("coverage")
    @classmethod
    def validate_coverage(cls, value: List[List[Any]]) -> List[List[Any]]:
        normalized_items: List[List[Any]] = []
        for item in value:
            if not isinstance(item, list) or len(item) != 4:
                raise ValueError("Each item in coverage must be a 4-item list.")
            ref_id, candidate_ids, result, certainty = item
            validated_item = CoverageItem(
                ref_id=ref_id,
                candidate_ids=candidate_ids,
                result=str(result),
                certainty=str(certainty),
            )
            normalized_items.append(
                [
                    validated_item.ref_id,
                    validated_item.candidate_ids,
                    validated_item.result,
                    validated_item.certainty,
                ]
            )
        return normalized_items

def normalize_payload_name(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"[\u00a0\u3000]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n-•*、，,;；")
    text = re.sub(r"^\s*[RrCc]\d+[\.\)）、:：]\s*", "", text)
    text = re.sub(r"^\s*(?:\d+|[一二三四五六七八九十]+)[\.\)）、:：]\s*", "", text)
    text = re.sub(r"^\s*[-•*]\s*", "", text)
    return text.strip()

def is_empty_cell(value: Any) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, float) and math.isnan(value):
            return True
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}

def payloads_from_json(value: Any) -> Optional[List[str]]:
    if isinstance(value, list):
        return [normalize_payload_name(item) for item in value if normalize_payload_name(item)]
    if isinstance(value, dict):
        for key in ("payloads", "载荷", "真实载荷", "参考载荷", "predictions", "predicted_payloads"):
            if key in value:
                return payloads_from_json(value[key])
        return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text.startswith(("[", "{")):
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None
    return payloads_from_json(parsed)

def parse_payload_cell(value: Any) -> List[str]:
    if is_empty_cell(value):
        return []

    json_payloads = payloads_from_json(value)
    if json_payloads is not None:
        return json_payloads

    text = str(value).strip()
    text = re.sub(r"^```(?:json|text)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    text = re.sub(r"\r\n?", "\n", text)

    payloads: List[str] = []
    for raw_part in re.split(r"[\n;；]+", text):
        part = normalize_payload_name(raw_part)
        if not part:
            continue
        if part in {"载荷", "载荷名称", "参考载荷", "推荐载荷", "配置载荷", "模型输出", "预测载荷"}:
            continue
        if re.match(r"^(?:载荷|载荷名称|参考载荷|推荐载荷|配置载荷|预测载荷)\s*[:：]", part):
            part = normalize_payload_name(re.split(r"[:：]", part, maxsplit=1)[1])
        if part:
            payloads.append(part)
    return payloads

def numbered_payloads(prefix: str, payloads: Sequence[str]) -> str:
    return "\n".join(f"{index}. {payload}" for index, payload in enumerate(payloads, start=1))

def build_system_prompt() -> str:
    return (
        "你是深空探测科学载荷配置评审模型。"
        "任务：依据科学目标，判断每个参考载荷代表的必要科学探测功能，是否被候选载荷集合覆盖。"
        "取消一对一限制：多个候选可联合覆盖一个参考；一个多功能候选也可参与多个参考的覆盖。"
        "M=候选组合能完整覆盖该参考载荷在当前科学目标下的主要必要功能；"
        "P=仅部分覆盖或无法完整覆盖。"
        "对每个参考载荷输出最小充分候选组合，不加入无必要关系的额外候选。"
        "C=判断明确；U=功能边界或名称能力描述存在不确定。"
        "只输出JSON，不输出载荷名称、解释、Markdown或指标。"
    )

def build_user_prompt(
    science_goal: str,
    reference_payloads: Sequence[str],
    candidate_payloads: Sequence[str],
    retry_error: str = "",
) -> str:
    retry_instruction = f"上次输出格式错误：{retry_error}\n请修正并重新输出合法 JSON。\n\n" if retry_error else ""
    return f"""科学目标：
{retry_instruction}{science_goal}

参考载荷集合（ref_id）：
{numbered_payloads("R", reference_payloads)}

候选载荷集合（candidate_id）：
{numbered_payloads("C", candidate_payloads)}

要求：
- 每个 ref_id 必须且只能输出一条记录。
- 只使用上方整数编号，不要输出载荷名称。
- candidate_id 列表表示共同覆盖该 ref_id 的最小充分候选组合。
- 若 result 为 M，candidate_id 列表不能为空。
- 若 result 为 P，可以输出部分相关候选组合，也可以输出空列表 []。
- 不计算 Precision、Recall 或 F1。

输出格式：
{{
  "coverage": [
    [ref_id, [candidate_id1, candidate_id2], "M或P", "C或U"]
  ]
}}
"""

def extract_first_json_object(text: str) -> Dict[str, Any]:
    stripped = str(text or "").strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
        raise ValueError(f"JSON root should be an object, got {type(parsed).__name__}.")
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    if start < 0:
        raise ValueError(f"No JSON object found in judge output: {stripped[:200]!r}")

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
                parsed = json.loads(stripped[start : pos + 1])
                if not isinstance(parsed, dict):
                    raise ValueError(f"JSON root should be an object, got {type(parsed).__name__}.")
                return parsed

    raise ValueError(f"No complete JSON object found in judge output: {stripped[:200]!r}")

def validate_judge_output(raw_result: Any) -> CoverageOutput:
    if isinstance(raw_result, CoverageOutput):
        return raw_result
    if isinstance(raw_result, str):
        raw_result = extract_first_json_object(raw_result)
    if hasattr(CoverageOutput, "model_validate"):
        return CoverageOutput.model_validate(raw_result)
    return CoverageOutput.parse_obj(raw_result)

def validate_coverage_semantics(
    judge_output: CoverageOutput,
    reference_count: int,
    candidate_count: int,
) -> CoverageOutput:
    expected_ref_ids = set(range(1, reference_count + 1))
    seen_ref_ids: List[int] = []

    for ref_id, candidate_ids, result, _certainty in judge_output.coverage:
        seen_ref_ids.append(ref_id)
        if ref_id < 1 or ref_id > reference_count:
            raise ValueError(f"ref_id={ref_id} is out of range 1..{reference_count}.")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise ValueError(f"ref_id={ref_id} contains duplicate candidate ids.")
        for candidate_id in candidate_ids:
            if candidate_id < 1 or candidate_id > candidate_count:
                raise ValueError(f"candidate_id={candidate_id} is out of range 1..{candidate_count}.")
        if result == "M" and not candidate_ids:
            raise ValueError(f"ref_id={ref_id} has result M but an empty candidate list.")

    seen_ref_set = set(seen_ref_ids)
    duplicated_ref_ids = sorted({ref_id for ref_id in seen_ref_ids if seen_ref_ids.count(ref_id) > 1})
    if duplicated_ref_ids:
        raise ValueError(f"Duplicate ref_id values: {duplicated_ref_ids}.")

    missing_ref_ids = sorted(expected_ref_ids - seen_ref_set)
    if missing_ref_ids:
        raise ValueError(f"Missing ref_id values: {missing_ref_ids}.")

    extra_ref_ids = sorted(seen_ref_set - expected_ref_ids)
    if extra_ref_ids:
        raise ValueError(f"Unexpected ref_id values: {extra_ref_ids}.")

    return judge_output

def coerce_and_validate_judge_output(
    raw_result: Any,
    reference_count: int,
    candidate_count: int,
) -> CoverageOutput:
    return validate_coverage_semantics(
        judge_output=validate_judge_output(raw_result),
        reference_count=reference_count,
        candidate_count=candidate_count,
    )

def safe_json_dump(value: Any) -> str:
    if isinstance(value, BaseModel):
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        else:
            value = value.dict()
    return json.dumps(value, ensure_ascii=False)

def extract_message_content(raw_response: Any) -> str:
    content = getattr(raw_response, "content", raw_response)
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)

def extract_token_usage(raw_response: Any) -> Dict[str, int]:
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    usage_metadata = getattr(raw_response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        prompt_tokens = int(usage_metadata.get("input_tokens") or usage_metadata.get("prompt_tokens") or 0)
        completion_tokens = int(
            usage_metadata.get("output_tokens") or usage_metadata.get("completion_tokens") or 0
        )
        total_tokens = int(usage_metadata.get("total_tokens") or prompt_tokens + completion_tokens)

    response_metadata = getattr(raw_response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        if isinstance(token_usage, dict):
            prompt_tokens = prompt_tokens or int(
                token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
            )
            completion_tokens = completion_tokens or int(
                token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
            )
            total_tokens = total_tokens or int(token_usage.get("total_tokens") or prompt_tokens + completion_tokens)

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

def format_token_status(token_usage: Dict[str, int]) -> str:
    if token_usage["total_tokens"]:
        return (
            f"tokens={token_usage['total_tokens']} "
            f"prompt_tokens={token_usage['prompt_tokens']} "
            f"completion_tokens={token_usage['completion_tokens']}"
        )
    return "tokens=unavailable prompt_tokens=unavailable completion_tokens=unavailable"

def print_llm_start(
    task_id: Any,
    model_name: str,
    attempt: int,
    mode: str,
    ref_count: int,
    candidate_count: int,
    prompt_chars: int,
    request_timeout: Any,
) -> None:
    print(
        f"[LLM START] task={task_id} model={model_name} attempt={attempt} mode={mode} "
        f"refs={ref_count} candidates={candidate_count} prompt_chars={prompt_chars} "
        f"timeout={request_timeout}s"
    )

def print_llm_done(
    task_id: Any,
    model_name: str,
    mode: str,
    raw_response: Any,
) -> None:
    token_status = format_token_status(extract_token_usage(raw_response))
    output_chars = len(extract_message_content(raw_response))
    print(
        f"[LLM DONE] task={task_id} model={model_name} mode={mode} "
        f"{token_status} output_chars={output_chars}"
    )

def print_llm_invalid(task_id: Any, model_name: str, mode: str, reason: Exception) -> None:
    print(f"[LLM INVALID] task={task_id} model={model_name} mode={mode} reason={str(reason)[:500]}")

def print_llm_fallback(task_id: Any, model_name: str, from_mode: str, to_mode: str) -> None:
    print(f"[LLM FALLBACK] task={task_id} model={model_name} from={from_mode} to={to_mode}")

def print_llm_retry(task_id: Any, model_name: str, attempt: int, reason: str) -> None:
    print(f"[LLM RETRY] task={task_id} model={model_name} attempt={attempt} reason={reason[:500]}")

def invoke_prompt_llm(
    prompt: Any,
    llm: Any,
    prompt_text: str,
    mode: str,
    attempt: int,
    task_id: Any,
    model_name: str,
    ref_count: int,
    candidate_count: int,
    request_timeout: Any,
    show_llm_status: bool,
) -> Any:
    if show_llm_status:
        print_llm_start(
            task_id=task_id,
            model_name=model_name,
            attempt=attempt,
            mode=mode,
            ref_count=ref_count,
            candidate_count=candidate_count,
            prompt_chars=len(prompt_text),
            request_timeout=request_timeout,
        )
    raw_response = llm.invoke(prompt.invoke({"input": prompt_text}))
    if show_llm_status:
        print_llm_done(
            task_id=task_id,
            model_name=model_name,
            mode=mode,
            raw_response=raw_response,
        )
    return raw_response

def create_judge_chain(
    judge_api_url: str,
    judge_api_key: str,
    judge_model_name: str,
) -> Dict[str, Any]:
    try:
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_openai import ChatOpenAI
    except ImportError as exc:
        raise RuntimeError(
            "Missing LangChain dependencies. Please install langchain-core and "
            "langchain-openai in the current PyCharm interpreter before running this evaluator."
        ) from exc

    request_timeout = REQUEST_TIMEOUT
    llm = ChatOpenAI(
        api_key=judge_api_key,
        base_url=judge_api_url,
        model=judge_model_name,
        temperature=TEMPERATURE,
        request_timeout=request_timeout,
        extra_body=EXTRA_BODY or None,
    )

    base_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", build_system_prompt()),
            ("human", "{input}"),
        ]
    )
    parser = JsonOutputParser(pydantic_object=CoverageOutput)
    json_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", build_system_prompt() + " 必须只输出一个合法 JSON 对象。"),
            ("human", "{input}\n\n格式说明：\n{format_instructions}"),
        ]
    ).partial(format_instructions=parser.get_format_instructions())

    chains: Dict[str, Any] = {
        "json_prompt": json_prompt,
        "text_prompt": base_prompt,
        "json_llm": llm.bind(response_format={"type": "json_object"}),
        "text_llm": llm,
        "request_timeout": request_timeout,
    }
    try:
        chains["structured"] = base_prompt | llm.with_structured_output(CoverageOutput, include_raw=True)
    except Exception:
        chains["structured"] = None
    return chains

def invoke_judge(
    chain_bundle: Dict[str, Any],
    science_goal: str,
    reference_payloads: Sequence[str],
    candidate_payloads: Sequence[str],
    task_id: Any,
    model_name: str,
    show_llm_status: bool = False,
) -> Tuple[CoverageOutput, str, str]:
    errors: List[str] = []
    retry_error = ""

    for attempt in range(2):
        if attempt > 0 and show_llm_status:
            print_llm_retry(task_id, model_name, attempt + 1, retry_error)
        prompt_text = build_user_prompt(
            science_goal=science_goal,
            reference_payloads=reference_payloads,
            candidate_payloads=candidate_payloads,
            retry_error=retry_error,
        )
        attempt_errors: List[str] = []

        try:
            raw_response = invoke_prompt_llm(
                prompt=chain_bundle["json_prompt"],
                llm=chain_bundle["json_llm"],
                prompt_text=prompt_text,
                mode="json",
                attempt=attempt + 1,
                task_id=task_id,
                model_name=model_name,
                ref_count=len(reference_payloads),
                candidate_count=len(candidate_payloads),
                request_timeout=chain_bundle.get("request_timeout", "unknown"),
                show_llm_status=show_llm_status,
            )
            raw_text = extract_message_content(raw_response)
            result = coerce_and_validate_judge_output(
                raw_result=raw_text,
                reference_count=len(reference_payloads),
                candidate_count=len(candidate_payloads),
            )
            return result, raw_text, "json"
        except Exception as exc:
            if show_llm_status:
                print_llm_invalid(task_id, model_name, "json", exc)
            attempt_errors.append(f"json={exc}")

        if show_llm_status:
            print_llm_fallback(task_id, model_name, "json", "text_json_extract")
        try:
            raw_response = invoke_prompt_llm(
                prompt=chain_bundle["text_prompt"],
                llm=chain_bundle["text_llm"],
                prompt_text=prompt_text,
                mode="text_json_extract",
                attempt=attempt + 1,
                task_id=task_id,
                model_name=model_name,
                ref_count=len(reference_payloads),
                candidate_count=len(candidate_payloads),
                request_timeout=chain_bundle.get("request_timeout", "unknown"),
                show_llm_status=show_llm_status,
            )
            raw_text = extract_message_content(raw_response)
            result = coerce_and_validate_judge_output(
                raw_result=raw_text,
                reference_count=len(reference_payloads),
                candidate_count=len(candidate_payloads),
            )
            return result, raw_text, "text_json_extract"
        except Exception as exc:
            if show_llm_status:
                print_llm_invalid(task_id, model_name, "text_json_extract", exc)
            attempt_errors.append(f"text={exc}")

        structured_chain = chain_bundle.get("structured")
        if structured_chain is not None:
            if show_llm_status:
                print_llm_fallback(task_id, model_name, "text_json_extract", "structured")
            try:
                if show_llm_status:
                    print_llm_start(
                        task_id=task_id,
                        model_name=model_name,
                        attempt=attempt + 1,
                        mode="structured",
                        ref_count=len(reference_payloads),
                        candidate_count=len(candidate_payloads),
                        prompt_chars=len(prompt_text),
                        request_timeout=chain_bundle.get("request_timeout", "unknown"),
                    )
                raw_result = structured_chain.invoke({"input": prompt_text})
                raw_response = raw_result.get("raw") if isinstance(raw_result, dict) else raw_result
                if show_llm_status:
                    print_llm_done(task_id, model_name, "structured", raw_response)
                parsed_result = raw_result.get("parsed") if isinstance(raw_result, dict) else raw_result
                result = coerce_and_validate_judge_output(
                    raw_result=parsed_result,
                    reference_count=len(reference_payloads),
                    candidate_count=len(candidate_payloads),
                )
                raw_output = extract_message_content(raw_response) if raw_response is not None else safe_json_dump(result)
                return result, raw_output, "structured"
            except Exception as exc:
                if show_llm_status:
                    print_llm_invalid(task_id, model_name, "structured", exc)
                attempt_errors.append(f"structured={exc}")

        errors.extend(f"attempt{attempt + 1}:{error}" for error in attempt_errors)
        retry_error = "; ".join(attempt_errors)[-1200:]

    raise RuntimeError("; ".join(errors))

def postprocess_coverages(
    judge_output: CoverageOutput,
    reference_payloads: Sequence[str],
    candidate_payloads: Sequence[str],
) -> List[Dict[str, Any]]:
    coverages: List[Dict[str, Any]] = []

    for ref_id, candidate_ids, result, certainty in judge_output.coverage:
        reference_payload = reference_payloads[ref_id - 1]
        candidate_payload_names = [candidate_payloads[candidate_id - 1] for candidate_id in candidate_ids]
        coverages.append(
            {
                "ref_id": ref_id,
                "reference_payload": reference_payload,
                "candidate_ids": list(candidate_ids),
                "candidate_payloads": candidate_payload_names,
                "result": result,
                "certainty": certainty,
            }
        )
    return coverages

def expand_coverage_pairs(
    coverages: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    pair_rows: List[Dict[str, Any]] = []
    for coverage in coverages:
        for candidate_id, candidate_payload in zip(coverage["candidate_ids"], coverage["candidate_payloads"]):
            pair_rows.append(
                {
                    "ref_id": coverage["ref_id"],
                    "reference_payload": coverage["reference_payload"],
                    "candidate_id": candidate_id,
                    "candidate_payload": candidate_payload,
                    "result": coverage["result"],
                    "certainty": coverage["certainty"],
                }
            )
    return pair_rows

def calculate_counts_and_metrics(
    reference_payloads: Sequence[str],
    candidate_payloads: Sequence[str],
    coverages: Sequence[Dict[str, Any]],
) -> Dict[str, float]:
    m_count = sum(1 for coverage in coverages if coverage["result"] == "M")
    p_count = sum(1 for coverage in coverages if coverage["result"] == "P")
    c_count = sum(1 for coverage in coverages if coverage["certainty"] == "C")
    u_count = sum(1 for coverage in coverages if coverage["certainty"] == "U")
    effective_candidate_ids = {
        candidate_id
        for coverage in coverages
        if coverage["result"] == "M"
        for candidate_id in coverage["candidate_ids"]
    }
    precision = len(effective_candidate_ids) / len(candidate_payloads) if candidate_payloads else 0.0
    recall = m_count / len(reference_payloads) if reference_payloads else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "M_count": m_count,
        "P_count": p_count,
        "C_count": c_count,
        "U_count": u_count,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def read_payload_result_excel(
    input_excel_path: Union[str, Path],
    sheet_name: str = "payload_config",
) -> pd.DataFrame:
    try:
        return pd.read_excel(input_excel_path, sheet_name=sheet_name, engine="openpyxl")
    except ImportError:
        workbook = load_workbook(input_excel_path, read_only=True, data_only=True)
        try:
            if sheet_name not in workbook.sheetnames:
                raise RuntimeError(f"Unable to read sheet {sheet_name!r} from {input_excel_path}.")
            worksheet = workbook[sheet_name]
            rows = list(worksheet.iter_rows(values_only=True))
        finally:
            workbook.close()
        if not rows:
            return pd.DataFrame()
        header = [str(item) if item is not None else "" for item in rows[0]]
        return pd.DataFrame(rows[1:], columns=header)
    except ValueError as exc:
        raise RuntimeError(f"Unable to read sheet {sheet_name!r} from {input_excel_path}: {exc}") from exc

def empty_dataframe(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))

def read_excel_sheet_or_empty(
    excel_path: Union[str, Path],
    sheet_name: str,
    columns: Sequence[str],
) -> pd.DataFrame:
    path = Path(excel_path)
    if not path.exists():
        return empty_dataframe(columns)

    try:
        df = pd.read_excel(path, sheet_name=sheet_name, engine="openpyxl")
    except (ImportError, ValueError):
        workbook = load_workbook(path, read_only=True, data_only=True)
        try:
            if sheet_name not in workbook.sheetnames:
                return empty_dataframe(columns)
            worksheet = workbook[sheet_name]
            rows = list(worksheet.iter_rows(values_only=True))
        finally:
            workbook.close()
        if not rows:
            return empty_dataframe(columns)
        header = [str(item) if item is not None else "" for item in rows[0]]
        df = pd.DataFrame(rows[1:], columns=header)

    for column in columns:
        if column not in df.columns:
            df[column] = ""
    return df.loc[:, list(columns)]

def load_existing_results(
    output_excel_path: Union[str, Path],
    resume: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not resume:
        return empty_dataframe(TASK_DETAIL_COLUMNS), empty_dataframe(JUDGE_PAIR_COLUMNS)
    task_details_df = read_excel_sheet_or_empty(output_excel_path, "task_details", TASK_DETAIL_COLUMNS)
    judge_pairs_df = read_excel_sheet_or_empty(output_excel_path, "judge_pairs", JUDGE_PAIR_COLUMNS)
    task_details_df = filter_current_schema_task_details(task_details_df)
    judge_pairs_df = filter_pair_rows_for_task_details(judge_pairs_df, task_details_df)
    return task_details_df, judge_pairs_df

def has_current_coverage_schema(row: Any) -> bool:
    if str(row.get("status", "")) != "ok":
        return False
    try:
        parsed = extract_first_json_object(str(row.get("judge_output", "")))
    except Exception:
        return False
    return isinstance(parsed.get("coverage"), list)

def filter_current_schema_task_details(task_details_df: pd.DataFrame) -> pd.DataFrame:
    if task_details_df.empty:
        return task_details_df
    keep_mask = task_details_df.apply(
        lambda row: str(row.get("status", "")) != "ok" or has_current_coverage_schema(row),
        axis=1,
    )
    return task_details_df.loc[keep_mask, TASK_DETAIL_COLUMNS].reset_index(drop=True)

def filter_pair_rows_for_task_details(
    judge_pairs_df: pd.DataFrame,
    task_details_df: pd.DataFrame,
) -> pd.DataFrame:
    if judge_pairs_df.empty:
        return judge_pairs_df
    if task_details_df.empty:
        return empty_dataframe(JUDGE_PAIR_COLUMNS)
    valid_keys = {
        row_key(row["序号"], row["model"])
        for _, row in task_details_df.iterrows()
        if str(row.get("status", "")) == "ok"
    }
    keep_mask = judge_pairs_df.apply(lambda row: row_key(row["序号"], row["model"]) in valid_keys, axis=1)
    return judge_pairs_df.loc[keep_mask, JUDGE_PAIR_COLUMNS].reset_index(drop=True)

def row_key(sequence_id: Any, model_name: Any) -> Tuple[str, str]:
    try:
        if isinstance(sequence_id, float) and sequence_id.is_integer():
            sequence_id = int(sequence_id)
    except AttributeError:
        pass
    return str(sequence_id), str(model_name)

def should_skip(task_details_df: pd.DataFrame, sequence_id: Any, model_name: str) -> bool:
    if task_details_df.empty:
        return False
    key = row_key(sequence_id, model_name)
    matching_rows = task_details_df[
        task_details_df.apply(lambda row: row_key(row["序号"], row["model"]) == key, axis=1)
    ]
    if matching_rows.empty:
        return False
    return str(matching_rows.iloc[-1].get("status", "")) == "ok"

def upsert_task_detail(df: pd.DataFrame, row: Dict[str, Any]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame([row], columns=TASK_DETAIL_COLUMNS)
    for column in TASK_DETAIL_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    key = row_key(row["序号"], row["model"])
    keep_mask = ~df.apply(lambda item: row_key(item["序号"], item["model"]) == key, axis=1)
    kept_df = df.loc[keep_mask, TASK_DETAIL_COLUMNS]
    return pd.concat([kept_df, pd.DataFrame([row], columns=TASK_DETAIL_COLUMNS)], ignore_index=True)

def replace_pair_rows(
    df: pd.DataFrame,
    sequence_id: Any,
    model_name: str,
    rows: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(list(rows), columns=JUDGE_PAIR_COLUMNS)
    for column in JUDGE_PAIR_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    key = row_key(sequence_id, model_name)
    keep_mask = ~df.apply(lambda item: row_key(item["序号"], item["model"]) == key, axis=1)
    kept_df = df.loc[keep_mask, JUDGE_PAIR_COLUMNS]
    new_df = pd.DataFrame(list(rows), columns=JUDGE_PAIR_COLUMNS)
    return pd.concat([kept_df, new_df], ignore_index=True)

def build_model_summary(task_details_df: pd.DataFrame, model_columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    models = list(model_columns) if model_columns is not None else list(task_details_df.get("model", pd.Series()).dropna().unique())

    for model_name in models:
        group = task_details_df[task_details_df["model"] == model_name] if not task_details_df.empty else task_details_df
        valid_group = group[group["status"] == "ok"] if not group.empty else group
        failed_count = int((group["status"] != "ok").sum()) if not group.empty else 0
        if valid_group.empty:
            rows.append(
                {
                    "model": model_name,
                    "valid_task_count": 0,
                    "failed_task_count": failed_count,
                    "precision": 0.0,
                    "recall": 0.0,
                    "f1": 0.0,
                }
            )
            continue

        rows.append(
            {
                "model": model_name,
                "valid_task_count": int(len(valid_group)),
                "failed_task_count": failed_count,
                "precision": float(valid_group["precision"].mean()),
                "recall": float(valid_group["recall"].mean()),
                "f1": float(valid_group["f1"].mean()),
            }
        )
    return pd.DataFrame(rows, columns=MODEL_SUMMARY_COLUMNS)

def write_result_workbook(
    output_excel_path: Union[str, Path],
    task_details_df: pd.DataFrame,
    judge_pairs_df: pd.DataFrame,
    model_summary_df: pd.DataFrame,
) -> None:
    output_path = Path(output_excel_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        task_details_df.to_excel(writer, sheet_name="task_details", index=False)
        judge_pairs_df.to_excel(writer, sheet_name="judge_pairs", index=False)
        model_summary_df.to_excel(writer, sheet_name="model_summary", index=False)
        for worksheet in writer.book.worksheets:
            style_worksheet(worksheet)

def style_worksheet(worksheet: Any) -> None:
    for cell in worksheet[1]:
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for row in worksheet.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(vertical="top", wrap_text=True)

    widths = {
        "A": 12,
        "B": 30,
        "C": 42,
        "D": 28,
        "E": 42,
        "F": 42,
        "G": 44,
        "H": 12,
        "I": 12,
        "J": 12,
        "K": 12,
        "L": 12,
        "M": 12,
        "N": 12,
        "O": 12,
        "P": 44,
    }
    for column_letter, width in widths.items():
        if worksheet.max_column >= worksheet[column_letter + "1"].column:
            worksheet.column_dimensions[column_letter].width = width
    worksheet.freeze_panes = "A2"

def evaluate_one_model_answer(
    chain_bundle: Dict[str, Any],
    sequence_id: Any,
    task_name: str,
    science_goal: str,
    model_name: str,
    reference_payloads: Sequence[str],
    candidate_payloads: Sequence[str],
    show_llm_status: bool = True,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    try:
        judge_output, raw_output, _output_mode = invoke_judge(
            chain_bundle=chain_bundle,
            science_goal=science_goal,
            reference_payloads=reference_payloads,
            candidate_payloads=candidate_payloads,
            task_id=sequence_id,
            model_name=model_name,
            show_llm_status=show_llm_status,
        )
        coverages = postprocess_coverages(judge_output, reference_payloads, candidate_payloads)
        pair_rows = expand_coverage_pairs(coverages)
        metrics = calculate_counts_and_metrics(reference_payloads, candidate_payloads, coverages)
        task_detail = {
            "序号": sequence_id,
            "任务": task_name,
            "科学目标": science_goal,
            "model": model_name,
            "参考载荷": "\n".join(reference_payloads),
            "模型输出载荷": "\n".join(candidate_payloads),
            "judge_output": raw_output,
            **metrics,
            "status": "ok",
            "error": "",
        }
        pair_rows = [
            {
                "序号": sequence_id,
                "任务": task_name,
                "model": model_name,
                **pair,
            }
            for pair in pair_rows
        ]
        return task_detail, pair_rows
    except Exception as exc:
        metrics = calculate_counts_and_metrics(reference_payloads, candidate_payloads, [])
        task_detail = {
            "序号": sequence_id,
            "任务": task_name,
            "科学目标": science_goal,
            "model": model_name,
            "参考载荷": "\n".join(reference_payloads),
            "模型输出载荷": "\n".join(candidate_payloads),
            "judge_output": "",
            **metrics,
            "status": "failed",
            "error": str(exc),
        }
        return task_detail, []

def print_model_summary_row(row: Dict[str, Any]) -> None:
    print(
        f"model={row['model']} "
        f"P={float(row['precision']):.4f} "
        f"R={float(row['recall']):.4f} "
        f"F1={float(row['f1']):.4f}"
    )


def count_completed_for_run(
    task_details_df: pd.DataFrame,
    input_df: pd.DataFrame,
    model_columns: Sequence[str],
    sequence_column: str,
) -> int:
    if task_details_df.empty:
        return 0
    valid_sequence_ids = {str(row_key(value, "")[0]) for value in input_df[sequence_column].tolist()}
    valid_models = {str(model_name) for model_name in model_columns}
    completed_count = 0
    for _, row in task_details_df.iterrows():
        if str(row.get("status", "")) != "ok":
            continue
        sequence_key = row_key(row.get("序号"), "")[0]
        model_name = str(row.get("model", ""))
        if sequence_key in valid_sequence_ids and model_name in valid_models:
            completed_count += 1
    return completed_count

def count_model_status(
    task_details_df: pd.DataFrame,
    model_name: str,
    input_df: pd.DataFrame,
    sequence_column: str,
) -> Dict[str, int]:
    valid_sequence_ids = {row_key(value, "")[0] for value in input_df[sequence_column].tolist()}
    completed_sequence_ids = set()
    if not task_details_df.empty:
        for _, row in task_details_df.iterrows():
            if str(row.get("model", "")) != str(model_name):
                continue
            if str(row.get("status", "")) != "ok":
                continue
            sequence_key = row_key(row.get("序号"), "")[0]
            if sequence_key in valid_sequence_ids:
                completed_sequence_ids.add(sequence_key)

    completed_count = len(completed_sequence_ids)
    total_count = len(valid_sequence_ids)
    return {
        "completed": completed_count,
        "pending": max(total_count - completed_count, 0),
    }

def print_progress_start(
    task_count: int,
    model_count: int,
    total_evaluations: int,
    existing_completed: int,
    judge_model_name: str,
    output_excel_path: Union[str, Path],
) -> None:
    print(f"Loaded {task_count} tasks and {model_count} model columns.")
    print(f"Total evaluations: {total_evaluations}")
    print(f"Existing completed evaluations: {existing_completed}")
    print(f"Pending evaluations: {max(total_evaluations - existing_completed, 0)}")
    print(f"Judge model: {judge_model_name}")
    print(f"Output: {output_excel_path}")

def print_model_start(
    model_index: int,
    model_count: int,
    model_name: str,
    status: Dict[str, int],
) -> None:
    print(
        f"[MODEL {model_index}/{model_count}] {model_name} "
        f"pending={status['pending']} completed={status['completed']}"
    )

def print_task_run(
    current_index: int,
    total_count: int,
    sequence_id: Any,
    model_name: str,
) -> None:
    print(f"[RUN {current_index}/{total_count}] task={sequence_id} model={model_name} seq={sequence_id}")

def print_task_status(task_detail: Dict[str, Any]) -> None:
    status = "OK" if str(task_detail.get("status", "")) == "ok" else "FAILED"
    print(
        f"[{status}] task={task_detail['序号']} "
        f"model={task_detail['model']} "
        f"M={int(task_detail['M_count'])} "
        f"P={int(task_detail['P_count'])} "
        f"C={int(task_detail['C_count'])} "
        f"U={int(task_detail['U_count'])} "
        f"P={float(task_detail['precision']):.4f} "
        f"R={float(task_detail['recall']):.4f} "
        f"F1={float(task_detail['f1']):.4f}"
    )

def print_save_status(output_excel_path: Union[str, Path]) -> None:
    print(f"[SAVED] {output_excel_path}")

def evaluate_payload_config_with_science_goal(
    input_excel_path: Union[str, Path],
    output_excel_path: Union[str, Path],
    judge_api_url: str,
    judge_api_key: str,
    judge_model_name: str,
    sheet_name: str = "payload_config",
    resume: bool = True,
    save_after_each: bool = True,
    verbose: bool = True,
    show_llm_status: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not judge_api_key.strip():
        raise RuntimeError('请在本脚本顶部填写 API_KEY；尚未发起模型请求。')
    validate_resume(input_excel_path, output_excel_path,
        {'model': judge_model_name, 'url': judge_api_url, 'temperature': TEMPERATURE,
         'timeout': REQUEST_TIMEOUT, 'extra_body': EXTRA_BODY}, resume)
    input_df = read_payload_result_excel(input_excel_path, sheet_name=sheet_name)
    if input_df.shape[1] < 5:
        raise ValueError(
            "Input Excel must contain at least 5 columns: 序号, 任务, 科学目标, 真实载荷, model output."
        )

    sequence_column = input_df.columns[0]
    task_column = input_df.columns[1]
    science_goal_column = input_df.columns[2]
    reference_payload_column = input_df.columns[3]
    model_columns = list(input_df.columns[4:])

    chain_bundle = create_judge_chain(
        judge_api_url=judge_api_url,
        judge_api_key=judge_api_key,
        judge_model_name=judge_model_name,
    )
    task_details_df, judge_pairs_df = load_existing_results(output_excel_path, resume=resume)

    total_evaluations = len(input_df) * len(model_columns)
    processed_count = count_completed_for_run(
        task_details_df=task_details_df,
        input_df=input_df,
        model_columns=model_columns,
        sequence_column=sequence_column,
    )
    if verbose:
        print_progress_start(
            task_count=len(input_df),
            model_count=len(model_columns),
            total_evaluations=total_evaluations,
            existing_completed=processed_count,
            judge_model_name=judge_model_name,
            output_excel_path=output_excel_path,
        )

    for model_index, model_name in enumerate(model_columns, start=1):
        model_name = str(model_name)
        if verbose:
            model_status = count_model_status(
                task_details_df=task_details_df,
                model_name=model_name,
                input_df=input_df,
                sequence_column=sequence_column,
            )
            print_model_start(
                model_index=model_index,
                model_count=len(model_columns),
                model_name=model_name,
                status=model_status,
            )

        for row_number, row in input_df.iterrows():
            sequence_id = row.get(sequence_column, row_number + 1)
            if should_skip(task_details_df, sequence_id, model_name):
                continue

            task_name = str(row.get(task_column, "")).strip()
            science_goal = str(row.get(science_goal_column, "")).strip()
            reference_payloads = parse_payload_cell(row.get(reference_payload_column))
            candidate_payloads = parse_payload_cell(row.get(model_name))

            if verbose:
                print_task_run(
                    current_index=processed_count + 1,
                    total_count=total_evaluations,
                    sequence_id=sequence_id,
                    model_name=model_name,
                )

            task_detail, pair_rows = evaluate_one_model_answer(
                chain_bundle=chain_bundle,
                sequence_id=sequence_id,
                task_name=task_name,
                science_goal=science_goal,
                model_name=model_name,
                reference_payloads=reference_payloads,
                candidate_payloads=candidate_payloads,
                show_llm_status=show_llm_status,
            )
            task_details_df = upsert_task_detail(task_details_df, task_detail)
            judge_pairs_df = replace_pair_rows(judge_pairs_df, sequence_id, model_name, pair_rows)
            processed_count += 1

            if verbose:
                print_task_status(task_detail)

            if save_after_each:
                model_summary_df = build_model_summary(task_details_df, model_columns=model_columns)
                write_result_workbook(output_excel_path, task_details_df, judge_pairs_df, model_summary_df)
                if verbose:
                    print_save_status(output_excel_path)

        model_summary_df = build_model_summary(task_details_df, model_columns=model_columns)
        model_rows = model_summary_df[model_summary_df["model"] == model_name]
        if not model_rows.empty:
            print_model_summary_row(model_rows.iloc[0].to_dict())

    model_summary_df = build_model_summary(task_details_df, model_columns=model_columns)
    write_result_workbook(output_excel_path, task_details_df, judge_pairs_df, model_summary_df)

    print("=" * 60)
    for _, row in model_summary_df.iterrows():
        print_model_summary_row(row.to_dict())

    return task_details_df, judge_pairs_df, model_summary_df

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


def main():
    if not API_KEY.strip():
        raise RuntimeError('请在本脚本顶部填写 API_KEY；尚未发起模型请求。')
    return evaluate_payload_config_with_science_goal(INPUT_PATH, OUTPUT_PATH, API_URL, API_KEY, MODEL_NAME,
        resume=RESUME, save_after_each=True, verbose=True, show_llm_status=False)


if __name__ == '__main__':
    main()
