#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent
RUN_ID = 'run01'
INPUT_DIR = ROOT / 'result/payload_judge_agreement'
OUTPUT_DIR = ROOT / 'outputs/payload_judge_agreement' / RUN_ID
EXPECTED_SCHEMES = 48
OUTPUT_NAME = "agreement_results.xlsx"
NUMBERED_LINE = re.compile(r"^\s*(\d+)\s*[.．、)）]\s*(.*?)\s*$")


class DataError(ValueError):
    pass


@dataclass
class Scheme:
    excel_row: int
    task_name: str
    reference_names: list[str]
    candidate_count: int
    coverage: dict[int, tuple[list[int], str]]
    original_f1: float | None


@dataclass
class InputData:
    role: str
    path: Path
    sheet: str
    json_column: str
    rows: list[Scheme]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judge", type=Path, required=True, help="完整评判结果 Excel")
    parser.add_argument("--expert1", type=Path, required=True, help="专家1评价 Excel")
    parser.add_argument("--expert2", type=Path, required=True, help="专家2评价 Excel")
    parser.add_argument("--consensus", type=Path, required=True, help="专家综合 Excel")
    parser.add_argument("--output-dir", type=Path, required=True, help="结果输出目录")
    if len(sys.argv) == 1:
        return argparse.Namespace(judge=INPUT_DIR / 'sample_48_judge_results.xlsx',
            expert1=INPUT_DIR / 'expert_1_coverage.xlsx',
            expert2=INPUT_DIR / 'expert_2_coverage.xlsx',
            consensus=INPUT_DIR / 'consensus_coverage.xlsx', output_dir=OUTPUT_DIR)
    return parser.parse_args()


def is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def nonempty_lines(value: Any, context: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, str):
        raise DataError(f"{context}：载荷列表必须是文本")
    lines = [line.strip() for line in value.splitlines() if line.strip()]
    if not lines and value.strip():
        lines = [value.strip()]
    matches = [NUMBERED_LINE.match(line) for line in lines]
    if any(matches) and not all(matches):
        raise DataError(f"{context}：载荷列表编号格式混杂")
    if all(matches) and matches:
        numbers = [int(match.group(1)) for match in matches if match]
        if numbers != list(range(1, len(lines) + 1)):
            raise DataError(f"{context}：载荷列表编号不连续或未从1开始")
        lines = [match.group(2) for match in matches if match]
        if any(not name for name in lines):
            raise DataError(f"{context}：载荷名称为空")
    return lines


def parse_coverage(raw: Any, context: str, candidate_count: int) -> dict[int, tuple[list[int], str]]:
    if not isinstance(raw, str) or not raw.strip():
        raise DataError(f"{context}：judge_output为空，评价尚未完成")
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DataError(f"{context}：JSON解析失败：{exc.msg}（字符位置{exc.pos}）") from exc
    if not isinstance(obj, dict) or not isinstance(obj.get("coverage"), list):
        raise DataError(f"{context}：JSON须含coverage数组")
    if not obj["coverage"]:
        raise DataError(f"{context}：coverage为空")
    result: dict[int, tuple[list[int], str]] = {}
    for position, entry in enumerate(obj["coverage"], 1):
        where = f"{context} coverage第{position}项"
        if not isinstance(entry, list) or len(entry) != 4:
            raise DataError(f"{where}：须为长度4的数组")
        ref_id, candidate_ids, tag, _ignored = entry
        if not is_int(ref_id) or ref_id < 1:
            raise DataError(f"{where}：参考载荷编号须为正整数")
        if ref_id in result:
            raise DataError(f"{where}：参考载荷编号{ref_id}重复")
        if not isinstance(candidate_ids, list):
            raise DataError(f"{where}：候选载荷编号须为数组")
        if any(not is_int(n) or n < 1 or n > candidate_count for n in candidate_ids):
            raise DataError(f"{where}：候选编号须为1至{candidate_count}的整数，实际为{candidate_ids!r}")
        if len(set(candidate_ids)) != len(candidate_ids):
            raise DataError(f"{where}：候选编号重复")
        if tag not in ("M", "P", "N"):
            raise DataError(f"{where}：非法标签{tag!r}，只允许M/P/N")
        if tag == "M" and not candidate_ids:
            raise DataError(f"{where}：M标签缺少参与完整覆盖的候选编号")
        result[ref_id] = (candidate_ids, tag)
    expected = set(range(1, len(result) + 1))
    if set(result) != expected:
        raise DataError(f"{context}：参考载荷编号存在遗漏或越界；实际{sorted(result)}，应为1至{len(result)}")
    return result


def read_input(path: Path, role: str, judge_candidate_counts: list[int] | None = None) -> InputData:
    if not path.is_file():
        raise DataError(f"{role}：找不到文件：{path}")
    try:
        book = load_workbook(path, read_only=True, data_only=True)
    except Exception as exc:
        raise DataError(f"{role}：无法读取Excel：{path}：{exc}") from exc
    try:
        candidates = []
        for sheet in book.worksheets:
            header = next(sheet.iter_rows(min_row=1, max_row=1, values_only=True), ())
            names = [str(x).strip() if x is not None else "" for x in header]
            json_cols = [i for i, name in enumerate(names) if "judge_output" in name]
            if len(json_cols) == 1 and "任务名称" in names and "参考载荷" in names and "模型输出载荷" in names:
                candidates.append((sheet, names, json_cols[0]))
        if len(candidates) != 1:
            raise DataError(f"{role}：应有且仅有一个含任务名称、参考载荷、模型输出载荷和judge_output列的Sheet；找到{len(candidates)}个")
        sheet, names, json_idx = candidates[0]
        if role == "裁判" and "F1值" not in names:
            raise DataError(f"{role}：Sheet「{sheet.title}」缺少原有F1值列")
        columns = {key: names.index(key) for key in ("任务名称", "参考载荷", "模型输出载荷")}
        f1_idx = names.index("F1值") if role == "裁判" else None
        rows: list[Scheme] = []
        for excel_row, values in enumerate(sheet.iter_rows(min_row=2, values_only=True), 2):
            if not any(value is not None and str(value).strip() for value in values):
                raise DataError(f"{role}「{path.name}」Sheet「{sheet.title}」Excel第{excel_row}行：空白结果行，不允许跳过")
            context = f"{role}「{path.name}」Sheet「{sheet.title}」Excel第{excel_row}行"
            def cell(index: int) -> Any:
                return values[index] if index < len(values) else None
            task_name = cell(columns["任务名称"])
            if not isinstance(task_name, str) or not task_name.strip():
                raise DataError(f"{context}：任务名称为空")
            ref_names = nonempty_lines(cell(columns["参考载荷"]), f"{context}参考载荷列")
            candidate_names = nonempty_lines(cell(columns["模型输出载荷"]), f"{context}模型输出载荷列")
            if not ref_names:
                raise DataError(f"{context}：参考载荷列表为空")
            scheme_index = len(rows)
            candidate_count = len(candidate_names) if judge_candidate_counts is None else judge_candidate_counts[scheme_index] if scheme_index < len(judge_candidate_counts) else -1
            if candidate_count < 0:
                raise DataError(f"{context}：专家文件方案数量超出裁判文件")
            if judge_candidate_counts is not None and len(candidate_names) != candidate_count:
                raise DataError(f"{context}：候选载荷数量{len(candidate_names)}与裁判文件的{candidate_count}不一致")
            coverage = parse_coverage(cell(json_idx), context, candidate_count)
            if len(ref_names) != len(coverage):
                raise DataError(f"{context}：参考载荷列表有{len(ref_names)}项，coverage有{len(coverage)}项")
            original_f1 = None
            if f1_idx is not None:
                raw_f1 = cell(f1_idx)
                try:
                    original_f1 = float(raw_f1)
                except (TypeError, ValueError) as exc:
                    raise DataError(f"{context}：原有F1值不是数字：{raw_f1!r}") from exc
                if not math.isfinite(original_f1) or not 0 <= original_f1 <= 1:
                    raise DataError(f"{context}：原有F1值须在0至1之间：{raw_f1!r}")
            rows.append(Scheme(excel_row, task_name.strip(), ref_names, candidate_count, coverage, original_f1))
        if len(rows) != EXPECTED_SCHEMES:
            raise DataError(f"{role}「{path.name}」Sheet「{sheet.title}」：有效结果{len(rows)}条，应为{EXPECTED_SCHEMES}条")
        return InputData(role, path, sheet.title, names[json_idx], rows)
    finally:
        book.close()


def validate_alignment(inputs: list[InputData]) -> int:
    total = 0
    for index, schemes in enumerate(zip(*(source.rows for source in inputs)), 1):
        baseline = set(schemes[0].coverage)
        total += len(baseline)
        for source, scheme in zip(inputs[1:], schemes[1:]):
            if set(scheme.coverage) != baseline:
                raise DataError(
                    f"方案{index}：{source.role}「{source.path.name}」Excel第{scheme.excel_row}行"
                    f"的参考编号{sorted(scheme.coverage)}与裁判Excel第{schemes[0].excel_row}行的{sorted(baseline)}不同"
                )
    return total


def score(scheme: Scheme) -> tuple[int, int, float, float, float]:
    m = [candidate_ids for candidate_ids, tag in scheme.coverage.values() if tag == "M"]
    effective = {candidate_id for ids in m for candidate_id in ids}
    fcr = len(m) / len(scheme.coverage)
    cur = len(effective) / scheme.candidate_count if scheme.candidate_count else 0.0
    f1 = 2 * fcr * cur / (fcr + cur) if fcr + cur else 0.0
    return len(m), len(effective), fcr, cur, f1


def safe_kappa(first: list[int], second: list[int], name: str, diagnostics: list[list[Any]]) -> float | str:
    from sklearn.metrics import cohen_kappa_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = float(cohen_kappa_score(first, second))
    if not math.isfinite(value):
        reason = "两组标签均为同一个恒定值，偶然一致率为1"
        diagnostics.append(["不可计算", name, "", reason])
        return "不可计算"
    return value


def safe_spearman(first: list[float], second: list[float], diagnostics: list[list[Any]]) -> tuple[float | str, float | str]:
    from scipy.stats import spearmanr
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        warnings.filterwarnings("ignore", message=".*constant.*", category=Warning)
        result = spearmanr(first, second)
    rho, p = float(result[0]), float(result[1])
    if not math.isfinite(rho):
        constant = []
        if len(set(first)) == 1:
            constant.append("裁判F1")
        if len(set(second)) == 1:
            constant.append("专家统一意见F1")
        reason = "、".join(constant) + "恒定，无法计算秩相关" if constant else "相关系数未定义"
        diagnostics.append(["不可计算", "Spearman", "", reason])
        return "不可计算", "不可计算"
    return rho, p


def calculate(inputs: list[InputData], total_refs: int) -> tuple[list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]], dict[str, Any]]:
    import numpy as np
    from sklearn.metrics import confusion_matrix
    judge, expert1, expert2, consensus = inputs
    diagnostics: list[list[Any]] = [["类别", "对象", "Excel行号", "说明"]]
    for source in inputs:
        diagnostics.append(["输入文件", source.role, "", f"{source.path.resolve()}；Sheet={source.sheet}；JSON列={source.json_column}"])
        diagnostics.append(["样本数量", source.role, "", f"{len(source.rows)}组；{sum(len(row.coverage) for row in source.rows)}项参考载荷判断；JSON解析成功{len(source.rows)}条"])
        counts = Counter(tag for row in source.rows for _ids, tag in row.coverage.values())
        diagnostics.append(["标签分布", source.role, "", f"M={counts['M']}，P={counts['P']}，N={counts['N']}，非M={counts['P'] + counts['N']}"])
    coverage_rows = [["方案编号", "任务名称", "参考载荷编号", "参考载荷名称", "裁判原始标签", "专家1原始标签", "专家2原始标签", "专家统一意见标签", "裁判与统一意见是否一致", "专家1与专家2是否一致", "裁判M对应的候选载荷编号", "统一专家意见M对应的候选载荷编号"]]
    score_rows = [["方案编号", "任务名称", "参考载荷数量", "候选载荷数量", "裁判M_count", "裁判有效候选载荷去重数", "裁判FCR", "裁判CUR", "裁判F1", "专家统一意见M_count", "专家统一意见有效候选载荷去重数", "专家统一意见FCR", "专家统一意见CUR", "专家统一意见F1", "裁判百分制得分", "人工百分制得分", "得分差值", "绝对误差"]]
    labels: dict[str, list[int]] = {source.role: [] for source in inputs}
    judge_f1: list[float] = []
    consensus_f1: list[float] = []
    f1_differences = 0
    empty_candidate_schemes = 0
    for scheme_id, schemes in enumerate(zip(*(source.rows for source in inputs)), 1):
        j, e1, e2, c = schemes
        for ref_id in sorted(j.coverage):
            entries = [scheme.coverage[ref_id] for scheme in schemes]
            for source, (_ids, tag) in zip(inputs, entries):
                labels[source.role].append(int(tag == "M"))
            j_ids, j_tag = entries[0]
            _e1_ids, e1_tag = entries[1]
            _e2_ids, e2_tag = entries[2]
            c_ids, c_tag = entries[3]
            coverage_rows.append([
                scheme_id, j.task_name, ref_id, j.reference_names[ref_id - 1], j_tag, e1_tag, e2_tag, c_tag,
                "是" if (j_tag == "M") == (c_tag == "M") else "否",
                "是" if (e1_tag == "M") == (e2_tag == "M") else "否",
                ", ".join(map(str, j_ids)) if j_tag == "M" else "",
                ", ".join(map(str, c_ids)) if c_tag == "M" else "",
            ])
        j_score = score(j)
        c_score = score(c)
        judge_f1.append(j_score[4])
        consensus_f1.append(c_score[4])
        if j.candidate_count == 0:
            empty_candidate_schemes += 1
            diagnostics.append(["警告", f"方案{scheme_id}无候选载荷", j.excel_row, "CUR及F1按0处理"])
        assert j.original_f1 is not None
        if not math.isclose(j.original_f1, j_score[4], rel_tol=0, abs_tol=1e-9):
            f1_differences += 1
            diagnostics.append(["F1差异", f"方案{scheme_id}（{j.task_name}）", j.excel_row, f"原始={j.original_f1:.12g}；重算={j_score[4]:.12g}；差={j_score[4] - j.original_f1:.12g}"])
        difference = (j_score[4] - c_score[4]) * 100
        score_rows.append([
            scheme_id, j.task_name, len(j.coverage), j.candidate_count,
            j_score[0], j_score[1], *j_score[2:],
            c_score[0], c_score[1], *c_score[2:],
            j_score[4] * 100, c_score[4] * 100, difference, abs(difference),
        ])
    diagnostics.append(["F1核查", "裁判原值与JSON重算", "", f"差异{f1_differences}组；比较容差1e-9"])
    diagnostics.append(["特殊情况", "无候选载荷", "", f"{empty_candidate_schemes}组"])
    if not f1_differences and not empty_candidate_schemes:
        diagnostics.append(["异常及警告", "全部数据", "", "未发现"])

    e1_labels, e2_labels = labels[expert1.role], labels[expert2.role]
    j_labels, c_labels = labels[judge.role], labels[consensus.role]
    k_experts = safe_kappa(e1_labels, e2_labels, "专家1与专家2 Kappa", diagnostics)
    k_judge = safe_kappa(j_labels, c_labels, "裁判与专家统一意见 Kappa", diagnostics)
    k_j_e1 = safe_kappa(j_labels, e1_labels, "裁判与专家1 Kappa", diagnostics)
    k_j_e2 = safe_kappa(j_labels, e2_labels, "裁判与专家2 Kappa", diagnostics)
    agreement_experts = float(np.mean(np.equal(e1_labels, e2_labels)))
    agreement_judge = float(np.mean(np.equal(j_labels, c_labels)))
    matrix = confusion_matrix(c_labels, j_labels, labels=[0, 1])
    rho, p = safe_spearman(judge_f1, consensus_f1, diagnostics)
    mae = float(np.mean(np.abs(np.subtract(judge_f1, consensus_f1))) * 100)
    for name, value in (("裁判与专家1 Kappa", k_j_e1), ("裁判与专家2 Kappa", k_j_e2), ("专家1与专家2直接一致率", agreement_experts), ("裁判与统一意见直接一致率", agreement_judge), ("Spearman p值", p), ("MAE（分）", mae)):
        diagnostics.append(["辅助统计", name, "", value])
    for actual in (0, 1):
        for predicted in (0, 1):
            diagnostics.append(["混淆矩阵", f"统一意见={actual}，裁判={predicted}", "", int(matrix[actual, predicted])])
    summary = [["评价对象", "样本数", "专家间κ", "评价模型与专家κ", "Spearmanρ", "MAE"],
               ["载荷配置（参考载荷功能覆盖判定）", total_refs, k_experts, k_judge, "—", "—"],
               ["载荷配置（综合得分）", len(judge.rows), "—", "—", rho, mae],
               [],
               ["说明", "功能覆盖判定将M作为完整覆盖、P/N作为非完整覆盖。"],
               ["说明", "Kappa为普通Cohen's Kappa。"],
               ["说明", "评价模型与专家Kappa以专家统一意见为对照。"],
               ["说明", "综合得分使用FCR和CUR的调和平均。"],
               ["说明", "MAE采用百分制，单位为分。"]]
    stats = {"total_refs": total_refs, "k_experts": k_experts, "k_judge": k_judge, "rho": rho, "p": p, "mae": mae, "f1_differences": f1_differences}
    return summary, coverage_rows, score_rows, diagnostics, stats


def write_sheet(workbook: Workbook, title: str, rows: list[list[Any]]) -> None:
    sheet = workbook.create_sheet(title)
    for row in rows:
        sheet.append(row)
    sheet.freeze_panes = "C2" if title != "数据校验" else "A2"
    if title in ("功能覆盖明细", "综合得分明细", "数据校验"):
        sheet.auto_filter.ref = f"A1:{get_column_letter(sheet.max_column)}{sheet.max_row}"
    header_fill = PatternFill("solid", fgColor="17365D")
    for cell in sheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
    sheet.row_dimensions[1].height = 32
    for column in sheet.columns:
        letter = column[0].column_letter
        if title == "功能覆盖明细" and letter in ("B", "D"):
            width = 38
        elif title == "数据校验" and letter == "D":
            width = 100
        elif title == "综合得分明细" and letter == "B":
            width = 36
        elif title == "论文结果汇总" and letter == "A":
            width = 42
        else:
            width = min(30, max(14, max((len(str(cell.value)) for cell in column if cell.value is not None), default=0) + 2))
        sheet.column_dimensions[letter].width = width
    for row in sheet.iter_rows(min_row=2):
        for cell in row:
            if isinstance(cell.value, float):
                cell.number_format = "0.0000"
            if title == "数据校验" and cell.row > 1 and sheet.cell(cell.row, 2).value == "Spearman p值" and cell.column == 4:
                cell.number_format = "0.0000E+00"
            if title == "功能覆盖明细" and cell.column in (9, 10) and cell.value == "否":
                cell.fill = PatternFill("solid", fgColor="FCE8E6")
            if title == "数据校验" and cell.column == 4:
                cell.alignment = Alignment(wrap_text=True, vertical="top")
    if title == "综合得分明细":
        for row in sheet.iter_rows(min_row=2):
            for col in (15, 16, 17, 18):
                row[col - 1].number_format = '0.0000"分"'
    if title == "论文结果汇总":
        sheet.column_dimensions["B"].width = 14
        for letter in "CDEF":
            sheet.column_dimensions[letter].width = 23
        for row in sheet.iter_rows(min_row=5):
            sheet.merge_cells(start_row=row[0].row, start_column=2, end_row=row[0].row, end_column=6)


def export_result(output_path: Path, tables: tuple[list[list[Any]], list[list[Any]], list[list[Any]], list[list[Any]]]) -> None:
    workbook = Workbook()
    workbook.remove(workbook.active)
    for title, rows in zip(("论文结果汇总", "功能覆盖明细", "综合得分明细", "数据校验"), tables):
        write_sheet(workbook, title, rows)
    workbook.properties.creator = "calculate_payload_judge_agreement.py"
    workbook.save(output_path)


def display(value: Any) -> str:
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def main() -> int:
    args = parse_args()
    try:
        judge = read_input(args.judge, "裁判")
        counts = [row.candidate_count for row in judge.rows]
        expert1 = read_input(args.expert1, "专家1", counts)
        expert2 = read_input(args.expert2, "专家2", counts)
        consensus = read_input(args.consensus, "专家统一意见", counts)
        inputs = [judge, expert1, expert2, consensus]
        total_refs = validate_alignment(inputs)
        summary, coverage_rows, score_rows, diagnostics, stats = calculate(inputs, total_refs)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output_dir / OUTPUT_NAME
        if output_path.exists():
            raise DataError('输出已存在，请设置新的 RUN_ID 或 --output-dir')
        export_result(output_path, (summary, coverage_rows, score_rows, diagnostics))
    except DataError as exc:
        print(f"数据校验失败：{exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"文件操作失败：{exc}", file=sys.stderr)
        return 3
    print(f"校验通过：{EXPECTED_SCHEMES}组方案，{stats['total_refs']}项参考载荷判断。")
    print(f"专家间Kappa：{display(stats['k_experts'])}；裁判与统一意见Kappa：{display(stats['k_judge'])}")
    p_text = f"{stats['p']:.4g}" if isinstance(stats["p"], float) else str(stats["p"])
    print(f"综合得分Spearman rho：{display(stats['rho'])}（p={p_text}）；百分制MAE：{display(stats['mae'])}分")
    print(f"裁判原有F1与重算F1差异：{stats['f1_differences']}组。")
    print(f"结果文件：{output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
