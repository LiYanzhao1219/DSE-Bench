# DSE-Bench: A Chinese-Language Benchmark for Deep-Space Exploration

 [Codebook](CODEBOOK.md)

## Description

DSE-Bench provides datasets and code for evaluating large language models on knowledge, specialized information processing, and representative design and planning tasks in Chinese-language deep-space exploration. The benchmark contains **10,236 instances across eight tasks and three capability dimensions**: Domain Knowledge, Domain Semantic Processing, and Task-Scenario Adaptation.

The repository includes task-specific prompts, response parsers, scoring scripts, available model outputs, and expert-agreement materials. Each evaluation task runs through a standalone Python script. The framework was developed from the processing requirements of 28 representative deep-space exploration activities.

## Dataset Information

| Capability dimension | Task | Instances | Dataset location |
|---|---|---:|---|
| Domain Knowledge | Factual knowledge | 1,200 | Four JSON files under `data/knowledge/factual_knowledge/` |
| Domain Knowledge | Conceptual knowledge | 727 | Six Excel workbooks under `data/knowledge/conceptual_knowledge/` |
| Domain Knowledge | Procedural knowledge | 375 | Four JSON files under `data/knowledge/procedural_knowledge/` |
| Domain Semantic Processing | Named entity recognition (NER) | 200 | `data/nlp/named_entity_recognition.json` |
| Domain Semantic Processing | Terminology translation | 7,554 | `data/nlp/terminology_translation.json` |
| Task-Scenario Adaptation | Exploration method recommendation | 80 | `data/task/exploration_method_recommendation.json` |
| Task-Scenario Adaptation | Scientific payload configuration | 60 | `data/task/scientific_payload_configuration.json` |
| Task-Scenario Adaptation | Payload observation-window planning | 40 | L1/L2 scenarios under `data/task/dataset/` |
| **Total** | | **10,236** | |

Conceptual knowledge comprises **582 Chinese technical concept explanations** across five subject workbooks and **145 abbreviation explanations** in `technical_abbreviations.xlsx`. They are separate code workflows within one benchmark task. Observation planning consists of 20 numbered scenarios (`01`–`20`), each with L1 and L2 settings. Companion files at the same scenario number and level form one instance.

JSON question banks contain arrays of records. Excel workbooks have a header row; empty rows are excluded from instance counts. Chinese questions, terms, labels, and references are retained to preserve the intended assessment conditions and NER character offsets. English field definitions, complete file inventories, units, and representative examples are provided in the [Codebook](CODEBOOK.md).

## Code Information

The `eval_` scripts generate model responses and, where applicable, score them immediately. Separate `score_` scripts evaluate saved responses. Concept explanation and payload configuration scoring use an LLM judge; abbreviation, translation, and observation scoring use local rules or programmatic checks.

| Task or subtask | Evaluation / response generation | Scoring |
|---|---|---|
| Factual knowledge | `eval_fact.py` | During evaluation |
| Procedural knowledge | `eval_program.py` | During evaluation |
| Technical concept explanation | `eval_concept.py` | `score_concept.py` |
| Abbreviation explanation | `eval_concept_words.py` | `score_concept_words.py` |
| NER | `eval_ner.py` | During evaluation |
| Terminology translation | `eval_term_translation.py` | `score_term_translation.py` |
| Exploration method recommendation | `eval_detect_methods.py` | During evaluation |
| Scientific payload configuration | `eval_payload_config.py` | `score_payload_config.py` |
| Observation planning | `eval_observation.py` | `score_observation.py` |

`score_payload_judge_agreement.py` is an additional utility for comparing payload-configuration judge results with expert assessments.

| Location | Contents |
|---|---|
| `data/` | Task datasets and planning inputs |
| `outputs/` | Saved responses where available, run records, and generated scores |
| `result/` | Archived evaluation results, expert assessments, and agreement statistics |
| `CODEBOOK.md` | Dataset structures, original-field meanings, examples, and evaluation interpretation |
| `requirements.txt` | Python dependencies |

Input paths are resolved relative to the script directory. Run the commands and data-loading examples below from the repository root. Script options are configured at the top of each file; there is no shared command-line interface for all tasks.

## Requirements

Use **Python 3.10 or later**. Install the dependencies with:

```sh
python -m pip install -r requirements.txt
```

| Dependencies | Purpose |
|---|---|
| `langchain-core`, `langchain-openai` | Prompt construction, model invocation, and compatible chat-model API integration |
| `pydantic>=2` | Response schemas and validation |
| `openpyxl`, `pandas` | Excel dataset and result handling |
| `numpy`, `scipy`, `scikit-learn` | Numerical processing and statistical or agreement analysis |

Loading datasets and running local scoring do not require a model API. Generating new responses requires a compatible model endpoint and credentials. The LLM-judge scoring scripts require their own endpoint, model, and credentials.

## Usage Instructions

### 1. Load the data without calling a model

For example, read the terminology translation dataset with Python's standard library:

```python
import json
from pathlib import Path

path = Path("data/nlp/terminology_translation.json")
records = json.loads(path.read_text(encoding="utf-8-sig"))
print(f"Instances: {len(records)}")
print(records[0]["zh"])  # Chinese source term
print(records[0]["en"])  # List of reference English equivalents
```

To inspect an Excel terminology workbook:

```python
from openpyxl import load_workbook

path = "data/knowledge/conceptual_knowledge/astronomy_terms.xlsx"
workbook = load_workbook(path, read_only=True, data_only=True)
try:
    sheet = workbook.active
    for term, reference in sheet.iter_rows(
        min_row=2, max_row=4, max_col=2, values_only=True
    ):
        print(term, reference)
finally:
    workbook.close()
```

The first two columns are `名词` (term) and `解释` (reference explanation). Use the original keys and column names when reading Chinese fields; the Codebook supplies their English meanings.

### 2. Configure a new evaluation

Edit the configuration at the top of the selected `eval_` script. For example:

```python
API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-max"
API_KEY = "YOUR_API_KEY"
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 300
EXTRA_BODY = {}
STREAMING = False
RUN_ID = "experiment01"
```

Replace the endpoint, model name, and key with your provider's values. Set `EXTRA_BODY`, `STREAMING`, and the task's `OUTPUT_MODE`, where available, to match the provider's supported response format. Select a new `RUN_ID` for a new experiment.

Runs with manifests validate their inputs, configuration, and code hashes before resuming. Changing dataset paths, data, or code can invalidate an existing run; use a new `RUN_ID` rather than reusing an incompatible directory. Keep generation and scoring paths aligned with the experiment you intend to evaluate.

### 3. Run tasks with immediate scoring

Factual knowledge, procedural knowledge, NER, and exploration method recommendation score responses during evaluation. For example:

```sh
python eval_ner.py
```

Use the corresponding script in the code table for the other tasks. The script prints its summary and saved output location. The current NER default matches normalized, deduplicated entity names rather than entity types or character spans.

### 4. Generate responses, then score them

Technical concept explanation, abbreviation explanation, terminology translation, and payload configuration save answers before separate scoring. For a concept-explanation experiment:

1. Configure `eval_concept.py` and generate responses:

   ```sh
   python eval_concept.py
   ```

2. In `score_concept.py`, set `INPUT_PATH` to the generated `answers.xlsx`, configure the judge API independently, and select the scoring `RUN_ID` and `OUTPUT_PATH`.
3. Run the scorer:

   ```sh
   python score_concept.py
   ```

Generated answer workbooks use `outputs/<task>/<RUN_ID>/<model_directory>/answers.xlsx`. The generation task directories are `concept`, `concept_words`, `term_translation`, and `payload_config`. The scorer defaults point to `run01/qwen3-max/answers.xlsx` under their respective task directories; update them to your actual run.

For abbreviation and translation scoring, retain the **complete generation directory**, including `answers.xlsx`, `samples.json`, and `run.json`. Their scorers require answer-status records: by default, `STATUS_PATH = None` resolves to `samples.json` beside the answer workbook. Set `STATUS_PATH` explicitly if the matching records are elsewhere. A workbook alone is insufficient for these two scoring workflows.

| Scoring workflow | Output directory under `outputs/` |
|---|---|
| Concept explanation | `concept_score/<RUN_ID>/all_models/` |
| Abbreviation explanation | `abbreviation_score/<RUN_ID>/all_models/` |
| Terminology translation | `translation_score/<RUN_ID>/all_models/` |
| Payload configuration | `payload_config_score/<RUN_ID>/all_models/` |

The concept and payload judge scripts expose an `OUTPUT_PATH`; the abbreviation and translation scorers construct the directories shown above from their scoring `RUN_ID`. Use the printed output location to find generated files.

### 5. Run observation planning at both levels

In `eval_observation.py`, set `LEVEL = "l1"`, configure the model and `RUN_ID`, then run:

```sh
python eval_observation.py
```

Change `LEVEL` to `"l2"` and run the same script again with the same experiment `RUN_ID`. L1 provides scientific requirements and observation windows. L2 adds mounting-plane, payload-power, and total-power constraints.

After generating both levels, configure `score_observation.py` with the matching `RUN_ID`, `INPUT_PATH`, and `OUTPUT_PATH`, then run:

```sh
python score_observation.py
```

Generation results are stored under `outputs/observation/<RUN_ID>/`, separated by level and model. Scores are stored under `outputs/observation_score/<RUN_ID>/`. To score archived responses directly under `outputs/observation/`, set `INPUT_PATH` to that directory and choose a separate scoring output location.

### 6. Optional: calculate payload judge–expert agreement

The following utility uses the expert-assessment workbooks under `result/payload_judge_agreement/` and writes to its configured `OUTPUT_DIR`:

```sh
python score_payload_judge_agreement.py
```

For custom input workbooks and an output directory, inspect the available arguments:

```sh
python score_payload_judge_agreement.py --help
```

## Methodology

1. **Construct task-specific datasets.** Extract facts and procedures from mission documentation, technical literature, and aerospace standards; pair concepts and translations with reference terminology. Use LLM-assisted question drafting where applicable, followed by source-based answer checks.
2. **Prepare annotations and scenarios.** Apply relevance filtering, terminology normalization, and duplicate removal as appropriate. Manually annotate NER passages, abstract identifying details in historical-mission scenarios, and simulate observation windows with scientific and engineering requirements.
3. **Generate and parse responses.** Supply the task-specific Chinese inputs to the configured model and validate the required answer structure. Keep reference answers separate from model inputs.
4. **Apply task-adapted scoring.** Use accuracy for reference-based knowledge and translation tasks, F1 for entity and exploration-method sets, LLM judging for technical concept explanations and payload functional coverage, and programmatic requirement/constraint checks for observation planning. Expert assessments support reliability analysis of selected judge-based metrics.

The [Codebook](CODEBOOK.md) explains field roles, normalization, payload functional coverage, and planning constraints. The accompanying paper provides the full construction methodology, quality-control results, and metric formulas.

## Citation

If you use DSE-Bench or its evaluation code in research, please cite the accompanying paper:

Yanzhao Li, Haolin Zhang, Wenlong Niu, Xiaoyi Du, Guangyu Wang, and Xiaodong Peng. Evaluating large language models for deep-space exploration: a task-requirement-driven Chinese-language benchmark.

The framework's 28 representative engineering activities and their supporting sources are documented in the [task source-verification table](28%20Typical%20Tasks%20Source%20Verification.md).

## Feedback and Contributions

For questions or proposed corrections, contact the author, Yanzhao Li, at [liyanzhao21@mails.ucas.ac.cn](mailto:liyanzhao21@mails.ucas.ac.cn). Include the task and script name, Python and dependency versions, a minimal example, and the relevant error message. For data corrections, identify the file and record ID or spreadsheet row and provide supporting technical evidence. Proposed code changes should describe their effect on data loading or scoring and the checks performed.
