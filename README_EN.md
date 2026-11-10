# Large Language Model Evaluation for Deep-Space Exploration

[中文](README_ZH.md)

This project provides datasets, evaluation scripts, scoring scripts, and result materials for evaluating large language models in Chinese-language deep-space exploration. It covers eight evaluation tasks across three dimensions: Domain Knowledge, Domain Semantic Processing, and Task-Scenario Adaptation. Each task runs through a standalone Python script.

## Tasks and Datasets

| Task | Evaluation Script | Scoring Script | Dataset or Scenarios |
|---|---|---|---|
| Factual knowledge | `evl_fact.py` | Scored during evaluation | Four JSON files under `data/knowledge/fact/` |
| Procedural knowledge | `evl_program.py` | Scored during evaluation | Four JSON files under `data/knowledge/program/` |
| Technical concept explanation | `evl_concept.py` | `cal_concept.py` | Five terminology workbooks under `data/knowledge/concept/` |
| Abbreviation explanation | `evl_concept_words.py` | `cal_concept_words.py` | `data/knowledge/concept/words.xlsx` |
| Named entity recognition (NER) | `evl_ner.py` | Scored during evaluation | `data/nlp/ner.json` |
| Terminology translation | `evl_term_translation.py` | `cal_term_translation.py` | `data/nlp/term_translation.json` |
| Exploration method recommendation | `evl_detect_methods.py` | Scored during evaluation | `data/task/detect_methods.json` |
| Scientific payload configuration | `evl_payload_config.py` | `cal_payload_config.py` | 60 instances in `data/task/payload_set.json` |
| Observation planning | `evl_observation.py` | `cal_observation.py` | L1/L2 scenarios numbered 01–20 under `data/task/dataset/` |

Conceptual knowledge comprises **582 Chinese technical concept explanation items and 145 abbreviation explanation items, totaling 727 items**. Technical concepts cover astro, orbit, posture, spacecraft, and survey.

Use `cal_payload_judge_agreement.py` to calculate agreement between the payload-configuration judge and experts.

## Installation

Python 3.10 or later is recommended. Run the following command from the repository root:

```powershell
python -m pip install -r requirements.txt
```

## Configuration

Before running a task, set the API endpoint, model name, API key, and run ID at the top of the selected script:

```python
API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL_NAME = 'qwen3-max'
API_KEY = ''
TEMPERATURE = 0.0
REQUEST_TIMEOUT = 300
EXTRA_BODY = {}
STREAMING = False
RUN_ID = 'run01'
```

Set `EXTRA_BODY`, `STREAMING`, and the script's `OUTPUT_MODE`, where available, according to the API requirements. Scoring scripts that use an LLM judge require separate API configuration for the judge model. File paths are relative to the script's location.

## Basic Usage

### Tasks with Immediate Scoring

Factual knowledge, procedural knowledge, NER, and exploration method recommendation calculate metrics and display scores during evaluation. Run the script for the desired task, for example:

```powershell
python evl_ner.py
```

### Generate Answers, Then Score

Technical concept explanation, abbreviation explanation, terminology translation, and payload configuration first generate answers and then use the corresponding scoring script. For example, run the technical concept explanation task with:

```powershell
python evl_concept.py
python cal_concept.py
```

Generated answers are saved to `outputs/<task>/<RUN_ID>/<model_directory>/answers.xlsx`. Before scoring, set the scoring script's `INPUT_PATH` to the corresponding answer file. The default configuration points to `run01/qwen3-max/answers.xlsx`. For other generation tasks, select the matching scripts from the task table.

### Observation Planning

In `evl_observation.py`, set `LEVEL` to `'l1'` and `'l2'` in turn, running the script once for each level:

```powershell
python evl_observation.py
```

After generating answers for both levels, run scoring:

```powershell
python cal_observation.py
```

Generated results are saved to `outputs/observation/<RUN_ID>/`, and scoring results to `outputs/observation_score/<RUN_ID>/`. Set the scoring script's `INPUT_PATH` to the corresponding generation directory. To score results stored under `outputs/observation/`, set it to that directory.

## Directory Structure

| Directory | Contents |
|---|---|
| `data/` | Task datasets and observation-planning scenarios |
| `outputs/` | Model answers, generation records, and scoring-script outputs |
| `result/` | Evaluation results, expert assessments, and agreement statistics |
