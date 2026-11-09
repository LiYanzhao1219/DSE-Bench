# Large Language Model Evaluation for Deep-Space Exploration

Each task runs directly from a standalone Python file. All execution scripts are in this directory. They do not use a shared entry point, model configuration, model factory, or common protocol class, and do not import other project scripts.

## Configuration and Execution

Python 3.10 or later is recommended. Install the dependencies from this directory:

```powershell
python -m pip install -r requirements.txt
```

Open the desired script in PyCharm, edit the configuration at the top, and run it. Alternatively, execute the file directly from the command line. All paths are relative to the script's own location, independent of the terminal's current working directory. Importing a script does not call a model or write files.

Configure `API_URL`, `MODEL_NAME`, `API_KEY`, `TEMPERATURE`, `REQUEST_TIMEOUT`, `EXTRA_BODY`, and `STREAMING` separately in each script. API keys are empty by default. Scripts do not read a shared configuration or `.env`, and do not automatically select a provider based on the model name. Judge scripts configure the judge model. If the API key is empty, execution stops before making a request. Set reasoning, streaming, and structured-output options according to the API being used. NER, translation, and payload generation provide an `OUTPUT_MODE` setting.

```python
API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL_NAME = 'qwen3-max'
API_KEY = ''
EXTRA_BODY = {}
STREAMING = False
RUN_ID = 'run01'
```

## Scripts and Datasets

| Task | Evaluation Script | Scoring Script | Dataset or Scenarios |
|---|---|---|---|
| Factual knowledge | `evl_fact.py` | Scored during evaluation | Four JSON files under `data/knowledge/fact/` |
| Procedural knowledge | `evl_program.py` | Scored during evaluation | Four JSON files under `data/knowledge/program/` |
| Technical concept explanation | `evl_concept.py` | `cal_concept.py` | Five terminology workbooks under `data/knowledge/concept/` |
| Abbreviation explanation | `evl_concept_words.py` | `cal_concept_words.py` | `data/knowledge/concept/words.xlsx` |
| NER | `evl_ner.py` | Scored during evaluation | `data/nlp/ner.json` |
| Terminology translation | `evl_term_translation.py` | `cal_term_translation.py` | `data/nlp/term_translation.json` |
| Exploration method recommendation | `evl_detect_methods.py` | Scored during evaluation | `data/task/detect_methods.json` |
| Scientific payload configuration | `evl_payload_config.py` | `cal_payload_config.py` | All 60 instances in `data/task/payload_set.json` |
| Observation planning | `evl_observation.py` | `cal_observation.py` | L1/L2 scenarios under `data/task/dataset/01—20/` |

Use `cal_payload_judge_agreement.py` to calculate agreement between the payload judge and experts. No new code for aggregating overall scores across the eight tasks or calculating expert agreement for technical concept explanations has been added in this revision.

Concept generation covers astro, orbit, posture, spacecraft, and survey by default, with header settings specified explicitly in `DATA_FILES`. The posture workbook has no header; its first item is “潮汐锁定” (tidal locking). Abbreviation generation reads the first three columns: domain, abbreviation, and reference explanation. The prompts follow the supplied original code and request both the English and Chinese full forms. Completely empty rows are ignored; missing fields in a nonempty item raise an error. The dataset is not rewritten. The current readers load 582 technical concepts and 145 abbreviations, totaling 727 items. Archived judge results cover 576 of the Chinese technical concepts.

## Four Tasks with Immediate Scoring

Factual knowledge, procedural knowledge, NER, and exploration method recommendation accumulate metrics in memory and display scores in the terminal. They do not save individual answers, per-item statuses, request records, or checkpoints. Running a script again starts a new evaluation. If the output directory already exists, a new directory such as `run01_2` is used automatically without overwriting the previous summary.

Multiple-choice tasks display accuracy for each input file and overall accuracy calculated across all samples. For multiple-select questions, the complete selected option set must match the reference. NER and exploration method recommendation average precision, recall, and F1 across samples, using F1 as the primary metric. NER uses exact matching between deduplicated sets of normalized entity text, without evaluating entity types or positions. An entity name only needs to be output once, even if it appears multiple times in the source text. Exploration methods are parsed based on whether their names appear in the response.

Non-judge requests have a 300-second timeout and at most three total attempts, with delays of 2 and 4 seconds between technical retries. After technical retries are exhausted, evaluation continues to the next item; the final status is `incomplete` and the official score is left empty. Deterministic configuration errors stop execution. Successfully returned responses that cannot be parsed receive zero and are not regenerated.

## Answer Generation and Separate Scoring

For technical concepts, abbreviations, translation, and payload configuration, generate answers first and then run the corresponding `cal` script:

```text
outputs/<concept or concept_words or term_translation or payload_config>/<RUN_ID>/<model_directory>/
    answers.xlsx
    run.json
    requests.json
    samples.json
    summary.json
```

The answer workbook contains the questions, reference answers, and a column for the current model. Request records contain raw responses or technical errors. Sample records contain per-item statuses. The run manifest records inputs, parameters, and code and SDK fingerprints. Special characters in model directory names are replaced, and a short hash is appended when necessary. Use the actual generated directory name.

With the same `RUN_ID` and `RESUME=True`, technical failures and unprocessed items are retried. Completed incorrect answers and format errors are not regenerated. If execution stops after receiving a raw response, local parsing is restored first. Use a new `RUN_ID` when inputs, item order, reference answers, model parameters, or code change. Only one process may write to a run directory at a time.

Set `INPUT_PATH` at the top of the scoring script. By default, it points to `run01/qwen3-max/answers.xlsx` for the corresponding generation task. Update the scoring input when changing the generation model or run ID.

- The concept judge maps levels 0–4 to a 0–100 score. The evaluated model's identity is not included in the prompt.
- The payload judge uses complete functional coverage (FCR) under the scientific objectives and candidate utilization (CUR), based on the union of effective candidate IDs. It calculates a harmonic mean for each scenario and then averages the scores per model.
- Both judge types average valid scores; failed items are not assigned zero. Only successfully scored items are skipped when resuming, and inputs and judge parameters are validated. Outputs are written to `outputs/concept_score/<RUN_ID>/all_models/` and `outputs/payload_config_score/<RUN_ID>/all_models/`, respectively.
- Translation and abbreviation scoring automatically read `samples.json` and `run.json` from the answer workbook's directory and verify the questions, references, models, and answers. Outputs are written to `outputs/translation_score/<RUN_ID>/all_models/` and `outputs/abbreviation_score/<RUN_ID>/all_models/`, including summary and detailed JSON files and `scores.xlsx`.

For external answers without generation records, set `STATUS_PATH` and provide a UTF-8 status sidecar file:

```json
{
  "protocol_version": 2,
  "input_sha256": "SHA256 of the answer xlsx file",
  "models": {"model_column_name": {"2": "ok", "3": "invalid_response", "4": "api_error"}}
}
```

Row numbers are actual Excel row numbers. Other allowed statuses are `config_error`, `execution_error`, and `unprocessed`; missing rows are treated as unprocessed. Format errors receive zero. Technical failures or unprocessed items leave the official score empty. Historical ERROR strings do not establish the failure type, so statuses are not fabricated automatically and archived workbooks are not automatically rescored.

## Observation Planning

Set `LEVEL='l1'` or `'l2'` in the generation script. It saves answers without calculating metrics:

```text
outputs/observation/<RUN_ID>/<l1 or l2>/<model_directory>/01.json … 20.json
```

The same directory stores the manifest, raw responses, and resume state. Completed answers are not overwritten, and returned invalid responses are not regenerated. Technical failures can be retried. L2 reads `max_power.txt` for each scenario and includes the power limit in the resume fingerprint. L1 does not read L2 power data. Prompts are included directly in the script and do not depend on external source files.

`cal_observation.py` runs entirely offline and requires neither a model SDK nor an API key. Its default is `INPUT_PATH=outputs/observation/<RUN_ID>`; change it to `outputs/observation` to analyze existing archives. The input directory should contain `l1/<model>/<scenario>.json` and `l2/<model>/<scenario>.json`. By default, model names are read from the plan JSON files; they can also be specified through `MODEL_NAMES`. Missing answers for specified models receive zero.

Each model is evaluated on a fixed set of 20 scenarios per level. Missing, blank, or unparseable answers receive zero. Invalid scientific-objective, window, or power inputs raise errors. Cached scores and the `generated` flag are ignored. Outputs are written to `outputs/observation_score/<RUN_ID>/`, including metric TXT files for both levels, per-item diagnostics and summary JSON files, and `scores.xlsx` with L1, L2, and combined sheets. Existing scoring outputs are not overwritten; change `OUTPUT_PATH` to use a new location.

The current scoring rules are retained: OSR is calculated from actual observation durations; shared data requirements use the maximum duration required by any objective; overlapping intervals for the same payload are merged; and declared duration values are used only for diagnostics. The engineering component C and scientific components are scored separately. C=0 does not zero the other components. L1 and L2 are combined into a 0–100 score across all 40 scenarios.

## Expert Agreement and Archives

By default, `INPUT_DIR` in `cal_payload_judge_agreement.py` reads the existing human evaluation materials under `result/payload_judge_agreement`. New statistics are written to `outputs/payload_judge_agreement/<RUN_ID>/`. Explicit command-line arguments are also supported. Records are aligned by row order across workbooks; confirm that configurations appear in the same order before use.

Existing `data`, `outputs`, and `result` directories remain in place. Existing datasets, answers, and scores are not automatically moved, rewritten, or recalculated.
