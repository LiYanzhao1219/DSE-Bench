# 深空探测领域大语言模型评测

[English](README_EN.md)

本项目提供面向中文深空探测领域的大语言模型评测数据、评测脚本、评分脚本和结果材料，涵盖领域知识、领域语义处理和任务场景适应三个维度的八类评测任务。各任务通过独立 Python 脚本运行。

## 任务与题库

| 任务 | 评测脚本 | 评分脚本 | 题库或场景 |
|---|---|---|---|
| 事实性知识 | `evl_fact.py` | 评测时评分 | `data/knowledge/fact/` 下四个 JSON |
| 程序性知识 | `evl_program.py` | 评测时评分 | `data/knowledge/program/` 下四个 JSON |
| 术语解释 | `evl_concept.py` | `cal_concept.py` | `data/knowledge/concept/` 下五个术语表 |
| 缩略词解释 | `evl_concept_words.py` | `cal_concept_words.py` | `data/knowledge/concept/words.xlsx` |
| 命名实体识别（NER） | `evl_ner.py` | 评测时评分 | `data/nlp/ner.json` |
| 术语翻译 | `evl_term_translation.py` | `cal_term_translation.py` | `data/nlp/term_translation.json` |
| 探测方式确认 | `evl_detect_methods.py` | 评测时评分 | `data/task/detect_methods.json` |
| 科学载荷配置 | `evl_payload_config.py` | `cal_payload_config.py` | `data/task/payload_set.json`，共 60 条 |
| 观测规划 | `evl_observation.py` | `cal_observation.py` | `data/task/dataset/` 下 01—20 的 L1/L2 场景 |

概念知识包含 **582 道中文术语解释题和 145 道缩略词解释题，共 727 题**。术语覆盖 astro、orbit、posture、spacecraft 和 survey 五类。

载荷配置的裁判与专家一致性统计使用 `cal_payload_judge_agreement.py`。

## 安装

建议使用 Python 3.10 或以上版本。在仓库根目录执行：

```powershell
python -m pip install -r requirements.txt
```

## 配置

运行前，在所选脚本顶部填写模型接口地址、模型名称和 API 密钥，并设置运行编号：

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

根据接口要求设置 `EXTRA_BODY`、`STREAMING` 和脚本提供的 `OUTPUT_MODE`。使用模型裁判的评分脚本需单独配置裁判模型的 API 参数。文件路径以脚本所在目录为基准。

## 基本运行

### 即时评分任务

事实性知识、程序性知识、NER 和探测方式确认在评测过程中计算指标并显示成绩。选择所需脚本运行，例如：

```powershell
python evl_ner.py
```

### 生成回答后评分

术语解释、缩略词解释、术语翻译和载荷配置先生成回答，再执行对应评分脚本。例如，术语解释任务运行：

```powershell
python evl_concept.py
python cal_concept.py
```

生成回答保存在 `outputs/<task>/<RUN_ID>/<model_directory>/answers.xlsx`。评分前，将评分脚本的 `INPUT_PATH` 设置为对应回答文件；默认配置对应 `run01/qwen3-max/answers.xlsx`。其他生成任务按任务表选择相应脚本。

### 观测规划

在 `evl_observation.py` 中将 `LEVEL` 分别设置为 `'l1'` 和 `'l2'`，各运行一次：

```powershell
python evl_observation.py
```

两级回答生成后，执行评分：

```powershell
python cal_observation.py
```

生成结果保存在 `outputs/observation/<RUN_ID>/`，评分结果保存在 `outputs/observation_score/<RUN_ID>/`。评分脚本的 `INPUT_PATH` 应指向对应的生成目录；对 `outputs/observation/` 中的结果评分时，将其设置为该目录。

## 目录说明

| 目录 | 内容 |
|---|---|
| `data/` | 各任务题库及观测规划场景 |
| `outputs/` | 模型回答、生成记录及评分脚本输出 |
| `result/` | 评测结果、专家评价及一致性统计材料 |

