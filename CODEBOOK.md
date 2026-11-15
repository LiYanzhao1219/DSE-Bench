# DSE-Bench Codebook

This codebook explains how to read the DSE-Bench datasets and how their fields are used by the evaluation code. It preserves the original field names and Chinese examples, with English definitions and explanations for readers who do not read Chinese. For installation, API configuration, and execution instructions, see the [README](README.md).

Dataset construction and metric definitions are described in the accompanying paper, *Evaluating large language models for deep-space exploration: a task-requirement-driven Chinese-language benchmark*. This document focuses on data interpretation rather than reproducing the paper's scoring formulas.

## 1. Dataset Overview

DSE-Bench evaluates large language models on Chinese-language knowledge, specialized information processing, and representative planning and design scenarios in deep-space exploration. It contains **10,236 instances across eight tasks and three capability dimensions**.

| Capability dimension | Evaluation task | Instances |
|---|---|---:|
| Domain Knowledge | Factual knowledge | 1,200 |
| Domain Knowledge | Conceptual knowledge | 727 |
| Domain Knowledge | Procedural knowledge | 375 |
| Domain Semantic Processing | Named entity recognition (NER) | 200 |
| Domain Semantic Processing | Terminology translation | 7,554 |
| Task-Scenario Adaptation | Exploration method recommendation | 80 |
| Task-Scenario Adaptation | Scientific payload configuration | 60 |
| Task-Scenario Adaptation | Payload observation-window planning | 40 |
| **Total** | | **10,236** |

Conceptual knowledge comprises **582 Chinese technical concept explanations and 145 English-abbreviation explanations**. These are two subtasks within one benchmark task. Observation planning consists of 20 numbered scenarios, each evaluated under L1 and L2 settings, giving 40 instances.

**Language convention.** The original Chinese content is retained because translation can change technical meaning and assessment difficulty. NER character offsets also refer directly to the original Chinese text. English explanations in this codebook are reading aids; the Chinese questions, labels, and reference answers remain the evaluation data. Use the original keys when loading files, even where an English equivalent is given below.

## 2. Dataset Organization

All paths are relative to the project root. Most JSON datasets are top-level arrays, with one object per instance. Excel counts exclude the header and empty rows. Saved model responses and scores are not additional benchmark instances.

| Task | Dataset path | Format | Instances | Contents |
|---|---|---|---:|---|
| Factual knowledge | `data/knowledge/factual_knowledge/celestial_bodies.json` | JSON array | 200 | Single-choice questions about celestial bodies |
| Factual knowledge | `data/knowledge/factual_knowledge/exploration_missions.json` | JSON array | 400 | Single-choice questions about missions |
| Factual knowledge | `data/knowledge/factual_knowledge/scientific_payloads.json` | JSON array | 400 | Single-choice questions about scientific payloads |
| Factual knowledge | `data/knowledge/factual_knowledge/launch_vehicles.json` | JSON array | 200 | Single-choice questions about launch vehicles |
| Concept explanations | `data/knowledge/conceptual_knowledge/astronomy_terms.xlsx` | XLSX, `Sheet1` | 143 | Astronomy terms and reference explanations |
| Concept explanations | `data/knowledge/conceptual_knowledge/orbital_dynamics_terms.xlsx` | XLSX, `Sheet2` | 122 | Orbital dynamics terms and explanations |
| Concept explanations | `data/knowledge/conceptual_knowledge/attitude_dynamics_and_control_terms.xlsx` | XLSX, `Sheet1` | 74 | Attitude dynamics and control terms |
| Concept explanations | `data/knowledge/conceptual_knowledge/spacecraft_engineering_terms.xlsx` | XLSX, `Sheet1` | 135 | Spacecraft engineering terms |
| Concept explanations | `data/knowledge/conceptual_knowledge/tracking_telemetry_and_command_terms.xlsx` | XLSX, `Sheet1` | 108 | Tracking, telemetry, and command terms |
| Abbreviation explanations | `data/knowledge/conceptual_knowledge/technical_abbreviations.xlsx` | XLSX, `Sheet1` | 145 | Domain context, abbreviations, and reference expansions |
| Procedural knowledge | `data/knowledge/procedural_knowledge/tracking_and_control_procedures.json` | JSON array | 50 | Tracking and control procedures |
| Procedural knowledge | `data/knowledge/procedural_knowledge/launch_procedures.json` | JSON array | 50 | Launch procedures |
| Procedural knowledge | `data/knowledge/procedural_knowledge/space_systems_development_procedures.json` | JSON array | 100 | Space systems development and engineering |
| Procedural knowledge | `data/knowledge/procedural_knowledge/spacecraft_testing_procedures.json` | JSON array | 175 | Spacecraft testing procedures |
| NER | `data/nlp/named_entity_recognition.json` | JSON array with nested annotations | 200 | Source passages, entity spans, and annotation metadata |
| Terminology translation | `data/nlp/terminology_translation.json` | JSON array | 7,554 | Chinese terms and reference English equivalents |
| Exploration method recommendation | `data/task/exploration_method_recommendation.json` | JSON array | 80 | Mission conditions and reference method sets |
| Scientific payload configuration | `data/task/scientific_payload_configuration.json` | JSON array | 60 | Mission requirements and reference payload combinations |
| Observation planning, L1 | `data/task/dataset/{NN}/l1/object_{NN}.json` and `detect_windows_no_plane.json` | Two JSON files per instance | 20 | Scientific objectives, duration requirements, and available windows |
| Observation planning, L2 | `data/task/dataset/{NN}/l2/object_{NN}.json`, `detect_windows.json`, and `max_power.txt` | Two JSON files and one TXT per instance | 20 | Requirements and windows, plus mounting-plane and power constraints |

`{NN}` ranges from `01` to `20`. Files under the same scenario number and level form **one planning instance**; individual objectives, payloads, and windows are not counted as separate instances.

All six Excel workbooks have a header in the first row. The loaders read the first two columns for technical concepts and the first three for abbreviations, skipping empty rows. Unnamed empty trailing columns are not data fields. The loaders use the active worksheet; its name is listed in the table above.

## 3. Field Definitions

Types use standard JSON terminology: `integer`, `number`, `string`, `object`, `array<T>`, and `null`. In a field path, `[]` denotes an array element; `<payload_name>` denotes a dictionary key containing a payload's actual name. These are documentation conventions, not literal keys.

### 3.1 Factual and Procedural Knowledge

The eight JSON files under `factual_knowledge/` and `procedural_knowledge/` share this structure.

| Field | Type | Meaning and use |
|---|---|---|
| `id` | integer | Item identifier within the file. Include the file path when identifying an item across datasets. |
| `question` | string | Chinese question stem supplied to the model. |
| `options` | `array<string>` | Ordered answer options. Each string includes its option letter and text, such as `A. …`. |
| `answer` | string | Reference option letters: one letter for factual questions, or concatenated letters such as `ACD` for procedural questions. |

A multi-select answer represents a set of correct options, not an ordering of procedural steps.

### 3.2 Conceptual Knowledge: Technical Concept Explanations

The five subject-specific terminology workbooks share two columns.

| Original column | Type | Meaning and use |
|---|---|---|
| `名词` | string | Technical term to be explained. |
| `解释` | string | Reference definition or explanation used for scoring. |

There is no separate ID column. Locate a record by workbook path, worksheet, and row number. Subject categories are represented by the workbooks. The code retains the internal category identifiers `astro`, `orbit`, `posture`, `spacecraft`, and `survey` through an explicit filename mapping.

### 3.3 Conceptual Knowledge: Abbreviation Explanations

Applies to `technical_abbreviations.xlsx`.

| Original column | Type | Meaning and use |
|---|---|---|
| `领域` | string | Domain context that disambiguates the abbreviation. |
| `缩写` | string | English abbreviation to be expanded. |
| `解释` | string | Reference English expansion and Chinese equivalent, stored together in one cell. |

The domain values are `理论部分` (theoretical topics), `测控相关` (tracking, telemetry, and command), `轨道力学` (orbital mechanics), `发动机部分` (propulsion engines), `运载火箭部分` (launch vehicles), `发射场部分` (launch sites), `空间站部分` (space stations), `导航部分` (navigation), `遥感部分` (remote sensing), and `卫星总体部分` (satellite systems engineering).

### 3.4 Named Entity Recognition

`data/nlp/named_entity_recognition.json` contains 200 passages and 3,451 entity-mention annotations in a Label Studio character-span format. A mention is an occurrence in a passage; repeated mentions need not correspond to distinct entity names.

| Field path | Type | Meaning and use |
|---|---|---|
| `id` | integer | Passage identifier. |
| `text` | string | Complete source passage, preserving its original characters and punctuation. |
| `label` | `array<object>` | Entity-span annotations for this passage. |
| `label[].start` | integer | Zero-based start character offset, inclusive. |
| `label[].end` | integer | End character offset, exclusive. |
| `label[].text` | string | Entity mention as it appears in the source passage. |
| `label[].labels` | `array<string>` | Entity type labels; each span has one label stored in an array. |
| `annotator` | integer | Internal annotator identifier. |
| `annotation_id` | integer | Internal annotation-record identifier. |
| `created_at` | string | Creation timestamp in ISO 8601 format, ending in `Z`. |
| `updated_at` | string | Update timestamp in the same format. |
| `lead_time` | number or null | Recorded annotation-time metadata; may be missing and is not used in scoring. |

Offsets define the half-open interval `[start, end)`, so `label[].text` equals `text[start:end]` using Python string indexing. They count characters, not UTF-8 bytes. Changing or translating the source text invalidates the original offsets. Spans may overlap, and annotations are not necessarily sorted by offset.

**The 29 predefined entity types are:**

| Original label | English meaning | Original label | English meaning |
|---|---|---|---|
| `探测任务` | Exploration mission | `任务别名` | Mission alias |
| `国家` | Country | `研制单位` | Developing organization |
| `探测对象` | Exploration target | `探测器` | Spacecraft or probe |
| `探测手段` | Exploration technique | `有效载荷` | Scientific payload |
| `发射时间` | Launch date or time | `发射场` | Launch site |
| `运载火箭` | Launch vehicle | `发射质量` | Launch mass |
| `任务负责人` | Mission lead | `科学家` | Scientist |
| `对应学科` | Associated scientific discipline | `任务经费` | Mission budget |
| `任务结果` | Mission outcome | `所属探测计划` | Parent exploration program |
| `设计寿命` | Design lifetime | `科学目标` | Scientific objectives |
| `尺寸` | Dimensions | `质量` | Mass |
| `形状` | Shape | `位置` | Position or location |
| `公转周期` | Orbital period | `自转周期` | Rotation period |
| `卫星` | Natural satellite | `表面颜色` | Surface color |
| `大气组成和气候` | Atmospheric composition and climate | | |

Annotation metadata supports traceability. It is not part of the model input or the reference entity-name matching procedure described in Section 6.2.

### 3.5 Terminology Translation

Applies to `data/nlp/terminology_translation.json`.

| Field | Type | Meaning and use |
|---|---|---|
| `id` | integer | Term identifier used to align batched inputs and responses. |
| `zh` | string | Chinese source term to translate. |
| `en` | `array<string>` | One or more reference English equivalents. |

`en` is a nonempty array even when only one equivalent is supplied. Its elements are alternative references; the model is asked to return one translation for each `id`.

### 3.6 Exploration Method Recommendation

Applies to `data/task/exploration_method_recommendation.json`.

| Original field | Type | Meaning and use |
|---|---|---|
| `id` | integer | Scenario identifier. |
| `任务` | string | Historical mission name retained for source identification. |
| `时间` | string | Approximate mission era or time context, rather than a uniformly formatted date. |
| `探测对象` | string | Target body or target description. |
| `科学目标` | string | Scientific objectives to be achieved. |
| `探测方式` | `array<string>` | Reference exploration-method set; multiple methods may be correct. |

The six allowed method labels are:

| Original value | English meaning |
|---|---|
| `飞越探测` | Flyby exploration |
| `环绕探测` | Orbital exploration |
| `着陆探测` | Lander-based exploration |
| `巡视探测` | Rover-based exploration |
| `伴飞探测` | Accompanying-flight exploration: flying alongside the target |
| `采样返回` | Sample return |

The parser recognizes these Chinese names. For example, `环绕探测，着陆探测` selects orbital and lander-based exploration; translating the labels in a model response would change what the existing parser recognizes.

### 3.7 Scientific Payload Configuration

Applies to `data/task/scientific_payload_configuration.json`.

| Original field | Type | Meaning and use |
|---|---|---|
| `id` | integer or string | Scenario identifier; preserve its original type. |
| `任务` | string | Historical mission name or alias. |
| `时间` | string | Approximate mission era or time setting. |
| `探测对象` | string | Target properties, class, or an abstracted description. |
| `科学目标` | string | Scientific objectives that determine which payload functions are needed. |
| `探测方式` | string | Given exploration-mode description. Unlike Section 3.6, this is free text, not an array of fixed labels. |
| `载荷` | `array<string>` | Reference scientific payload combination. An entry can include a name, abbreviation, technical parameters, or a functional description. |

Read each payload entry as a complete string. Embedded wavelengths, resolutions, or capability descriptions are not separate numeric fields. Structured engineering constraints are provided in the observation-planning data described next.

### 3.8 Payload Observation-Window Planning

L1 provides scientific requirements and available observation windows. L2 additionally specifies mounting-plane groups, payload operating powers, and a total concurrent-power limit.

#### 3.8.1 Scientific requirements: `object_{NN}.json`

The top-level array contains scientific objectives. Both levels use the same structure.

| Field path | Type | Meaning and use |
|---|---|---|
| `[].name` | string | Scientific objective name. |
| `[].loadData` | `array<object>` | Payload observation requirements associated with this objective. |
| `[].loadData[].pay_load` | string | Required payload name, corresponding to a key in the window file. Preserve the spelling `pay_load`. |
| `[].loadData[].require_time/min` | integer | Required cumulative observation duration in **minutes**. The slash is part of the literal key. |

Convert input durations from minutes to seconds before comparing them with scheduled time intervals.

#### 3.8.2 Payload parameters and observation windows

The filenames are `detect_windows_no_plane.json` for L1 and `detect_windows.json` for L2. Each file is a dictionary keyed by payload name, not a list of independent samples.

| Field path | Type | Meaning and use |
|---|---|---|
| `<payload_name>` | object | Parameter object for the payload named by this key. |
| `<payload_name>.observation_windows` | `array<array<string>>` | Available windows. Each inner array contains exactly a start and an end timestamp. |
| `<payload_name>.observation_windows[][0]` | string | Window start timestamp. |
| `<payload_name>.observation_windows[][1]` | string | Window end timestamp. |
| `<payload_name>.plane` | string | L2 mounting-plane or pointing group, used for simultaneous-observation and attitude-adjustment constraints. |
| `<payload_name>.power_consumption` | number | L2 payload operating power in watts (W). |

Timestamps use `YYYY-MM-DD HH:MM:SS` on a common time axis within each scenario. No time zone is encoded. Starts precede ends; adjacent or overlapping availability windows can be merged when checking containment.

#### 3.8.3 Power limit: `max_power.txt`

Each L2 file contains one positive number, without a header: the maximum combined operating power of simultaneously active payloads, in W. Read the value from the corresponding scenario rather than applying one limit to every scenario.

#### 3.8.4 Model-generated plan

The following fields describe the expected JSON response, rather than additional input-dataset fields.

| Field path | Expected type | Meaning and use |
|---|---|---|
| `achieved_objectives` | `array<string>` | Names of scientific objectives the model claims it can achieve. |
| `observation_plan` | `array<object>` | Scheduled observation segments; a payload may have multiple segments. |
| `observation_plan[].name` | string | Payload name matching the window dictionary. |
| `observation_plan[].start_time` | string | Observation start timestamp. |
| `observation_plan[].end_time` | string | Observation end timestamp. |
| `observation_plan[].duration` | number | Declared observation duration in **seconds**. |
| `adjustment_plan` | `array<object>` | Attitude-adjustment intervals; empty when none are required. |
| `adjustment_plan[].name` | string | Name of the adjustment event. |
| `adjustment_plan[].start_time` | string | Adjustment start timestamp. |
| `adjustment_plan[].end_time` | string | Adjustment end timestamp. |
| `adjustment_plan[].duration` | number | Declared adjustment duration in **seconds**. |

Here, `name` identifies a payload or an adjustment event according to its parent object; in the requirements file, it identifies a scientific objective. Planned timestamps use the same format as the input windows.

## 4. Representative Examples

The nine examples below cover all eight tasks, with separate examples for the two conceptual-knowledge subtasks. Original Chinese records are preserved. Single-record JSON examples omit the dataset's outer array unless the array is needed to show a planning structure.

### 4.1 Factual knowledge

Source: `data/knowledge/factual_knowledge/celestial_bodies.json`, `id=1`.

```json
{
  "id": 1,
  "question": "根据探测数据，海王星大气层呈现蓝色的主要原因是什么？",
  "options": [
    "A. 表面覆盖着液态水海洋",
    "B. 大气中含有微量的甲烷吸收了红光",
    "C. 表面岩石反射太阳光",
    "D. 极光活动的持续影响"
  ],
  "answer": "B"
}
```

**English reading:** Why does Neptune's atmosphere appear blue? The options propose (A) a liquid-water ocean, (B) atmospheric methane absorbing red light, (C) sunlight reflected by surface rocks, or (D) persistent auroral effects. The reference answer is `B`. The model receives the question and options, not the reference letter.

### 4.2 Procedural knowledge

Source: `data/knowledge/procedural_knowledge/tracking_and_control_procedures.json`, `id=1`.

```json
{
  "id": 1,
  "question": "基于地面测控系统实施轨道控制的完整流程包括哪些环节？",
  "options": [
    "A. 采集测站测量数据与精密轨道确定",
    "B. 轨道控制规划与实施上行注入",
    "C. 控制机构执行与控后精密轨道确定",
    "D. 轨道控制效果标定"
  ],
  "answer": "ABCD"
}
```

**English reading:** Select all stages in a complete ground-based orbit-control process: (A) tracking measurements and precise orbit determination; (B) maneuver planning and command uplink; (C) execution and post-maneuver orbit determination; and (D) assessment of the maneuver outcome. All four options are correct. `ABCD` encodes the complete answer set.

### 4.3 Technical concept explanation

Source: `data/knowledge/conceptual_knowledge/astronomy_terms.xlsx`, `Sheet1`, row 2.

| 名词 | 解释 |
|---|---|
| 阿波罗群 | 近地小行星中的一种，其轨道半长径大于地球轨道半径且近日距小于1.017AU，半长轴大于1AU |

**English reading:** `阿波罗群` means the Apollo group of near-Earth asteroids. The reference describes semimajor axes greater than 1 AU and perihelion distances below 1.017 AU. `名词` supplies the term to the model; `解释` supplies the reference meaning to the scorer.

### 4.4 Abbreviation explanation

Source: `data/knowledge/conceptual_knowledge/technical_abbreviations.xlsx`, `Sheet1`, row 2.

| 领域 | 缩写 | 解释 |
|---|---|---|
| 理论部分 | EKF | Extended Kalman Filter 拓展卡尔曼滤波 |

**English reading:** In the domain `理论部分` (theoretical topics), `EKF` stands for *Extended Kalman Filter*. The reference cell also contains its Chinese equivalent. The domain context and abbreviation form the model input; the combined expansion is the reference.

### 4.5 Named entity recognition

Source: `data/nlp/named_entity_recognition.json`, `id=170`. This is the complete record.

```json
{
  "text": "通过月球探测工程的实施，开展月球着陆巡视区的月表特性原位分析、形貌探测、结构构造综合探测，以及月表环境探测和月基天文观测。",
  "id": 170,
  "label": [
    {"start": 2, "end": 4, "text": "月球", "labels": ["探测对象"]},
    {"start": 2, "end": 8, "text": "月球探测工程", "labels": ["所属探测计划"]},
    {"start": 14, "end": 16, "text": "月球", "labels": ["探测对象"]},
    {"start": 14, "end": 21, "text": "月球着陆巡视区", "labels": ["位置"]},
    {"start": 22, "end": 60, "text": "月表特性原位分析、形貌探测、结构构造综合探测，以及月表环境探测和月基天文观测", "labels": ["科学目标"]},
    {"start": 26, "end": 30, "text": "原位分析", "labels": ["探测手段"]},
    {"start": 54, "end": 60, "text": "月基天文观测", "labels": ["探测手段"]}
  ],
  "annotator": 1,
  "annotation_id": 480,
  "created_at": "2026-03-11T21:23:47.893318Z",
  "updated_at": "2026-03-11T21:24:53.438318Z",
  "lead_time": 65.545
}
```

**English reading:** The passage describes a lunar exploration program conducting in-situ analysis of surface properties, morphology and geological structure surveys, environmental investigations, and Moon-based astronomical observations in lunar landing and rover regions.

The mentions include `月球` (the Moon), `月球探测工程` (lunar exploration program), `月球着陆巡视区` (lunar landing and rover region), `原位分析` (in-situ analysis), and `月基天文观测` (Moon-based astronomical observation). The long span labeled `科学目标` records the scientific objectives. `月球` occurs twice, and its spans overlap with longer annotations. These positions are retained in the file, while scoring deduplicates entity names within the passage.

### 4.6 Terminology translation

Source: `data/nlp/terminology_translation.json`, `id=1`.

```json
{
  "id": 1,
  "zh": "11年太阳活动周",
  "en": [
    "11-year solar cycle"
  ]
}
```

**English reading:** The source term `11年太阳活动周` has the reference translation *11-year solar cycle*. Even this single reference is stored in an array. The model receives `id` and `zh`; it does not receive `en`.

### 4.7 Exploration method recommendation

Source: `data/task/exploration_method_recommendation.json`, `id=1`.

```json
{
  "id": 1,
  "任务": "火星-1960A（Mars 1960A）",
  "时间": "20世纪六十年代",
  "探测对象": "火星",
  "科学目标": "探测地火之间宇宙射线、磁场、微陨石等行星际空间环境，勘测火星大气与电离层状况、地表形貌物质特征，搜寻火星生命存在线索，同时验证远距离深空通信及长期太空飞行相关技术",
  "探测方式": [
    "飞越探测"
  ]
}
```

**English reading:** This scenario places a Mars mission in the 1960s. Its objectives include investigating cosmic rays, magnetic fields, and micrometeoroids between Earth and Mars; studying the Martian atmosphere, ionosphere, surface morphology, and materials; seeking indications of life; and testing deep-space communication and long-duration flight technologies. The reference method is `飞越探测` (flyby exploration).

The historical name, Mars 1960A, remains in the record for identification. The model is asked to choose methods from the six labels using the era, target, and objectives, without being given the mission name or reference choice.

### 4.8 Scientific payload configuration

Source: `data/task/scientific_payload_configuration.json`, `id=1`.

```json
{
  "id": 1,
  "任务": "火星观测者（Mars Observer）",
  "时间": "20世纪九十年代",
  "探测对象": "一稀薄大气类地天体，表面富含氧化铁，存在明显的流水侵蚀遗迹和季节性极冠",
  "科学目标": "研究该天体的地质学、地质物理学和气候。",
  "探测方式": "环绕探测",
  "载荷": [
    "伽马射线光谱仪",
    "火星观测者相机",
    "热发射光谱仪",
    "压力调制器红外辐射计",
    "激光高度计",
    "射电科学实验",
    "磁强计",
    "电子反射计"
  ]
}
```

**English reading:** Configure payloads for a 1990s orbital mission studying the geology, geophysics, and climate of a terrestrial body. The target description specifies a thin atmosphere, an iron-oxide-rich surface, evidence of fluvial erosion, and seasonal polar caps.

In order, the reference entries denote a gamma-ray spectrometer, Mars Observer camera, thermal emission spectrometer, pressure-modulator infrared radiometer, laser altimeter, radio science experiment, magnetometer, and electron reflectometer. They provide a reference combination of scientific capabilities; functionally equivalent instruments or combinations may also receive credit.

### 4.9 Observation-window planning

Source: scenario `01`, L2. The extracts show the second objective in `object_01.json` and its two associated payload entries in `detect_windows.json`. Other objectives and payloads are omitted; the full instance requires all companion files.

**Scientific requirement extract:**

```json
[
  {
    "name": "小天体自转参数测定",
    "loadData": [
      {
        "pay_load": "中视场彩色相机",
        "require_time/min": 2000
      },
      {
        "pay_load": "窄视场导航敏感器",
        "require_time/min": 1500
      }
    ]
  }
]
```

**Associated payload parameters and windows:**

```json
{
  "中视场彩色相机": {
    "plane": "-Z",
    "power_consumption": 30,
    "observation_windows": [
      ["2026-05-26 00:21:00", "2026-05-28 15:02:36"],
      ["2026-06-20 10:57:23", "2026-06-23 01:38:59"]
    ]
  },
  "窄视场导航敏感器": {
    "plane": "-Z",
    "power_consumption": 50,
    "observation_windows": [
      ["2026-05-25 05:10:12", "2026-05-28 00:32:30"],
      ["2026-06-21 01:27:30", "2026-06-23 20:49:47"]
    ]
  }
}
```

**Complete contents of `max_power.txt` for this scenario:**

```text
250
```

**English reading:** Determine a small body's rotational parameters using `中视场彩色相机` (medium-field color camera) for 2,000 minutes and `窄视场导航敏感器` (narrow-field navigation sensor) for 1,500 minutes. Their operating powers are 30 W and 50 W. Each timestamp pair gives an available window, and the scenario limits total concurrent payload power to 250 W. Required durations are in minutes, while durations in a generated plan are in seconds.

## 5. Data Sources and Preprocessing

The benchmark datasets were constructed from publicly available deep-space exploration mission materials, technical literature, aerospace standards, and authoritative terminology resources. Task-specific preprocessing included information extraction, domain-relevance filtering, terminology normalization, duplicate removal, and manual verification, as applicable. Knowledge questions were developed with LLM assistance and checked against reference sources. Named entities were manually annotated, while historical mission information was adapted into exploration-method and payload-configuration scenarios. Observation-planning scenarios were developed using mission-informed scientific requirements and simulated observation windows, followed by feasibility checks. The original Chinese-language content was retained throughout preprocessing. Further details on dataset construction and quality control are provided in the accompanying paper.

## 6. Evaluation Data Interpretation

### 6.1 Inputs, references, and code entry points

Reference answers are withheld from the evaluated model. For the two historical-mission scenario tasks, `任务` is retained for identification but excluded from the generation prompt.

| Task or subtask | Model input | Reference and evaluation |
|---|---|---|
| Factual knowledge | `question`, `options` | `answer`; single-choice accuracy |
| Procedural knowledge | `question`, `options` | `answer`; the full selected set must match |
| Technical concept explanation | `名词` | `解释`; technical correctness and completeness on a 0-4 scale |
| Abbreviation explanation | `领域`, `缩写` | `解释`; a matching English or Chinese expansion is accepted after normalization |
| NER | `text` and the predefined entity-type scope | Entity names from `label[].text`; deduplicated name matching and F1 |
| Terminology translation | `id`, `zh` | Any reference in `en`, including rule-recognized variants |
| Exploration method recommendation | `时间`, `探测对象`, `科学目标`, and the six candidate methods | `探测方式`; set-based Precision, Recall, and F1 |
| Scientific payload configuration | `时间`, `探测对象`, `科学目标`, `探测方式` | Scientific functions represented by `载荷` |
| Observation planning | Requirements, windows, and the level-specific engineering parameters | Constraint, objective, and observation-duration satisfaction |

For concept explanations, the judge receives the term, reference explanation, and model answer; exact wording is not required. Translation scoring normalizes case, punctuation, word forms, and content-word sets, and accepts recognized initialisms or aliases. It therefore measures agreement with reference expressions and the implemented matching rules. See paper Eqs. (1)-(3) and Table 7.

Input paths are resolved relative to the script directory. The `eval_` scripts either score responses immediately or save them for a separate `score_` script, as shown above. The README explains the corresponding output paths and run settings.

### 6.2 NER: span annotations versus name-based scoring

The NER dataset contains entity spans, labels, and annotation metadata. However, the evaluation focuses on extracted entity names rather than entity types or positions. Reference and predicted names are normalized and deduplicated within each passage, then compared using exact string matching. Performance is measured by macro-averaged F1 across passages.
### 6.3 Scientific payload configuration: functional coverage

Scientific payload configurations are evaluated based on functional coverage rather than exact instrument-name matching. An LLM judge assesses whether the proposed payload combination fulfills the scientific functions represented by the reference payloads, allowing functionally equivalent instruments and joint coverage. The evaluation combines the functional coverage ratio (FCR) and candidate utilization ratio (CUR). Detailed definitions are provided in the accompanying paper.
### 6.4 Observation planning: requirements and constraints

Observation plans are evaluated against scientific requirements and engineering constraints rather than a single reference schedule. The evaluation considers observation durations, available time windows, payload power limits, and attitude-adjustment constraints where applicable. Objective completion is determined from scheduled observation durations rather than the model-reported achieved_objectives field. The overall score combines hard-constraint satisfaction (HCSR), scientific-objective satisfaction (OSR), and payload observation-duration satisfaction (PRSR) with equal weights. Detailed scoring rules are provided in the accompanying paper.
- **Observation duration:** Convert required minutes to seconds. Recompute scheduled duration from timestamps and merge a payload's observation intervals to avoid double-counting overlaps. A declared `duration` does not replace the timestamp-based calculation.
- **Objective completion:** All payload-duration requirements for an objective must be met. Completion is computed from the schedule, not accepted from `achieved_objectives` alone.
- **Shared observations:** Repeated requirements for a payload within one objective are added. Across objectives, the payload-duration satisfaction calculation uses the largest objective-specific requirement for that payload, allowing shared observations. Satisfaction is capped at 1.
- **Time and conflicts:** Timestamps must be valid and starts must precede ends. Observations must fit within the corresponding available windows. A payload's observation segments must not overlap, and adjustments must not conflict with observations.
- **L2 engineering conditions:** Payloads in different mounting-plane groups cannot observe simultaneously. Switching groups requires an attitude-adjustment interval. Concurrent payload power must remain within that scenario's `max_power.txt` limit.

The three components are the **hard-constraint satisfaction rate (HCSR)**, **objective satisfaction ratio (OSR)**, and **payload requirement satisfaction ratio (PRSR)**. In the code, the corresponding per-scenario keys are `constraint_ratio_first`, `objective_ratio_first`, and `payload_avg_ratio_first`; task-level averages give the reported components. Hard-constraint satisfaction is binary for each schedule: all applicable constraints pass, or the schedule fails that component.

The planning score gives equal weight to the three components. A schedule that violates a hard constraint may still receive partial credit for meeting scientific requirements, so the composite score is not the proportion of fully feasible schedules. See paper Eqs. (7)-(10).
