# 深空探测领域大语言模型评测

每种题型直接运行一个独立 Python 文件。执行脚本均在本目录，不使用统一入口、模型配置、模型工厂或公共协议类，不导入其他项目脚本。

## 配置与运行

建议使用 Python 3.10 或以上版本，在本目录安装依赖：

```powershell
python -m pip install -r requirements.txt
```

在 PyCharm 中打开所需脚本，修改顶部配置后 Run；也可在命令行直接执行对应文件。所有路径基于脚本自身位置，不依赖终端当前目录。导入不会请求模型或写文件。

模型调用由每个脚本单独配置 `API_URL`、`MODEL_NAME`、`API_KEY`、`TEMPERATURE`、`REQUEST_TIMEOUT`、`EXTRA_BODY`、`STREAMING`。密钥默认留空，不读取统一配置或 `.env`，不根据模型名自动选择服务商。裁判脚本配置的是裁判模型。密钥为空时在请求前停止。思考、流式和结构化输出参数由使用者按接口设置；NER、翻译及载荷生成提供 `OUTPUT_MODE`。

```python
API_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL_NAME = 'qwen3-max'
API_KEY = ''
EXTRA_BODY = {}      
STREAMING = False
RUN_ID = 'run01'
```

## 脚本与题库

| 任务 | 评测脚本 | 评分脚本 | 题库或场景 |
|---|---|---|---|
| 事实性知识 | `evl_fact.py` | 同时评分 | `data/knowledge/fact/` 下四个 JSON |
| 程序性知识 | `evl_program.py` | 同时评分 | `data/knowledge/program/` 下四个 JSON |
| 术语解释 | `evl_concept.py` | `cal_concept.py` | `data/knowledge/concept/` 下五个术语表 |
| 缩略词解释 | `evl_concept_words.py` | `cal_concept_words.py` | `data/knowledge/concept/words.xlsx` |
| NER | `evl_ner.py` | 同时评分 | `data/nlp/ner.json` |
| 术语翻译 | `evl_term_translation.py` | `cal_term_translation.py` | `data/nlp/term_translation.json` |
| 探测方式确认 | `evl_detect_methods.py` | 同时评分 | `data/task/detect_methods.json` |
| 科学载荷配置 | `evl_payload_config.py` | `cal_payload_config.py` | `data/task/payload_set.json`，全部60条 |
| 观测规划 | `evl_observation.py` | `cal_observation.py` | `data/task/dataset/01—20/` 下 L1/L2 场景 |

载荷裁判与专家一致性统计用 `cal_payload_judge_agreement.py`。本次未新增八任务总分或术语解释专家一致性统计。

术语生成默认覆盖 astro、orbit、posture、spacecraft、survey，在 `DATA_FILES` 明确表头配置。posture 没有表头，首题为“潮汐锁定”。缩略词读取前三列：领域、缩写、参考解释；提示词沿用提供的原始代码，要求英文和中文全称。忽略全空行，有效题目缺字段时报错，不改写题库。当前读取582道术语、145道缩略词，共727题；归档裁判覆盖其中576道中文术语。

## 四种即时评分任务

事实性知识、程序性知识、NER、探测方式确认只在内存中累计指标，在终端显示成绩，不保存逐题回答、逐题状态、请求记录或断点记录。再次运行重新评测；目录已存在时自动使用 `run01_2` 等新目录，不覆盖原汇总。

选择题显示各文件准确率及总体准确率，总体按全部样本计算。多选题要求完整选项集合相同。NER、探测方式按逐样本 P/R/F1 取平均，以 F1 为主指标。NER使用规范化后的实体文本去重集合进行精确匹配，不评价实体类型或位置；同一实体名称即使在原文中出现多次，也只需输出一次。方式解析仍按名称出现判断。

非裁判请求单次超时300秒、最多尝试3次，技术重试间隔2、4秒。技术失败耗尽后继续下一题，最终标记 `incomplete`、正式成绩为空；确定性的配置错误停止。完整返回但无法解析的回答计零，不重新生成。

## 生成回答与独立评分

术语、缩略词、翻译及载荷先生成回答，再执行对应 `cal` 文件：

```text
outputs/<concept 或 concept_words 或 term_translation 或 payload_config>/<RUN_ID>/<模型目录>/
    answers.xlsx
    run.json
    requests.json
    samples.json
    summary.json
```

回答表含题目、参考答案和当前模型列；请求记录含原始响应或技术错误；样本记录含逐题状态；运行清单含输入、参数、代码及 SDK 指纹。模型目录中的特殊字符会替换，必要时增加短哈希，以实际输出目录为准。

同一个 `RUN_ID` 且 `RESUME=True` 时补跑技术失败和未处理项，已完成错误答案和格式错误不重新生成。收到原始响应后中断，优先恢复本地解析。输入、顺序、参考答案、模型参数或代码变化时使用新 `RUN_ID`。同一运行目录只允许单进程写入。

评分脚本顶部设置 `INPUT_PATH`，默认对应生成任务的 `run01/qwen3-max/answers.xlsx`。改变生成模型或运行编号后同步修改评分输入。

- 术语裁判保持0—4级转百分制，提示词不提供被测模型身份。
- 载荷裁判保持科学目标约束下的完整覆盖FCR、有效候选编号并集CUR，逐任务调和平均后再取模型均值。
- 两类裁判保持有效评分均分，失败项不计零；只跳过成功项，校验输入及裁判参数。输出分别为 `outputs/concept_score/<RUN_ID>/all_models/`、`outputs/payload_config_score/<RUN_ID>/all_models/`。
- 翻译和缩略词评分自动读取回答表同目录的 `samples.json` 和 `run.json`，核对题目、参考答案、模型与回答。输出为 `outputs/translation_score/<RUN_ID>/all_models/`、`outputs/abbreviation_score/<RUN_ID>/all_models/`，包含汇总及明细JSON、`scores.xlsx`。

外部回答没有生成记录时，设置 `STATUS_PATH`，提供UTF-8状态侧文件：

```json
{
  "protocol_version": 2,
  "input_sha256": "回答xlsx文件的SHA256",
  "models": {"模型列名": {"2": "ok", "3": "invalid_response", "4": "api_error"}}
}
```

行号为实际Excel行号，其他允许状态为 `config_error`、`execution_error`、`unprocessed`；缺少行按未处理计。格式错误计零，技术失败或未处理使正式成绩为空。历史ERROR字符串不能确定失败类型，不自动补造状态或重评归档表。

## 观测规划

生成脚本设置 `LEVEL='l1'` 或 `'l2'`，只保存回答，不计算指标：

```text
outputs/observation/<RUN_ID>/<l1 或 l2>/<模型目录>/01.json … 20.json
```

同目录保存清单、原始响应及续跑状态。完成回答不覆盖；已返回无效回答不重新生成；技术失败可以补请求。L2读取每场景 `max_power.txt`，功率纳入续跑指纹；L1不读取L2功率。提示词直接包含在脚本内，不依赖外部源文件。

`cal_observation.py` 完全离线，不需要模型SDK或密钥。默认 `INPUT_PATH=outputs/observation/<RUN_ID>`；分析已有归档时改为 `outputs/observation`。目录下应有 `l1/<模型>/<场景>.json` 和 `l2/<模型>/<场景>.json`。默认从方案JSON读取模型名单，也可设置 `MODEL_NAMES`，指定模型缺失的回答计零。

每模型每级固定20场景，缺失、空白或无法解析均计零；科学目标、窗口、功率输入错误会报错。忽略缓存分数和 `generated` 标志。输出到 `outputs/observation_score/<RUN_ID>/`，包含两级指标TXT、逐题诊断和汇总JSON，以及L1/L2/combined页的 `scores.xlsx`。已有评分输出不覆盖，请修改 `OUTPUT_PATH`。

保持当前计分：OSR按实际观测时长计算，共享数据需求取各目标最大需求，同一载荷重叠区间取并集，声明的duration只作诊断。工程指标C与科学指标分别计分，C=0不清空其他分量。L1/L2合并共40题输出百分制成绩。

## 专家一致性与归档

`cal_payload_judge_agreement.py` 的 `INPUT_DIR` 默认读取现有 `result/payload_judge_agreement`，这是人工评价材料；新统计写到 `outputs/payload_judge_agreement/<RUN_ID>/`，也保留显式命令行参数。仍按各表行序对齐，使用前须确认方案顺序一致。

现有 `data`、`outputs`、`result` 原位保留，不自动搬动、改写或重算已有题库、回答和成绩。

