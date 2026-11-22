# 训练数据湖 · 可重复数据集划分工具

输入**样本、主体关系、时间字段**，在网页设定训练/验证/测试目标比例，
生成可直接下载的三份清单。核心约束：

1. **同一主体、以及通过指定类型关系间接连通的主体必须在同一份清单里**——
   先生成关系连通组，再以组为最小单位分配，绝不按单个样本随机拆分；
2. **时间约束优先于比例**——截止日之后的样本不能流入早期训练集；
   组内跨越时间边界时按 `shift`（整组后移）或 `isolate`（整组隔离）处理，不拆组；
3. **冻结可复现**——种子、样本版本、关系版本、目标比例、时间边界与分配规则
   全部进入 `freeze_id`（SHA-256），冻结后重新生成逐字节一致；
4. **旧清单不可变**——后来发现两组有关联时，原清单原样保留并标注潜在泄漏，
   只生成新候选版，经人工确认后发布，发布版本只增不改。

纯 Python 标准库 + 原生 HTML/JS 实现，无任何第三方依赖。

## 快速开始

```bash
# 启动 Web 工作台（默认 http://127.0.0.1:8000，数据存 ./data）
python3 -m splitter --host 127.0.0.1 --port 8000

# 跑内置示例，实际产出清单 CSV 到 examples/output/（不是只给比例图）
python3 examples/run_example.py            # shift 规则
python3 examples/run_example.py --rule isolate

# 独立小规模穷举校验：枚举全部组级分配，验证最优性与全部硬约束
python3 examples/verify.py

# 单元测试（55 个，含独立暴力穷举交叉验证）
python3 -m unittest discover -s tests -v
```

## 工作流（页面五个标签页）

1. **数据与配置**：上传样本/关系 CSV，或一键载入内置示例；
   设定种子、目标比例、两个时间截止日、跨边界规则与穷举上限；
2. **连通组与分配解释**：目标 vs 实际比例条形图、连通组规模条形图、
   每组的时间资格与逐条分配理由，明确展示**大组为什么让比例偏离目标**；
3. **清单下载**：train/val/test/quarantine 四份 CSV 直接下载，
   页面展示 freeze_id、输入内容哈希与全部冻结配置；
4. **关系修订与泄漏**：撤回关系（active=false）或合并/新增关系；
   基于修订生成候选版并查看跨集合冲突，旧版文件不被触碰；
5. **版本对比与发布**：生成历史、逐样本新旧分配对比、确认发布（v1、v2… 不可变）。

## 输入 CSV 格式

样本（其余列原样透传进产出清单）：

```csv
sample_id,subject_id,event_date,amount
s_S01_1,S01,2025-01-08,302
```

关系（`active` 可选，缺省 true；撤回即置 false）：

```csv
rel_id,src_subject,dst_subject,rel_type,active
rel_c01_c02,C01,C02,family,true
```

## 算法管线

```
样本+关系
  → 并查集连通组（无向图；非 active 边与类型过滤后的边不参与；
     端点不存在的脏关系跳过并记录）
  → 组级时间评价（组内最晚样本决定资格；跨边界组按 shift/isolate 处理）
  → 两层组级分配：
      第一层 时间强制（shift 组直接进最早合格集合，不参与比例优化）
      第二层 自由组：≤ exhaustive_limit 时穷举全部可行映射取 SSE 最小，
              否则确定性贪心（大组优先 LPT + 最大缺口优先，种子哈希破平局）
  → 冻结清单（freeze_id = SHA256(规范化 JSON 的全部冻结项)）
```

时间资格是后缀链：在 `train_cutoff ≤ val_cutoff` 下，train 合格必然
val/test 都合格；含截止日后样本的组整体失去早期集合资格，**不会为凑比例
被挪回去**。

## 大组为什么必然造成比例偏差

分配单位是组而不是样本。一个占可分配样本 43.5% 的超大组只能整体落入一个
集合，任何集合接收它都会产生约等于其半数样本量级的偏差——这是结构性的，
调种子无法消除，种子只影响多个等优方案之间的确定性平局。页面的组规模图与
`allocation_report.md` 会逐条说明每个组如何改变各集合缺口。

## 内置示例（69 样本 / 24 主体 / 12 关系）

- **超大关联组**：M01..M10 经 9 条 household 边连通，30 个样本（43.5%），
  最晚样本 2025-11-18 越过两个时间边界，整体只能进 test；
- **间接三层关系**：C01 -family→ C02 -household→ C03 -same_address→ C04，
  C01 与 C04 无直接边但三跳间接连通，必须同组；
- **迟到关系修订**：发布后发现 C04 与 S01 实为同一人，新增 same_as 边把
  test 中的 chain 组与 train 中的 S01 连通 → 检出 1 处跨集合冲突、
  12 个受影响样本；v1 清单保留，v2 候选版带泄漏报告生成。

## 目录结构

```
splitter/
  dataset.py     数据模型、CSV 读写、内容哈希
  graph.py       并查集连通组（组 ID 由成员 SHA-256 派生）
  timesplit.py   时间边界、组级资格、shift/isolate
  allocate.py    组级贪心 + 小规模穷举、偏差解释
  manifest.py    冻结配置、freeze_id、清单生成
  revision.py    关系撤回/合并、泄漏分析、版本对比
  store.py       文件式不可变快照存储（datasets/generations/publications）
  api.py         http.server 标准库 HTTP 接口
web/static/      单页前端（原生 HTML/CSS/JS，条形图带悬停解释）
tests/           55 个 unittest（含独立暴力穷举交叉验证）
examples/
  run_example.py 实际产出清单与报告
  verify.py      独立穷举校验器
  output/        已生成的示例清单（v1 旧版 + v2 候选版 + 泄漏报告 + 对比）
```

## 可复现性说明

- 所有 CSV/JSON 输出确定性排序（sample_id、rel_id、JSON 键排序），
  换行统一为 `\n`，哈希不受平台影响；
- `freeze_id` 只包含冻结项，不含时间戳；运行时间单独存于
  `generated_at.json`，重复运行除该文件外逐字节一致（已校验）；
- 相同输入重复调用生成接口返回相同 `generation_id`，存储层幂等。
