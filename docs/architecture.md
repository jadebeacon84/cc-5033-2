# 掩码流程解耦：架构与等价性说明

本次重构只调整掩码的流转方式，不改变任何修复算法。边界（core/band/keep）规则、
纹理保护、斑块合成与双调和插值的选择逻辑均保持原样。

## 三个明确版本关联的阶段

代码位于 `src/restoration/scratch.py`、`src/restoration/services.py`。

1. **划痕候选 `CandidateProposal`**
   检测器输出的不可变提案，带内容哈希 `proposal_id`。提案本身不能修改任何掩码，
   必须经显式“并入”动作才会进入修订历史。

2. **人工修订 `MaskRevisionStore` / `FileMaskRevisionStore`**
   只追加的修订历史。每次笔刷、并入候选、清空、反相都生成一个递增修订号，
   记录来源（origin）和所依据的提案 id。支持 `undo()`，初始空修订是下限。
   工作选区只由该存储持有；对外返回的掩码是可写副本，历史中的数组只读。

3. **修复像素求值 `FrozenMask` + `repair_with_frozen_mask`**
   修复时从头修订冻结出不可变快照（含 `mask_id`、来源修订号、提案链接）。
   处理器只接收快照，拿不到修订存储，因此**不能直接修改界面选区**。导出与预览
   （`mask_frozen.png`、`mask_used.png` 及各预览图）共享同一份冻结掩码。

## 处理器边界与兼容适配

- `services.repair_with_frozen_mask` 负责分区、重建、合成、保留核对与推测标记，
  复用原有 `maskzones`、`methods`、`composite`，未新增修复算法。
- `pipeline.run` 保留为旧入口兼容适配层：加载/检测掩码后立即冻结，再委托处理器。
  CLI 与既有调用方无需改接即可获得相同行为。
- Web 界面（`webui.py`）改用修订存储，所有按钮调用真实修复实现。

## 等价性如何被限定与验证

等价性用两条证据限定，测试在 `tests/test_equivalence.py`、`tests/test_webui_workflow.py`：

- **掩码身份**：报告中的 `mask.mask_id` 是掩码内容的 SHA-256；旧入口与新处理器
  对同一输入产出相同 id，落盘的 `mask_frozen.png` 重新读取后身份一致。
- **输出差值**：同图同参数下，旧入口与冻结处理器的 7 个 PNG 逐字节相同
  （新增的只有 `mask_frozen.png`，其内容与原 `mask_used.png` 相同）；
  跨边缘差分 `difference.cross_edge_changed_samples == 0`，掩码外保留核对逐样本通过。

对旧交付样例 `out/deliverable` 的实测结果：所有原有 PNG 哈希与重构前基线一致，
掩码外改动为 0，保留核对通过。

## 错误不得被默认值遮盖

- 图形界面修复请求缺少任一参数时返回 HTTP 400 并列出缺失字段；
- 空掩码执行修复返回 HTTP 400，不产生“无改动即成功”的误导性结果；
- 冻结快照身份与内容不符会在构造时直接抛出 `ValueError`。
