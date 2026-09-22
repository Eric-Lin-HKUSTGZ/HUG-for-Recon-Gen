# HO3D evaluation protocol

当前 `train_ho3d.yaml` 与 `train_ho3d_smoke.yaml` 的 `trainer.val` 指向官方 `splits_v2/ho3d_eval.txt`，训练过程中用该 test/evaluation 集的指标选择 `model_best.pt`。`trainer.test` 也评测同一列表。

因此这是 **test-selection protocol**：最终指标不是独立测试结果，必须在报告中明确说明。`ho3d_val.txt` 仍保留为训练池按序列留出的验证划分，但当前配置不用于 best checkpoint 选择。
