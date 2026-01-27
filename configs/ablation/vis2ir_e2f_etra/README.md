# E2F-ETRA 消融配置说明

本目录包含 E2F-ETRA 的消融实验配置。每个 YAML 都基于
`configs/flow/vis2ir_e2f_etra.yml`，只修改下述条目。

定义：
- Baseline：`baseline.yml`
- 最终方法（Final）：与本目录的 baseline 相同（ETRL + Flow + ETRA 引导）

所有配置都包含 `precompute` 选项，用于可选加载预计算 IR0：
```
precompute:
  ir0_root: /path/to/ir0
  use_split_subdir: true
  use_only: false
```

## baseline.yml
- 作用：参考基线配置。
- 相对 baseline：无变化。
- 相对最终方法：无变化（与最终方法一致）。

## no_etra.yml
- 作用：去除 ETRA 引导。
- 相对 baseline：`etra.enabled: false`，`etra.loss_weight: 0.0`。
- 相对最终方法：移除 ETRA 引导。

## etra_w01.yml / etra_w05.yml / etra_w10.yml
- 作用：测试 ETRA 引导强度。
- 相对 baseline：`etra.loss_weight` 分别为 0.1 / 0.5 / 1.0。
- 相对最终方法：仅改变 ETRA 损失权重。

## etrl_gt.yml（Oracle）
- 作用：上界实验（IR0 直接使用 GT）。
- 相对 baseline：`ablation.etrl_mode: gt`（ir0 = GT ir）。
- 相对最终方法：IR0 为 oracle，不再由 ETRL 预测。
- 说明：仅作上界参考，不是公平对比。

## etrl_zero.yml
- 作用：去除 ETRL，IR0 置零。
- 相对 baseline：`ablation.etrl_mode: zero`。
- 相对最终方法：IR0 为全零。

## etrl_noise.yml
- 作用：去除 ETRL，IR0 使用随机噪声。
- 相对 baseline：`ablation.etrl_mode: noise`（包含噪声参数）。
- 相对最终方法：IR0 为噪声。

## etrl_constant.yml
- 作用：去除 ETRL，IR0 固定常数。
- 相对 baseline：`ablation.etrl_mode: constant`，`ablation.etrl_constant: 0.5`。
- 相对最终方法：IR0 为常数。

## etrl_trainable.yml
- 作用：端到端微调 ETRL。
- 相对 baseline：`ablation.etrl_trainable: true`，设置 `ablation.etrl_lr`。
- 相对最终方法：ETRL 由冻结改为可训练。
