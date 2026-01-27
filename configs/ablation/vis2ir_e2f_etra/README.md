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

---

## 消融实验表（论文用）

| 编号 | 配置文件 | 实验目的 | 相对 Baseline 改动 | 相对最终方法改动 | 预计算 IR0 | 备注 |
|---|---|---|---|---|---|---|
| A0 | baseline.yml | 基线参考 | 无 | 无 | 是 | 最终方法 |
| A1 | no_etra.yml | 去除 ETRA 引导 | `etra.enabled=false`，`etra.loss_weight=0.0` | 去除 ETRA 引导 | 是 |  |
| A2 | etra_w01.yml | 低权重 ETRA 引导 | `etra.loss_weight=0.1` | 调整 ETRA 权重 | 是 |  |
| A3 | etra_w05.yml | 中权重 ETRA 引导 | `etra.loss_weight=0.5` | 调整 ETRA 权重 | 是 |  |
| A4 | etra_w10.yml | 高权重 ETRA 引导 | `etra.loss_weight=1.0` | 调整 ETRA 权重 | 是 |  |
| B1 | etrl_zero.yml | 去除 ETRL（零 IR0） | `ablation.etrl_mode=zero` | IR0 置零 | 否 |  |
| B2 | etrl_noise.yml | 去除 ETRL（噪声 IR0） | `ablation.etrl_mode=noise` | IR0 为噪声 | 否 |  |
| B3 | etrl_constant.yml | 去除 ETRL（常数 IR0） | `ablation.etrl_mode=constant` | IR0 为常数 | 否 |  |
| B4 | etrl_gt.yml | 上界（Oracle IR0） | `ablation.etrl_mode=gt` | IR0 为 GT | 否 | 仅作上界 |
| C1 | etrl_trainable.yml | 端到端微调 ETRL | `ablation.etrl_trainable=true` | ETRL 可训练 | 是 |  |
