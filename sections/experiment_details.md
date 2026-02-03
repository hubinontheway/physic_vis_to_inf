# 实验细节说明

本节给出实验设置与实现细节，旨在保证实验描述的学术规范性与可复现性。除非另有说明，所有设置均遵循默认配置。

实验在 VEDIA、AVIID 与 M3FD_Detection 数据集上进行，图像统一处理为 256×256 分辨率，可见光 3 通道与红外 1 通道均归一化至 [0,1]。训练采用三阶段流程：首先以红外图像训练 ETRA 物理属性解耦模块，其后训练 ETRL 物理属性对齐模块并以冻结的 ETRA 施加物理一致性约束，最后以 ETRL 输出为初值进行条件流匹配细化。默认超参数为 batch size 8、评测 batch size 4，ETRA 学习率 1e-4，ETRL/Flow 学习率 2e-4，余弦退火调度（warmup 1000，min lr 1e-5），ETRA/ETRL 训练 200k 步、Flow 训练 100k 步，随机种子 123；主干采用 UnetPlusPlus 与 efficientnet-b5，ETRA 损失权重为 rec 1.0、ssim 0.2、tv_low 0.5、tv_eps 0.1、edge 0.2、prior 0.05，ETRL 物理一致性项权重为 0.3。推理阶段使用 Euler 采样 50 步（atol/rtol=1e-5），评测指标为 PSNR、SSIM、LPIPS、FID 与 KID，所有实验均在单张 NVIDIA A100 GPU 上完成。

## 1. 数据集与划分
本文在 VEDIA、AVIID 与 M3FD_Detection 数据集上进行实验。训练/验证/测试样本数分别为（N_tr / N_val / N_te），若采用官方划分则沿用原始分割；若使用自定义划分，则说明具体划分策略与随机种子（默认为 123）。可见光与红外图像为一一配对关系，若存在时序偏移或对齐修正，应明确给出处理方式。原始分辨率与场景类型（如城市道路、郊区、天空背景等）需在实验部分说明。

## 2. 数据预处理与增强
所有图像统一处理为固定分辨率 `image_size=256`，可见光为 3 通道输入，红外为 1 通道真值，像素强度归一化至 `[0,1]`。若进行几何对齐、中心裁剪或随机裁剪，应说明具体策略与参数。数据增强（如随机翻转、颜色抖动或噪声扰动）若启用，需要报告增强方式及其概率设置。

## 3. 训练流程（分阶段）
训练采用三阶段流程：第一阶段仅使用红外图像训练 ETRA（物理属性解耦）模块，输出温度、发射率、透过率、路径辐射与反射项等五通道物理属性图；第二阶段训练 ETRL（物理属性对齐）模块，以可见光为输入生成红外，并引入冻结的 ETRA 作为物理教师施加一致性约束；第三阶段训练 Flow（流模型细化）模块，以 ETRL 输出为初值进行条件流匹配学习，实现少步采样细化。

## 4. 训练超参数
默认超参数设置如下：`batch_size=8`，`eval_batch_size=4`；ETRA 学习率为 `1e-4`，ETRL/Flow 学习率为 `2e-4`；学习率调度采用余弦退火（`warmup_steps=1000`，`min_lr=1e-5`）。训练步数方面，ETRA/ETRL 为 `200000` 步，Flow 为 `100000` 步。随机种子设为 123。优化器方面，ETRA/ETRL 使用 AdamW，Flow 使用 Adam（实现见 `models/*.py`）。

## 5. 模型结构与损失设置
主干网络采用 `UnetPlusPlus` 并使用 `efficientnet-b5` 作为编码器。ETRA 的低通核大小为 `lowpass_kernel=16`。ETRA 损失权重设置为：`rec=1.0`，`ssim=0.2`，`tv_low=0.5`，`tv_eps=0.1`，`edge=0.2`，`prior=0.05`。ETRL 中物理一致性项权重为 `etra.loss_weight=0.3`。Flow 阶段以速度场回归的均方误差作为训练目标。

## 6. 推理与采样设置
推理阶段采用 Euler 方法进行常微分方程积分，采样步数为 `steps=50`，容差设置为 `atol=1e-5`、`rtol=1e-5`。若进行不同步数或不同积分器的对比实验，应在实验部分明确报告。

## 7. 评价指标与协议
评测指标包括 PSNR、SSIM、LPIPS、FID 与 KID（实现见 `utils/metrics.py`）。所有结果在测试集上统计，预测输出裁剪到 `[0,1]`。指标均报告平均值，若 KID 含标准差需额外说明。

## 8. 运行环境
所有实验均在单张 NVIDIA A100 GPU 上完成。

## 9. 可复现性记录
关键配置文件包括 `configs/etra/etra_flat.yml`、`configs/vis2ir/vis2ir_etrl.yml` 与 `configs/flow/vis2ir_flow_VEDAI_advanced.yml`。应给出训练与评测命令、日志目录与 checkpoint 保存策略（例如 `max_checkpoints`），以便他人复现实验。

若包含消融实验，请在本节补充对应变量设置、对照组配置与结果汇总方式。
