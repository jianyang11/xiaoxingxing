# 阶段性结果报告（2026-07-14）

## 数据
- SBDB 42,204 NEO → cc≤3, arc≥365d → 10,907 → Earth MOID<0.05 au → 3,424 目标（含 6×6 cometary 协方差）。
- 地面真值：REBOUND IAS15 + REBOUNDx 完整 GR，太阳+8行星+月球（Horizons 初值，历元 JD 2461000.5），100 年，10 个十年窗口，克隆按协方差多元正态采样（train/val 96 克隆、2.4d 采样；test 256 克隆、1.8d 采样），接近极小值抛物线细化。
- Horizons 交叉验证：Apophis 10 年误差 无GR 2.57e6 km → 全GR 4.7e4 km；gr_potential 不合格（6.4e6 km）。残差主要来自 JPL 动力学包含 16 颗大质量小行星摄动与 Yarkovsky。
- 因预算截止，本轮训练用 858 颗（train 688 / val 69 / test 101）；生成脚本断点续跑，可继续扩到 3,424。

## 模型
- 输入（35 维）：cometary 要素（角度 sin/cos、轨道相位）+ log10 MOID + log10 周期 + 共振接近度 + 协方差 Cholesky 符号-log 编码（训练用对角 6 项）。
- MDN：residual MLP（hidden 96×2 blocks，dropout 0.25，wd 5e-3），5 高斯分量，window embedding；3 seeds deep ensemble；早停（val NLL）。
- 教训：全协方差 21 维 + 大模型在 562 颗样本上严重过拟合（val NLL 0.46→3.4）；对角协方差 + 强正则 + MOID 特征解决。

## 测试集指标（101 颗未见过的小行星 × 10 窗口，vs 256 克隆 MC）
| 指标 | 神经代理 | 气候学基线 |
|---|---|---|
| CRPS (log10 au) | **0.226** | 0.297 |
| Wasserstein-1 | **0.462** | 0.567 |
| ensemble NLL | 0.462 | — |
| q50 RMSE | 0.409 dex | — |
| 单天体推理耗时 | ~0.03 s | 85 s (REBOUND MC) |
| 加速比 | **~2,700×**（批量化后 >10⁴） | — |

- 校准近乎完美（calibration.png 对角线、PIT 平坦）→ 不确定性量化可信。
- 中位数量化散点沿对角线但散布 ~0.4 dex → 条件均值技能有限，受训练集规模（688 颗）限制；扩数据是首要改进方向。

## 图
results/figs/: pit_hist.png, calibration.png, quantile_scatter.png, example_distributions.png

## 后续（新会话继续）
1. `python src/batch_propagate.py --workers 2 --loop` 续跑至 3,424 颗（约 30 CPU 小时）。
2. `python src/dataset.py` 重建 → 重训（可增大 hidden/加回全协方差）。
3. 增补基线（二体 MOID 预测、无 UQ 回归）、Apophis/Bennu 案例研究、撞击阈值可靠性曲线。
4. 论文 paper/draft.tex 填充 Results/Discussion。
