# 项目 TODO：撞击概率估计的神经代理模型（方向3）

**目标**：用深度学习加速近地小行星（NEA）撞击概率/不确定性传播计算，替代 Sentry 式蒙特卡洛。
学习轨道传播算子 + 不确定性量化。数据全部本地用 REBOUND 生成 + JPL SBDB 公开轨道。
目标期刊：Icarus / CMDA / AAS journals。

## 阶段0：环境搭建
- [x] 安装 Miniconda（用户要求所有 Python 通过 anaconda 环境）
- [x] 创建 conda 环境 `nea`（python 3.11），安装 rebound 5.0.1, torch 2.13(CPU), numpy 等
- [x] 验证 REBOUND（最小 N 体测试通过 + Horizons 行星初值缓存 planets_2461000.5.bin）
- [x] 安装 REBOUNDx（GR 修正；注意：reboundx 会把 rebound 降到 4.6.0，已重建行星缓存）
- [x] 确认硬件：2 CPU / 7GB RAM / ~110GB 磁盘，无 GPU → CPU 训练，规模据此设计

## 阶段1：科学方案设计
- [x] 文献调研（arXiv API）：确认方向空白，最接近为 Hefele 2020 二分类（写入 design.md §1.2）
- [x] 明确科学问题：给定 NEA 初始轨道要素 + 协方差，预测未来 N 年的深度接近距离分布 / 撞击概率代理量
- [x] 设计输入/输出表示（design.md §3.1）
- [x] 选定模型架构：MDN + deep ensemble（design.md §3.2，备选 flow/分位数）
- [x] 确定基线：MC ground truth / 二体 MOID / 无 UQ 回归（design.md §3.3）
- [x] 参数依据写入 docs/design.md（IAS15、克隆数收敛测试、100 年窗口对齐 Sentry）

## 阶段2：真实数据获取
- [x] SBDB bulk：42204 个 NEO → 筛选 cc≤3 & arc≥365d → 10907 → MOID<0.05au 分层选 3424 目标
- [~] 逐个拉取 6×6 协方差（sbdb.api cov=mat，6线程，断点续拉，ETA ~1.5h，API 间歇 502 已加重试）
- [x] 保存可复现原始数据（data/raw/nea_elements.csv, targets.csv, cov/*.json）

## 阶段3：仿真数据生成（REBOUND）
- [x] 单体克隆传播脚本 src/propagate.py（太阳+8行星+月球，IAS15+GR，克隆为测试粒子，抛物线细化最小距离）
- [x] Horizons 交叉验证：无 GR 时 Apophis 10yr 误差 257万km → 加 GR 后 4.7万km；gr_potential 不可用（640万km）；残差源于 Horizons 含小行星摄动/Yarkovsky（写入报告）
- [x] 断点续跑（每天体一个 npz，tmp+rename 原子写入）
- [x] 修复多进程 worker 状态污染 bug（'Primary has no mass'）：maxtasksperchild=1
- [~] 批量生成进行中：train/val 96 克隆 2.4d 采样，test 256 克隆 1.8d，100 年，~4/min，预计 ~14h
- [x] train/val/test 按 spkid 哈希 80/8/12 划分（防泄漏）
- [ ] 克隆数收敛性检验（用 test 天体 256 克隆子采样对比分位数）

## 阶段4：模型训练
- [x] 数据集构建 src/dataset.py（要素 sin/cos 编码 + 协方差 Cholesky 符号 log 编码，归一化统计量存 checkpoint）
- [x] MDN 模型 src/model.py（residual MLP + 8 高斯分量，window embedding）；训练脚本 src/train.py（可续训）
- [ ] 小规模训练跑通（用已完成的部分仿真数据）
- [ ] 训练不确定性量化头（MDN 或 deep ensemble）
- [ ] 训练曲线、超参记录、断点续训（checkpoint）
- [ ] 迭代改进直至测试集精度达标（位置误差、分布覆盖率校准）

## 阶段5：评估与科学验证
- [ ] 与蒙特卡洛 ground truth 对比：精度（接近距离分布 KL/CRPS、校准曲线）与加速比
- [ ] 用真实 PHA（如 Apophis、Bennu）做案例研究
- [ ] 生成论文级图表（matplotlib，出版质量）
- [ ] 撰写结果报告 docs/results.md

## 阶段6：论文与交付
- [ ] 撰写论文草稿（LaTeX，AAS/Icarus 格式）paper/
- [ ] 整理代码仓库结构（README、环境文件、可复现脚本）
- [ ] 全部推送到 https://github.com/jianyang11/xiaoxingxing.git（确认推送方式，有问题问用户）
- [ ] 最终自检：核对 todos 全部完成，结果质量达标

## 进度日志
- 2026-07-14: 项目启动，clone 空仓库，开始环境搭建。
