# 项目 TODO：撞击概率估计的神经代理模型（方向3）

**目标**：用深度学习加速近地小行星（NEA）撞击概率/不确定性传播计算，替代 Sentry 式蒙特卡洛。
学习轨道传播算子 + 不确定性量化。数据全部本地用 REBOUND 生成 + JPL SBDB 公开轨道。
目标期刊：Icarus / CMDA / AAS journals。

## 阶段0：环境搭建
- [x] 安装 Miniconda（用户要求所有 Python 通过 anaconda 环境）
- [x] 创建 conda 环境 `nea`（python 3.11），安装 rebound 5.0.1, torch 2.13(CPU), numpy 等
- [ ] 验证 REBOUND（跑最小 N 体测试 + Horizons 行星初值缓存）
- [x] 确认硬件：2 CPU / 7GB RAM / ~110GB 磁盘，无 GPU → CPU 训练，规模据此设计

## 阶段1：科学方案设计
- [x] 文献调研（arXiv API）：确认方向空白，最接近为 Hefele 2020 二分类（写入 design.md §1.2）
- [x] 明确科学问题：给定 NEA 初始轨道要素 + 协方差，预测未来 N 年的深度接近距离分布 / 撞击概率代理量
- [x] 设计输入/输出表示（design.md §3.1）
- [x] 选定模型架构：MDN + deep ensemble（design.md §3.2，备选 flow/分位数）
- [x] 确定基线：MC ground truth / 二体 MOID / 无 UQ 回归（design.md §3.3）
- [x] 参数依据写入 docs/design.md（IAS15、克隆数收敛测试、100 年窗口对齐 Sentry）

## 阶段2：真实数据获取
- [ ] 从 JPL SBDB API 拉取 NEA/PHA 轨道要素 + 协方差矩阵（公开、免费）
- [ ] 数据清洗与统计（要素分布、条件数检查）
- [ ] 保存为可复现的原始数据文件（data/raw/）

## 阶段3：仿真数据生成（REBOUND）
- [ ] 写单个小行星克隆传播脚本（太阳+8大行星+月球，IAS15，含近地接近记录）
- [ ] 少样本测试（~10 个小行星 × 少量克隆）验证物理正确性（能量守恒、与 JPL Horizons 对比某颗已知小行星轨道）
- [ ] 与 JPL Horizons 星历交叉验证传播精度（误差量化写入报告）
- [ ] 实现断点续跑（每个任务独立输出文件 + 完成标记）
- [ ] 多进程并行批量生成完整训练集（数千个真实/采样轨道 × 时间序列输出）
- [ ] 生成验证集/测试集（与训练集不重叠的小行星）

## 阶段4：模型训练
- [ ] 数据集构建与归一化（保存归一化统计量）
- [ ] 训练确定性传播代理（先小规模跑通，再全量）
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
