# 方案文档：近地小行星撞击风险的神经不确定性传播代理
**Neural Surrogate for Orbital Uncertainty Propagation of Near-Earth Asteroids**

目标期刊：Icarus / Celestial Mechanics and Dynamical Astronomy / AAS Journals (PSJ/AJ)

## 1. 科学问题与新颖性

### 1.1 背景
- 当前撞击监测系统（JPL Sentry-II, ESA CLOMON2/Aegis）依赖蒙特卡洛式方法：从轨道解协方差中采样数千个"虚拟小行星"（克隆），逐个 N 体数值传播 ~100 年，统计与地球的接近/撞击情况。计算代价高（每个新发现天体需数千次长期积分）。
- Rubin/LSST 时代 NEA 发现率将增长 ~5-10 倍，实时撞击监测的算力压力急剧上升，加速需求真实存在。

### 1.2 文献调研结论（2026-07-14，arXiv API 检索）
- "impact monitoring" + "neural network"：0 篇直接相关。
- "uncertainty propagation" + "deep learning" + orbit：0 篇直接相关。
- 最接近的已有工作：
  - Hefele, Bortolussi & Zwart 2020 (A&A 634, A45)：普通 MLP 对 JPL 表格数据做"撞击体/非撞击体"二分类，无不确定性传播、无分布预测。
  - 2025 年若干 GNN/XAI "危险小行星分类"论文：均基于 Kaggle 表格数据，无动力学内容。
  - 航天器轨道领域有 ML 辅助共轭评估（conjunction assessment）的工作，但针对地球卫星短弧、不涉及行星际百年尺度 N 体传播。
- **空白点（本工作贡献）**：学习"不确定性传播算子"本身——输入 NEA 轨道要素+协方差（历元），直接输出未来数十年内地球最小接近距离的完整概率分布（带校准 UQ），一次前向推理替代数百次 N 体克隆积分。

### 1.3 可交付的科学结论
1. 神经代理在保持接近距离分布精度（CRPS、覆盖率校准）的同时，相对克隆蒙特卡洛加速 ≥10³ 倍。
2. 在真实 PHA（Apophis 等）上的案例验证。
3. 对 LSST 时代撞击监测流水线的算力意义量化。

## 2. 数据

### 2.1 真实轨道数据（输入分布锚定于真实 NEA 群体）
- JPL SBDB Query API（https://ssd-api.jpl.nasa.gov/sbdb_query.api）：全部 NEO 的密切轨道要素（e, a, q, i, Ω, ω, M, epoch）+ 1σ 不确定度；公开、免费、可复现。
- 每个目标的完整 6×6 协方差矩阵：SBDB per-object API（cov=mat）。
- DART/Sentry 对照：JPL Sentry API 提供当前风险列表对象的撞击概率，作为极端案例参照。

### 2.2 仿真数据生成（ground truth）
- 工具：REBOUND 5.x，IAS15 积分器（自适应步长、机器精度、可处理近距离交会——这是接近距离统计的关键；比 WHFast/MERCURIUS 更适合本任务的精度要求）。
- 动力学模型：太阳 + 8 大行星 + 月球（10 个大质量体），初始状态取自 NASA Horizons（REBOUND 内置接口）。小行星克隆作为无质量测试粒子，同一模拟中批量传播（行星积分开销摊薄）。
- 相对论修正：对百年尺度、非极端近日点的 NEA 接近距离统计影响小于协方差本身的不确定性；首版不含 GR，在验证阶段用 Apophis 对比 Horizons 量化误差，若显著则引入 REBOUNDx gr 项。
- 传播时长：100 年（对齐 Sentry 监测窗口）。输出：每个克隆与地球的所有 <0.05 au 接近事件（时间、距离），以及每 10 年窗口的最小距离。
- 克隆数：基线 250/天体（经验：Sentry-II 论文中 LOV 采样量级；首先做收敛性测试确定够用的克隆数）。
- 规模（受本机 2 CPU 限制，先做单例计时测试后最终确定）：目标 ~1500-2500 个真实 NEA + 协方差采样扩增，train/val/test 按天体划分（防泄漏）。

### 2.3 验证
- 用 Horizons 星历对若干知名 NEA（Apophis, Bennu, 2023 DW）做传播精度交叉验证（同一初值、对比若干年后位置）。
- 克隆数收敛性测试（分布统计量随 N 收敛）。

## 3. 模型

### 3.1 输入表示
- 历元密切轨道要素（对 a,e,i,Ω,ω,M 做适当归一化与角度 sin/cos 编码）。
- 协方差矩阵的 Cholesky 下三角 21 个元素（log 缩放）。
- 预测窗口编码（哪个 10 年窗）。

### 3.2 输出/架构
- 主模型：混合密度网络（MDN，K 个高斯分量）预测 log10(地球最小接近距离/au) 的条件分布——分布可能多峰（共振回归路径），MDN 天然适配。
- 认知不确定性：5-成员 deep ensemble。
- 规模：CPU 可训（<10⁶ 参数 MLP/residual MLP）；若容量不足升级为 set-transformer 风格窗口联合预测。

### 3.3 基线与消融
- 基线1：直接 MC（ground truth，同时给出加速比）。
- 基线2：二体/开普勒近似 + MOID 解析计算（新颖性对照：证明学习到的是真 N 体效应）。
- 基线3：普通回归（无 UQ）— 证明 UQ 头的必要性。

## 4. 评估指标
- 分布精度：CRPS、负对数似然、与 MC 分布的 Wasserstein 距离。
- 校准：PIT 直方图、覆盖率-置信度曲线（expected calibration error）。
- 风险相关量：min-distance < 阈值（0.05/0.01/0.002 au）概率的可靠性图。
- 速度：单天体推理时间 vs 250 克隆 IAS15 传播时间。

## 5. 计算预算与工程
- 本机：2 CPU / 7GB RAM / ~50GB 磁盘 → 所有批量任务先小样本计时，再定总规模；数据以 parquet/npz 存储，预计 <5GB。
- 断点续跑：每个天体一个输出文件 + manifest；训练用 checkpoint。
- 并行：multiprocessing 2 工作进程；克隆在单模拟内批量。
- 全程日志 + 进度打印；关键中间结果出图目视检查。

## 6. 风险与预案（不降级核心目标）
- R1 CPU 算力不足以生成足够训练集 → 对策：单模拟多测试粒子摊薄开销；缩短单窗口传播但增加天体数；必要时把传播窗口设计成课程式（先 30 年验证方法学，再扩 100 年）。
- R2 MDN 校准不足 → 换分位数回归 / normalizing flow 头，或 conformal 校准后处理。
- R3 REBOUND-Horizons 接口限流 → 行星初值只需查询一次并缓存；小行星初值用 SBDB 要素自行转换。
- R4 分布过于集中在"无接近"（不平衡）→ 按 MOID 分层抽样 NEA，保证近接近样本充足。
