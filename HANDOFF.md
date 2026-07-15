# 交接说明（给下一个 Devin 会话）

## 项目
NEA 撞击风险神经代理：学习"轨道不确定性传播算子"。输入小行星轨道要素+6×6协方差，
MDN deep ensemble 直接输出未来 100 年（10 个十年窗口）地球最小接近距离 log10(d_min/au)
的概率分布，替代 Sentry 式克隆蒙特卡洛 N 体传播。目标期刊 Icarus/CMDA/AAS。
详见 docs/design.md、docs/results.md、todos.md。

## 仓库状态（全部已 push 到 git）
- `src/fetch_sbdb.py`：SBDB 数据拉取（已完成：data/raw/targets.csv 3424 目标，data/raw/cov/*.json 3416 个协方差）
- `src/propagate.py`：单天体克隆 MC（REBOUND IAS15 + REBOUNDx 完整 GR；gr_potential 不合格勿用）
- `src/batch_propagate.py`：断点续跑批量生成。**未完成**：data/sim/ 目前 ~1100/3424 颗，
  继续跑：`conda activate nea && python src/batch_propagate.py --workers 2 --loop`（约 3.7 颗/分钟）
- `src/dataset.py` → data/dataset.npz（重建后重训）
- `src/train.py`：当前最优配置 `--epochs 14 --hidden 96 --blocks 2 --comp 5 --dropout 0.25 --wd 5e-3 --diag-cov`，seeds 0/1/2
- `src/evaluate.py`：CRPS/NLL/W1/校准/加速比 + 图 → results/
- 环境：conda env `nea`（rebound 4.6.0 + reboundx 4.6.2 + torch CPU）；行星缓存 data/raw/planets_2461000.5.bin（rebound 4.6 格式）

## 当前结果（858 颗训练时）
CRPS 0.226 vs 气候学 0.297；校准近乎完美（PIT 平坦）；加速 ~2700×。见 results/metrics.json、results/figs/。

## 下一步（按序）
1. 续跑 batch_propagate 至 3424 颗（每 ~1h `git add data && git commit && git push`）
2. `python src/dataset.py` 重建 → 删 checkpoints/*.pt → 重训 3 seeds（数据多了可试 hidden 128 / 全协方差）
3. `python src/evaluate.py` → 目视检查 results/figs/，确认 CRPS/校准改善
4. 补充：Apophis/Bennu 案例研究、撞击阈值（d<0.05 au 概率）可靠性曲线、克隆数收敛检验
5. 填充论文 paper/draft.tex 的 Results/Discussion，撰写 Introduction（新颖性：最接近工作是 Hefele+2020 二分类）
6. 每个里程碑及时 push

## 注意事项
- 用户要求：不编造数据、参数要有依据、少样本试通再大批量、断点续跑、频繁 push、迭代到结果理想
- 多进程 REBOUND 必须 maxtasksperchild=1（否则 worker 状态污染报 'Primary has no mass'）
- pkill 时注意别匹配到自己的命令行
- git push 用户仓库 jianyang11/xiaoxingxing 需要用户提供 PAT（Devin GitHub 集成无权限）
