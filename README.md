# NEA Impact-Risk Neural Surrogate（近地小行星撞击风险神经代理）

用深度学习学习"轨道不确定性传播算子"：输入 NEA 轨道要素 + 协方差，直接输出未来 100 年地球最小接近距离的概率分布，替代 Sentry 式克隆蒙特卡洛 N 体传播。

- 方案文档：`docs/design.md`
- 任务清单：`todos.md`
- 代码：`src/`；数据：`data/`（大文件不入库，脚本可复现）；结果：`results/`；论文：`paper/`

## 环境
```bash
conda create -n nea python=3.11
conda activate nea
pip install numpy scipy pandas astropy matplotlib requests tqdm rebound
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
