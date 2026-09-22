# 服务器操作指南（给人工操作用）

两台机器：`.78`（主）、`.79`（备/并行）。均 `ssh intern3@10.103.22.xx`（id_ed25519 免密）。
只动 `~/research/SingGuard/` 下的东西；跑前 `nvidia-smi` 挑空闲卡，别人占用的卡不要碰。

## 环境速查

```bash
PY=/data/intern3/research/SingGuard/.venv/bin/python   # 远程 venv python（torch cu128 + vllm 0.11 + tf 4.57.1）
cd ~/research/SingGuard/eval                            # 工作目录
```

## 常用命令

```bash
# 挑卡（找 used 接近 0 的卡）
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# 冒烟（每数据集 5 条，验证链路；结果不可比）
CUDA_VISIBLE_DEVICES=6 $PY runner.py --models sing-guard-8b --datasets vlguard --smoke 5

# 正式跑（全部数据集，断点续跑，中断重跑即续）
CUDA_VISIBLE_DEVICES=6 nohup $PY runner.py --models sing-guard-8b > logs/run_sg8b.log 2>&1 &

# 多模型顺序跑（SingGuard 三行）
CUDA_VISIBLE_DEVICES=6 nohup $PY runner.py --models sing-guard-2b,sing-guard-4b,sing-guard-8b > logs/run_sg.log 2>&1 &

# 汇总（runner 结束也会自动跑；随时手动刷）
$PY aggregate.py     # -> outputs/table4_repro.csv + table4_compare.csv（含与论文的 Δ）

# 看进度 / 日志
tail -f logs/run_sg8b.log                     # runner 日志
tail -f logs/vllm_sing-guard-8b.log           # vLLM 服务器日志
wc -l results/preds/*.jsonl                   # 每个模型×数据集已完成样本数
cat results/results.csv | column -t -s,       # 已有指标

# 换推理模式 / 输出长度（调试用）
... --thinking slow --max-tokens 1024          # slow 模式
... --engine hf                                # transformers 后端（慢，作对照）
... --gpu-util 0.60                            # 共享卡时降低显存占用
```

## 数据/模型位置

- 数据：`eval/data/<key>/test.jsonl`（已采样 1000 条）+ `data/extracted/`、`data/raw/...`（图像）
- 模型：`eval/models/<key>/`（sing-guard-{2b,4b,8b}、qwen3-vl-{4b,8b}）
- 结果：`results/results.csv`（每跑完一个格自动追加）→ `aggregate.py` 汇总成 Table 4 形状

## 当前已知状态（2026-09-22）

- 环境：`.78` 的 venv 已修好（torch 2.8+cu128 / vllm 0.11 / transformers 4.57.1，flashinfer 已卸载）
- 链路：vLLM 启动/推理/解析/CSV 自动汇总全通
- **待解**：sing-guard-8b 在 fast 模式下判决过宽（README 炸弹示例判 safe）；带图 unsafe 判定正常。
  正在测 fast-slow/slow 模式是否复现论文行为（探针数据集 `data/probe/`）
- 数据 6/8 列就绪（MMDS 两列缺公开数据）

## .79 部署（并行用）

```bash
# 1) 代码：从 .78 内网克隆（.79 无公网，GitHub 不可达）
ssh intern3@10.103.22.79 'git clone intern3@10.103.22.78:research/SingGuard research/SingGuard'
# 或带分支：git clone -b eval/table4-repro intern3@10.103.22.78:research/SingGuard ...

# 2) 数据+模型：从 .78 rsync（内网快）
ssh intern3@10.103.22.79 'mkdir -p research/SingGuard/eval'
rsync -a intern3@10.103.22.78:research/SingGuard/eval/data/ intern3@10.103.22.79:research/SingGuard/eval/data/
rsync -a intern3@10.103.22.78:research/SingGuard/eval/models/ intern3@10.103.22.79:research/SingGuard/eval/models/

# 3) 环境：与 .78 相同配方（tuna 源可达即可）
/data/intern3/.local/bin/uv venv ~/research/SingGuard/.venv --python 3.10   # .79 若无 uv: curl 安装或从 .78 拷 ~/.local/bin/uv
UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv pip install --python ~/research/SingGuard/.venv/bin/python \
  "vllm==0.11.0" "transformers==4.57.1" "modelscope" pyyaml pandas openai pillow tqdm tabulate pyarrow
# 注意：vllm 0.11 会自动带 torch 2.8+cu128（驱动 535 可用）；不要装 flashinfer
```

版本控制：本地 Arch 上 `git push fork eval/table4-repro` → .78 上 `git pull fork`（.78 可达 GitHub？
不可达时用本地 rsync 推送或在 .78 `git remote add fork git@github.com:...` 走代理）。
