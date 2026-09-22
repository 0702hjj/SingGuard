# SingGuard Table 4 复现手册

复现 [技术报告](../singguard.pdf) Table 4（多模态安全基准 F1，8 列）：
VLGuard / JailBreakV / SPA-VL / MMDS-Q / MMDS-R / VLSBench / MM-Safety / BeaverTails-V。

> 仓库官方未发布评测代码（`eval/` 原为空），本目录是自建 harness。评测协议对齐论文
> 4.1 节：生成式 guard 接口、解析行首 safe/unsafe、解析失败计错、unsafe 类二分类 F1、
> 响应侧对 (query, image, response) 联合判定。SingGuard 行用官方 checkpoint + fast 模式
> （论文明确 fast 用于 high-throughput benchmark evaluation）。
> **注意**：baseline 的提示词论文未公开，适配器按各模型卡近似，baseline 数字与论文
> 有偏差是预期内的；SingGuard 三行应最接近论文。

## 架构（本地无公网限制的远程服务器 → 非对称流程）

```
本地 Arch（有公网）                        远程 A6000（仅国内镜像/ModelScope 可达）
────────────────────                      ─────────────────────────────────────
uv sync                                   uv sync --extra gpu
prepare_data.py  # HF 下载数据集      →    rsync 过去
download_models.py --models <HF-only> →    download_models.py   # ModelScope 直连
                                          runner.py   # vLLM 评测（无网络依赖）
                                          aggregate.py # 汇总 → outputs/*.csv
```

- 数据集全部在**本地**下载归一化（`eval/data/`），连同代码 rsync 到远程；
- 模型优先 ModelScope（SingGuard 3 个、Qwen3-VL 2 个可直接在远程下载）；
  HF-only 的模型（GuardReasoner-VL、LlavaGuard）在本地下载到 `eval/models/<key>/`
  再 rsync（或先试远程 `HF_ENDPOINT=https://hf-mirror.com`，不通再走本地）。

## 一、远程环境（一次性）

```bash
ssh intern3@10.103.22.78
cd ~/research/SingGuard/eval
uv sync --extra gpu          # torch/vllm/transformers/modelscope（清华源可加速：见 pyproject 注释）
```

A6000 (Ampere, 48GB)：所有默认模型 bf16 加载无压力；`vllm` 日志在 `logs/vllm_<model>.log`。

## 二、本地数据准备（Arch，有公网）

```bash
cd ~/projects/research/SingGuard/eval
uv sync                                        # 只装轻依赖（pandas/yaml/hf_hub…）
uv run python scripts/prepare_data.py          # 下载 8 个数据集 → 归一化 → 采样 1000/集
uv run python scripts/inspect_data.py vlguard  # 检查 schema / 标签分布（强烈建议逐个看）
```

要点：

- loader 是按公开 schema 写的（`configs/datasets.yaml` 里 `verified: false`），
  schema 对不上会**报错而不是瞎猜**；报错后按 `inspect_data.py` 的输出改
  `scripts/prepare_data.py` 顶部 `QUERY_COLS/RESPONSE_COLS/LABEL_COLS` 或对应 loader。
- MMDS（→ MMDS-Q/MMDS-R 两列）来自 [LLaVAShield 项目](https://leost123456.github.io)，
  无稳定公开镜像：拿到数据后按 `load_mmds` docstring 归一化成
  `data/raw/mmds/mmds_normalized.csv`（列：query,image_path,response,label,side）。
  拿不到就先跳过，harness 会自动缺列。
- 采样确定性：seed=42、按标签分层；`data/<key>/manifest.json` 记录来源规模与 unsafe 数。

## 三、同步到远程

```bash
# 本地执行（数据 + 代码 + 已下载的 HF-only 模型）
rsync -av --exclude .venv --exclude models ~/projects/research/SingGuard/eval/ \
      intern3@10.103.22.78:~/research/SingGuard/eval/

# 远程下载 ModelScope 可达的模型（SingGuard×3 + Qwen3-VL×2）
ssh intern3@10.103.22.78 'cd ~/research/SingGuard/eval && uv run python scripts/download_models.py'
```

HF-only 模型本地下载后单独 rsync：

```bash
uv run --with huggingface_hub python scripts/download_models.py --models guardreasoner-vl-7b,llavaguard-7b
rsync -av ~/projects/research/SingGuard/eval/models/ intern3@10.103.22.78:~/research/SingGuard/eval/models/
```

## 四、远程评测

```bash
ssh intern3@10.103.22.78
cd ~/research/SingGuard/eval
tmux new -s eval

# 冒烟：每数据集 5 条，确认解析/图像通路没问题（结果不可比，看 unparsable 比例）
uv run python runner.py --models sing-guard-8b --smoke 5

# 正式：SingGuard 三行（每模型 8 列 ≈ 2–4 h，断点续跑，中断重跑即续）
uv run python runner.py --models sing-guard-2b,sing-guard-4b,sing-guard-8b

# baseline（全部可跑模型）
uv run python runner.py --models all

# 某模型 vLLM 加载失败时退回 transformers
uv run python runner.py --models llavaguard-7b --engine hf
```

- 断点续跑：预测逐条落 `results/preds/<model>__<dataset>.jsonl`，重跑自动跳过已完成；
- 监控：`watch -n 5 nvidia-smi`；日志 `logs/`；每跑完一个 (模型,数据集) 即追加
  `results/results.csv`；
- Ctrl-C 中断安全（tmux 里 Ctrl-b d 脱离会话即可挂后台）。

## 五、汇总对比

```bash
uv run python aggregate.py
# outputs/table4_repro.csv    复现宽表（行=模型，列=8 数据集 + Avg）
# outputs/table4_compare.csv  与论文数字逐列对比 + Δ
```

`configs/table4_paper.csv` 是论文原始数字（含全部 17 行，包括未跑的闭源行，便于对照）。

## 六、范围与已知偏差

| 项 | 状态 |
| --- | --- |
| SingGuard-2B/4B/8B | 官方 checkpoint + 官方模板 + fast 模式，最接近论文 |
| Qwen3-VL-4B/8B、GuardReasoner-VL-7B、LlavaGuard | 提示词为近似，数字会有偏差 |
| Qwen3-VL-235B | 单卡 A6000 装不下，默认跳过（可接 DashScope API 自行扩展） |
| GPT-5.1 / Gemini3-Pro | 需要 API key，默认跳过 |
| ShieldGemma-2 / LlamaGuard3-Vision / LlamaGuard4 | HF gated：网页点同意 + `HF_TOKEN` 后在 `models.yaml` 启用 |
| SafeGuard-VL / LLaVAShield | 公开渠道待确认，`models.yaml` 已留位（`enabled: false`） |
| MMDS-Q/R | 数据需手动获取（见上） |
| 采样 | 论文未说明是否子采样；此处固定 seed=42 每集 1000 条，±1–2 点波动属正常 |

## 七、目录速查

```
configs/        models.yaml datasets.yaml table4_paper.csv（论文参考数字）
scripts/        download_models.py prepare_data.py inspect_data.py
sgeval/         adapters(提示词/解析) engine(vLLM/hf) parsing metrics
runner.py       评测主入口（断点续跑）
aggregate.py    汇总 + 对比论文
data/           raw/<key>/ 原始；<key>/test.jsonl 归一化采样后（gitignored）
models/         checkpoint（gitignored）
results/        preds/ + results.csv（gitignored）
outputs/        table4_repro.csv table4_compare.csv（gitignored）
```
