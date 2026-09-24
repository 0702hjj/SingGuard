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
... --thinking fast --max-tokens 1024          # fast 模式（模板只实现了 fast/fast-slow；`slow` 会被静默渲染成 fast，已禁）
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

---

# 踩坑记录（论文未提及、我们自行解决的问题）

按主题归类。日期 2026-09-22，环境：.78（8×A6000, 驱动 535.86 / CUDA 12.2, ubuntu）。

## 1. 模型与数据下载（CN 网络环境）

| 问题 | 现象 | 解法 |
|---|---|---|
| ModelScope 速度剧烈波动 | 同一模型 65kB/s ~ 7.7MB/s | 测速后择优：服务器直连快就服务器下，慢就本地下完 tar-ssh 推送（实测 campus ssh ~16MB/s） |
| HF 大文件传输频繁中断 | `SSL: UNEXPECTED_EOF` / xet CDN 报错 | `HF_HUB_DISABLE_XET=1` + snapshot 断点续传；vlsbench 最终用 curl `-C -` 循环扛下来 |
| gated 数据集 | VLGuard 401 | 网页同意条款 + read token（`~/.cache/huggingface/token` 自动生效） |
| **论文引用名 ≠ HF 实际仓库 id** | VLSBench / SPA-VL / MM-SafetyBench 均 404 | 正确 id：`Foreshhh/vlsbench`（非 ys-zong）、`sqrti/SPA-VL`（非 dsty）、`PKU-Alignment/MM-SafetyBench`（非 isXinLiu） |
| Qwen 基线在 ModelScope 无 base 版 | `Qwen/Qwen3-VL-{4,8}B` 404 | 用 Instruct 版（生成式 guard 基线本应如此）：`Qwen/Qwen3-VL-*-Instruct` |
| JailBreakV 图像不完整 | HF 仓库仅 ~300/28000 张图（完整图在 Google Drive） | 强制纳入全部本地可得图（338/1000）+ 确定性文本样本填充，README 记录偏差 |
| 中断的下载分片保留正式文件名 | 启发式完整性检查（有 config+safetensors 即"完整"）误判 | 下载器成功后写 `.complete` 标记；推送/运行只认标记 |

## 2. GPU 环境兼容（最大时间黑洞）

| 问题 | 现象 | 解法 |
|---|---|---|
| **驱动 535 (CUDA 12.2) vs torch cu130** | `torch._C._cuda_init: driver too old (found 12020)`，任何后端都碰不了 GPU | 换 cu12x 栈：`vllm==0.11.0`（自动带 torch 2.8.0+cu128）。CUDA 12.x minor-version compatibility 使 cu128 轮子跑在 12.2 驱动上 |
| flashinfer JIT 编译失败 | `Ninja build failed`（现场编译 kernel，服务器工具链不配） | 直接卸载 flashinfer——可选加速件，vllm 回退自带 kernel；顺带 `VLLM_USE_FLASHINFER_SAMPLER=0` |
| transformers 5.x 与 vllm 0.11 不兼容 | `Qwen2Tokenizer has no attribute all_special_tokens_extended`（5.x 删除了该 API） | 钉 `transformers==4.57.1`（对 Qwen3-VL 支持完备且与 vllm 0.11 同时代） |
| 共享 GPU | 空闲卡随时被别人占用（GPU6 曾中途被占 15.8G） | 跑前 `nvidia-smi` 挑卡 + `--gpu-util 0.6` 共享跑；只杀自己 venv 路径的进程 |

## 3. 推理接口细节（论文/README 没写或误导的）

| 问题 | 现象 | 解法 |
|---|---|---|
| **fast 模式行为异常** | 论文 2.6 称 fast 用于 high-throughput benchmark evaluation，但 checkpoint 在 fast 模式下**连 README 自己的炸弹示例都判 safe** | 用默认 **fast-slow** 模式 + `max_tokens=1024`（README 官方示例的输出格式即 hybrid 格式，其 max_new_tokens=1024） |
| hybrid 输出的首行是临时判决 | 4.1 节"parse the leading safe/unsafe decision token"按字面实现会取到临时值；真实判决在推理后的 `<answer>` | 解析器 **`<answer>` 优先、首行兜底**。佐证：2.5 节 RL 奖励"decodes the complete valid response"且首个 token 被 mask——计分从来不用首 token |
| thinking_type 传参方式 | `processor.apply_chat_template(..., thinking_type="fast")` 被**静默忽略**（warning 一行） | 必须走字典参数 `chat_template_kwargs={"thinking_type": ...}`；vLLM 侧是 `extra_body={"chat_template_kwargs": ...}` |
| 模板文件位置 | guard 提示词在独立 `chat_template.jinja`（tokenizer_config.json 的 chat_template 字段为空，新版惯例） | vLLM 启动显式 `--chat-template`（0.11 实测能自动读，显式更稳） |
| **VLGuard 必须 query-side 评测** | 数据集 gold-unsafe 项配的是无害拒绝回复；模型卡明示 "Refusals and safe redirections can be classified as safe"，带上回复后整列 recall 塌到 0（实证：8B 0/1000 标记） | VLGuard 判定只喂 (query, image)，不带 response（论文 4.1：query-side 与 response-side 分开评） |
| **`thinking_type=slow` 是假模式** | 模板只判断 `== 'fast-slow'`，其余值（含 slow、拼写错误）都落到 fast 格式块——`<thinking_type>slow</thinking_type>` 配 fast 提示词，训练分布外 | runner 的 `--thinking` 限定为 fast/fast-slow；OPS 命令已修正 |
| **评测协议：首 token 才是判决** | 论文 4.1 原文 "parse the leading safe/unsafe decision token"，`<answer>` 仅用于归因；实测（用已存 raw 重算）首 token 与我们旧口径在 99.7% 记录一致且各列 ≥，个别列 +0.10 | 解析器已改为**首 token 优先、`<answer>` 兜底**，并容忍 code fence（基线常把判决包在 ``` 里）|
| **8192 上下文 → 13 条大图样本确定性 400** | beavertails-v 有 13 张图 >8175 visual token（preprocessor 与官方 Qwen3-VL 一致、不降采样；模型自身上限 262144），5 个模型上全部 HTTP 400 被计错（~1.3%/列）| `max_model_len` 提到 32768（KV 仅 144KB/token，成本可忽略）|
| vllm 0.29 CLI 参数变更 | `--disable-log-requests` 不存在 → 启动失败 exit 2 | 删掉该 flag（新版本默认不记请求日志） |

## 4. 数据集真实 schema（论文只给名字，全部要自己摸）

- **VLGuard**：实际是 `test.json`（平铺 list：`safe` 布尔 + `instr-resp[].instruction/safe_instruction/response`）+ `test.zip` 图像包（558 safe / 442 unsafe）
- **BeaverTails-V**：`is_response_safe` 列是 **yes/no 字符串**（不是布尔），按每类别 `data/<category>/train.parquet` 组织
- **SPA-VL**：test 拆成 `harm-*.parquet` / `help-*.parquet`，标签在文件名里；1.3GB train.zip 不需要
- **VLSBench**：无标签列（全攻击集，6 危害类别）；parquet 版内嵌图像，另有冗余 imgs.tar 勿下
- **MM-SafetyBench**：`data/<13 主题>/{SD,SD_TYPO,TYPO,Text_only}.parquet`。论文规模 5,040 = 13 主题 x 3 图像变体 x ~129 —— **Text_only 是对照消融变体，论文未用**，loader 须排除（旧版误含 242 条）
- **SPA-VL**：HF test config = harm(265, unsafe) + help(265, safe)；论文标 7,530 = 上述 530 + validation(7,000)，而 validation 是 chosen/rejected 偏好对、**无二分类标签**。我们评测官方 test config，口径差异已记录

## 4b. 数据集规模口径对照（论文 vs 本仓库）

| 数据集 | 论文规模 | 我们的池子 | 采样 | 说明 |
|---|---|---|---|---|
| VLGuard | 1,000 | 1,000 | 全量 | query-side（见上）|
| JailBreakV | 28,000 | 28,000（图像仅 ~360 可用） | 1,000 | 官方图像大部分在 Google Drive，未公开镜像 |
| SPA-VL | 7,530 | 530（官方 test config） | 全量 | 7,000 validation 无标签 |
| VLSBench | 2,241 | 2,240 | 1,000 | |
| MM-Safety | 5,040 | 5,040（图像变体） | 1,000 | 已排除 Text_only |
| BeaverTails-V | 1,180 | 1,180 | 1,000 | response-side |

## 4c. 解码口径

- README 全部示例用 `do_sample=False`（greedy），我们采用 greedy；
- 模型自带 `generation_config.json` 却推荐采样（temperature 0.7 / top_p 0.8 / top_k 20）——论文未说明评测用哪种，偏差已记录。

## 5. 与论文行为的实质差异（复现的核心发现）

**已定位并修复的协议差异**：VLGuard 曾因携带 assistant 回复导致整列 recall 塌到 0
（8B: 0/1000 标记）——该数据集的 gold label 指 query+image 侧，而 gold-unsafe 项配的是
无害拒绝回复，模型按模型卡明示（"Refusals and safe redirections can be classified as
safe"）判 safe。改为 query-side 评测后同批样本 F1 0.0 → 0.8685（300 条 A/B 实证，
precision=1.0）。教训：**评测前先确认每个基准是 query-side 还是 response-side**（论文 4.1
只给了一句通则，未逐数据集标注）。

**已定性的核心差异（发布权重 vs 论文数字）**：论文 4.2 自己承认 JailBreakV/VLSBench/MM-Safety
三列 "saturated by guards that specifically train on jailbreak templates"——论文这些列的 0.88~0.99
要求模型近乎全判 unsafe（全 unsafe 集下 F1=2R/(1+R)，LLaVAShield 0.9939 ⇒ R≈0.99）。实测发布权重：
8B 是审慎的 policy judge（对白底黑字 "Crohn's disease" 图 + 正当医疗问题判 safe、VLSBench R=0.525），
2B 反而近乎全喊 unsafe（VLSBench R=0.988、SPA-VL R=0.9925）；**两个尺寸都不匹配论文数字**，而论文
Table 12 显示模式/配置差异只值 ~1 个点。→ 结论：Table 4 的绝对值不可用发布权重复现，复现应报告
排序/趋势 + 逐列 Δ，并注明该差异来源（refusal-template 敏感度 vs policy-grounded 判定）。

**待全量确认的剩余偏差**：query-side 修复后，复现值与论文报告仍可能有余量（如 8B
VLGuard 论文 0.9511）——baseline 提示词未公开、子采样口径未知、checkpoint 行为差异都是
来源，报告中需逐列如实记录 Δ。

> **2026-09-24 订正**：上面"绝对值不可用发布权重复现"的结论**已被证伪**。补上 MMDS
> 的正确口径后，8B 全 8 列 Avg 0.8828（论文 0.9092，Δ −0.026）、2B 0.8978（Δ +0.005），
> 9 列中 7 列 |Δ| ≤ 0.03。当时的"两个尺寸都不匹配"里有很大一块其实是 MMDS 采样错误
> （见 §6.1）拉低了均值。剩余缺口集中在 SPA-VL（−0.08，协议歧义）与 JailBreakV
> （−0.064，图像只有 360/28000）两列，均已单独归因。

## 6. 2026-09-24 复盘：四个静默污染数据的 bug

共同特征：**都不报错、都不影响其它列、都只能靠交叉验证发现**。报告数字若与此前不一致，
先怀疑这四条。

### 6.1 MMDS 必须用官方 test split（影响最大）

- 语料 `mmds.jsonl` 自带官方 `set` 字段：**train 4045 / val 109 / test 330**，论文评的是 test。
- 旧 loader 完全忽略该字段，从全量 4,484 条分层采样 1,000 → **约 93% 是训练集**，
  且标签分布不同（全量池 31% unsafe，test split 52%）。
- 后果：同一模型、同一 harness，**MMDS-Q 0.7722 → 0.9571**（8B；论文 0.9851），
  MMDS-R 0.7209 → 0.8908（论文 0.8963）。
- 教训：数据集自带 split 字段时**先查 `set`/`split` 列**，不要假定全量池即评测集。
- 归档：全量池版本仍可跑，键名降级为 `mmds-*-pool`，预测留在 `*__mmds-*-pool.jsonl`。

### 6.2 每个模型必须喂它自己的 chat template

- vLLM 只读 `tokenizer_config.json` 的内联 `chat_template` 或显式 `--chat-template`，
  否则**静默替换成通用后备模板**。两种发布惯例把真模板放在别处：
  `chat_template.jinja`（SingGuard/LlamaGuard4/SafeGuard-VL，vLLM 不自动读）与
  `chat_template.json`（LLaVA-NeXT 系）。
- 中招者：**LlavaGuard**（只有 `chat_template.json` 且无内联）→ 26% 样本返回
  **缺失开头键的 JSON 残片**（如 ` The image violates category: "Safe" ... "}`）。
- 修法：`sgeval.engine.resolve_chat_template` 依次解析 .jinja / .json，并把模板哈希并入
  cfg 指纹（有内联模板的模型短路返回空指纹，保证既有 tag 逐字节不变、不误触发重跑）。

### 6.3 基线要用它自己发布的 prompt（三次踩同一个坑）

同一个教训在三个模型上各犯一次，都是"我们自造提示词 → 输出格式不对 → 解析失败记错"：

| 模型 | 我们的错误做法 | 官方实际要求 |
|---|---|---|
| GuardReasoner-VL | 自造 instruction 放 user 消息 | **system message** 装 INSTRUCTION + `Human user:\n<image>\n{query}\n\nAI assistant:\n{response}` 转录；结论在 `<result>` 块，词汇是 harmful/unharmful |
| LlavaGuard | 自造 "answer with predicted_label" | 5.4KB 的 **O1–O9 政策全文**（Should not/Can 条目 + assessment steps + JSON 模板），判决键是 **`rating`** |
| LLaVAShield | （已正确）| 官方 prompt 组装已 vendored |

- 修法：模板/prompt **逐字 vendored 进 `sgeval/vendor/`**，不要手抄、不要"condensed"。
- GuardReasoner 修复后 unparsable 496–568/1000 → **0**，VLGuard 0.2973 → 0.8822。
- 配套机制：`Adapter.version` 并入 cfg 指纹（默认空 → 不破坏既有 tag）。**改 prompt
  就是换了一个问题**，必须让该模型的缓存失效，否则会拿旧答案冒充新结果。

### 6.4 `recompute.py` 的列错位

`DictWriter(fieldnames=list(rows_out[0]))` 用了**新行的键序**，而 results.csv 已有自己的
表头 → 追加的整行**错位**：F1 落进 `n`、gold 计数落进 `thinking`/`max_tokens`，真正的
`f1` 为空。报告随后按 ts 取"最新行"，于是静默退回到**被淘汰的旧 MMDS 数字**。
修法：追加前先从文件读表头。

### 6.5 其它

- **`\toprule` 转义**：`"c}\toprule"` 非 raw string → `\t` 变真 TAB →
  `table4_compare.tex` 一直编译不过。生成 LaTeX 的字符串一律用 raw。
- **GPU 抢占**：`.79` 的 GPU3 在一次检查间隔内被别的用户占满（15MiB → 39GB/92%）。
  申请 GPU 前**重新查一次**，并用保守 `--gpu-util` 给邻居留余量。
- **跨机传输**：`.79 → .78` 无 SSH 通道（Permission denied, publickey）。**不要经本机中转**
  （~5.6MB/s 且依赖本地机器在线）——两台机器各自都能连 hf-mirror/ModelScope，**让目标机自己下**。
- **上下文上限要按 checkpoint 自己的 config 定**：LLaVAShield 的 16K 是
  `tokenizer_model_max_length`（分词器设置），`max_position_embeddings` 才是 32768。
  按 16K 跑会让 **103/330 条 MMDS-Q 提示（全是 unsafe 标签，占 unsafe 的 61%）**被 400 拒绝
  并记错——恰好压掉该列要测的召回。
- **vLLM `device_map="auto"`** 需要 `accelerate`（离线环境没有）→ 分类器路径改用
  `.to("cuda")`。
