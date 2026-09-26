# 跨节点跑 Qwen3-VL-235B：探索方案

目标：用 `.78` 与 `.79` 各 4 张空闲卡（共 8 卡、跨 2 节点）跑起 FP8 版
`Qwen/Qwen3-VL-235B-A22B-Instruct-FP8`（权重 221.3 GiB，24 分片）。

规模硬约束（已核实）：`num_attention_heads=64` → TP 世界大小必须整除 64；94 层、128 专家、
`num_key_value_heads=4`。单节点 8 卡是唯一可行的本地配置，因此跨节点是"用 4+4 凑 8"。

---

## 两条路线（由 Phase 0 决定走哪条）

| 路线 | 配置 | 跨节点流量 | 适用前提 |
|---|---|---|---|
| **A. TP=8** | `--tensor-parallel-size 8` | 每层 2 次 all-reduce × 94 层 = 188 次/token | **必须有 RDMA/IB**（或 100G+ RoCE） |
| **B. PP=2 × TP=4** | `--pipeline-parallel-size 2 --tensor-parallel-size 4` | 只传层间激活（每 token 约 94/2 次边界传输） | 普通以太网也勉强可用 |

显存核算（FP8 权重 221.3 GiB）：
- 路线 A：221.3 ÷ 8 = **27.7 GiB/卡**
- 路线 B：221.3 ÷ 2 = 110.6 GiB/节点 → ÷4 卡 = **27.7 GiB/卡**
两者单卡占用相同；差别只在跨节点通信模式。**B 是弱互联下的现实选择。**

---

## Phase 0 — 只读探测（每节点 ~10 分钟，零风险，先做这个）

两个节点各跑一遍，把输出贴回来即可判定路线。

```bash
# 1) 有没有 InfiniBand / RoCE 设备
ls /sys/class/infiniband/ 2>/dev/null; ibstat 2>/dev/null | head -40; ibv_devinfo 2>/dev/null | head -40

# 2) 网卡与链路速率
ip -br link
for i in $(ls /sys/class/net | grep -v lo); do
  echo "== $i"; ethtool $i 2>/dev/null | grep -E "Speed|Duplex|Link detected"
done

# 3) GPU 与网卡的拓扑关系（看 GPUs 是否直连 IB HCA）
nvidia-smi topo -m

# 4) 节点内 NVLink（决定 intra-node TP=4 的效率）
nvidia-smi nvlink -s 2>/dev/null | head -20

# 5) 现成的测试工具在不在
which ib_send_bw ib_write_bw iperf3 all_reduce_perf 2>/dev/null
python -c "import torch; print('NCCL', torch.cuda.nccl.version())"
```

**判定标准**

| 观测 | 结论 |
|---|---|
| `/sys/class/infiniband/` 非空 且 `ibstat` 显示 Active，速率 ≥100 Gb/s | 走 **路线 A（TP=8）** |
| 只有以太网，速率 ≤25 Gb/s | 走 **路线 B（PP=2×TP=4）** |
| 只有以太网且 <10 Gb/s | **放弃**，改用 API 通道 |

---

## Phase 1 — 打通前置条件

### 1.1 免密 SSH（你来做，双向）
```bash
# 两端各执行一次，然后互推公钥
ssh-keygen -t ed25519 -N '' -f ~/.ssh/id_ed25519     # 若已有则跳过
ssh-copy-id intern3@10.103.22.78                       # 在 .79 上执行
ssh-copy-id intern3@10.103.22.79                       # 在 .78 上执行
# 验证（应无密码输出主机名）
ssh intern3@10.103.22.79 hostname
```

### 1.2 环境对齐（关键，失败率高）
```bash
# 两端分别导出，diff 比对关键包
for n in 78 79; do ssh intern3@10.103.22.$n \
  '/data/intern3/research/SingGuard/.venv/bin/pip freeze | grep -iE "^(vllm|torch|transformers|ray|numpy|flashinfer)"' \
  > /tmp/env_$n.txt; done; diff /tmp/env_78.txt /tmp/env_79.txt && echo "环境一致"
```
**已知差异风险**：`.78` 驱动 535（CUDA 12.2）、`.79` 驱动 595（CUDA 13.2）。
NCCL 库本身来自同一个 venv（torch 2.8+cu128）所以版本一致，但跨节点不同驱动是
Phase 1.5 冒烟必须验证的点。若失败，尝试 `NCCL_CUMEM_ENABLE=0`。

### 1.3 权重可见性（易被忽略的硬门槛）
vLLM 的加载器在每个 rank 上都要能打开**整个** checkpoint 目录（索引 + 分片），
不是只读自己那份。所以：
- 若两节点有**共享文件系统**（NFS/Lustre）→ 放一份即可（首选，检查 `df -hT` 里是否有 nfs/lustre）
- 否则 → **两个节点各存一份完整权重，237.6 GB × 2**。`.78` 剩 3.3T、`.79` 剩 4.4T，容量够。
  注意：`.79` 上那份还没下（之前只下到 `.78`，12 GB 后已停）

### 1.4 安装 Ray（两端）
```bash
/data/intern3/research/SingGuard/.venv/bin/pip install "ray[default]" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 1.5 NCCL 冒烟测试（**决定性一步**，先于任何模型加载）
不加载模型，只用 8 张卡跑一次 all-reduce，直接量出互联带宽。
```bash
# 起点：.78 做 head
ssh intern3@10.103.22.78 'cd ~/research/SingGuard/eval && \
  NCCL_DEBUG=WARN /data/intern3/research/SingGuard/.venv/bin/python nccl_probe.py \
  --head --iface <两台互通的网卡名>'
# .79 加入
ssh intern3@10.103.22.79 'cd ~/research/SingGuard/eval && \
  NCCL_DEBUG=WARN /data/intern3/research/SingGuard/.venv/bin/python nccl_probe.py \
  --worker --addr 10.103.22.78 --iface <同上>'
```
`nccl_probe.py` 由我写好同步过去（用 torch.distributed + NCCL 后端，做 2 节点 × 4 卡的
all_reduce，报告 GB/s 与每次迭代耗时）。

**判定**：
| all_reduce 结果 | 结论 |
|---|---|
| ≥ 20 GB/s | 路线 A（TP=8）可行 |
| 3–20 GB/s | 走路线 B（PP=2×TP=4） |
| < 3 GB/s 或挂起 | 放弃跨节点，改 API |

---

## Phase 2 — 启动（先小后大）

1. **先跑 1 个数据集 + `--limit 20`** 验证端到端能出结果，别看 F1，看能否跑通
2. 再放开全量 8 个数据集
3. Ray 集群：
```bash
# .78（head）
ray start --head --port=6379 --num-gpus=4 --dashboard-host=127.0.0.1
# .79（worker）
ray start --address=10.103.22.78:6379 --num-gpus=4

# .78 上启动（路线 A）
vllm serve models/qwen3-vl-235b-fp8 \
  --tensor-parallel-size 8 --distributed-executor-backend ray \
  --served-model-name guard --port 8280 --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90

# 路线 B 则换成
#   --pipeline-parallel-size 2 --tensor-parallel-size 4
```
我们的 harness 已支持 `tp`，但**尚未支持 PP 与 Ray 后端**，需要我加
`--pipeline-parallel-size` 与 `--distributed-executor-backend` 两个配置项。

---

## Phase 3 — 判定与止损

- 指标：端到端 **tokens/s**（用 20 条样本测）。参照：单节点 8 卡正常应 >200 tok/s
- **止损线**：若 < 5 tok/s，8 个数据集（约 6,600 请求）需要 >100 小时，不划算 → 转 API
- 无论成功失败，把结论写回 OPS.md

---

## 风险清单（按严重度）

1. **无 RDMA**：TP=8 直接不可用（已由 Phase 0 判定）。PP 可绕开但吞吐低
2. **`.79` 资源紧张**：load 11.11 / 54 用户，可用内存仅 149 GB。加载 110 GB 权重时若页缓存不足会非常慢，需要耐心或等负载下降
3. **驱动版本不一致**（535 vs 595）：NCCL 可能报错，Phase 1.5 会暴露
4. **权重需双份**：若无共享 FS，多下 237.6 GB（`.79` 上还没下）
5. **vLLM 的 PP 对 MoE 支持成熟度**：不如 TP，可能需要 `--enable-expert-parallel` 或直接失败
6. **共享节点礼仪**：Ray 会占满节点间网络；`.79` 上有 54 个其他用户，启动前应确认
7. **成本对比**：整套探索 + 双份下载（约 5 小时）成本远高于 API 方案（约 ¥19、2 小时）

## 我的建议

**先做 Phase 0**——只读、10 分钟、零风险，就能知道有没有 RDMA。有 RDMA 就值得往下走；
没有的话，路线 B（PP）虽然理论可行，但考虑 `.79` 的负载和驱动差异，**投入产出比不如直接走 API**。

---

# Phase 0 实测结果（2026-09-26）：跨节点路线判定为不可行

## 硬件事实

| 项 | 实测 |
|---|---|
| InfiniBand / RoCE | ❌ `/sys/class/infiniband` 为空，两台均无 RDMA 设备 |
| 网卡 | Intel I350（千兆）+ Intel 82599ES（万兆）；**无 Mellanox/ConnectX** |
| 节点内 NVLink | ❌ A6000 无桥接器（all links inActive）；且 NVLink 是**服务器内**总线，跨机本就不可用 |
| 当前互联 | 1 GbE，**实测 87.4 MB/s**，RTT ~2.3 ms |
| 万兆口 | `ens29f0/f1` = 82599ES **未接线**（NO-CARRIER），且配 IP 需 root（无 sudo） |
| 共享文件系统 | ❌ 两台都无 |

## 结论

- **路线 A（TP=8）**：188 次 all-reduce/token × 2.5 ms ≈ **0.47 s/token ≈ 2 tok/s** → 判死
- **路线 B（PP=2×TP=4）**：理论可行但需 vLLM MoE PP（不成熟）、权重两份（237 GB）、
  `.79` 可用内存仅 162 GB 且 load ~9 → 投入产出比远不如 API
- **10 GbE 若接通**：路线 A 可达 ~25 tok/s，但**需要管理员接线并配 IP**（无 sudo）

## 真正的替代路线：llama.cpp + GGUF（单节点，已验证）

跨节点失败的根因是**互联**，而 llama.cpp 的 MoE 专家卸载**根本不需要跨节点**：

- llama.cpp v0.5.0 **原生支持 Qwen3-VL**（`tools/mtmd/models/qwen3vl.cpp`）
- GGUF Q4_K_M 共 142 GB + mmproj 0.75 GB，ModelScope（国内）直下
- `.78` 有 455 GB 空闲内存 → `--cpu-moe` / `--n-cpu-moe` / `-ot` 把专家放内存
- `llama-server` 提供 OpenAI 兼容接口 → 复用已实现的 `openai_compatible` 引擎

### 构建踩坑（已解决）

| 问题 | 现象 | 解法 |
|---|---|---|
| 系统 cmake 3.16 太旧 | `ggml-cuda` 要求 ≥3.18 | `uv venv` + `uv pip install cmake`（→4.4.3），无需 root |
| **cuBLAS 版本错配** | 链接报 `cublasSetWorkspace_v2` undefined | CMake 默认找到了 Ubuntu 的 **CUDA 10.2** `libcublas`，而 nvcc 是 12.2 → 显式指定 `-DCUDA_cublas_LIBRARY=/usr/local/cuda-12.2/lib64/libcublas.so.12` |
| GitHub 限速 | `raw.githubusercontent` 仅 2.4 KB/s | 37 MB 源码走本机下载再传；或 GitCode 镜像 |

### 验证结果（2B 模型端到端）

```
llama-server  v0.5.0-dev  CUDA backend  ✓
text  probe: "OK. How can I assist you today"                     ✓
IMAGE probe: "A dog peeking out from behind a curtain."           ✓  (用 VLGuard 真实图片)
```
