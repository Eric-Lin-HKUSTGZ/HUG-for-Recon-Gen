# PointNeXt 中的 FPS（最远点采样）详解

> 对应代码：`src/models/pointnext.py` 中的 `_fps_indices`（第 29–44 行）
> 用途：SetAbstraction 每一级下采样时，从当前点集中选出"质心"（centroid）

---

## 1. FPS 在 PointNeXt 中扮演什么角色

PointNeXt 编码器的每一级 SetAbstraction（SA）都做一次下采样：

```
xyz  (B,N,3) ──SA1──▶ xyz1 (B,1024,3) ──SA2──▶ xyz2 (B,256,3)
            ──SA3──▶ xyz3 (B,64,3)  ──SA4──▶ xyz4 (B,16,3)
```

**所谓"质心"就是 FPS 从当前点集中选出的代表点**。关键事实：

- 质心**不是平均/聚类出来的**，而是原有点的**子集**（坐标原值，米制，不做修改）；
- FPS 只输出**下标** `(B, K)`，再由 `_gather` 取出坐标：`new_xyz = gather(xyz, idx)`；
- 选出质心后，SA 模块再对每个质心做 kNN 分组 + 边特征 MLP + max-pool，把邻域信息聚合到质心上，得到 `new_feat`。

四级 SA 的质心数：`N → 1024 → 256 → 64 → 16`，逐级嵌套（`xyz4 ⊂ xyz3 ⊂ xyz2 ⊂ xyz1 ⊂ xyz`）。

---

## 2. 算法流程（代码逐行对应）

```python
def _fps_indices(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """Batched FPS: (B, N, 3) → (B, K) local indices"""
    B, N, _ = xyz.shape
    start = torch.randint(0, N, (B, 1), device=device)   # ① 随机起点
    selected = [start]
    dist = torch.full((B, N), float("inf"))              # ② 运行最小距离
    for _ in range(k - 1):                               # ③ 串行选 K-1 个
        latest_xyz = xyz.gather(...)                     #    (B, 1, 3)
        d = (xyz - latest_xyz).norm(dim=-1)              #    (B, N) 到新点的距离
        dist = torch.min(dist, d)                        #    就地更新最小值
        farthest = dist.argmax(dim=-1, keepdim=True)     #    选 dist 最大者
        selected.append(farthest)
    return torch.cat(selected, dim=-1)                   # (B, K)
```

### 2.1 核心数据结构：`dist` 为什么只需 (B, N)

`dist[b, i]` 的含义是 **"点 i 到当前已选集合的最小距离"**。不需要保存点到每个已选点的距离（那要 `(B, N, K)` 显存），因为 min 运算可折叠：

```
min(到集合{a,b,c}的距离) = min( min(d(i,a), d(i,b)), d(i,c) )
                          └──── 上一轮已存入 dist[i] ────┘
```

每轮只需 `dist = min(dist, 到新点的距离)`，显存 O(N)，每轮计算 O(N)。

### 2.2 单步示例

5 个点 {P0..P4}，随机起点 P2，再选 2 个：

```
第1轮: d    = [5.0, 3.0, 0, 2.0, 8.0]      (到 P2 的距离)
       dist = min(inf…, d) = [5.0, 3.0, 0, 2.0, 8.0]
       argmax → P4          selected = [P2, P4]

第2轮: d    = [7.0, 4.0, 8.0, 6.0, 0]      (到 P4 的距离)
       dist = min(旧值, d) = [5.0, 3.0, 0, 2.0, 0]
       argmax → P0          selected = [P2, P4, P0]
```

---

## 3. 为什么是 max-min（maximin 准则）

"点到集合的距离"取的是**最小值**（到最近已选点的距离），"选下一个"取的是这个最小距离的**最大值**。合称 **maximin 准则**。

对比三种"点到集合距离"的定义：

| 定义 | 后果 |
|---|---|
| **min（采用）** | 靠近任何已选点的候选都被判为"近"，新点必然落在覆盖空隙里 → 均匀铺开 |
| mean | 紧挨已选点的候选只要离其他点远，均值仍大，可能被选中 → 扎堆 |
| max | 只要求离一个已选点远 → 逻辑上选反，往点堆里挤 |

数学上，`dist[i] = min_j d(i, s_j)` 是点 i 到集合 S 的**点-集距离**（豪斯多夫意义），每步 argmax 在最大化当前采样集的**覆盖半径缩减**。

---

## 4. 为什么能保证均匀铺开

**覆盖半径定义**：`R = max(dist)`，即所有点到最近质心距离的最大值。R 越小 ⇒ 没有大片空白 ⇒ 越均匀。

### 直觉：每步精确消灭最大的窟窿

`argmax(dist)` 选中的正是**当前覆盖最差的点**（最大空白区的"中心"）。选完后该点 dist 归零，其余点的 dist 因 min 更新只减不增，故 **R 单调下降**——每轮针对性地补当前最大的空白区。

### 严格性质：packing–covering 对偶

设第 j 个质心被选中时的覆盖半径为 R_j，由单调性 `R_1 ≥ R_2 ≥ … ≥ R_K`，可证：

1. **打包性（packing）**：任意两个质心间距 ≥ R_K。
   （s_j 被选中时它到所有先选点的距离 = R_j ≥ R_K）
   → 质心永不扎堆。
2. **覆盖性（covering）**：选完 K 个后，任何点到最近质心 ≤ R_K。
   → 不存在比 R_K 更大的空白区。

同一个 R_K 同时卡住两端，这就是"均匀"的硬保证。该贪心即 Gonzalez (1985) k-center 算法，是最优 k-center 的 **2 倍近似**（覆盖半径至多为最优解的 2 倍）。

### 对比随机采样

随机采样 K 个点没有任何机制阻止采样点彼此相邻，"某区域重复采、某区域漏采"以正概率发生，覆盖半径无保证。FPS 把随机性压缩到只剩第一个点，之后每步确定性消灭最大窟窿。

### 一维小例子

[0,10] 区间密集点，K=3，随机起点 s₁=3：

```
dist = 到 3 的距离        → 最远 10 (dist=7) → 选 s₂=10
dist = min(到3, 到10)     → 最远 0  (dist=3) → 选 s₃=0
最终 {0, 3, 10}，R=3，区间被均匀覆盖
```

---

## 5. 复杂度与性能注意点

### 循环次数

FPS 串行：第 i 个质心依赖前 i−1 个。四级合计循环：

| 阶段 | K | 循环次数 | 每轮计算 |
|---|---|---|---|
| SA1 | 1024 | 1023 | `(B, N)` |
| SA2 | 256  | 255  | `(B, 1024)` |
| SA3 | 64   | 63   | `(B, 256)` |
| SA4 | 16   | 15   | `(B, 64)` |
| **合计** | | **1356** | |

N=4096 时距离计算总量 ≈ 4.5M 次，**FLOPs 可忽略**。

### 真正的瓶颈：Python 层串行循环

每次迭代发射 4~5 个小 CUDA kernel（gather / sub / norm / min / argmax），1356 次循环 ≈ **约 7000 次 kernel 启动**，启动开销（µs 级）占主导，每次前向仅 FPS 就达数十~一百多 ms。

### 附带热点：kNN 的 4D 张量

`_knn_indices` 显式构造 `diff = (B, M, N, 3)`（SA1：M=1024, N=4096 → 约 50MB/batch），全网 11 次 kNN 调用，显存压力大。

### 优化方向（按收益排序）

1. **换回 CUDA 版 FPS**（`torch_cluster.fps` / `pointnet2_ops`）：循环在单个 kernel 内完成，快 1~2 个数量级。当前用纯 PyTorch 是因为系统 glibc < 2.32 二进制不兼容（见文件头注释）；
2. kNN 用 `torch.cdist` 或分块 topk，避免物化 4D 张量；
3. 减小 SA1 质心数 1024→512（砍近半循环，需精度实验）；
4. `torch.compile` 融合循环体小算子，减少启动开销。

---

## 6. 一句话总结

> FPS = 随机起点 + 反复执行"找离已选集合最远的点"，用 O(N) 的运行最小值数组增量维护；
> 质心是真实点的子集（非平均/插值）；maximin 准则保证 packing（质心互不扎堆）+ covering（无大空白区），是最优 k-center 的 2 倍近似；
> 实现上 FLOPs 便宜，但 1356 次 Python 层串行循环的 kernel 启动开销是主要性能瓶颈。
