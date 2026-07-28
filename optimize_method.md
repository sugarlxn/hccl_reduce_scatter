结论：当前性能差距的主因不是双缓冲参数，而是大数据算法的工作分配方式与真实硬件带宽不匹配。v2.4 仍把 ReduceScatter 设计成“Server 内代算多个目标块，再跨 Server 交换 partial”，导致 layer-0 重复搬运和规约，同时只使用极少量 Clos peer。对 8+4 的影响尤其严重。

## 性能差距

根据 [test_result.csv](/home/ubuntu/hccl_reduce_scatter/test_result.csv:2)：

| 拓扑/大小 | Benchmark | v2.4 | v2.4 / Benchmark |
|---|---:|---:|---:|
| 2×8 / 512KB | 19 μs | 35 μs | 1.84× |
| 2×8 / 512MB | 1.27 ms | 1.84 ms | 1.45× |
| 2×8 / 400MB+4B | 0.979 ms | 1.43 ms | 1.46× |
| 8+4 / 512KB | 24 μs | 37 μs | 1.54× |
| 8+4 / 512MB | 1.85 ms | 3.51 ms | 1.90× |
| 8+4 / 400MB+4B | 1.40 ms | 2.73 ms | 1.95× |

400MB 和 512MB 的比例基本线性，说明大包已经是稳定的数据面瓶颈，而不是 launch 抖动。继续微调两段 tile 比例，收益会很有限；v2.3→v2.4 也只改善了约 1%～4%。

## 当前算法的问题

### 1. 2×8：每个 NPU 在本机计算两个目标块

当前计划给每个 rank 分配：

- 自己的输出块；
- 对端 Server 配对 rank 的输出块；
- 最后只和一个跨 Server peer 交换 partial。

见 [reduce_scatter.cc](/home/ubuntu/hccl_reduce_scatter/op_host/reduce_scatter.cc:164)。

因此每个 NPU 对两个完整目标块分别读取本 Server 的 8 个输入贡献。虽然跨 Server 数据量很小，但 layer-0 的搬运、scratch 写入和本地规约几乎翻倍。

这对传统“跨机慢、机内快”的网络合理，但这里：

- 每个 NPU 都可直接使用 Clos；
- Clos 接入带宽约为单条 layer-0 链路的 4 倍；
- 当前却只让每个 NPU 使用一个 Clos peer。

减少跨机字节数换来的，是大量 layer-0 重复工作和 Clos 空闲。

### 2. 8+4：4 卡侧形成严重热点

4 卡侧每个 rank 被分配三个 target：

- 自己；
- 8 卡侧一个 rank；
- 8 卡侧另一个 rank。

见 [reduce_scatter.cc](/home/ubuntu/hccl_reduce_scatter/op_host/reduce_scatter.cc:179)。

对于每个 42.67 MiB 输出块，4 卡侧每个 rank 要为三个 target 各读取另外三个本机 NPU：

\[
3\ targets \times 3\ peers \times 42.67\text{ MiB}
\]

也就是每个 layer-0 peer 链路承载三个输出块。8 卡侧只需要计算一到两个 target，负载明显不对称，最终整体时间由 4 卡侧决定。

这很好地解释了为什么：

- 2×8 只慢约 45%；
- 8+4 却慢约 90%～95%。

问题不是 Clos 出口不足，而是算法主动把 4 卡侧变成了代理规约节点。

### 3. 两个 IO Die 没有真正并行

当前确实注册了 layer-0 和 layer-1 两个 CCU kernel，但执行时都下发到同一个 thread：

- 先 launch local kernel；
- 再 launch cross kernel。

见 [exec_op.cc](/home/ubuntu/hccl_reduce_scatter/op_host/exec_op.cc:147)。

而且 cross 阶段依赖 local partial，因此两个 Die 的执行是串联关系。当前只申请一个 thread，也印证了这一点：[reduce_scatter.cc](/home/ubuntu/hccl_reduce_scatter/op_host/reduce_scatter.cc:338)。

实际执行结构是：

```text
layer-0：多个 target 的局部规约
                    ↓ 完成后
layer-1：一个 peer partial 的 ReadReduce
```

理想结构应当是：

```text
layer-0 CCU：规约本 Server 对“自己输出块”的贡献 ─┐
                                                ├─ 本地合并两个 partial
layer-1 CCU：规约远端 Server 对“自己输出块”的贡献 ─┘
```

两边可以在不同 CCU、不同 thread 上同时运行。

### 4. staging 后的本地规约仍是串行 N−1 遍

大包路径虽然并行读取不同 peer，但数据到达后采用：

```cpp
LocalCopy(source 0)
EventWait
LocalReduce(source 1)
EventWait
LocalReduce(source 2)
EventWait
...
```

见 [ccu_kernel.cc](/home/ubuntu/hccl_reduce_scatter/op_kernel_ccu/ccu_kernel.cc:372)。

每个 source 都单独产生一次规约和等待。2×8 每个 target 是 7 次 `LocalReduce`；8+4 的 4 卡侧是每 target 3 次、共三个 target。

本地 HCCL 的 CCU ReduceScatter 实现已经提供了更高效的模式：

- 多 peer 并行 `Read` 到连续 scratch；
- 在 CCU buffer 中做 group reduce；
- 或对连续 scratch 做树形/成组规约；
- 多个 tile 软件展开，先批量 Read，再批量 Wait/Reduce。

参考 [/home/workspace/hccl/src/ops/reduce_scatter/template/ccu/kernel/ccu_kernel_reduce_scatter_mesh1d_mem2mem.cc](/home/workspace/hccl/src/ops/reduce_scatter/template/ccu/kernel/ccu_kernel_reduce_scatter_mesh1d_mem2mem.cc:178)。

### 5. v2.0 不能否定全 Clos 方案

v2.0 全部使用 layer-1，却退化到 4.90 ms/4.55 ms。原因是它采用每个 source：

```text
ReadReduce → EventWait → 下一个 source
```

见当前保留的直接 mesh 实现 [ccu_kernel.cc](/home/ubuntu/hccl_reduce_scatter/op_kernel_ccu/ccu_kernel.cc:96)。

它串行执行 15 或 11 次远端规约，无法体现 Clos 多 peer 并行和 4× 接入带宽。正确对比对象应是“并行 Clos Read + 本地 group reduce”，而不是 v2.0 的串行 `ReadReduce`。

## 优化方向

### P0：改成双 Die 并行的 own-target partial reduction

这是最值得优先实现的方向。

每个 rank 只计算自己的输出块：

- layer-0 kernel：读取本 Server 所有 source 对自己输出块的贡献，生成 local partial；
- layer-1 kernel：通过每个远端 NPU 自身的 Clos 接口读取远端 source，生成 remote partial；
- 两个 kernel 使用主、从两个 thread 并发运行；
- join 后做一次确定性的本地 partial 合并。

收益：

- 2×8：每个 rank 不再代算对端 target，layer-0 工作量近似减半；
- 8+4：4 卡侧从三个 target 降到一个，直接消除最大热点；
- Clos peer 从当前每 rank 1～2 个扩展到所有远端 source；
- layer-0 与 layer-1 两个 IO Die 同时工作；
- 每个对端仍只申请一个 channel，符合约束。

HCCL 自带的双 Die CCU 模板也是使用两个 thread 并发 launch 两个 kernel，再合并 partial，可参考 [/home/workspace/hccl/src/ops/reduce_scatter/template/ccu/ccu_temp_reduce_scatter_mesh_1D_2die_mem2mem.cc](/home/workspace/hccl/src/ops/reduce_scatter/template/ccu/ccu_temp_reduce_scatter_mesh_1D_2die_mem2mem.cc:145)。

这一路径最有希望把：

- 2×8 大包从 1.84 ms 拉向 1.3 ms；
- 8+4 大包从 3.51 ms 显著拉向 2 ms 附近。

### P1：并行 Read + CCU group reduce，替换串行 LocalReduce

每个 partial kernel 内部采用：

1. 对所有 source 同时发出 `Read`；
2. 写入连续 scratch slot；
3. 单次 masked `EventWait`；
4. 使用 CCU buffer group reduce，或树形连续区域规约；
5. 输出一个 partial。

FP32 确定性可以通过固定的树形括号顺序保证；规则只要求相同输入多次执行结果一致，不要求严格按 rank 0→N−1 的线性累加顺序。

这会减少：

- `LocalReduce` 指令数量；
- EventWait 数量；
- scratch DRAM 的重复读写；
- CCU 指令调度开销。

### P2：kernel 内微切片和软件流水，不按整个 recvBytes staging

400MB CCL buffer 无法一次容纳 12/16 个完整 512MB 用例 slot，因此不要恢复现有“每 rank 一个完整 slot”的平铺 staging。

建议使用固定微 tile，例如 2～8 MiB：

```text
bank 0：并行读取 tile k
bank 1：规约 tile k-1
```

两个 kernel 分配不重叠的 scratch 区域，并在一个 kernel launch 内部循环所有 tile。这样可以：

- 避免 512MB 用例因 CCL buffer 不足产生多次 host launch；
- 避免 400MB+4B 的 4 字节尾块单独 launch；
- 持续重叠网络搬运与本地规约；
- 控制 scratch 占用在 400MB 内。

### P3：小包单独使用 Clos NHR/halving-doubling

512KB 总输入下，每 rank 实际输出只有：

- 2×8：32 KiB；
- 8+4：约 42.7 KiB。

此时数据量不是主要因素，当前 35/37 μs 主要来自：

- 两个 kernel launch；
- 多 peer 地址/token 握手；
- local/cross 两阶段串联。

小包可以单独尝试单 layer-1 NHR：

- 2×8 约 4 个逻辑阶段；
- channel 数和同步数远少于 15-peer mesh；
- 一个 CCU kernel；
- 每步只接收一个确定 peer，天然满足确定性。

不建议让大包和小包共用同一个算法分支。

## 推荐实施顺序

1. 先实现双 thread、双 CCU 并发，每个 rank 只计算 own target。
2. 第一版允许两个 kernel 各自产生一个 DRAM partial，最后一次本地规约，先验证拓扑收益。
3. 再将两个 partial kernel 改成并行 Read + group reduce。
4. 引入微 tile 双缓冲和 kernel 内循环。
5. 最后单独优化 512KB 的 NHR/单 kernel 路径。

不建议继续优先调整当前 `tileBytes` 的 1:1 比例或增加 staging bank。v2.4 的结果已经说明，这类局部流水优化无法解决 target 委派、4 卡侧热点和双 Die 串行这三个结构性瓶颈。