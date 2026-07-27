ReduceScatter（规约-分散）算子可以理解为 **先对多个 Rank 上的数据进行逐元素规约，再把规约结果按块分发给各个 Rank**。

图中共有 4 个 Rank，分别为 rank 0、rank 1、rank 2 和 rank 3。每个 Rank 都持有一份输入数据：

```text
rank 0：in0
rank 1：in1
rank 2：in2
rank 3：in3
```

每份输入通常会被划分为 4 个数据块。ReduceScatter 首先对所有 Rank 中位置相同的数据块执行规约操作，例如求和：

$$
R_j = in0_j \oplus in1_j \oplus in2_j \oplus in3_j
$$

其中：

* ($j$) 表示第 ($j$) 个数据块；
* ($\oplus$) 表示规约操作，可以是求和、最大值、最小值或乘积等；
* ($R_j$) 表示第 ($j$) 个数据块的规约结果。

随后，将不同的数据块分别发送给对应的 Rank：

```text
rank 0 获得 out0 = Reduce(in0[0], in1[0], in2[0], in3[0])
rank 1 获得 out1 = Reduce(in0[1], in1[1], in2[1], in3[1])
rank 2 获得 out2 = Reduce(in0[2], in1[2], in2[2], in3[2])
rank 3 获得 out3 = Reduce(in0[3], in1[3], in2[3], in3[3])
```

如果规约操作为求和，则可以表示为：

$$
out_0 = in0_0 + in1_0 + in2_0 + in3_0
$$

$$
out_1 = in0_1 + in1_1 + in2_1 + in3_1
$$

$$
out_2 = in0_2 + in1_2 + in2_2 + in3_2
$$

$$
out_3 = in0_3 + in1_3 + in2_3 + in3_3
$$

最终，每个 Rank 只保留完整规约结果中的一部分：

```text
输入阶段：
rank 0：完整输入 in0
rank 1：完整输入 in1
rank 2：完整输入 in2
rank 3：完整输入 in3

ReduceScatter 后：
rank 0：out0
rank 1：out1
rank 2：out2
rank 3：out3
```

因此，ReduceScatter 的输出不是每个 Rank 都得到完整的规约结果，而是每个 Rank 得到其中一个不同的数据分片。它可以看作：

```text
ReduceScatter = Reduce + Scatter
```

但实际实现通常不会先生成完整的 Reduce 结果再执行 Scatter，而是将规约与数据分发过程融合，以减少中间数据存储和通信开销。

总结：

> ReduceScatter 算子首先将每个 Rank 的输入数据划分为与 Rank 数量相等的数据块，然后对所有 Rank 中相同位置的数据块执行逐元素规约操作。规约完成后，第 (i) 个数据块的结果保留在 rank (i) 上。因此，在包含 (P) 个 Rank 的通信组中，每个 Rank 输入一份完整数据，最终仅输出完整规约结果的 (1/P)。图中共有4个Rank，经过ReduceScatter操作后，rank 0、rank 1、rank 2和rank 3分别获得out0、out1、out2和out3。
