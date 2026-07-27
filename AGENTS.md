# **粤港澳赛区（ReduceScatter）- 决赛** 
## 一、赛题描述
实现集合通信算子ReduceScatter的操作接口，将通信域内所有rank的输入数据均分成rank size份，然后分别取每个rank的rank size之一份数据进行归约操作（如sum）。最后，将结果按照编号分散到各个rank的输出buffer。其计算原理如:[ReduceScatter](./reducescatter.md)

## 实验拓扑
评测环境为4*8卡昇腾Ascend 950的仿真环境，拓扑如下所示，横向为Mesh互联，纵向为Clos组网。

拓扑由4个Server组成，每个Server内包含8个NPU。
Server内8个NPU之间为Full-Mesh互联，Server间通过Clos网络互通。
每个NPU连接Clos网络的带宽大致是Server内单条直连链路带宽的4倍。
基于完整32卡集群构建3种子拓扑：2 * 8、4 * 1、8 + 4。
拓扑如文件所示：[topo](./topo.md)

## 项目介绍

- [README](./README.md)
- [CCU demo](/home/workspace/hccl/examples/05_custom_ops_allgather/ccu/README.md)
- [hccl](/home/workspace/hccl/README.md)
- [hcomm](/home/workspace/hcomm/README.md)

## 二、核心定义与约束
### 2.1 函数原型与参数说明
```cpp
HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType, HcclReduceOp op, HcclComm comm, aclrtStream stream)
```
| 参数名         | 输入/输出 | 描述                                                                             |
| ----------- | ------ | ------------------------------------------------------------------------------ |
| `sendBuf`   | 输入    | 源数据 Buffer 地址。                                                                 |
| `recvBuf`   | 输出    | 目的数据 Buffer 地址，集合通信结果输出至该 Buffer 中。                                            |
| `recvCount` | 输入    | 参与 ReduceScatter 操作的 `recvBuf` 数据大小。`sendBuf` 的数据大小等于 `recvCount × rank size`。 |
| `dataType`  | 输入    | ReduceScatter 操作的数据类型，类型为 `HcclDataType`。                                      |
| `op`        | 输入    | Reduce 操作类型，例如 `sum`、`prod`、`max`、`min`。                                       |
| `comm`      | 输入    | 集合通信操作所在的通信域。                                                                  |
| `stream`    | 输入    | 当前 Rank 使用的 Stream。                                                            |

接口调用成功时返回 `HCCL_SUCCESS`，其他返回值表示调用失败。


### 2.2 算子约束

所有rank的recvCount、dataType、op均相同。
针对一个对端仅能申请1个channel进行通信。基于赛题提供的拓扑，每两个npu之间仅有一条物理链路，因此与同一个对端只使用一条channel完成通信的收/发、同步等操作即可，使用多个channel不会带来额外的性能收益。

### 2.3 规则要求

- 基于给定拓扑，实现集合通信ReduceScatter算子功能，主要包括控制面的资源申请和数据面的算法逻辑实现。	
- **限定使用CCU通信引擎**
- 算子实现需满足确定性要求，即在相同输入下（特别是浮点数输入），多次通信计算得到的输出结果相同
- 选手仅允许修改 `custom.h`、`reduce_scatter.cc`、`exec_op.cc` 、`ccu_kernel.cc`共 4 个文件内容。

### 评分指标

赛题将从以下三个维度进行评分：

1. 功能分：通过功能测试用例验证，通过得分，不通过得0分。
2. 性能分：通过性能测试用例验证，不通过得0分，通过则按照带宽用量计分，带宽最高得满分，按照排名依次递减。
如不满足2.3中的规则要求，则不得分。
3. 通信域覆盖3种不同拓扑类型，数据类型覆盖float32，（输入）数据size覆盖512KB、512MB、400MB+4B，Reduce类型为sum。

### 用例说明

功能用例：

- 共9个功能用例（数据量分别为512KB、512MB、400M+4B），覆盖三种拓扑类型。
- 代码提交后，等待实时判分。

性能用例：

- 共9个性能用例（数据量分别为512KB、512MB、400M+4B），覆盖三种拓扑类型。
- 每天统一出1次性能分，取第二天凌晨3点前最后1次提交的代码进行性能评分。
- 需保证功能用例全部通过才能参与性能评分。

| 测试点     | 测试点1      | 测试点2      | 测试点3         | 测试点4      | 测试点5      | 测试点6         | 测试点7      | 测试点8      | 测试点9         | 测试点10     | 测试点11     | 测试点12        | 测试点13     | 测试点14     | 测试点15        | 测试点16     | 测试点17     | 测试点18        |
| ------- | --------- | --------- | ------------ | --------- | --------- | ------------ | --------- | --------- | ------------ | --------- | --------- | ------------ | --------- | --------- | ------------ | --------- | --------- | ------------ |
| 拓扑与数据大小 | 2×8，512KB | 2×8，512MB | 2×8，400MB+4B | 4×1，512KB | 4×1，512MB | 4×1，400MB+4B | 8+4，512KB | 8+4，512MB | 8+4，400MB+4B | 2×8，512KB | 2×8，512MB | 2×8，400MB+4B | 4×1，512KB | 4×1，512MB | 4×1，400MB+4B | 8+4，512KB | 8+4，512MB | 8+4，400MB+4B |
| 测试类型    | 功能测试      | 功能测试      | 功能测试         | 功能测试      | 功能测试      | 功能测试         | 功能测试      | 功能测试      | 功能测试         | 性能测试      | 性能测试      | 性能测试         | 性能测试      | 性能测试      | 性能测试         | 性能测试      | 性能测试      | 性能测试         |


### 拓展信息

信息同步：

1. 对于ReduceScatter算子，用例的数据量指的是输入内存的大小。
2. 特殊情况：针对ReduceScatter算子4B的功能用例，为了确保每张卡的输出内存至少有1个float32的数据，会把每张卡的输入内存的大小扩展到4B*rankSize（也就是64B）。
3. 对于512KB及以上的数据量，输入内存不会再乘以rankSize

### 注意事项

#### 赛题为2层拓扑，允许使用layer-0和layer-1的链路
- 通过HcclRankGraphGetLayers接口可查出当前通信域的netLayers，包含0和1两层。可根据算法需要,选择对应层级的链路。
- CCU模式，要同时使用layer-0和layer-1的链路时，需要将不同层的通信任务放到不同的CCU Kernel中。
  - 一个Ascend 950 NPU包含2个IO Die，不同IO Die包含不同的网络设备。
  - 在赛题给定的拓扑形态下，layer-0的网络设备和layer-1的网络设备分别分布在2个不同的IO Die上。
  - 每个IO Die都有一个CCU，一个CCU不能跨Die使用另一个IO Die的网络设备进行通信。
  - 在开发运行于某个CCU上的程序(ccu kernel)时，需要保证当一个ccu kernel使用的网络设备都在同一个IO Die上，CCU翻译器会自动根据它使用的网络设备推导出翻译后的指令序列应该部署到哪个CCU设备上。

#### 规约类算子(如Allreduce/ReduceScatter算子)如何保证确定性?
- 可通过cclBuffer中转，先将不同的数据存到cclBuffer不同地址段中，再按顺序对不同的数据做本地规约操作。
- 也可通过算法设计来保证确定性，比如:
  - Halving-Doubling\RING等算法因为每一步只接收一个对端发来的数据，天然保证确定性。
  - MESH算法，可将数据进一步切片，不同的对端，在同一步内针对不同的内存段做读/写/规约的动作。

#### 考虑小数据量场景任务数量对性能的影响
- 小数据量下，通信任务执行时间短，通信任务数量过多，会导致任务下发的时延较大，影响最终性能。

#### 比赛经验tip:
1. notify的多打一问题：hccl-vm@checker插件当前还无法校验拦截notify的多打一问题（对于同一个notify资源的多组record和wait任务，需要保证时序，避免多个record通知同一个wait，导致另一个wait任务等待超时），会导致性能出分异常。建议同学们自查下算法逻辑；补充说明：Record任务本质上是往对应的notify寄存器上写1，Wait任务是确认notify寄存器是否被写1，写1了就可以继续执行后续任务同时将对应notify寄存器重置为0，两次Record都提前向同一个寄存器写了1，只有第一个Wait能正常完成并将寄存器重置为0，第二个Wait会一直等待（因为没有新的record再将该寄存器写1了）

2. 一个对端申请多个channel
有个约束限制需要跟大家同步一下：针对一个对端仅能申请1个channel进行通信。请大家自查下算法逻辑，避免一个对端申请多个channel导致性能出分异常。基于赛题提供的拓扑，每两个npu之间仅有一条物理链路，因此与同一个对端只使用一条channel完成通信的收/发、同步等操作即可，使用多个channel不会带来额外的性能收益。

3. CCU模式，要同时使用layer-0和layer-1的链路时，需要将不同层的通信任务放到不同的CCU Kernel
一个Ascend 950 NPU包含2个IO Die，不同IO Die包含不同的网络设备。
在赛题给定的拓扑形态下，layer-0的网络设备和layer-1的网络设备分别分布在2个不同的IO Die上。
每个IO Die都有一个CCU，一个CCU不能跨Die使用另一个IO Die的网络设备进行通信。
在开发运行于某个CCU上的程序(ccu kernel)时，需要保证当一个ccu kernel使用的网络设备都在同一个IO Die上，CCU翻译器会自动根据它使用的网络设备推导出翻译后的指令序列应该部署到哪个CCU设备上。

4. AICPU模式下，部分数据面接口存在功能问题，如下接口暂无法使用：
  - HcommWriteWithNotifyOnThread
  - HcommWriteReduceWithNotifyOnThread


## 编译方法

```shell
bash build.sh
```
完整的Ascend 环境安装在路径: `/home/workspace`
包括：`Ascend`、`hccl`、`hcomm`

编译完成后需要运行 `cp_build.sh` 将编译产物拷贝到对应目录


## 仿真-checker
```shell
  source /home/workspace/Ascend/cann-9.1.0/set_env.sh
  source /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/script/hccl_config.sh   # 设 RANK_TABLE_FILE 等运行变量
 cd /home/workspace/hcomm/test/hccl_vm/hccl_vm_install/bin && ./hccl-vm start ascend950_cluster_32_server_normal.yaml
    (hvm)$> hccl-vm mock-comm 128
    (hvm)$> mpirun --allow-run-as-root --oversubscribe -np 16 /home/workspace/Ascend/cann-9.1.0/tools/hccl_test/bin/reduce_scatter_test -b 512K -e 512K -d fp32 -o sum -w 0 -n 1 -c 1 > /home/ubuntu/hccl_reduce_scatter/reduce_scatter_test.log
    (hvm)$> hccl-vm plugin run @checker
    (hvm)$> exit
  更多集群配置与用例见 /home/workspace/hcomm/test/hccl_vm/README-Competition.md §4
```

