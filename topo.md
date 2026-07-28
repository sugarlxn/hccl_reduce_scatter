## 评测环境与基础拓扑

评测环境为 **4×8 卡昇腾 Ascend 950 仿真集群**，共包含 4 个 Server 和 32 个 NPU。每个 Server 内部署 8 个 NPU，Server 内部的 NPU 之间采用 Full-Mesh 方式互联，不同 Server 之间通过 Clos 网络互通。

4 台 Server 的网络接入方式保持一致：

* **Server1**：NPU0～NPU7 均直接连接到 Clos 网络；
* **Server2**：NPU0～NPU7 均直接连接到 Clos 网络；
* **Server3**：NPU0～NPU7 均直接连接到 Clos 网络；
* **Server4**：NPU0～NPU7 均直接连接到 Clos 网络。

因此，集群中的任意 NPU 在进行跨 Server 通信时，均可以直接通过自身的 Clos 网络接口发送和接收数据，不需要先通过 Server 内 Full-Mesh 网络转发至其他 NPU。

对于直接连接 Clos 网络的 NPU，其 Clos 接入带宽约为 **Server 内单条 NPU 直连链路带宽的 4 倍**。

整体拓扑可抽象表示为：

```text
                        Clos Network
┌───────────────────────┼───────────────────────┐
│                       │                       │
│                       │                       │

Server1
[NPU0][NPU1][NPU2][NPU3][NPU4][NPU5][NPU6][NPU7]
   │     │     │     │     │     │     │     │
   └────────── 8 个 NPU 均直接接入 Clos ──────────┘

Server2
[NPU0][NPU1][NPU2][NPU3][NPU4][NPU5][NPU6][NPU7]
   │     │     │     │     │     │     │     │
   └────────── 8 个 NPU 均直接接入 Clos ──────────┘

Server3
[NPU0][NPU1][NPU2][NPU3][NPU4][NPU5][NPU6][NPU7]
   │     │     │     │     │     │     │     │
   └────────── 8 个 NPU 均直接接入 Clos ──────┘

Server4
[NPU0][NPU1][NPU2][NPU3][NPU4][NPU5][NPU6][NPU7]
   │     │     │     │     │     │     │     │
   └────────── 8 个 NPU 均直接接入 Clos ──────┘
```

基于完整的 32 卡集群，构建 **2×8、4×1 和 8+4** 三种子拓扑。

---

## 1. 2×8 拓扑

2×8 拓扑固定使用 **Server1 和 Server2**，两台 Server 均使用全部 8 个 NPU，因此总计包含 16 个 NPU：

$$
2 \times 8 = 16
$$

拓扑可表示为：

```text
Server1
[NPU0][NPU1][NPU2][NPU3][NPU4][NPU5][NPU6][NPU7]
   │     │     │     │     │     │     │     │
   └────────── 8 个 NPU 均直接接入 Clos ──────┘
                         ⇅
                    Clos Network
                         ⇅
Server2
[NPU0][NPU1][NPU2][NPU3][NPU4][NPU5][NPU6][NPU7]
   │     │     │     │     │     │     │     │
   └────────── 8 个 NPU 均直接接入 Clos ──────┘
```

该拓扑在设备规模和网络接入方式上具有对称性：

* Server1 和 Server2 均有 8 个 NPU 参与通信；
* 两台 Server 中的所有参与 NPU 均直接接入 Clos 网络；
* 每个 NPU 都可以独立完成跨 Server 数据传输；
* 跨 Server 通信不需要经过本机其他 NPU 中转。

因此，2×8 拓扑同时包含：

1. Server 内部的 Full-Mesh 通信；
2. Server 之间的 Clos 网络通信；
3. 多个 NPU 并行接入 Clos 网络时的链路竞争与带宽共享。

该拓扑适合评测层次化集合通信、双 Server 多卡并行通信、Server 内与 Server 间通信协同，以及多个 Clos 接入链路并行工作时的集合通信性能。

---

## 2. 4×1 拓扑

4×1 拓扑从 4 台 Server 中各选取 1 个 NPU，具体选择各 Server 的 **NPU0**，共包含 4 个 NPU：

$$
4 \times 1 = 4
$$

拓扑可表示为：

```text
Server1-NPU0 ─┐
Server2-NPU0 ─┼── Clos Network
Server3-NPU0 ─┤
Server4-NPU0 ─┘
```

由于 Server1、Server2、Server3 和 Server4 的 NPU0 均直接连接到 Clos 网络，因此，4×1 拓扑中的所有参与 NPU 都能够直接进行跨 Server 通信，不需要经过 Server 内其他 NPU 转发。

该拓扑的主要特点是：

* 每台 Server 仅有一个 NPU 参与通信；
* 所有参与 NPU 均直接接入 Clos 网络；
* 通信过程主要依赖跨 Server Clos 网络；
* Server 内 Full-Mesh 网络基本不参与数据交换；
* 集合通信需要覆盖 4 台不同的 Server。

因此，4×1 拓扑适合用于评测纯跨 Server 通信、Clos 网络基础延迟、多 Server 集合通信性能，以及通信规模随 Server 数量增加时的扩展性。

---

## 3. 8+4 拓扑

8+4 拓扑固定使用：

* **Server1 的全部 8 个 NPU**；
* **Server2 的 NPU0、NPU1、NPU2 和 NPU3**。

因此，总共包含 12 个 NPU：

$$
8 + 4 = 12
$$

拓扑可表示为：

```text
Server1
[NPU0][NPU1][NPU2][NPU3][NPU4][NPU5][NPU6][NPU7]
   │     │     │     │     │     │     │     │
   └────────── 8 个 NPU 均直接接入 Clos ──────┘
                         ⇅
                    Clos Network
                         ⇅
Server2
[NPU0][NPU1][NPU2][NPU3]
   │     │     │     │
   └4 个 NPU 均直接接入 Clos ┘
```

在 Server2 选取的 4 个 NPU 中：

* NPU0 直接连接 Clos 网络；
* NPU1 直接连接 Clos 网络；
* NPU2 直接连接 Clos 网络；
* NPU3 直接连接 Clos 网络。

因此，Server2 中参与该子拓扑的 4 个 NPU 均可以独立完成跨 Server 数据传输，不需要通过其他 NPU进行数据汇聚或转发。

该拓扑在参与设备数量上具有明显的非对称性，但在网络接入能力上不存在单出口限制：

* Server1 一侧包含 8 个 NPU；
* Server2 一侧包含 4 个 NPU；
* 两侧所有参与 NPU 均直接接入 Clos 网络；
* Server1 可使用 8 条 NPU Clos 接入链路；
* Server2 可使用 4 条 NPU Clos 接入链路；
* 两侧的总体网络注入能力可能因参与 NPU 数量不同而存在差异。

因此，8+4 拓扑适合用于评测非对称资源规模、不同 NPU 数量下的通信负载分配、两侧网络注入能力差异，以及不均衡参与规模对集合通信算法性能的影响。

---

## 三种子拓扑对比

| 拓扑类型 | 使用的 Server      | 参与 NPU                                    | NPU 总数 | 直接接入 Clos 的参与 NPU     |
| ---- | --------------- | ----------------------------------------- | -----: | --------------------- |
| 2×8  | Server1、Server2 | 两台 Server 各使用 NPU0～NPU7                   |     16 | 16 个参与 NPU 均直接接入 Clos |
| 4×1  | Server1～Server4 | 每台 Server 使用 NPU0                         |      4 | 4 个参与 NPU 均直接接入 Clos  |
| 8+4  | Server1、Server2 | Server1 使用 NPU0～NPU7；Server2 使用 NPU0～NPU3 |     12 | 12 个参与 NPU 均直接接入 Clos |

## 总结

> 评测环境为 4×8 卡昇腾 Ascend 950 仿真集群，共包含 4 个 Server 和 32 个 NPU。每个 Server 内的 8 个 NPU 采用 Full-Mesh 方式互联，不同 Server 之间通过 Clos 网络互通。Server1、Server2、Server3 和 Server4 中的 NPU0～NPU7 均直接连接到 Clos 网络。因此，任意 NPU 在进行跨 Server 通信时，都可以直接通过自身的 Clos 网络接口完成数据传输，不需要经过 Server 内其他 NPU 转发。每个 NPU 的 Clos 接入带宽约为 Server 内单条 NPU 直连链路带宽的 4 倍。基于完整的 32 卡集群，构建了 2×8、4×1 和 8+4 三种子拓扑。其中，2×8 拓扑使用 Server1 和 Server2 的全部 NPU；4×1 拓扑使用 4 台 Server 各自的 NPU0；8+4 拓扑使用 Server1 的全部 8 个 NPU以及 Server2 的 NPU0、NPU1、NPU2 和 NPU3。三种子拓扑中的所有参与 NPU 均直接接入 Clos 网络，其差异主要体现在参与 Server 数量、每台 Server 的参与 NPU 数量以及两侧网络注入能力上。
