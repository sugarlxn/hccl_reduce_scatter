## 评测环境与基础拓扑

评测环境为 **4×8 卡昇腾 Ascend 950 仿真集群**，共包含 4 个 Server 和 32 个 NPU。每个 Server 内部署 8 个 NPU，Server 内部的 NPU 之间采用 Full-Mesh 方式互联，不同 Server 之间通过 Clos 网络互通。

需要注意的是，各 Server 接入 Clos 网络的方式并不完全相同：

* **Server1**：8 个 NPU 均直接连接到 Clos 网络；
* **Server2、Server3、Server4**：仅 **NPU0、NPU4 和 NPU7** 直接连接到 Clos 网络；
* Server2、Server3 和 Server4 中其余未直连 Clos 的 NPU，在进行跨 Server 通信时，需要先通过 Server 内 Full-Mesh 网络将数据传输至本机的 NPU0、NPU4 或 NPU7，再经 Clos 网络完成跨 Server 数据传输。

对于直接连接 Clos 网络的 NPU，其 Clos 接入带宽约为 **Server 内单条 NPU 直连链路带宽的 8 倍**。

整体拓扑可抽象表示为：

```text
                         Clos Network
        ┌────────────────────┼────────────────────┐
        │                    │                    │
Server1 │  NPU0 NPU1 NPU2 NPU3 NPU4 NPU5 NPU6 NPU7
        │    │    │    │    │    │    │    │    │
        │    └────┴────┴────┴────┴────┴────┴────┘
        │          8 个 NPU 均直连 Clos
        │
Server2 │  NPU0 NPU1 NPU2 NPU3 NPU4 NPU5 NPU6 NPU7
        │    │                   │              │
        │    └────── Clos ───────┴──── Clos ────┘
        │        仅 NPU0、NPU4、NPU7 直连 Clos
        │
Server3 │  NPU0 NPU1 NPU2 NPU3 NPU4 NPU5 NPU6 NPU7
        │    │                   │              │
        │    └────── Clos ───────┴──── Clos ────┘
        │        仅 NPU0、NPU4、NPU7 直连 Clos
        │
Server4 │  NPU0 NPU1 NPU2 NPU3 NPU4 NPU5 NPU6 NPU7
        │    │                   │              │
        │    └────── Clos ───────┴──── Clos ────┘
             仅 NPU0、NPU4、NPU7 直连 Clos
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
   └────────── 8 个 NPU 均直连 Clos ──────────┘
                         ⇅
                    Clos Network
                         ⇅
Server2
[NPU0][NPU1][NPU2][NPU3][NPU4][NPU5][NPU6][NPU7]
   │                       │                 │
   └──── NPU0 ─────────── NPU4 ────────── NPU7
                 直连 Clos
```

该拓扑虽然在逻辑上是对称的两台 8 卡 Server，但其跨 Server 网络接入能力并不对称：

* Server1 的 8 个 NPU 均能够直接接入 Clos；
* Server2 只有 NPU0、NPU4 和 NPU7 能直接接入 Clos；
* Server2 中的 NPU1、NPU2、NPU3、NPU5 和 NPU6 进行跨 Server 通信时，需要先通过 Server 内 Full-Mesh 网络，将数据转发至 NPU0、NPU4 或 NPU7。

因此，2×8 拓扑同时包含：

1. Server 内 Full-Mesh 通信；
2. Server 间 Clos 通信；
3. Server2 内部的跨机出口汇聚和转发。

该拓扑适合评测层次化集合通信、非对称跨机出口以及 Server 内转发对通信性能的影响。

---

## 2. 4×1 拓扑

4×1 拓扑从 4 台 Server 中各选取 1 个 NPU，具体选择各 Server 的 **NPU0**，共包含 4 个 NPU：

```text
Server1-NPU0 ─┐
Server2-NPU0 ─┼── Clos Network
Server3-NPU0 ─┤
Server4-NPU0 ─┘
```

由于 Server1、Server2、Server3 和 Server4 的 NPU0 均直接连接到 Clos 网络，因此，4×1 拓扑中的所有参与 NPU 都能够直接进行跨 Server 通信，不需要经过 Server 内其他 NPU 转发。

该拓扑的主要特点是：

* 每台 Server 仅有一个 NPU 参与通信；
* 所有参与 NPU 均直连 Clos；
* 通信过程主要依赖跨 Server Clos 网络；
* Server 内 Full-Mesh 网络基本不参与数据交换。

因此，4×1 拓扑适合用于评测纯跨 Server 通信、Clos 网络延迟以及多 Server 集合通信性能。

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
   └────────── 8 个 NPU 均直连 Clos ──────────┘
                         ⇅
                    Clos Network
                         ⇅
Server2
[NPU0][NPU1][NPU2][NPU3]
   │
   │
仅 NPU0 直连 Clos
```

在 Server2 选取的 4 个 NPU 中：

* NPU0 直接连接 Clos 网络；
* NPU1、NPU2 和 NPU3 不直接连接 Clos 网络；
* NPU1、NPU2 和 NPU3 的跨 Server 数据需要先通过 Server2 内部的 Full-Mesh 网络传输至 NPU0，再由 NPU0 接入 Clos 网络。

需要说明的是，虽然 Server2 的 NPU4 和 NPU7 也具备 Clos 直连能力，但它们不属于该 8+4 子拓扑中的参与 NPU。因此，在严格限定子拓扑参与设备的情况下，Server2 一侧的跨 Server 通信主要由 NPU0 承担。

该拓扑具有明显的非对称性：

* Server1 一侧包含 8 个 NPU，且全部直连 Clos；
* Server2 一侧仅包含 4 个 NPU，其中只有 NPU0 直连 Clos；
* Server2 的 NPU0 同时承担本地数据汇聚和跨 Server 数据传输任务；
* Server2 一侧可能形成较明显的出口竞争和通信瓶颈。

因此，8+4 拓扑适合用于评测非对称资源规模、单出口汇聚、Server 内转发以及 Clos 出口竞争对集合通信性能的影响。

---

## 三种子拓扑对比

| 拓扑类型 | 使用的 Server      | 参与 NPU                             | NPU 总数 | 直接接入 Clos 的参与 NPU                  |
| ---- | --------------- | ---------------------------------- | -----: | ---------------------------------- |
| 2×8  | Server1、Server2 | 两台 Server 各 8 个 NPU                |     16 | Server1：8 个；Server2：NPU0、NPU4、NPU7 |
| 4×1  | Server1～Server4 | 每台 Server 的 NPU0                   |      4 | 4 个参与 NPU 均直连 Clos                 |
| 8+4  | Server1、Server2 | Server1 全部 8 卡；Server2 的 NPU0～NPU3 |     12 | Server1：8 个；Server2：仅 NPU0         |

## 总结

> 评测环境为4×8卡昇腾Ascend 950仿真集群，共包含4个Server和32个NPU。每个Server内的8个NPU采用Full-Mesh方式互联，不同Server之间通过Clos网络互通。Server1中的8个NPU均直接连接到Clos网络，而Server2、Server3和Server4中仅NPU0、NPU4和NPU7直接连接到Clos网络。未直连Clos的NPU在进行跨Server通信时，需要先通过Server内Full-Mesh网络将数据传输至本机的Clos出口NPU，再完成跨Server传输。直连Clos的NPU，其Clos接入带宽约为Server内单条NPU直连链路带宽的8倍。基于完整32卡集群，构建了2×8、4×1和8+4三种子拓扑。其中，2×8拓扑使用Server1和Server2的全部NPU；4×1拓扑使用4台Server各自的NPU0；8+4拓扑使用Server1的全部8个NPU以及Server2的NPU0、NPU1、NPU2和NPU3。
