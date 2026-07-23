/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hccl/hccl_res_expt.h>
#include <hccl/hccl_rank_graph.h>
#include <hccl/hccl_diag.h>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "launch_aicpu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 2;
constexpr uint32_t EXPECTED_RANK_SIZE = 16;
constexpr uint32_t EXPECTED_LOCAL_RANK_SIZE = 8;
// One coordinator, one dedicated Clos worker and seven full-mesh workers.
constexpr uint32_t AICPU_THREAD_NUM = EXPECTED_LOCAL_RANK_SIZE + 1;

HcclResult FindLink(HcclComm comm, uint32_t localRank, uint32_t remoteRank, CommLink &selected)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    CHK_PRT_RET(layerNum == 0 || layers == nullptr, HCCL_ERROR("No network layer is available"), HCCL_E_INTERNAL);
    for (uint32_t layerIdx = 0; layerIdx < layerNum; ++layerIdx) {
        CommLink *links = nullptr;
        uint32_t linkNum = 0;
        HcclResult ret = HcclRankGraphGetLinks(comm, layers[layerIdx], localRank, remoteRank, &links, &linkNum);
        if (ret != HCCL_SUCCESS || links == nullptr || linkNum == 0) {
            continue;
        }
        selected = links[0];
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("No link from rank %u to rank %u", localRank, remoteRank);
    return HCCL_E_INTERNAL;
}

HcclResult CheckParam(const OpParam &param)
{
    CHK_PRT_RET(param.rankSize == 0, HCCL_ERROR("rankSize must be positive"), HCCL_E_PARA);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("Only FP32 is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(param.reduceType != HCCL_REDUCE_SUM, HCCL_ERROR("Only SUM is supported"), HCCL_E_NOT_SUPPORT);
    constexpr uint64_t typeSize = sizeof(float);
    CHK_PRT_RET(param.count > UINT64_MAX / typeSize / param.rankSize,
        HCCL_ERROR("ReduceScatter data size overflows uint64_t"), HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult GetLocalTopology(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    uint32_t *layers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &layers, &layerNum));
    for (uint32_t layerIdx = 0; layerIdx < layerNum; ++layerIdx) {
        uint32_t *ranks = nullptr;
        uint32_t rankNum = 0;
        CHK_RET(HcclRankGraphGetRanksByLayer(comm, layers[layerIdx], &ranks, &rankNum));
        if (rankNum != EXPECTED_LOCAL_RANK_SIZE || ranks == nullptr) {
            continue;
        }
        resCtx.localRanks.assign(ranks, ranks + rankNum);
        for (uint32_t idx = 0; idx < rankNum; ++idx) {
            if (ranks[idx] == param.myRank) {
                resCtx.localRankIndex = idx;
                break;
            }
        }
        break;
    }
    CHK_PRT_RET(param.rankSize != EXPECTED_RANK_SIZE || resCtx.localRanks.size() != EXPECTED_LOCAL_RANK_SIZE ||
            resCtx.localRankIndex == INVALID_VALUE_RANKID,
        HCCL_ERROR("Expected a 2x8 topology, rankSize=%u, localRankSize=%zu", param.rankSize,
            resCtx.localRanks.size()),
        HCCL_E_NOT_SUPPORT);
    resCtx.partnerRank = param.myRank < EXPECTED_LOCAL_RANK_SIZE ? param.myRank + EXPECTED_LOCAL_RANK_SIZE :
                                                                   param.myRank - EXPECTED_LOCAL_RANK_SIZE;
    return HCCL_SUCCESS;
}
} // namespace

HcclResult HcclReduceScatter(void *sendBuf, void *recvBuf, uint64_t recvCount, HcclDataType dataType,
    HcclReduceOp op, HcclComm comm, aclrtStream stream)
{
    CHK_PTR_NULL(sendBuf);
    CHK_PTR_NULL(recvBuf);
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(stream);

    // 构造算子参数
    OpParam param;
    sprintf(param.tag, "%s", "hccl_custom_reduce_scatter");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.reduceType = op;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;

    // 注册算子信息
    HcclDfxOpInfo dfxInfo;
    char commName[COMM_INDENTIFIER_MAX_LENGTH];
    CHK_RET(HcclGetCommName(comm, commName));
    CHK_RET(HcclDfxRegOpInfoByCommId(commName, reinterpret_cast<void *>(&dfxInfo)));

    // ==============================================
    // STEP 1: 解析拓扑信息
    // ==============================================
    CHK_RET(HcclGetRankId(comm, &param.myRank));
    CHK_RET(HcclGetRankSize(comm, &param.rankSize));
    CHK_RET(CheckParam(param));

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine aicpuTsEngine = CommEngine::COMM_ENGINE_AICPU_TS;
    CommEngine cpuTsEngine = CommEngine::COMM_ENGINE_CPU_TS;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 thread，并申请 Notify；同时导出为 AICPU 上可用的 thread
    CHK_RET(HcclThreadAcquireWithStream(comm, cpuTsEngine, stream, 1, &param.cpuThread));
    CHK_RET(HcclThreadExportToCommEngine(comm, 1, &param.cpuThread, aicpuTsEngine, &param.cpuThreadOnAicpu));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, aicpuTsEngine, &ctx, &size) == HCCL_SUCCESS) {
        // AICPU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;

        // Host 资源已经存在，复用资源
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = 0;
        CHK_RET(HcclEngineCtxGet(comm, param.tag, cpuTsEngine, &hostCtx, &hostCtxSize));
        ThreadHandle *aicpuThread = static_cast<ThreadHandle *>(hostCtx);
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};
        CHK_RET(GetLocalTopology(comm, param, resCtxHost));

        // ==============================================
        // STEP 2.2: 申请资源Thread和Channel
        // ==============================================

        // Thread 0 coordinates reductions, thread 1 drives the Clos link, and threads 2..8 each drive one
        // full-mesh peer. Notify 0 remains the host/device rendezvous (and worker start gate); notify 1..7 on
        // thread 0 are worker completion gates.
        uint32_t threadNum = AICPU_THREAD_NUM;
        uint32_t notifyNumPerThread = EXPECTED_LOCAL_RANK_SIZE;

        resCtxHost.threads.resize(threadNum);
        CHK_RET(HcclThreadAcquire(comm, aicpuTsEngine, threadNum, notifyNumPerThread, resCtxHost.threads.data()));
        // 将 threads[0] 导出为 CPU 上可用的 thread，用于 Host 与 Device 同步
        resCtxHost.aicpuThread = resCtxHost.threads[0];
        CHK_RET(HcclThreadExportToCommEngine(comm, 1, &resCtxHost.aicpuThread, cpuTsEngine, &param.aicpuThreadOnCpu));

        // 每个对端仅申请一个 channel；DATA 和 READY notify 分别负责数据到达与工作区复用同步。
        const uint32_t channelNum = param.rankSize - 1;
        if (channelNum > 0) {
            std::vector<HcclChannelDesc> channelDescs(channelNum);
            CHK_RET(HcclChannelDescInit(channelDescs.data(), channelNum));
            resCtxHost.channels.resize(channelNum);
            uint32_t channelIdx = 0;
            for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
                if (remoteRank == param.myRank) {
                    continue;
                }
                CommLink link;
                CHK_RET(FindLink(comm, param.myRank, remoteRank, link));
                HcclChannelDesc &desc = channelDescs[channelIdx];
                desc.remoteRank = remoteRank;
                desc.channelProtocol = link.linkAttr.linkProtocol;
                desc.localEndpoint = link.srcEndpointDesc;
                desc.remoteEndpoint = link.dstEndpointDesc;
                desc.notifyNum = CHANNEL_NOTIFY_NUM;
                resCtxHost.channels[channelIdx].remoteRank = remoteRank;
                resCtxHost.channels[channelIdx].notifyNum = CHANNEL_NOTIFY_NUM;
                ++channelIdx;
            }
            std::vector<ChannelHandle> handles(channelNum);
            CHK_RET(HcclChannelAcquire(comm, aicpuTsEngine, channelDescs.data(), channelNum, handles.data()));
            for (uint32_t idx = 0; idx < channelNum; ++idx) {
                void *remoteBuffer = nullptr;
                uint64_t remoteBufferSize = 0;
                CHK_RET(HcclChannelGetHcclBuffer(comm, handles[idx], &remoteBuffer, &remoteBufferSize));
                CHK_PRT_RET(remoteBuffer == nullptr || remoteBufferSize == 0,
                    HCCL_ERROR("Invalid remote HCCL buffer for rank %u", resCtxHost.channels[idx].remoteRank),
                    HCCL_E_INTERNAL);
                resCtxHost.channels[idx].handle = handles[idx];
                resCtxHost.channels[idx].remoteCclMem = CommBuffer{remoteBuffer, remoteBufferSize};
            }
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 AICPU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, aicpuTsEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, aicpuTsEngine, param.tag, seq.data(), seqSize, 0));
        // 申请 CPU 通信引擎上下文，存放 aicpuThread 句柄
        void *hostCtx = nullptr;
        uint64_t hostCtxSize = sizeof(ThreadHandle);
        const void *aicpuThreadPtr = static_cast<const void *>(&resCtxHost.aicpuThread);
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, cpuTsEngine, hostCtxSize, &hostCtx));
        CHK_RET(HcclEngineCtxCopy(comm, cpuTsEngine, param.tag, aicpuThreadPtr, hostCtxSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 AICPU Kernel
    // ==============================================
    CHK_RET(ops_hccl::LaunchAICPUKernel(param, stream));
    return HCCL_SUCCESS;
}
