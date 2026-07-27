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
#include <hccl/hccl_ccu_res.h>
#include <ccu/ccu_launch.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <vector>

#include "log.h"
#include "common.h"
#include "custom.h"
#include "hccl.h"
#include "exec_op.h"
#include "ccu_kernel.h"

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;

HcclResult BuildChannelDesc(HcclComm comm, uint32_t netLayer, uint32_t myRank, uint32_t remoteRank,
    HcclChannelDesc &desc)
{
    CommLink *links = nullptr;
    uint32_t linkNum = 0;
    CHK_RET(HcclRankGraphGetLinks(comm, netLayer, myRank, remoteRank, &links, &linkNum));

    const CommLink *selected = nullptr;
    for (uint32_t i = 0; i < linkNum; ++i) {
        if (links[i].linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_UBC_CTP) {
            selected = &links[i];
            break;
        }
    }
    CHK_PRT_RET(selected == nullptr,
        HCCL_ERROR("No CCU UBC_CTP link on layer %u from rank %u to rank %u", netLayer, myRank, remoteRank),
        HCCL_E_NOT_FOUND);

    CHK_RET(HcclChannelDescInit(&desc, 1));
    desc.remoteRank = remoteRank;
    desc.notifyNum = CHANNEL_NOTIFY_NUM;
    desc.channelProtocol = selected->linkAttr.linkProtocol;
    desc.localEndpoint = selected->srcEndpointDesc;
    desc.remoteEndpoint = selected->dstEndpointDesc;
    return HCCL_SUCCESS;
}

HcclResult AcquireChannelsAtLayer(HcclComm comm, const OpParam &param, uint32_t netLayer,
    const std::vector<uint32_t> &peerRanks, std::vector<ChannelHandle> &channels)
{
    std::vector<HcclChannelDesc> descs(peerRanks.size());
    channels.resize(peerRanks.size());
    for (uint32_t i = 0; i < peerRanks.size(); ++i) {
        CHK_RET(BuildChannelDesc(comm, netLayer, param.myRank, peerRanks[i], descs[i]));
    }
    if (!descs.empty()) {
        CHK_RET(HcclChannelAcquire(comm, CommEngine::COMM_ENGINE_CCU, descs.data(),
            static_cast<uint32_t>(descs.size()), channels.data()));
    }
    return HCCL_SUCCESS;
}

HcclResult AcquireMeshChannels(HcclComm comm, const OpParam &param, std::vector<ChannelHandle> &channels)
{
    uint32_t *netLayers = nullptr;
    uint32_t layerNum = 0;
    CHK_RET(HcclRankGraphGetLayers(comm, &netLayers, &layerNum));
    CHK_PRT_RET(layerNum == 0, HCCL_ERROR("Rank graph contains no network layer"), HCCL_E_INTERNAL);

    // Use the highest layer that covers the whole communicator. In all three
    // competition topologies this is Clos, keeping every channel on one IO Die.
    uint32_t netLayer = netLayers[0];
    for (uint32_t i = 1; i < layerNum; ++i) {
        netLayer = std::max(netLayer, netLayers[i]);
    }

    std::vector<uint32_t> peers;
    for (uint32_t remoteRank = 0; remoteRank < param.rankSize; ++remoteRank) {
        if (remoteRank != param.myRank) {
            peers.push_back(remoteRank);
        }
    }
    return AcquireChannelsAtLayer(comm, param, netLayer, peers, channels);
}

HcclResult RegisterKernel(HcclComm comm, const OpParam &param, const std::vector<ChannelHandle> &channels,
    AlgResourceCtx &resCtx)
{
    CcuKernelInfo kernelInfo{};
    const char kernelName[] = "CcuKernel";
    (void)std::memcpy(kernelInfo.kernelFuncName, kernelName, sizeof(kernelName));
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuKernel);

    auto kernelArg = std::make_shared<CcuReduceScatterKernelArg>();
    kernelArg->rankSize = param.rankSize;
    kernelArg->rankId = param.myRank;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t i = 0; i < channels.size(); ++i) {
        kernelArg->channels[i] = channels[i];
    }
    kernelInfo.setKernelArg(kernelArg);

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Expected one CCU instruction instance, got %u", insNum), HCCL_E_INTERNAL);

    resCtx.ccuKernels.resize(1);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    const void *kernelArgs[] = {kernelInfo.kernelArg};
    constexpr uint32_t DIE_ID_AUTO = 0;
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, DIE_ID_AUTO, kernelInfo.kernelFuncName, kernelInfo.kernelFunc,
        kernelArgs, 1, &resCtx.ccuKernels[0]));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
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
    sprintf(param.tag, "%s", "hccl_custom_reducescatter");
    param.inputPtr = sendBuf;
    param.outputPtr = recvBuf;
    param.count = recvCount;
    param.dataType = dataType;
    param.opType = HcclCMDType::HCCL_CMD_REDUCE_SCATTER;
    param.reduceType = op;

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
    CHK_PRT_RET(param.rankSize == 0 || param.rankSize > MAX_RANK_SIZE,
        HCCL_ERROR("Unsupported rank size %u", param.rankSize), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(dataType != HCCL_DATA_TYPE_FP32, HCCL_ERROR("Only float32 is supported"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(op != HCCL_REDUCE_SUM, HCCL_ERROR("Only sum reduction is supported"), HCCL_E_NOT_SUPPORT);
    // Every NPU in the updated topology has a direct layer-1 Clos endpoint.
    // Keep all peer channels and the kernel on that IO Die.
    (void)snprintf(param.tag, sizeof(param.tag), "hccl_custom_reducescatter_clos_v2");

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，并申请 1 个 notify
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, 1, &param.cpuThread));

    void *ctx = nullptr;
    uint64_t size = 0;
    if (HcclEngineCtxGet(comm, param.tag, ccuEngine, &ctx, &size) == HCCL_SUCCESS) {
        // CCU 资源已经存在，复用资源
        HCCL_INFO("Engine context already exists");
        param.resCtx = ctx;
        param.ctxSize = size;
    } else {
        // Device 资源不存在，资源构建
        AlgResourceCtx resCtxHost;

        // 从通信域获取 HCCL Buffer（Device上的内存，默认总大小400MB）
        void *cclBufferAddr;
        uint64_t cclBufferSize;
        CHK_RET(HcclGetHcclBuffer(comm, &cclBufferAddr, &cclBufferSize));
        resCtxHost.localBuffer = CommBuffer{cclBufferAddr, cclBufferSize};

        // ==============================================
        // STEP 2.2: 申请资源：Thread、Channel、CCU Kernel
        // ==============================================

        uint32_t threadNum = 1;
        resCtxHost.threads.resize(threadNum);
        resCtxHost.threads[0] = param.cpuThread;
        resCtxHost.ccuThread = param.cpuThread;

        if (param.rankSize > 1) {
            std::vector<ChannelHandle> channels;
            CHK_RET(AcquireMeshChannels(comm, param, channels));
            CHK_RET(RegisterKernel(comm, param, channels, resCtxHost));
        }

        // ==============================================
        // STEP 2.3: 申请通信引擎上下文
        // ==============================================
        // 申请 CCU 通信引擎上下文，存放 AlgResourceCtx 信息
        std::vector<char> seq = resCtxHost.Serialize();
        uint64_t seqSize = seq.size();
        param.ctxSize = seqSize;
        CHK_RET(HcclEngineCtxCreate(comm, param.tag, ccuEngine, param.ctxSize, &param.resCtx));
        CHK_RET(HcclEngineCtxCopy(comm, ccuEngine, param.tag, seq.data(), seqSize, 0));
    }

    // ==============================================
    // STEP 3: 下发 CCU Kernel
    // ==============================================
    CHK_RET(ops_hccl::ExecOp(param));
    return HCCL_SUCCESS;
}
