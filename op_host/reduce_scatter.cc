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

namespace ops_hccl {
CcuResult CcuStagingKernel(CcuKernelArg arg);
CcuResult CcuLocalReduceKernel(CcuKernelArg arg);
CcuResult CcuCrossReduceKernel(CcuKernelArg arg);
CcuResult CcuPartialReduceKernel(CcuKernelArg arg);
CcuResult CcuGroupReducePartialKernel(CcuKernelArg arg);
CcuResult CcuMergePartialKernel(CcuKernelArg arg);
} // namespace ops_hccl

namespace {
constexpr uint32_t CHANNEL_NOTIFY_NUM = 1;
constexpr uint64_t SMALL_INPUT_BYTES = 512 * 1024;
constexpr uint64_t HIERARCHICAL_MIN_INPUT_BYTES = 1024 * 1024;

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

HcclResult RegisterMeshKernel(HcclComm comm, const OpParam &param, const std::vector<ChannelHandle> &channels,
    ReduceScatterAlgorithm algorithm, AlgResourceCtx &resCtx)
{
    CcuKernelInfo kernelInfo{};
    const bool direct = algorithm == ReduceScatterAlgorithm::DIRECT_MESH;
    const char *kernelName = direct ? "CcuKernel" : "CcuStagingKernel";
    (void)snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", kernelName);
    kernelInfo.kernelFunc =
        direct ? reinterpret_cast<void *>(ops_hccl::CcuKernel) : reinterpret_cast<void *>(ops_hccl::CcuStagingKernel);

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
    resCtx.algorithm = algorithm;
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult BuildHierarchyPlan(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    uint32_t *localRanks = nullptr;
    uint32_t localRankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &localRanks, &localRankNum));
    CHK_PRT_RET(localRankNum == 0 || localRankNum > MAX_LOCAL_RANK_SIZE,
        HCCL_ERROR("Invalid layer-0 group size %u", localRankNum), HCCL_E_NOT_SUPPORT);
    resCtx.localRanks.assign(localRanks, localRanks + localRankNum);
    std::sort(resCtx.localRanks.begin(), resCtx.localRanks.end());

    const auto localIt = std::find(resCtx.localRanks.begin(), resCtx.localRanks.end(), param.myRank);
    CHK_PRT_RET(localIt == resCtx.localRanks.end(), HCCL_ERROR("Local rank is absent from layer-0 group"),
        HCCL_E_INTERNAL);
    const uint32_t localIndex = static_cast<uint32_t>(localIt - resCtx.localRanks.begin());

    std::vector<uint32_t> remoteRanks;
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!std::binary_search(resCtx.localRanks.begin(), resCtx.localRanks.end(), rank)) {
            remoteRanks.push_back(rank);
        }
    }

    resCtx.targetRanks = {param.myRank};
    if (param.rankSize == 16) {
        CHK_PRT_RET(localRankNum != 8 || remoteRanks.size() != 8,
            HCCL_ERROR("Unexpected 2x8 hierarchy"), HCCL_E_INTERNAL);
        resCtx.targetRanks.push_back(remoteRanks[localIndex]);
        resCtx.crossPeers.push_back(remoteRanks[localIndex]);
        resCtx.crossSendTargetIndices.push_back(1);
    } else if (param.rankSize == 12 && localRankNum == 8) {
        CHK_PRT_RET(remoteRanks.size() != 4, HCCL_ERROR("Unexpected 8+4 hierarchy"), HCCL_E_INTERNAL);
        const uint32_t remoteIndex = localIndex % 4;
        resCtx.crossPeers.push_back(remoteRanks[remoteIndex]);
        if (localIndex < 4) {
            resCtx.targetRanks.push_back(remoteRanks[remoteIndex]);
            resCtx.crossSendTargetIndices.push_back(1);
        }
    } else if (param.rankSize == 12 && localRankNum == 4) {
        CHK_PRT_RET(remoteRanks.size() != 8, HCCL_ERROR("Unexpected 8+4 hierarchy"), HCCL_E_INTERNAL);
        resCtx.targetRanks.push_back(remoteRanks[localIndex]);
        resCtx.targetRanks.push_back(remoteRanks[localIndex + 4]);
        resCtx.crossPeers.push_back(remoteRanks[localIndex]);
        resCtx.crossPeers.push_back(remoteRanks[localIndex + 4]);
        resCtx.crossSendTargetIndices = {1, 2};
    } else {
        HCCL_ERROR("Unsupported hierarchical topology: rankSize=%u, localSize=%u", param.rankSize, localRankNum);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult BuildOwnTargetPlan(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    uint32_t *localRanks = nullptr;
    uint32_t localRankNum = 0;
    CHK_RET(HcclRankGraphGetRanksByLayer(comm, 0, &localRanks, &localRankNum));
    CHK_PRT_RET(localRankNum == 0 || localRankNum > MAX_LOCAL_RANK_SIZE,
        HCCL_ERROR("Invalid layer-0 group size %u", localRankNum), HCCL_E_NOT_SUPPORT);

    resCtx.localRanks.assign(localRanks, localRanks + localRankNum);
    std::sort(resCtx.localRanks.begin(), resCtx.localRanks.end());
    CHK_PRT_RET(!std::binary_search(resCtx.localRanks.begin(), resCtx.localRanks.end(), param.myRank),
        HCCL_ERROR("Local rank is absent from layer-0 group"), HCCL_E_INTERNAL);

    resCtx.crossPeers.clear();
    for (uint32_t rank = 0; rank < param.rankSize; ++rank) {
        if (!std::binary_search(resCtx.localRanks.begin(), resCtx.localRanks.end(), rank)) {
            resCtx.crossPeers.push_back(rank);
        }
    }
    CHK_PRT_RET(resCtx.crossPeers.empty(), HCCL_ERROR("Dual-die plan has no remote sources"),
        HCCL_E_NOT_SUPPORT);
    resCtx.targetRanks = {param.myRank};
    return HCCL_SUCCESS;
}

std::shared_ptr<CcuPartialReduceKernelArg> MakePartialKernelArg(const OpParam &param,
    const std::vector<ChannelHandle> &channels, uint32_t sourceCount, bool includeLocalSource,
    uint32_t localSourceIndex)
{
    auto kernelArg = std::make_shared<CcuPartialReduceKernelArg>();
    kernelArg->sourceCount = sourceCount;
    kernelArg->localSourceIndex = localSourceIndex;
    kernelArg->includeLocalSource = includeLocalSource;
    kernelArg->splitMerge = false;
    kernelArg->mergeFirstHalf = false;
    kernelArg->dataType = param.dataType;
    kernelArg->reduceOp = param.reduceType;
    kernelArg->channelCount = static_cast<uint32_t>(channels.size());
    for (uint32_t i = 0; i < channels.size(); ++i) {
        kernelArg->channels[i] = channels[i];
    }
    return kernelArg;
}

HcclResult RegisterDualDiePartialKernels(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    std::vector<uint32_t> localPeers;
    for (uint32_t rank : resCtx.localRanks) {
        if (rank != param.myRank) {
            localPeers.push_back(rank);
        }
    }
    const uint32_t localIndex = static_cast<uint32_t>(
        std::find(resCtx.localRanks.begin(), resCtx.localRanks.end(), param.myRank) - resCtx.localRanks.begin());

    std::vector<ChannelHandle> localChannels;
    std::vector<ChannelHandle> crossChannels;
    CHK_RET(AcquireChannelsAtLayer(comm, param, 0, localPeers, localChannels));
    CHK_RET(AcquireChannelsAtLayer(comm, param, 1, resCtx.crossPeers, crossChannels));

    CcuKernelInfo localInfo{};
    CcuKernelInfo crossInfo{};
    CcuKernelInfo mergeInfo{};
    const bool useGroupReduce = param.rankSize == 16;
    const char *partialKernelName =
        useGroupReduce ? "CcuGroupReducePartialKernel" : "CcuPartialReduceKernel";
    (void)snprintf(localInfo.kernelFuncName, sizeof(localInfo.kernelFuncName), "%s", partialKernelName);
    (void)snprintf(crossInfo.kernelFuncName, sizeof(crossInfo.kernelFuncName), "%s", partialKernelName);
    (void)snprintf(mergeInfo.kernelFuncName, sizeof(mergeInfo.kernelFuncName), "%s", "CcuMergePartialKernel");
    localInfo.kernelFunc = useGroupReduce ? reinterpret_cast<void *>(ops_hccl::CcuGroupReducePartialKernel) :
                                           reinterpret_cast<void *>(ops_hccl::CcuPartialReduceKernel);
    crossInfo.kernelFunc = useGroupReduce ? reinterpret_cast<void *>(ops_hccl::CcuGroupReducePartialKernel) :
                                           reinterpret_cast<void *>(ops_hccl::CcuPartialReduceKernel);
    mergeInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuMergePartialKernel);
    auto localKernelArg = MakePartialKernelArg(param, localChannels,
        static_cast<uint32_t>(resCtx.localRanks.size()), true, localIndex);
    auto crossKernelArg = MakePartialKernelArg(param, crossChannels,
        static_cast<uint32_t>(resCtx.crossPeers.size()), false, 0);
    localKernelArg->splitMerge = useGroupReduce;
    localKernelArg->mergeFirstHalf = true;
    crossKernelArg->splitMerge = useGroupReduce;
    crossKernelArg->mergeFirstHalf = false;
    localInfo.setKernelArg(localKernelArg);
    crossInfo.setKernelArg(crossKernelArg);
    auto mergeKernelArg = std::make_shared<CcuMergePartialKernelArg>();
    mergeKernelArg->dataType = param.dataType;
    mergeKernelArg->reduceOp = param.reduceType;
    mergeKernelArg->channelCount = 0;
    mergeInfo.setKernelArg(mergeKernelArg);

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Expected one CCU instruction instance, got %u", insNum), HCCL_E_INTERNAL);

    resCtx.ccuKernels.resize(useGroupReduce ? 2 : 3);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    const void *localArgs[] = {localInfo.kernelArg};
    const void *crossArgs[] = {crossInfo.kernelArg};
    const void *mergeArgs[] = {mergeInfo.kernelArg};
    constexpr uint32_t DIE_ID_AUTO = 0;
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, DIE_ID_AUTO, localInfo.kernelFuncName, localInfo.kernelFunc,
        localArgs, 1, &resCtx.ccuKernels[0]));
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, DIE_ID_AUTO, crossInfo.kernelFuncName, crossInfo.kernelFunc,
        crossArgs, 1, &resCtx.ccuKernels[1]));
    if (!useGroupReduce) {
        CHK_RET_CCU(HcommCcuKernelRegister(insHandle, DIE_ID_AUTO, mergeInfo.kernelFuncName,
            mergeInfo.kernelFunc, mergeArgs, 1, &resCtx.ccuKernels[2]));
    }
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    resCtx.algorithm = ReduceScatterAlgorithm::DUAL_DIE_PARTIAL;
    return HCCL_SUCCESS;
}

HcclResult RegisterSmallClosParallelKernel(HcclComm comm, const OpParam &param, AlgResourceCtx &resCtx)
{
    std::vector<ChannelHandle> channels;
    CHK_RET(AcquireMeshChannels(comm, param, channels));

    CcuKernelInfo kernelInfo{};
    (void)snprintf(kernelInfo.kernelFuncName, sizeof(kernelInfo.kernelFuncName), "%s", "CcuPartialReduceKernel");
    kernelInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuPartialReduceKernel);
    kernelInfo.setKernelArg(MakePartialKernelArg(
        param, channels, param.rankSize, true, param.myRank));

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
    resCtx.algorithm = ReduceScatterAlgorithm::SMALL_CLOS_PARALLEL;
    return HCCL_SUCCESS;
}

[[maybe_unused]] HcclResult RegisterHierarchicalKernels(HcclComm comm, const OpParam &param, bool useStaging,
    AlgResourceCtx &resCtx)
{
    std::vector<uint32_t> localPeers;
    for (uint32_t rank : resCtx.localRanks) {
        if (rank != param.myRank) {
            localPeers.push_back(rank);
        }
    }
    const uint32_t localIndex = static_cast<uint32_t>(
        std::find(resCtx.localRanks.begin(), resCtx.localRanks.end(), param.myRank) - resCtx.localRanks.begin());

    std::vector<ChannelHandle> localChannels;
    std::vector<ChannelHandle> crossChannels;
    CHK_RET(AcquireChannelsAtLayer(comm, param, 0, localPeers, localChannels));
    CHK_RET(AcquireChannelsAtLayer(comm, param, 1, resCtx.crossPeers, crossChannels));

    CcuKernelInfo localInfo{};
    (void)snprintf(localInfo.kernelFuncName, sizeof(localInfo.kernelFuncName), "%s", "CcuLocalReduceKernel");
    localInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuLocalReduceKernel);
    auto localArg = std::make_shared<CcuLocalReduceKernelArg>();
    localArg->groupSize = static_cast<uint32_t>(resCtx.localRanks.size());
    localArg->groupRankId = localIndex;
    localArg->targetCount = static_cast<uint32_t>(resCtx.targetRanks.size());
    localArg->useStaging = useStaging;
    localArg->dataType = param.dataType;
    localArg->reduceOp = param.reduceType;
    localArg->channelCount = static_cast<uint32_t>(localChannels.size());
    for (uint32_t i = 0; i < localChannels.size(); ++i) {
        localArg->channels[i] = localChannels[i];
    }
    localInfo.setKernelArg(localArg);

    CcuKernelInfo crossInfo{};
    (void)snprintf(crossInfo.kernelFuncName, sizeof(crossInfo.kernelFuncName), "%s", "CcuCrossReduceKernel");
    crossInfo.kernelFunc = reinterpret_cast<void *>(ops_hccl::CcuCrossReduceKernel);
    auto crossArg = std::make_shared<CcuCrossReduceKernelArg>();
    crossArg->dataType = param.dataType;
    crossArg->reduceOp = param.reduceType;
    crossArg->channelCount = static_cast<uint32_t>(crossChannels.size());
    crossArg->sendCount = static_cast<uint32_t>(resCtx.crossSendTargetIndices.size());
    for (uint32_t i = 0; i < crossChannels.size(); ++i) {
        crossArg->channels[i] = crossChannels[i];
    }
    for (uint32_t sendIdx = 0; sendIdx < crossArg->sendCount; ++sendIdx) {
        const uint32_t targetRank = resCtx.targetRanks[resCtx.crossSendTargetIndices[sendIdx]];
        const auto peerIt = std::find(resCtx.crossPeers.begin(), resCtx.crossPeers.end(), targetRank);
        CHK_PRT_RET(peerIt == resCtx.crossPeers.end(), HCCL_ERROR("Cross target has no channel"), HCCL_E_INTERNAL);
        crossArg->sendChannelIndices[sendIdx] = static_cast<uint32_t>(peerIt - resCtx.crossPeers.begin());
    }
    crossInfo.setKernelArg(crossArg);

    CcuInsHandle insHandle = 0;
    uint32_t insNum = 0;
    CHK_RET(HcclCommQueryCcuIns(comm, &insHandle, &insNum));
    CHK_PRT_RET(insNum != 1, HCCL_ERROR("Expected one CCU instruction instance, got %u", insNum), HCCL_E_INTERNAL);
    resCtx.ccuKernels.resize(2);
    CHK_RET_CCU(HcommCcuKernelRegisterStart(insHandle));
    const void *localArgs[] = {localInfo.kernelArg};
    const void *crossArgs[] = {crossInfo.kernelArg};
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, 0, localInfo.kernelFuncName, localInfo.kernelFunc, localArgs, 1,
        &resCtx.ccuKernels[0]));
    CHK_RET_CCU(HcommCcuKernelRegister(insHandle, 0, crossInfo.kernelFuncName, crossInfo.kernelFunc, crossArgs, 1,
        &resCtx.ccuKernels[1]));
    CHK_RET_CCU(HcommCcuKernelRegisterEnd(insHandle));
    resCtx.algorithm =
        useStaging ? ReduceScatterAlgorithm::HIERARCHICAL_STAGING : ReduceScatterAlgorithm::HIERARCHICAL;
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
    const uint64_t inputBytes = recvCount * sizeof(float) * param.rankSize;
    const bool useDualDie = (param.rankSize == 16 || param.rankSize == 12) && inputBytes > SMALL_INPUT_BYTES;
    const bool useSmallClosParallel =
        (param.rankSize == 16 || param.rankSize == 12) && inputBytes <= SMALL_INPUT_BYTES;
    const bool useDirectMesh = param.rankSize == 4 && inputBytes <= SMALL_INPUT_BYTES;
    const char *algorithmTag = useDualDie ?
        (param.rankSize == 16 ? "dual_die_group_oneshot_v3" : "dual_die_own_v1") :
        (useSmallClosParallel ? "small_clos_parallel_v1" : (useDirectMesh ? "direct_v3" : "stage_v3"));
    (void)snprintf(param.tag, sizeof(param.tag), "hccl_custom_reducescatter_%s", algorithmTag);

    // ==============================================
    // STEP 2: 创建资源
    // ==============================================
    CommEngine ccuEngine = CommEngine::COMM_ENGINE_CCU;

    // ==============================================
    // STEP 2.1: 申请用于 Host/Device 同步的通信资源
    // ==============================================
    // 将用户传入的 stream 转换为 CCU 通信引擎中的 thread，并申请 1 个 notify。
    const uint32_t threadNotifyNum = 1;
    CHK_RET(HcclThreadAcquireWithStream(comm, ccuEngine, stream, threadNotifyNum, &param.cpuThread));

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

        const uint32_t threadNum = useDualDie ? 2 : 1;
        resCtxHost.threads.resize(threadNum);
        resCtxHost.threads[0] = param.cpuThread;
        if (useDualDie) {
            CHK_RET(HcclThreadAcquire(comm, ccuEngine, 1, threadNotifyNum, &resCtxHost.threads[1]));
        }
        resCtxHost.ccuThread = param.cpuThread;

        if (param.rankSize > 1) {
            if (useDualDie) {
                CHK_RET(BuildOwnTargetPlan(comm, param, resCtxHost));
                CHK_RET(RegisterDualDiePartialKernels(comm, param, resCtxHost));
            } else if (useSmallClosParallel) {
                CHK_RET(RegisterSmallClosParallelKernel(comm, param, resCtxHost));
            } else {
                std::vector<ChannelHandle> channels;
                CHK_RET(AcquireMeshChannels(comm, param, channels));
                const auto algorithm = useDirectMesh ? ReduceScatterAlgorithm::DIRECT_MESH :
                                                       ReduceScatterAlgorithm::STAGING_MESH;
                CHK_RET(RegisterMeshKernel(comm, param, channels, algorithm, resCtxHost));
            }
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
