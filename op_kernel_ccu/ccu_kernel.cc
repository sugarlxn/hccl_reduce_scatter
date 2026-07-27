/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <hcomm/hcomm_primitives.h>

#include <vector>

#include "custom.h"
#include "ccu_kernel.h"
#include "log.h"

namespace ccu = ::AscendC::ccu;

#define CCU_RETURN_IF_ERROR(call) \
    do { \
        CcuResult ccuRet = static_cast<CcuResult>(call); \
        if (ccuRet != CCU_SUCCESS) { \
            return ccuRet; \
        } \
    } while (0)

namespace ops_hccl {
CcuResult CcuKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuReduceScatterKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->rankSize < 2 || kernelArg->channelCount != kernelArg->rankSize - 1) {
        return CCU_E_PARA;
    }

    constexpr uint32_t REMOTE_CCL_ADDR_ID = 1;
    constexpr uint32_t REMOTE_CCL_TOKEN_ID = 2;
    constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
    constexpr uint16_t CCL_ADDR_READY = 1U << 1;
    constexpr uint16_t CCL_TOKEN_READY = 1U << 2;
    constexpr uint16_t DATA_READY = 1U << 3;
    constexpr uint16_t REDUCE_DONE = 1U << 4;

    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable inputToken;
    ccu::Variable localCclAddr;
    ccu::Variable localCclToken;
    ccu::Variable chunkBytes;
    ccu::Variable mySlotOffset;
    std::vector<ccu::Variable> sourceAddrs(kernelArg->rankSize);
    std::vector<ccu::Variable> stageAddrs(kernelArg->rankSize);
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(localCclAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(localCclToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunkBytes, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(mySlotOffset, argId++));
    for (uint32_t peer = 0; peer < kernelArg->rankSize; ++peer) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(sourceAddrs[peer], argId++));
    }
    for (uint32_t sourceRank = 0; sourceRank < kernelArg->rankSize; ++sourceRank) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(stageAddrs[sourceRank], argId++));
    }

    std::vector<ccu::Variable> remoteCclAddr(kernelArg->rankSize);
    std::vector<ccu::Variable> remoteCclToken(kernelArg->rankSize);
    uint32_t channelIdx = 0;
    for (uint32_t peer = 0; peer < kernelArg->rankSize; ++peer) {
        if (peer == kernelArg->rankId) {
            remoteCclAddr[peer] = localCclAddr;
            remoteCclToken[peer] = localCclToken;
            continue;
        }
        remoteCclAddr[peer] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIdx], REMOTE_CCL_ADDR_ID);
        remoteCclToken[peer] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIdx], REMOTE_CCL_TOKEN_ID);
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIdx], localCclAddr,
            REMOTE_CCL_ADDR_ID,
            CHANNEL_NOTIFY_INDEX, CCL_ADDR_READY));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIdx], localCclToken,
            REMOTE_CCL_TOKEN_ID,
            CHANNEL_NOTIFY_INDEX, CCL_TOKEN_READY));
        ++channelIdx;
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX,
            static_cast<uint16_t>(CCL_ADDR_READY | CCL_TOKEN_READY)));
    }

    ccu::Event transferEvent;
    ccu::LocalAddr localSrc;
    localSrc.addr = sourceAddrs[kernelArg->rankId];
    localSrc.token = inputToken;

    ccu::LocalAddr localStage;
    localStage.addr = stageAddrs[kernelArg->rankId];
    localStage.token = localCclToken;
    CCU_RETURN_IF_ERROR(ccu::LocalCopy(localStage, localSrc, chunkBytes, transferEvent,
        static_cast<uint16_t>(1U << kernelArg->rankId)));

    // Send only the chunk owned by each peer into that peer's source slot.
    channelIdx = 0;
    for (uint32_t peer = 0; peer < kernelArg->rankSize; ++peer) {
        if (peer == kernelArg->rankId) {
            continue;
        }
        ccu::LocalAddr src;
        src.addr = sourceAddrs[peer];
        src.token = inputToken;

        ccu::RemoteAddr dst;
        dst.addr = remoteCclAddr[peer];
        dst.token = remoteCclToken[peer];
        dst.addr += mySlotOffset;
        CCU_RETURN_IF_ERROR(ccu::Write(kernelArg->channels[channelIdx], dst, src, chunkBytes, transferEvent,
            static_cast<uint16_t>(1U << peer)));
        ++channelIdx;
    }

    const uint16_t allRanksMask = static_cast<uint16_t>((1U << kernelArg->rankSize) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, allRanksMask));
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX, DATA_READY));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX, DATA_READY));
    }

    // Fixed rank order makes floating-point sum deterministic.
    ccu::LocalAddr dst;
    dst.addr = outputAddr;
    dst.token = outputToken;
    ccu::LocalAddr staged;
    staged.addr = stageAddrs[0];
    staged.token = localCclToken;
    ccu::Event reduceEvent;
    CCU_RETURN_IF_ERROR(ccu::LocalCopy(dst, staged, chunkBytes, reduceEvent));
    CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvent));
    for (uint32_t sourceRank = 1; sourceRank < kernelArg->rankSize; ++sourceRank) {
        staged.addr = stageAddrs[sourceRank];
        CCU_RETURN_IF_ERROR(
            ccu::LocalReduce(dst, staged, chunkBytes, kernelArg->dataType, kernelArg->reduceOp, reduceEvent));
        CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvent));
    }

    // Protect the shared staging slots before the next sliced launch.
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX, REDUCE_DONE));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX, REDUCE_DONE));
    }

    return CCU_SUCCESS;
}

CcuResult CcuLocalReduceKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuLocalReduceKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->groupSize < 2 || kernelArg->groupSize > MAX_LOCAL_RANK_SIZE ||
        kernelArg->channelCount != kernelArg->groupSize - 1 || kernelArg->targetCount == 0 ||
        kernelArg->targetCount > MAX_HIERARCHICAL_TARGETS) {
        return CCU_E_PARA;
    }

    constexpr uint32_t REMOTE_INPUT_ADDR_ID = 1;
    constexpr uint32_t REMOTE_INPUT_TOKEN_ID = 2;
    constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
    constexpr uint16_t INPUT_ADDR_READY = 1U << 1;
    constexpr uint16_t INPUT_TOKEN_READY = 1U << 2;

    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable chunkBytes;
    std::vector<ccu::Variable> targetOffsets(kernelArg->targetCount);
    std::vector<ccu::Variable> dstAddrs(kernelArg->targetCount);
    std::vector<ccu::Variable> dstTokens(kernelArg->targetCount);
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunkBytes, argId++));
    for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(targetOffsets[target], argId++));
    }
    for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(dstAddrs[target], argId++));
    }
    for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(dstTokens[target], argId++));
    }

    std::vector<ccu::Variable> remoteInputAddr(kernelArg->groupSize);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->groupSize);
    uint32_t channelIdx = 0;
    for (uint32_t source = 0; source < kernelArg->groupSize; ++source) {
        if (source == kernelArg->groupRankId) {
            remoteInputAddr[source] = inputAddr;
            remoteInputToken[source] = inputToken;
            continue;
        }
        remoteInputAddr[source] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIdx], REMOTE_INPUT_ADDR_ID);
        remoteInputToken[source] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIdx], REMOTE_INPUT_TOKEN_ID);
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIdx], inputAddr,
            REMOTE_INPUT_ADDR_ID, CHANNEL_NOTIFY_INDEX, INPUT_ADDR_READY));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIdx], inputToken,
            REMOTE_INPUT_TOKEN_ID, CHANNEL_NOTIFY_INDEX, INPUT_TOKEN_READY));
        ++channelIdx;
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX,
            static_cast<uint16_t>(INPUT_ADDR_READY | INPUT_TOKEN_READY)));
    }

    // Targets use disjoint destinations. Issue the same source step for all
    // targets in parallel, then wait before advancing to the next source so
    // every FP32 result still accumulates in local-rank order.
    std::vector<ccu::Event> reduceEvents(kernelArg->targetCount);
    for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
        ccu::LocalAddr dst;
        dst.addr = dstAddrs[target];
        dst.token = dstTokens[target];
        if (kernelArg->groupRankId == 0) {
            ccu::LocalAddr src;
            src.addr = inputAddr;
            src.addr += targetOffsets[target];
            src.token = inputToken;
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(dst, src, chunkBytes, reduceEvents[target]));
        } else {
            ccu::RemoteAddr src;
            src.addr = remoteInputAddr[0];
            src.addr += targetOffsets[target];
            src.token = remoteInputToken[0];
            CCU_RETURN_IF_ERROR(ccu::Read(kernelArg->channels[0], dst, src, chunkBytes, reduceEvents[target]));
        }
    }
    for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
        CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvents[target]));
    }

    for (uint32_t source = 1; source < kernelArg->groupSize; ++source) {
        for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
            ccu::LocalAddr dst;
            dst.addr = dstAddrs[target];
            dst.token = dstTokens[target];
            if (source == kernelArg->groupRankId) {
                ccu::LocalAddr src;
                src.addr = inputAddr;
                src.addr += targetOffsets[target];
                src.token = inputToken;
                CCU_RETURN_IF_ERROR(ccu::LocalReduce(dst, src, chunkBytes, kernelArg->dataType,
                    kernelArg->reduceOp, reduceEvents[target]));
            } else {
                const uint32_t sourceChannel = source < kernelArg->groupRankId ? source : source - 1;
                ccu::RemoteAddr src;
                src.addr = remoteInputAddr[source];
                src.addr += targetOffsets[target];
                src.token = remoteInputToken[source];
                CCU_RETURN_IF_ERROR(ccu::ReadReduce(kernelArg->channels[sourceChannel], dst, src, chunkBytes,
                    kernelArg->dataType, kernelArg->reduceOp, reduceEvents[target]));
            }
        }
        for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
            CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvents[target]));
        }
    }
    return CCU_SUCCESS;
}

CcuResult CcuCrossReduceKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuCrossReduceKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount > MAX_CROSS_PEERS ||
        kernelArg->sendCount > kernelArg->channelCount) {
        return CCU_E_PARA;
    }

    constexpr uint32_t REMOTE_OUTPUT_ADDR_ID = 1;
    constexpr uint32_t REMOTE_OUTPUT_TOKEN_ID = 2;
    constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
    constexpr uint16_t OUTPUT_ADDR_READY = 1U << 1;
    constexpr uint16_t OUTPUT_TOKEN_READY = 1U << 2;
    constexpr uint16_t REDUCE_DONE = 1U << 3;

    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable cclToken;
    ccu::Variable chunkBytes;
    std::vector<ccu::Variable> sendAddrs(kernelArg->sendCount);
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(cclToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunkBytes, argId++));
    for (uint32_t sendIdx = 0; sendIdx < kernelArg->sendCount; ++sendIdx) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(sendAddrs[sendIdx], argId++));
    }

    std::vector<ccu::Variable> remoteOutputAddr(kernelArg->channelCount);
    std::vector<ccu::Variable> remoteOutputToken(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        remoteOutputAddr[i] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], REMOTE_OUTPUT_ADDR_ID);
        remoteOutputToken[i] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], REMOTE_OUTPUT_TOKEN_ID);
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i], outputAddr, REMOTE_OUTPUT_ADDR_ID,
            CHANNEL_NOTIFY_INDEX, OUTPUT_ADDR_READY));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i], outputToken, REMOTE_OUTPUT_TOKEN_ID,
            CHANNEL_NOTIFY_INDEX, OUTPUT_TOKEN_READY));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX,
            static_cast<uint16_t>(OUTPUT_ADDR_READY | OUTPUT_TOKEN_READY)));
    }

    ccu::Event writeEvent;
    for (uint32_t sendIdx = 0; sendIdx < kernelArg->sendCount; ++sendIdx) {
        const uint32_t channel = kernelArg->sendChannelIndices[sendIdx];
        ccu::LocalAddr src;
        src.addr = sendAddrs[sendIdx];
        src.token = cclToken;
        ccu::RemoteAddr dst;
        dst.addr = remoteOutputAddr[channel];
        dst.token = remoteOutputToken[channel];
        CCU_RETURN_IF_ERROR(ccu::WriteReduce(kernelArg->channels[channel], dst, src, chunkBytes,
            kernelArg->dataType, kernelArg->reduceOp, writeEvent, static_cast<uint16_t>(1U << sendIdx)));
    }
    if (kernelArg->sendCount > 0) {
        const uint16_t sendMask = static_cast<uint16_t>((1U << kernelArg->sendCount) - 1U);
        CCU_RETURN_IF_ERROR(ccu::EventWait(writeEvent, sendMask));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyRecord(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX, REDUCE_DONE));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX, REDUCE_DONE));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl

#undef CCU_RETURN_IF_ERROR
