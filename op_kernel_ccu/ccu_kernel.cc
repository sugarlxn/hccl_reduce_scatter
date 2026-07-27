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

} // namespace ops_hccl

#undef CCU_RETURN_IF_ERROR
