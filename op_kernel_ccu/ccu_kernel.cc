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

    constexpr uint32_t REMOTE_INPUT_ADDR_ID = 1;
    constexpr uint32_t REMOTE_INPUT_TOKEN_ID = 2;
    constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
    constexpr uint16_t INPUT_ADDR_READY = 1U << 1;
    constexpr uint16_t INPUT_TOKEN_READY = 1U << 2;

    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable sourceOffset;
    ccu::Variable chunkBytes;
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(sourceOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunkBytes, argId++));

    std::vector<ccu::Variable> remoteInputAddr(kernelArg->rankSize);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->rankSize);
    uint32_t channelIdx = 0;
    for (uint32_t peer = 0; peer < kernelArg->rankSize; ++peer) {
        if (peer == kernelArg->rankId) {
            remoteInputAddr[peer] = inputAddr;
            remoteInputToken[peer] = inputToken;
            continue;
        }
        remoteInputAddr[peer] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[channelIdx], REMOTE_INPUT_ADDR_ID);
        remoteInputToken[peer] =
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

    // Each rank starts with its local contribution. Subsequent source ranks use
    // a fixed cyclic order, so FP32 accumulation is deterministic. At every
    // step sourceRank=(rankId+step)%rankSize is a permutation: every Clos port
    // serves exactly one peer and all ranks progress in parallel.
    ccu::LocalAddr dst;
    dst.addr = outputAddr;
    dst.token = outputToken;

    ccu::LocalAddr localSrc;
    localSrc.addr = inputAddr;
    localSrc.addr += sourceOffset;
    localSrc.token = inputToken;
    ccu::Event reduceEvent;
    CCU_RETURN_IF_ERROR(ccu::LocalCopy(dst, localSrc, chunkBytes, reduceEvent));
    CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvent));

    for (uint32_t step = 1; step < kernelArg->rankSize; ++step) {
        const uint32_t sourceRank = (kernelArg->rankId + step) % kernelArg->rankSize;
        const uint32_t sourceChannel = sourceRank < kernelArg->rankId ? sourceRank : sourceRank - 1;
        ccu::RemoteAddr remoteSrc;
        remoteSrc.addr = remoteInputAddr[sourceRank];
        remoteSrc.addr += sourceOffset;
        remoteSrc.token = remoteInputToken[sourceRank];
        CCU_RETURN_IF_ERROR(ccu::ReadReduce(kernelArg->channels[sourceChannel], dst, remoteSrc, chunkBytes,
            kernelArg->dataType, kernelArg->reduceOp, reduceEvent));
        CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvent));
    }

    return CCU_SUCCESS;
}

CcuResult CcuStagingKernel(CcuKernelArg arg)
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
            REMOTE_CCL_ADDR_ID, CHANNEL_NOTIFY_INDEX, CCL_ADDR_READY));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIdx], localCclToken,
            REMOTE_CCL_TOKEN_ID, CHANNEL_NOTIFY_INDEX, CCL_TOKEN_READY));
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
    ccu::Variable cclToken;
    ccu::Variable chunkBytes;
    ccu::Variable tileBytes;
    ccu::Variable tailBytes;
    std::vector<ccu::Variable> targetOffsets(kernelArg->targetCount);
    constexpr uint32_t STAGING_BANK_COUNT = 2;
    std::vector<ccu::Variable> stageAddrs(
        kernelArg->useStaging ? STAGING_BANK_COUNT * kernelArg->groupSize : 0);
    std::vector<ccu::Variable> dstAddrs(kernelArg->targetCount);
    std::vector<ccu::Variable> dstTokens(kernelArg->targetCount);
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, argId++));
    if (kernelArg->useStaging) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(cclToken, argId++));
    }
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunkBytes, argId++));
    if (kernelArg->useStaging) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(tileBytes, argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(tailBytes, argId++));
    }
    for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(targetOffsets[target], argId++));
    }
    for (uint32_t source = 0; source < stageAddrs.size(); ++source) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(stageAddrs[source], argId++));
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

    if (kernelArg->useStaging) {
        // Prime bank 0 for the first target. Reads from different peers use
        // independent links and are issued before the single masked wait.
        ccu::Event transferEvent;
        ccu::Event reduceEvent;
        const uint16_t groupMask = static_cast<uint16_t>((1U << kernelArg->groupSize) - 1U);
        for (uint32_t source = 0; source < kernelArg->groupSize; ++source) {
            ccu::LocalAddr stage;
            stage.addr = stageAddrs[source];
            stage.token = cclToken;
            const uint16_t mask = static_cast<uint16_t>(1U << source);
            if (source == kernelArg->groupRankId) {
                ccu::LocalAddr src;
                src.addr = inputAddr;
                src.addr += targetOffsets[0];
                src.token = inputToken;
                CCU_RETURN_IF_ERROR(ccu::LocalCopy(stage, src, tileBytes, transferEvent, mask));
            } else {
                const uint32_t sourceChannel = source < kernelArg->groupRankId ? source : source - 1;
                ccu::RemoteAddr src;
                src.addr = remoteInputAddr[source];
                src.addr += targetOffsets[0];
                src.token = remoteInputToken[source];
                CCU_RETURN_IF_ERROR(
                    ccu::Read(kernelArg->channels[sourceChannel], stage, src, tileBytes, transferEvent, mask));
            }
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, groupMask));

        // A two-bank pipeline spans target boundaries:
        //   fetch current tail | reduce current head
        //   fetch next head    | reduce current tail
        // Every reduction still consumes sources in the fixed [0, groupSize)
        // order, so floating-point results remain deterministic.
        for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
            CCU_IF(tailBytes != 0) {
                for (uint32_t source = 0; source < kernelArg->groupSize; ++source) {
                    ccu::LocalAddr stage;
                    stage.addr = stageAddrs[kernelArg->groupSize + source];
                    stage.token = cclToken;
                    const uint16_t mask = static_cast<uint16_t>(1U << source);
                    if (source == kernelArg->groupRankId) {
                        ccu::LocalAddr src;
                        src.addr = inputAddr;
                        src.addr += targetOffsets[target];
                        src.addr += tileBytes;
                        src.token = inputToken;
                        CCU_RETURN_IF_ERROR(ccu::LocalCopy(stage, src, tailBytes, transferEvent, mask));
                    } else {
                        const uint32_t sourceChannel = source < kernelArg->groupRankId ? source : source - 1;
                        ccu::RemoteAddr src;
                        src.addr = remoteInputAddr[source];
                        src.addr += targetOffsets[target];
                        src.addr += tileBytes;
                        src.token = remoteInputToken[source];
                        CCU_RETURN_IF_ERROR(ccu::Read(
                            kernelArg->channels[sourceChannel], stage, src, tailBytes, transferEvent, mask));
                    }
                }
            }

            ccu::LocalAddr dst;
            dst.addr = dstAddrs[target];
            dst.token = dstTokens[target];
            ccu::LocalAddr staged;
            staged.addr = stageAddrs[0];
            staged.token = cclToken;
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(dst, staged, tileBytes, reduceEvent));
            CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvent));
            for (uint32_t source = 1; source < kernelArg->groupSize; ++source) {
                staged.addr = stageAddrs[source];
                CCU_RETURN_IF_ERROR(
                    ccu::LocalReduce(dst, staged, tileBytes, kernelArg->dataType, kernelArg->reduceOp, reduceEvent));
                CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvent));
            }

            CCU_IF(tailBytes != 0)
            {
                CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, groupMask));

                if (target + 1 < kernelArg->targetCount) {
                    for (uint32_t source = 0; source < kernelArg->groupSize; ++source) {
                        ccu::LocalAddr stage;
                        stage.addr = stageAddrs[source];
                        stage.token = cclToken;
                        const uint16_t mask = static_cast<uint16_t>(1U << source);
                        if (source == kernelArg->groupRankId) {
                            ccu::LocalAddr src;
                            src.addr = inputAddr;
                            src.addr += targetOffsets[target + 1];
                            src.token = inputToken;
                            CCU_RETURN_IF_ERROR(ccu::LocalCopy(stage, src, tileBytes, transferEvent, mask));
                        } else {
                            const uint32_t sourceChannel =
                                source < kernelArg->groupRankId ? source : source - 1;
                            ccu::RemoteAddr src;
                            src.addr = remoteInputAddr[source];
                            src.addr += targetOffsets[target + 1];
                            src.token = remoteInputToken[source];
                            CCU_RETURN_IF_ERROR(ccu::Read(
                                kernelArg->channels[sourceChannel], stage, src, tileBytes, transferEvent, mask));
                        }
                    }
                }

                dst.addr = dstAddrs[target];
                dst.addr += tileBytes;
                staged.addr = stageAddrs[kernelArg->groupSize];
                CCU_RETURN_IF_ERROR(ccu::LocalCopy(dst, staged, tailBytes, reduceEvent));
                CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvent));
                for (uint32_t source = 1; source < kernelArg->groupSize; ++source) {
                    staged.addr = stageAddrs[kernelArg->groupSize + source];
                    CCU_RETURN_IF_ERROR(ccu::LocalReduce(
                        dst, staged, tailBytes, kernelArg->dataType, kernelArg->reduceOp, reduceEvent));
                    CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvent));
                }

                if (target + 1 < kernelArg->targetCount) {
                    CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, groupMask));
                }
            }
        }
        return CCU_SUCCESS;
    }

    // Give each independent target a different cyclic source order. Each
    // result still has a fixed deterministic order, while one round uses
    // different full-mesh links instead of serializing all targets on one
    // peer/channel (especially important for the 4-rank side of 8+4).
    std::vector<ccu::Event> reduceEvents(kernelArg->targetCount);
    for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
        const uint32_t source = target;
        ccu::LocalAddr dst;
        dst.addr = dstAddrs[target];
        dst.token = dstTokens[target];
        if (kernelArg->groupRankId == source) {
            ccu::LocalAddr src;
            src.addr = inputAddr;
            src.addr += targetOffsets[target];
            src.token = inputToken;
            CCU_RETURN_IF_ERROR(ccu::LocalCopy(dst, src, chunkBytes, reduceEvents[target]));
        } else {
            const uint32_t sourceChannel = source < kernelArg->groupRankId ? source : source - 1;
            ccu::RemoteAddr src;
            src.addr = remoteInputAddr[source];
            src.addr += targetOffsets[target];
            src.token = remoteInputToken[source];
            CCU_RETURN_IF_ERROR(
                ccu::Read(kernelArg->channels[sourceChannel], dst, src, chunkBytes, reduceEvents[target]));
        }
    }
    for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
        CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvents[target]));
    }

    for (uint32_t step = 1; step < kernelArg->groupSize; ++step) {
        for (uint32_t target = 0; target < kernelArg->targetCount; ++target) {
            const uint32_t source = (target + step) % kernelArg->groupSize;
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

CcuResult CcuSmallStagingTreeKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuPartialReduceKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->sourceCount < 2 || kernelArg->sourceCount > MAX_RANK_SIZE ||
        !kernelArg->includeLocalSource || kernelArg->localSourceIndex >= kernelArg->sourceCount ||
        kernelArg->channelCount != kernelArg->sourceCount - 1) {
        return CCU_E_PARA;
    }

    constexpr uint32_t REMOTE_INPUT_ADDR_ID = 1;
    constexpr uint32_t REMOTE_INPUT_TOKEN_ID = 2;
    constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
    constexpr uint16_t INPUT_ADDR_READY = 1U << 1;
    constexpr uint16_t INPUT_TOKEN_READY = 1U << 2;

    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable scratchToken;
    ccu::Variable sourceOffset;
    ccu::Variable chunkBytes;
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(scratchToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(sourceOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(chunkBytes, argId++));

    std::vector<ccu::Variable> stageAddrs(kernelArg->sourceCount);
    for (uint32_t source = 0; source < kernelArg->sourceCount; ++source) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(stageAddrs[source], argId++));
    }

    std::vector<ccu::Variable> remoteInputAddr(kernelArg->sourceCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->sourceCount);
    uint32_t channelIdx = 0;
    for (uint32_t source = 0; source < kernelArg->sourceCount; ++source) {
        if (source == kernelArg->localSourceIndex) {
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

    // Pull all remote contributions concurrently. The first one lands directly
    // in output; the others use disjoint CCL-buffer slots.
    const uint32_t firstRemoteSource = kernelArg->localSourceIndex == 0 ? 1 : 0;
    ccu::Event transferEvent;
    channelIdx = 0;
    for (uint32_t source = 0; source < kernelArg->sourceCount; ++source) {
        if (source == kernelArg->localSourceIndex) {
            continue;
        }
        ccu::LocalAddr dst;
        dst.addr = source == firstRemoteSource ? outputAddr : stageAddrs[source];
        dst.token = source == firstRemoteSource ? outputToken : scratchToken;
        ccu::RemoteAddr src;
        src.addr = remoteInputAddr[source];
        src.addr += sourceOffset;
        src.token = remoteInputToken[source];
        CCU_RETURN_IF_ERROR(ccu::Read(kernelArg->channels[channelIdx], dst, src, chunkBytes,
            transferEvent, static_cast<uint16_t>(1U << channelIdx)));
        ++channelIdx;
    }
    const uint16_t allChannelsMask =
        static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
    CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, allChannelsMask));

    // Build a stable leaf order and reduce disjoint pairs at each tree level.
    // The tree is fixed for a rank size, satisfying deterministic FP32 output.
    std::vector<ccu::LocalAddr> work(kernelArg->sourceCount);
    uint32_t workIndex = 0;
    for (uint32_t source = 0; source < kernelArg->sourceCount; ++source) {
        if (source == kernelArg->localSourceIndex) {
            continue;
        }
        work[workIndex].addr = source == firstRemoteSource ? outputAddr : stageAddrs[source];
        work[workIndex].token = source == firstRemoteSource ? outputToken : scratchToken;
        ++workIndex;
    }
    work[workIndex].addr = inputAddr;
    work[workIndex].addr += sourceOffset;
    work[workIndex].token = inputToken;

    ccu::Event reduceEvent;
    uint32_t remain = kernelArg->sourceCount;
    while (remain > 1) {
        const uint32_t reducePieces = remain / 2;
        const uint32_t sourceBase = remain - reducePieces;
        for (uint32_t piece = 0; piece < reducePieces; ++piece) {
            CCU_RETURN_IF_ERROR(ccu::LocalReduce(work[piece], work[sourceBase + piece], chunkBytes,
                kernelArg->dataType, kernelArg->reduceOp, reduceEvent,
                static_cast<uint16_t>(1U << piece)));
        }
        CCU_RETURN_IF_ERROR(
            ccu::EventWait(reduceEvent, static_cast<uint16_t>((1U << reducePieces) - 1U)));
        remain -= reducePieces;
    }
    return CCU_SUCCESS;
}

CcuResult FetchTiledPartial(const CcuTiledPartialReduceKernelArg *kernelArg,
    const ccu::Variable &outputAddr, const ccu::Variable &outputToken,
    const ccu::Variable &scratchToken, const ccu::Variable &sourceOffset,
    const ccu::Variable &tileOffset, const ccu::Variable &tileBytes, uint32_t bank,
    const std::vector<ccu::Variable> &stageAddrs,
    const std::vector<ccu::Variable> &remoteInputAddr,
    const std::vector<ccu::Variable> &remoteInputToken, ccu::Event &transferEvent)
{
    const uint32_t firstRemoteSource =
        kernelArg->includeLocalSource && kernelArg->localSourceIndex == 0 ? 1 : 0;
    uint32_t channelIdx = 0;
    for (uint32_t source = 0; source < kernelArg->sourceCount; ++source) {
        if (kernelArg->includeLocalSource && source == kernelArg->localSourceIndex) {
            continue;
        }

        ccu::LocalAddr stage;
        if (source == firstRemoteSource) {
            stage.addr = outputAddr;
            stage.addr += tileOffset;
            stage.token = outputToken;
        } else {
            stage.addr = stageAddrs[bank * kernelArg->sourceCount + source];
            stage.token = scratchToken;
        }
        const uint16_t mask = static_cast<uint16_t>(1U << channelIdx);

        ccu::RemoteAddr src;
        src.addr = remoteInputAddr[source];
        src.addr += sourceOffset;
        src.addr += tileOffset;
        src.token = remoteInputToken[source];
        CCU_RETURN_IF_ERROR(
            ccu::Read(kernelArg->channels[channelIdx], stage, src, tileBytes, transferEvent, mask));
        ++channelIdx;
    }
    return CCU_SUCCESS;
}

CcuResult ReduceTiledPartial(const CcuTiledPartialReduceKernelArg *kernelArg,
    const ccu::Variable &outputAddr, const ccu::Variable &outputToken,
    const ccu::Variable &inputAddr, const ccu::Variable &inputToken,
    const ccu::Variable &scratchToken, const ccu::Variable &sourceOffset,
    const ccu::Variable &tileOffset, const ccu::Variable &tileBytes, uint32_t bank,
    const std::vector<ccu::Variable> &stageAddrs, ccu::Event &reduceEvent)
{
    std::vector<ccu::LocalAddr> work(kernelArg->sourceCount);
    const uint32_t firstRemoteSource =
        kernelArg->includeLocalSource && kernelArg->localSourceIndex == 0 ? 1 : 0;
    uint32_t workIndex = 0;
    for (uint32_t source = 0; source < kernelArg->sourceCount; ++source) {
        if (kernelArg->includeLocalSource && source == kernelArg->localSourceIndex) {
            continue;
        }
        if (source == firstRemoteSource) {
            work[workIndex].addr = outputAddr;
            work[workIndex].addr += tileOffset;
            work[workIndex].token = outputToken;
        } else {
            work[workIndex].addr = stageAddrs[bank * kernelArg->sourceCount + source];
            work[workIndex].token = scratchToken;
        }
        ++workIndex;
    }
    if (kernelArg->includeLocalSource) {
        work[workIndex].addr = inputAddr;
        work[workIndex].addr += sourceOffset;
        work[workIndex].addr += tileOffset;
        work[workIndex].token = inputToken;
    }

    // Fixed tree shape preserves deterministic FP32 accumulation.
    uint32_t remain = kernelArg->sourceCount;
    while (remain > 1) {
        const uint32_t reducePieces = remain / 2;
        const uint32_t sourceBase = remain - reducePieces;
        for (uint32_t piece = 0; piece < reducePieces; ++piece) {
            CCU_RETURN_IF_ERROR(ccu::LocalReduce(work[piece], work[sourceBase + piece], tileBytes,
                kernelArg->dataType, kernelArg->reduceOp, reduceEvent,
                static_cast<uint16_t>(1U << piece)));
        }
        CCU_RETURN_IF_ERROR(
            ccu::EventWait(reduceEvent, static_cast<uint16_t>((1U << reducePieces) - 1U)));
        remain -= reducePieces;
    }
    return CCU_SUCCESS;
}

CcuResult CcuTiledPartialReduceKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuTiledPartialReduceKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->sourceCount == 0 ||
        kernelArg->sourceCount > MAX_RANK_SIZE ||
        kernelArg->channelCount !=
            kernelArg->sourceCount - (kernelArg->includeLocalSource ? 1U : 0U) ||
        (kernelArg->includeLocalSource &&
            kernelArg->localSourceIndex >= kernelArg->sourceCount)) {
        return CCU_E_PARA;
    }

    constexpr uint32_t REMOTE_INPUT_ADDR_ID = 1;
    constexpr uint32_t REMOTE_INPUT_TOKEN_ID = 2;
    constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
    constexpr uint16_t INPUT_ADDR_READY = 1U << 1;
    constexpr uint16_t INPUT_TOKEN_READY = 1U << 2;

    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable scratchToken;
    ccu::Variable sourceOffset;
    ccu::Variable tileCapacity;
    ccu::Variable pairCount;
    ccu::Variable hasOddTile;
    ccu::Variable tailBytes;
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(scratchToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(sourceOffset, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(tileCapacity, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(pairCount, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(hasOddTile, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(tailBytes, argId++));
    std::vector<ccu::Variable> stageAddrs(2 * kernelArg->sourceCount);
    for (uint32_t slot = 0; slot < stageAddrs.size(); ++slot) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(stageAddrs[slot], argId++));
    }

    std::vector<ccu::Variable> remoteInputAddr(kernelArg->sourceCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->sourceCount);
    uint32_t channelIdx = 0;
    for (uint32_t source = 0; source < kernelArg->sourceCount; ++source) {
        if (kernelArg->includeLocalSource && source == kernelArg->localSourceIndex) {
            remoteInputAddr[source] = inputAddr;
            remoteInputToken[source] = inputToken;
            continue;
        }
        remoteInputAddr[source] =
            ccu::GetResByChannel<ccu::Variable>(
                kernelArg->channels[channelIdx], REMOTE_INPUT_ADDR_ID);
        remoteInputToken[source] =
            ccu::GetResByChannel<ccu::Variable>(
                kernelArg->channels[channelIdx], REMOTE_INPUT_TOKEN_ID);
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIdx],
            inputAddr, REMOTE_INPUT_ADDR_ID, CHANNEL_NOTIFY_INDEX, INPUT_ADDR_READY));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[channelIdx],
            inputToken, REMOTE_INPUT_TOKEN_ID, CHANNEL_NOTIFY_INDEX, INPUT_TOKEN_READY));
        ++channelIdx;
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX,
            static_cast<uint16_t>(INPUT_ADDR_READY | INPUT_TOKEN_READY)));
    }

    ccu::Event transferEvent;
    ccu::Event reduceEvent;
    const uint16_t sourceMask =
        static_cast<uint16_t>((1U << kernelArg->channelCount) - 1U);
    ccu::Variable tileOffset;
    ccu::Variable one;
    tileOffset = 0;
    one = 1;

    CCU_IF(pairCount != UINT64_MAX)
    {
        CCU_RETURN_IF_ERROR(FetchTiledPartial(kernelArg, outputAddr, outputToken,
            scratchToken, sourceOffset, tileOffset, tileCapacity, 0, stageAddrs,
            remoteInputAddr, remoteInputToken, transferEvent));
        CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, sourceMask));
    }

    CCU_WHILE(pairCount != UINT64_MAX)
    {
        ccu::Variable nextOffset;
        nextOffset = tileOffset;
        nextOffset += tileCapacity;
        CCU_RETURN_IF_ERROR(FetchTiledPartial(kernelArg, outputAddr, outputToken,
            scratchToken, sourceOffset, nextOffset, tileCapacity, 1, stageAddrs,
            remoteInputAddr, remoteInputToken, transferEvent));
        CCU_RETURN_IF_ERROR(ReduceTiledPartial(kernelArg, outputAddr, outputToken,
            inputAddr, inputToken, scratchToken, sourceOffset, tileOffset, tileCapacity,
            0, stageAddrs, reduceEvent));
        CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, sourceMask));

        tileOffset += tileCapacity;
        tileOffset += tileCapacity;
        pairCount += one;

        CCU_IF(pairCount != UINT64_MAX)
        {
            CCU_RETURN_IF_ERROR(FetchTiledPartial(kernelArg, outputAddr, outputToken,
                scratchToken, sourceOffset, tileOffset, tileCapacity, 0, stageAddrs,
                remoteInputAddr, remoteInputToken, transferEvent));
        }
        CCU_RETURN_IF_ERROR(ReduceTiledPartial(kernelArg, outputAddr, outputToken,
            inputAddr, inputToken, scratchToken, sourceOffset, nextOffset, tileCapacity,
            1, stageAddrs, reduceEvent));
        CCU_IF(pairCount != UINT64_MAX)
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, sourceMask));
        }
    }

    CCU_IF(hasOddTile != 0)
    {
        CCU_RETURN_IF_ERROR(FetchTiledPartial(kernelArg, outputAddr, outputToken,
            scratchToken, sourceOffset, tileOffset, tileCapacity, 0, stageAddrs,
            remoteInputAddr, remoteInputToken, transferEvent));
        CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, sourceMask));
        CCU_IF(tailBytes != 0)
        {
            ccu::Variable tailOffset;
            tailOffset = tileOffset;
            tailOffset += tileCapacity;
            CCU_RETURN_IF_ERROR(FetchTiledPartial(kernelArg, outputAddr, outputToken,
                scratchToken, sourceOffset, tailOffset, tailBytes, 1, stageAddrs,
                remoteInputAddr, remoteInputToken, transferEvent));
        }
        CCU_RETURN_IF_ERROR(ReduceTiledPartial(kernelArg, outputAddr, outputToken,
            inputAddr, inputToken, scratchToken, sourceOffset, tileOffset, tileCapacity,
            0, stageAddrs, reduceEvent));
        CCU_IF(tailBytes != 0)
        {
            CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, sourceMask));
        }
        tileOffset += tileCapacity;
    }

    CCU_IF(tailBytes != 0)
    {
        CCU_IF(hasOddTile == 0)
        {
            CCU_RETURN_IF_ERROR(FetchTiledPartial(kernelArg, outputAddr, outputToken,
                scratchToken, sourceOffset, tileOffset, tailBytes, 1, stageAddrs,
                remoteInputAddr, remoteInputToken, transferEvent));
            CCU_RETURN_IF_ERROR(ccu::EventWait(transferEvent, sourceMask));
        }
        CCU_RETURN_IF_ERROR(ReduceTiledPartial(kernelArg, outputAddr, outputToken,
            inputAddr, inputToken, scratchToken, sourceOffset, tileOffset, tailBytes,
            1, stageAddrs, reduceEvent));
    }
    return CCU_SUCCESS;
}

CcuResult CcuPartialReduceKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuPartialReduceKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->sourceCount == 0 || kernelArg->sourceCount > MAX_RANK_SIZE ||
        kernelArg->stripeCount == 0 || kernelArg->stripeCount > kernelArg->sourceCount ||
        kernelArg->channelCount != kernelArg->sourceCount - (kernelArg->includeLocalSource ? 1U : 0U) ||
        (kernelArg->includeLocalSource && kernelArg->localSourceIndex >= kernelArg->sourceCount)) {
        return CCU_E_PARA;
    }

    constexpr uint32_t REMOTE_INPUT_ADDR_ID = 1;
    constexpr uint32_t REMOTE_INPUT_TOKEN_ID = 2;
    constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
    constexpr uint16_t INPUT_ADDR_READY = 1U << 1;
    constexpr uint16_t INPUT_TOKEN_READY = 1U << 2;

    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable inputAddr;
    ccu::Variable inputToken;
    ccu::Variable sourceOffset;
    std::vector<ccu::Variable> stripeOffsets(kernelArg->stripeCount);
    std::vector<ccu::Variable> stripeBytes(kernelArg->stripeCount);
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(inputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(sourceOffset, argId++));
    for (uint32_t stripe = 0; stripe < kernelArg->stripeCount; ++stripe) {
        CCU_RETURN_IF_ERROR(ccu::LoadArg(stripeOffsets[stripe], argId++));
        CCU_RETURN_IF_ERROR(ccu::LoadArg(stripeBytes[stripe], argId++));
    }

    std::vector<ccu::Variable> remoteInputAddr(kernelArg->sourceCount);
    std::vector<ccu::Variable> remoteInputToken(kernelArg->sourceCount);
    uint32_t channelIdx = 0;
    for (uint32_t source = 0; source < kernelArg->sourceCount; ++source) {
        if (kernelArg->includeLocalSource && source == kernelArg->localSourceIndex) {
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

    // Split the partial into independent stripes. In every round,
    // (stripe + round) % sourceCount selects distinct sources, so each channel
    // is used at most once and every operation writes a disjoint range. Across
    // sourceCount rounds every stripe consumes every source in a fixed cyclic
    // order, preserving deterministic FP32 accumulation without HBM staging.
    ccu::Event stripeEvent;
    const uint16_t allStripesMask = static_cast<uint16_t>(
        kernelArg->stripeCount == 16 ? 0xFFFFU : ((1U << kernelArg->stripeCount) - 1U));
    for (uint32_t round = 0; round < kernelArg->sourceCount; ++round) {
        for (uint32_t stripe = 0; stripe < kernelArg->stripeCount; ++stripe) {
            const uint32_t source = (stripe + round) % kernelArg->sourceCount;
            const uint16_t mask = static_cast<uint16_t>(1U << stripe);

            ccu::LocalAddr dst;
            dst.addr = outputAddr;
            dst.addr += stripeOffsets[stripe];
            dst.token = outputToken;

            if (kernelArg->includeLocalSource && source == kernelArg->localSourceIndex) {
                ccu::LocalAddr src;
                src.addr = inputAddr;
                src.addr += sourceOffset;
                src.addr += stripeOffsets[stripe];
                src.token = inputToken;
                if (round == 0) {
                    CCU_RETURN_IF_ERROR(ccu::LocalCopy(dst, src, stripeBytes[stripe], stripeEvent, mask));
                } else {
                    CCU_RETURN_IF_ERROR(ccu::LocalReduce(dst, src, stripeBytes[stripe],
                        kernelArg->dataType, kernelArg->reduceOp, stripeEvent, mask));
                }
            } else {
                const uint32_t sourceChannel =
                    kernelArg->includeLocalSource && source > kernelArg->localSourceIndex ? source - 1 : source;
                ccu::RemoteAddr src;
                src.addr = remoteInputAddr[source];
                src.addr += sourceOffset;
                src.addr += stripeOffsets[stripe];
                src.token = remoteInputToken[source];
                if (round == 0) {
                    CCU_RETURN_IF_ERROR(ccu::Read(kernelArg->channels[sourceChannel], dst, src,
                        stripeBytes[stripe], stripeEvent, mask));
                } else {
                    CCU_RETURN_IF_ERROR(ccu::ReadReduce(kernelArg->channels[sourceChannel], dst, src,
                        stripeBytes[stripe], kernelArg->dataType, kernelArg->reduceOp, stripeEvent, mask));
                }
            }
        }
        CCU_RETURN_IF_ERROR(ccu::EventWait(stripeEvent, allStripesMask));
    }
    return CCU_SUCCESS;
}

CcuResult CcuMergePartialKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuMergePartialKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount != 1) {
        return CCU_E_PARA;
    }

    // HcommCcuKernelRegister's dieId argument is currently reserved. Referencing
    // one channel makes the translator place each merge kernel on the same IO
    // Die as its corresponding partial-reduce kernel. No network instruction is
    // emitted for this anchor.
    constexpr uint32_t DIE_ANCHOR_VARIABLE_ID = 0;
    const ccu::Variable dieAnchor =
        ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[0], DIE_ANCHOR_VARIABLE_ID);
    (void)dieAnchor;

    ccu::Variable outputAddr;
    ccu::Variable outputToken;
    ccu::Variable crossPartialAddr;
    ccu::Variable cclToken;
    ccu::Variable recvBytes;
    uint32_t argId = 0;
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(outputToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(crossPartialAddr, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(cclToken, argId++));
    CCU_RETURN_IF_ERROR(ccu::LoadArg(recvBytes, argId++));

    ccu::LocalAddr output;
    output.addr = outputAddr;
    output.token = outputToken;
    ccu::LocalAddr crossPartial;
    crossPartial.addr = crossPartialAddr;
    crossPartial.token = cclToken;
    ccu::Event reduceEvent;
    CCU_RETURN_IF_ERROR(ccu::LocalReduce(output, crossPartial, recvBytes,
        kernelArg->dataType, kernelArg->reduceOp, reduceEvent));
    CCU_RETURN_IF_ERROR(ccu::EventWait(reduceEvent));
    return CCU_SUCCESS;
}

CcuResult CcuCrossReduceKernel(CcuKernelArg arg)
{
    auto *kernelArg = static_cast<CcuCrossReduceKernelArg *>(arg);
    if (kernelArg == nullptr || kernelArg->channelCount == 0 || kernelArg->channelCount > MAX_CROSS_PEERS ||
        kernelArg->sendCount > kernelArg->channelCount) {
        return CCU_E_PARA;
    }

    constexpr uint32_t REMOTE_PARTIAL_ADDR_ID = 1;
    constexpr uint32_t REMOTE_PARTIAL_TOKEN_ID = 2;
    constexpr uint32_t CHANNEL_NOTIFY_INDEX = 0;
    constexpr uint16_t PARTIAL_ADDR_READY = 1U << 1;
    constexpr uint16_t PARTIAL_TOKEN_READY = 1U << 2;
    constexpr uint16_t CROSS_READ_DONE = 1U << 3;

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

    std::vector<ccu::Variable> remotePartialAddr(kernelArg->channelCount);
    std::vector<ccu::Variable> remotePartialToken(kernelArg->channelCount);
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        remotePartialAddr[i] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], REMOTE_PARTIAL_ADDR_ID);
        remotePartialToken[i] =
            ccu::GetResByChannel<ccu::Variable>(kernelArg->channels[i], REMOTE_PARTIAL_TOKEN_ID);

        ccu::Variable advertisedAddr;
        ccu::Variable advertisedToken;
        advertisedAddr = outputAddr;
        advertisedToken = outputToken;
        for (uint32_t sendIdx = 0; sendIdx < kernelArg->sendCount; ++sendIdx) {
            if (kernelArg->sendChannelIndices[sendIdx] == i) {
                advertisedAddr = sendAddrs[sendIdx];
                advertisedToken = cclToken;
            }
        }
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i], advertisedAddr,
            REMOTE_PARTIAL_ADDR_ID, CHANNEL_NOTIFY_INDEX, PARTIAL_ADDR_READY));
        CCU_RETURN_IF_ERROR(ccu::WriteVariableWithNotify(kernelArg->channels[i], advertisedToken,
            REMOTE_PARTIAL_TOKEN_ID, CHANNEL_NOTIFY_INDEX, PARTIAL_TOKEN_READY));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(ccu::NotifyWait(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX,
            static_cast<uint16_t>(PARTIAL_ADDR_READY | PARTIAL_TOKEN_READY)));
    }

    // Each rank needs one peer partial. A pull completes locally, eliminating
    // the post-write notify barrier while preserving the same reduction order.
    ccu::LocalAddr dst;
    dst.addr = outputAddr;
    dst.token = outputToken;
    ccu::RemoteAddr src;
    src.addr = remotePartialAddr[0];
    src.token = remotePartialToken[0];
    ccu::Event readEvent;
    CCU_RETURN_IF_ERROR(ccu::ReadReduce(kernelArg->channels[0], dst, src, chunkBytes,
        kernelArg->dataType, kernelArg->reduceOp, readEvent));
    CCU_RETURN_IF_ERROR(ccu::EventWait(readEvent));

    // A local ReadReduce completion only proves that this rank stopped reading
    // its peer. Complete a symmetric handshake so the host-side ACK also proves
    // every peer stopped reading our advertised partial before bank reuse.
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyRecord(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX, CROSS_READ_DONE));
    }
    for (uint32_t i = 0; i < kernelArg->channelCount; ++i) {
        CCU_RETURN_IF_ERROR(
            ccu::NotifyWait(kernelArg->channels[i], CHANNEL_NOTIFY_INDEX, CROSS_READ_DONE));
    }
    return CCU_SUCCESS;
}

} // namespace ops_hccl

#undef CCU_RETURN_IF_ERROR
