/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <ccu/ccu_res.h>
#include <ccu/ccu_launch.h>
#include <hcomm/hcomm_primitives.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "custom.h"
#include "exec_op.h"
#include "log.h"

namespace {
constexpr uint64_t GROUP_REDUCE_MICRO_BYTES = 4096;
constexpr uint64_t GROUP_REDUCE_PARALLEL = 16;

constexpr uint64_t MaskThrough(uint16_t end)
{
    return (uint64_t{1} << (end + 1)) - 1;
}

uint64_t PackParallelParam(uint64_t repeatNum, uint64_t repeatLoopIndex, uint64_t totalLoopNum)
{
    constexpr uint16_t REPEAT_BITS = 7;
    constexpr uint16_t REPEAT_SHIFT = 55;
    constexpr uint16_t LOOP_INDEX_BITS = 7;
    constexpr uint16_t LOOP_INDEX_SHIFT = 48;
    constexpr uint16_t LOOP_COUNT_BITS = 7;
    constexpr uint16_t LOOP_COUNT_SHIFT = 41;
    return ((repeatNum & MaskThrough(REPEAT_BITS)) << REPEAT_SHIFT) |
        ((repeatLoopIndex & MaskThrough(LOOP_INDEX_BITS)) << LOOP_INDEX_SHIFT) |
        ((totalLoopNum & MaskThrough(LOOP_COUNT_BITS)) << LOOP_COUNT_SHIFT);
}

std::vector<uint64_t> CalcGroupReduceSize(uint64_t size)
{
    const uint64_t parallelBytes = GROUP_REDUCE_MICRO_BYTES * GROUP_REDUCE_PARALLEL;
    const uint64_t loopCount = size / parallelBytes;
    const uint64_t remaining = size - loopCount * parallelBytes;
    const uint64_t microCount = remaining / GROUP_REDUCE_MICRO_BYTES;
    const uint64_t residual = remaining - microCount * GROUP_REDUCE_MICRO_BYTES;

    uint64_t parallelParam = 0;
    uint64_t tailBytes = 0;
    if (microCount != 0 && residual == 0) {
        parallelParam = PackParallelParam(microCount - 1, 0, 1);
        tailBytes = GROUP_REDUCE_MICRO_BYTES;
    } else if (microCount == 0 && residual != 0) {
        parallelParam = PackParallelParam(0, 0, 1);
        tailBytes = residual;
    } else if (microCount != 0) {
        parallelParam = PackParallelParam(microCount - 1, 1, 2);
        tailBytes = residual;
    }
    return {loopCount * parallelBytes, loopCount, parallelParam, tailBytes};
}
} // namespace

namespace ops_hccl {
HcclResult ExecOp(const OpParam &param)
{
    char *ctx = static_cast<char *>(param.resCtx);
    std::vector<char> seq(ctx, ctx + param.ctxSize);
    AlgResourceCtx resCtx;
    resCtx.DeSerialize(seq);

    const auto sizeIt = SIZE_TABLE.find(param.dataType);
    CHK_PRT_RET(sizeIt == SIZE_TABLE.end(), HCCL_ERROR("Unsupported data type %d", param.dataType),
        HCCL_E_NOT_SUPPORT);
    const uint64_t dataTypeSize = sizeIt->second;
    const uint64_t recvBytes = param.count * dataTypeSize;
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }

    if (param.rankSize == 1) {
        CHK_RET(HcommLocalCopyOnThread(resCtx.threads[0], param.outputPtr, param.inputPtr, recvBytes));
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(resCtx.ccuKernels.empty() || resCtx.threads.empty(), HCCL_ERROR("Incomplete CCU resources"),
        HCCL_E_INTERNAL);

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t cclAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t cclToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, recvBytes * param.rankSize, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(outputAddr, recvBytes, &outputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(cclAddr, resCtx.localBuffer.size, &cclToken));

    if (resCtx.algorithm == ReduceScatterAlgorithm::DUAL_DIE_PARTIAL ||
        resCtx.algorithm == ReduceScatterAlgorithm::SMALL_CLOS_PARALLEL) {
        const bool dualDie = resCtx.algorithm == ReduceScatterAlgorithm::DUAL_DIE_PARTIAL;
        const bool groupReduce = dualDie && param.rankSize == 16;
        CHK_PRT_RET(resCtx.ccuKernels.size() != (dualDie ? (groupReduce ? 2U : 3U) : 1U) ||
                resCtx.threads.size() != (dualDie ? 2U : 1U),
            HCCL_ERROR("Incomplete partial-reduce resources"), HCCL_E_INTERNAL);

        constexpr uint64_t PREFERRED_TILE_BYTES = 8 * 1024 * 1024;
        uint64_t tileCapacity = recvBytes;
        uint64_t localSourceCount = param.rankSize;
        uint64_t crossSourceCount = 0;
        uint64_t crossPartialAddr = 0;
        uint64_t crossScratchAddr = 0;

        if (dualDie && !groupReduce) {
            localSourceCount = resCtx.localRanks.size();
            crossSourceCount = resCtx.crossPeers.size();
            CHK_PRT_RET(localSourceCount == 0 || crossSourceCount == 0,
                HCCL_ERROR("Invalid dual-die source groups"), HCCL_E_INTERNAL);
            CHK_PRT_RET(resCtx.localBuffer.size <= recvBytes,
                HCCL_ERROR("CCL buffer cannot hold the remote partial"), HCCL_E_INTERNAL);
            const uint64_t scratchSlotCount = 2 * (localSourceCount + crossSourceCount);
            tileCapacity = std::min<uint64_t>(
                PREFERRED_TILE_BYTES, (resCtx.localBuffer.size - recvBytes) / scratchSlotCount);
        }
        tileCapacity = std::min(tileCapacity, recvBytes);
        tileCapacity = (tileCapacity / dataTypeSize) * dataTypeSize;
        CHK_PRT_RET(tileCapacity == 0, HCCL_ERROR("CCL buffer is too small for partial reduction"),
            HCCL_E_INTERNAL);

        const uint64_t localScratchBytes = 2 * localSourceCount * tileCapacity;
        if (dualDie && !groupReduce) {
            const uint64_t crossScratchBytes = 2 * crossSourceCount * tileCapacity;
            crossScratchAddr = cclAddr + localScratchBytes;
            crossPartialAddr = crossScratchAddr + crossScratchBytes;
            CHK_PRT_RET(crossPartialAddr + recvBytes > cclAddr + resCtx.localBuffer.size,
                HCCL_ERROR("Partial-reduce scratch exceeds the CCL buffer"), HCCL_E_INTERNAL);
        } else if (!dualDie) {
            CHK_PRT_RET(localScratchBytes > resCtx.localBuffer.size,
                HCCL_ERROR("Small-message scratch exceeds the CCL buffer"), HCCL_E_INTERNAL);
        }

        const uint64_t fullTileCount = recvBytes / tileCapacity;
        const uint64_t pairCount = fullTileCount / 2;
        const uint64_t pairRepeat = UINT64_MAX - pairCount;
        const uint64_t hasOddTile = fullTileCount % 2;
        const uint64_t tailBytes = recvBytes % tileCapacity;
        const uint64_t sourceOffset = param.myRank * recvBytes;
        auto makePartialArgs = [&](uint64_t partialOutputAddr, uint64_t partialOutputToken,
                                   uint64_t scratchBase, uint64_t sourceCount) {
            std::vector<uint64_t> args = {
                partialOutputAddr, partialOutputToken, inputAddr, inputToken, cclToken, sourceOffset,
                tileCapacity, pairRepeat, hasOddTile, tailBytes};
            for (uint64_t bank = 0; bank < 2; ++bank) {
                for (uint64_t source = 0; source < sourceCount; ++source) {
                    args.push_back(scratchBase + (bank * sourceCount + source) * tileCapacity);
                }
            }
            return args;
        };
        std::vector<uint64_t> localArgs;
        if (groupReduce) {
            CHK_PRT_RET(resCtx.localBuffer.size < recvBytes,
                HCCL_ERROR("CCL buffer cannot hold the cross partial"), HCCL_E_INTERNAL);
            localSourceCount = resCtx.localRanks.size();
            crossSourceCount = resCtx.crossPeers.size();
            CHK_PRT_RET(localSourceCount != 8 || crossSourceCount != 8,
                HCCL_ERROR("Unexpected 2x8 source groups"), HCCL_E_INTERNAL);
            crossPartialAddr = cclAddr;
            const std::vector<uint64_t> groupSize = CalcGroupReduceSize(recvBytes);
            localArgs = {outputAddr, outputToken, inputAddr, inputToken, sourceOffset};
            localArgs.insert(localArgs.end(), groupSize.begin(), groupSize.end());
            const uint64_t firstHalfBytes = (recvBytes / (2 * dataTypeSize)) * dataTypeSize;
            const std::vector<uint64_t> mergeArgs = {
                outputAddr, outputToken, crossPartialAddr, cclToken, 0, firstHalfBytes};
            localArgs.insert(localArgs.end(), mergeArgs.begin(), mergeArgs.end());
        } else {
            localArgs = makePartialArgs(outputAddr, outputToken, cclAddr, localSourceCount);
        }

        if (!dualDie) {
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], localArgs.data(),
                static_cast<uint32_t>(localArgs.size())));
            return HCCL_SUCCESS;
        }

        std::vector<uint64_t> crossArgs;
        if (groupReduce) {
            const std::vector<uint64_t> groupSize = CalcGroupReduceSize(recvBytes);
            crossArgs = {crossPartialAddr, cclToken, inputAddr, inputToken, sourceOffset};
            crossArgs.insert(crossArgs.end(), groupSize.begin(), groupSize.end());
            const uint64_t firstHalfBytes = (recvBytes / (2 * dataTypeSize)) * dataTypeSize;
            const std::vector<uint64_t> mergeArgs = {
                outputAddr, outputToken, crossPartialAddr, cclToken, firstHalfBytes,
                recvBytes - firstHalfBytes};
            crossArgs.insert(crossArgs.end(), mergeArgs.begin(), mergeArgs.end());
        } else {
            crossArgs = makePartialArgs(crossPartialAddr, cclToken, crossScratchAddr, crossSourceCount);
        }

        // Main and slave threads target different IO Dies. The pre/post notify
        // pair brackets both launches without serializing their data paths.
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resCtx.threads[1], 0)));
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], localArgs.data(),
            static_cast<uint32_t>(localArgs.size())));
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[1], resCtx.ccuKernels[1], crossArgs.data(),
            static_cast<uint32_t>(crossArgs.size())));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resCtx.threads[0], 0)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], 0)));
        if (!groupReduce) {
            const std::vector<uint64_t> mergeArgs = {
                outputAddr, outputToken, crossPartialAddr, cclToken, recvBytes};
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[2], mergeArgs.data(),
                static_cast<uint32_t>(mergeArgs.size())));
        }
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == ReduceScatterAlgorithm::DIRECT_MESH) {
        CHK_PRT_RET(resCtx.ccuKernels.size() != 1, HCCL_ERROR("Incomplete direct-mesh resources"),
            HCCL_E_INTERNAL);
        for (uint64_t chunkOffset = 0; chunkOffset < recvBytes; chunkOffset += MAX_DATA_SIZE) {
            const uint64_t chunkBytes = std::min<uint64_t>(MAX_DATA_SIZE, recvBytes - chunkOffset);
            const uint64_t sourceOffset = param.myRank * recvBytes + chunkOffset;
            const std::vector<uint64_t> taskArgs = {
                outputAddr + chunkOffset, outputToken, inputAddr, inputToken, sourceOffset, chunkBytes};
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size())));
        }
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == ReduceScatterAlgorithm::STAGING_MESH) {
        CHK_PRT_RET(resCtx.ccuKernels.size() != 1, HCCL_ERROR("Incomplete staging-mesh resources"),
            HCCL_E_INTERNAL);
        uint64_t slotStride = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size / param.rankSize);
        slotStride = (slotStride / dataTypeSize) * dataTypeSize;
        CHK_PRT_RET(slotStride == 0, HCCL_ERROR("CCL buffer is too small"), HCCL_E_INTERNAL);

        for (uint64_t chunkOffset = 0; chunkOffset < recvBytes; chunkOffset += slotStride) {
            const uint64_t chunkBytes = std::min(slotStride, recvBytes - chunkOffset);
            std::vector<uint64_t> taskArgs = {outputAddr + chunkOffset, outputToken, inputToken, cclAddr, cclToken,
                chunkBytes, param.myRank * slotStride};
            for (uint32_t peer = 0; peer < param.rankSize; ++peer) {
                taskArgs.push_back(inputAddr + peer * recvBytes + chunkOffset);
            }
            for (uint32_t sourceRank = 0; sourceRank < param.rankSize; ++sourceRank) {
                taskArgs.push_back(cclAddr + sourceRank * slotStride);
            }
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], taskArgs.data(),
                static_cast<uint32_t>(taskArgs.size())));
        }
        return HCCL_SUCCESS;
    }

    const bool hierarchicalStaging = resCtx.algorithm == ReduceScatterAlgorithm::HIERARCHICAL_STAGING;
    CHK_PRT_RET((resCtx.algorithm != ReduceScatterAlgorithm::HIERARCHICAL && !hierarchicalStaging) ||
            resCtx.ccuKernels.size() != 2 || resCtx.localRanks.empty() || resCtx.targetRanks.empty(),
        HCCL_ERROR("Incomplete hierarchical resources"), HCCL_E_INTERNAL);
    const uint32_t localSize = static_cast<uint32_t>(resCtx.localRanks.size());
    const uint32_t targetCount = static_cast<uint32_t>(resCtx.targetRanks.size());
    constexpr uint64_t STAGING_BANK_COUNT = 2;
    const uint64_t stagingSlotCount = hierarchicalStaging ? STAGING_BANK_COUNT * localSize : 0;
    const uint64_t partialSlotWidth = hierarchicalStaging ? STAGING_BANK_COUNT : 1;
    const uint64_t totalSlotCount =
        stagingSlotCount + partialSlotWidth * std::max<uint64_t>(targetCount - 1, 1);
    uint64_t slotStride = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size / totalSlotCount);
    slotStride = (slotStride / dataTypeSize) * dataTypeSize;
    CHK_PRT_RET(slotStride == 0, HCCL_ERROR("CCL buffer is too small"), HCCL_E_INTERNAL);

    const uint64_t chunkCapacity = hierarchicalStaging ? STAGING_BANK_COUNT * slotStride : slotStride;
    for (uint64_t chunkOffset = 0; chunkOffset < recvBytes; chunkOffset += chunkCapacity) {
        const uint64_t chunkBytes = std::min(chunkCapacity, recvBytes - chunkOffset);
        std::vector<uint64_t> localArgs = {inputAddr, inputToken};
        if (hierarchicalStaging) {
            localArgs.push_back(cclToken);
        }
        localArgs.push_back(chunkBytes);
        if (hierarchicalStaging) {
            // Keep both banks close to the same size. The local kernel pipelines
            // bank 1 traffic with bank 0 reduction and, across targets, the next
            // bank 0 traffic with the current bank 1 reduction. An oversized
            // first bank leaves too little work in the second bank to hide the
            // following transfer.
            const uint64_t tileBytes =
                std::min(slotStride, ((chunkBytes + dataTypeSize) / (2 * dataTypeSize)) * dataTypeSize);
            localArgs.push_back(tileBytes);
            localArgs.push_back(chunkBytes - tileBytes);
        }
        for (uint32_t targetRank : resCtx.targetRanks) {
            localArgs.push_back(targetRank * recvBytes + chunkOffset);
        }
        if (hierarchicalStaging) {
            for (uint32_t bank = 0; bank < STAGING_BANK_COUNT; ++bank) {
                for (uint32_t sourceIdx = 0; sourceIdx < localSize; ++sourceIdx) {
                    localArgs.push_back(cclAddr + (bank * localSize + sourceIdx) * slotStride);
                }
            }
        }
        localArgs.push_back(outputAddr + chunkOffset);
        for (uint32_t targetIdx = 1; targetIdx < targetCount; ++targetIdx) {
            localArgs.push_back(
                cclAddr + (stagingSlotCount + partialSlotWidth * (targetIdx - 1)) * slotStride);
        }
        localArgs.push_back(outputToken);
        for (uint32_t targetIdx = 1; targetIdx < targetCount; ++targetIdx) {
            localArgs.push_back(cclToken);
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], localArgs.data(),
            static_cast<uint32_t>(localArgs.size())));

        std::vector<uint64_t> crossArgs = {outputAddr + chunkOffset, outputToken, cclToken, chunkBytes};
        for (uint32_t targetIdx : resCtx.crossSendTargetIndices) {
            crossArgs.push_back(
                cclAddr + (stagingSlotCount + partialSlotWidth * (targetIdx - 1)) * slotStride);
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[1], crossArgs.data(),
            static_cast<uint32_t>(crossArgs.size())));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
