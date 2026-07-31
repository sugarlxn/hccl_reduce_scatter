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

    if (resCtx.algorithm == ReduceScatterAlgorithm::SMALL_STAGING_TREE) {
        CHK_PRT_RET(resCtx.ccuKernels.size() != 1 || resCtx.threads.size() != 1,
            HCCL_ERROR("Incomplete small staging-tree resources"), HCCL_E_INTERNAL);
        CHK_PRT_RET(recvBytes > resCtx.localBuffer.size / param.rankSize,
            HCCL_ERROR("Small staging-tree scratch exceeds the CCL buffer"), HCCL_E_INTERNAL);

        const uint64_t sourceOffset = param.myRank * recvBytes;
        std::vector<uint64_t> taskArgs = {
            outputAddr, outputToken, inputAddr, inputToken, cclToken, sourceOffset, recvBytes};
        for (uint64_t source = 0; source < param.rankSize; ++source) {
            taskArgs.push_back(cclAddr + source * recvBytes);
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size())));
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == ReduceScatterAlgorithm::DUAL_DIE_PARTIAL ||
        resCtx.algorithm == ReduceScatterAlgorithm::SMALL_CLOS_PARALLEL) {
        const bool dualDie = resCtx.algorithm == ReduceScatterAlgorithm::DUAL_DIE_PARTIAL;
        CHK_PRT_RET(resCtx.ccuKernels.size() != (dualDie ? 4U : 1U) ||
                resCtx.threads.size() != (dualDie ? 2U : 1U),
            HCCL_ERROR("Incomplete partial-reduce resources"), HCCL_E_INTERNAL);

        constexpr uint64_t PREFERRED_TILE_BYTES = 8 * 1024 * 1024;
        uint64_t tileCapacity = recvBytes;
        uint64_t localSourceCount = param.rankSize;
        uint64_t crossSourceCount = 0;
        uint64_t crossPartialAddr = 0;
        uint64_t crossScratchAddr = 0;

        if (dualDie) {
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
        if (dualDie) {
            const uint64_t crossScratchBytes = 2 * crossSourceCount * tileCapacity;
            crossScratchAddr = cclAddr + localScratchBytes;
            crossPartialAddr = crossScratchAddr + crossScratchBytes;
            CHK_PRT_RET(crossPartialAddr + recvBytes > cclAddr + resCtx.localBuffer.size,
                HCCL_ERROR("Partial-reduce scratch exceeds the CCL buffer"), HCCL_E_INTERNAL);
        } else {
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
        const std::vector<uint64_t> localArgs =
            makePartialArgs(outputAddr, outputToken, cclAddr, localSourceCount);

        if (!dualDie) {
            CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], localArgs.data(),
                static_cast<uint32_t>(localArgs.size())));
            return HCCL_SUCCESS;
        }

        const std::vector<uint64_t> crossArgs =
            makePartialArgs(crossPartialAddr, cclToken, crossScratchAddr, crossSourceCount);

        // Main and slave threads target different IO Dies. Keep the proven
        // partial-reduce path unchanged, then use a two-way barrier before the
        // two Dies merge disjoint halves of the result.
        constexpr uint32_t START_NOTIFY = 0;
        constexpr uint32_t SECONDARY_NOTIFY = 1;
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(
                resCtx.threads[0], resCtx.threads[1], START_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThreadWithDefaultTimeout(
                resCtx.threads[1], START_NOTIFY)));
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], localArgs.data(),
            static_cast<uint32_t>(localArgs.size())));
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[1], resCtx.ccuKernels[1], crossArgs.data(),
            static_cast<uint32_t>(crossArgs.size())));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(
                resCtx.threads[0], resCtx.threads[1], SECONDARY_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThreadWithDefaultTimeout(
                resCtx.threads[1], SECONDARY_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(
                resCtx.threads[1], resCtx.threads[0], START_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThreadWithDefaultTimeout(
                resCtx.threads[0], START_NOTIFY)));

        const uint64_t localMergeBytes = (param.count / 2) * dataTypeSize;
        const uint64_t crossMergeBytes = recvBytes - localMergeBytes;
        const std::vector<uint64_t> localMergeArgs = {
            outputAddr, outputToken, crossPartialAddr, cclToken, localMergeBytes};
        const std::vector<uint64_t> crossMergeArgs = {
            outputAddr + localMergeBytes, outputToken,
            crossPartialAddr + localMergeBytes, cclToken, crossMergeBytes};
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[2],
            localMergeArgs.data(), static_cast<uint32_t>(localMergeArgs.size())));
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[1], resCtx.ccuKernels[3],
            crossMergeArgs.data(), static_cast<uint32_t>(crossMergeArgs.size())));

        // The operation is ordered by the main thread, so do not let it retire
        // before the slave Die has completed its half.
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(
                resCtx.threads[1], resCtx.threads[0], SECONDARY_NOTIFY)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThreadWithDefaultTimeout(
                resCtx.threads[0], SECONDARY_NOTIFY)));
        return HCCL_SUCCESS;
    }

    if (resCtx.algorithm == ReduceScatterAlgorithm::STRIPED_SINGLE_DIE) {
        CHK_PRT_RET(resCtx.ccuKernels.size() != 1 || resCtx.threads.size() != 1,
            HCCL_ERROR("Incomplete striped single-die resources"), HCCL_E_INTERNAL);

        const uint64_t stripeCount =
            std::min<uint64_t>(param.rankSize, MAX_SINGLE_DIE_STRIPES);
        const uint64_t sourceOffset = param.myRank * recvBytes;
        std::vector<uint64_t> taskArgs = {
            outputAddr, outputToken, inputAddr, inputToken, sourceOffset};
        const uint64_t stripeElements = param.count / stripeCount;
        const uint64_t largeStripeCount = param.count % stripeCount;
        uint64_t stripeOffset = 0;
        for (uint64_t stripe = 0; stripe < stripeCount; ++stripe) {
            const uint64_t stripeBytes =
                (stripeElements + (stripe < largeStripeCount ? 1U : 0U)) * dataTypeSize;
            taskArgs.push_back(stripeOffset);
            taskArgs.push_back(stripeBytes);
            stripeOffset += stripeBytes;
        }
        CHK_PRT_RET(stripeOffset != recvBytes, HCCL_ERROR("Invalid stripe partition"),
            HCCL_E_INTERNAL);
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], taskArgs.data(),
            static_cast<uint32_t>(taskArgs.size())));
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
            resCtx.ccuKernels.size() != 2 || resCtx.threads.size() != 2 ||
            resCtx.localRanks.empty() || resCtx.targetRanks.empty(),
        HCCL_ERROR("Incomplete hierarchical resources"), HCCL_E_INTERNAL);
    const uint32_t localSize = static_cast<uint32_t>(resCtx.localRanks.size());
    const uint32_t targetCount = static_cast<uint32_t>(resCtx.targetRanks.size());
    constexpr uint64_t STAGING_BANK_COUNT = 2;
    constexpr uint64_t PIPELINE_BANK_COUNT = 2;
    constexpr uint64_t PIPELINE_TILE_BYTES = 4 * 1024 * 1024;
    const uint64_t stagingSlotCount = hierarchicalStaging ? STAGING_BANK_COUNT * localSize : 0;
    const uint64_t partialTargetCount = std::max<uint64_t>(targetCount - 1, 1);
    const uint64_t partialSlotCount = PIPELINE_BANK_COUNT * partialTargetCount;
    const uint64_t totalSlotCount = stagingSlotCount + partialSlotCount;
    const uint64_t tileCapacity = std::min<uint64_t>(
        PIPELINE_TILE_BYTES, resCtx.localBuffer.size / totalSlotCount);
    const uint64_t alignedTileCapacity = (tileCapacity / dataTypeSize) * dataTypeSize;
    CHK_PRT_RET(alignedTileCapacity == 0, HCCL_ERROR("CCL buffer is too small"), HCCL_E_INTERNAL);

    const uint64_t tileCount = (recvBytes + alignedTileCapacity - 1) / alignedTileCapacity;
    for (uint64_t tileIndex = 0; tileIndex < tileCount; ++tileIndex) {
        const uint64_t bank = tileIndex % PIPELINE_BANK_COUNT;
        const uint64_t chunkOffset = tileIndex * alignedTileCapacity;
        const uint64_t chunkBytes = std::min(alignedTileCapacity, recvBytes - chunkOffset);

        // Reusing a bank is legal only after the layer-1 kernel has completed
        // its symmetric channel handshake, proving the remote peer has also
        // stopped reading our partial from that bank.
        if (tileIndex >= PIPELINE_BANK_COUNT) {
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resCtx.threads[0], bank)));
        }

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
                ((chunkBytes + dataTypeSize) / (2 * dataTypeSize)) * dataTypeSize;
            localArgs.push_back(tileBytes);
            localArgs.push_back(chunkBytes - tileBytes);
        }
        for (uint32_t targetRank : resCtx.targetRanks) {
            localArgs.push_back(targetRank * recvBytes + chunkOffset);
        }
        if (hierarchicalStaging) {
            for (uint32_t bank = 0; bank < STAGING_BANK_COUNT; ++bank) {
                for (uint32_t sourceIdx = 0; sourceIdx < localSize; ++sourceIdx) {
                    localArgs.push_back(
                        cclAddr + (bank * localSize + sourceIdx) * alignedTileCapacity);
                }
            }
        }
        localArgs.push_back(outputAddr + chunkOffset);
        for (uint32_t targetIdx = 1; targetIdx < targetCount; ++targetIdx) {
            const uint64_t partialSlot =
                stagingSlotCount + bank * partialTargetCount + targetIdx - 1;
            localArgs.push_back(cclAddr + partialSlot * alignedTileCapacity);
        }
        localArgs.push_back(outputToken);
        for (uint32_t targetIdx = 1; targetIdx < targetCount; ++targetIdx) {
            localArgs.push_back(cclToken);
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], localArgs.data(),
            static_cast<uint32_t>(localArgs.size())));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[0], resCtx.threads[1], bank)));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThreadWithDefaultTimeout(resCtx.threads[1], bank)));

        std::vector<uint64_t> crossArgs = {outputAddr + chunkOffset, outputToken, cclToken, chunkBytes};
        for (uint32_t targetIdx : resCtx.crossSendTargetIndices) {
            const uint64_t partialSlot =
                stagingSlotCount + bank * partialTargetCount + targetIdx - 1;
            crossArgs.push_back(cclAddr + partialSlot * alignedTileCapacity);
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[1], resCtx.ccuKernels[1], crossArgs.data(),
            static_cast<uint32_t>(crossArgs.size())));
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(resCtx.threads[1], resCtx.threads[0], bank)));
    }

    // Join the last one or two in-flight layer-1 tiles onto the user stream.
    const uint64_t firstOutstanding =
        tileCount > PIPELINE_BANK_COUNT ? tileCount - PIPELINE_BANK_COUNT : 0;
    for (uint64_t tileIndex = firstOutstanding; tileIndex < tileCount; ++tileIndex) {
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyWaitOnThreadWithDefaultTimeout(
                resCtx.threads[0], tileIndex % PIPELINE_BANK_COUNT)));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
