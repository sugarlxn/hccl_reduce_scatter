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
    const uint64_t stagingSlotCount = hierarchicalStaging ? localSize : 0;
    const uint64_t totalSlotCount = stagingSlotCount + std::max<uint64_t>(targetCount - 1, 1);
    uint64_t slotStride = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size / totalSlotCount);
    slotStride = (slotStride / dataTypeSize) * dataTypeSize;
    CHK_PRT_RET(slotStride == 0, HCCL_ERROR("CCL buffer is too small"), HCCL_E_INTERNAL);

    for (uint64_t chunkOffset = 0; chunkOffset < recvBytes; chunkOffset += slotStride) {
        const uint64_t chunkBytes = std::min(slotStride, recvBytes - chunkOffset);
        std::vector<uint64_t> localArgs = {inputAddr, inputToken};
        if (hierarchicalStaging) {
            localArgs.push_back(cclToken);
        }
        localArgs.push_back(chunkBytes);
        for (uint32_t targetRank : resCtx.targetRanks) {
            localArgs.push_back(targetRank * recvBytes + chunkOffset);
        }
        if (hierarchicalStaging) {
            for (uint32_t sourceIdx = 0; sourceIdx < localSize; ++sourceIdx) {
                localArgs.push_back(cclAddr + sourceIdx * slotStride);
            }
        }
        localArgs.push_back(outputAddr + chunkOffset);
        for (uint32_t targetIdx = 1; targetIdx < targetCount; ++targetIdx) {
            localArgs.push_back(cclAddr + (stagingSlotCount + targetIdx - 1) * slotStride);
        }
        localArgs.push_back(outputToken);
        for (uint32_t targetIdx = 1; targetIdx < targetCount; ++targetIdx) {
            localArgs.push_back(cclToken);
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[0], localArgs.data(),
            static_cast<uint32_t>(localArgs.size())));

        std::vector<uint64_t> crossArgs = {outputAddr + chunkOffset, outputToken, cclToken, chunkBytes};
        for (uint32_t targetIdx : resCtx.crossSendTargetIndices) {
            crossArgs.push_back(cclAddr + (stagingSlotCount + targetIdx - 1) * slotStride);
        }
        CHK_RET_CCU(HcommCcuKernelLaunch(resCtx.threads[0], resCtx.ccuKernels[1], crossArgs.data(),
            static_cast<uint32_t>(crossArgs.size())));
    }

    return HCCL_SUCCESS;
}
} // namespace ops_hccl
