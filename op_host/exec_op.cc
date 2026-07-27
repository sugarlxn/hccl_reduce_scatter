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
    // 反序列化
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

    // One staging slot per source rank. Each primitive remains below 256 MiB.
    uint64_t slotStride = std::min<uint64_t>(MAX_DATA_SIZE, resCtx.localBuffer.size / param.rankSize);
    slotStride = (slotStride / dataTypeSize) * dataTypeSize;
    CHK_PRT_RET(slotStride == 0, HCCL_ERROR("CCL buffer is too small"), HCCL_E_INTERNAL);

    const uint64_t inputAddr = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t outputAddr = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t cclAddr = reinterpret_cast<uint64_t>(resCtx.localBuffer.addr);
    uint64_t inputToken = 0;
    uint64_t outputToken = 0;
    uint64_t cclToken = 0;
    CHK_RET_CCU(HcommCcuGetMemToken(inputAddr, recvBytes * param.rankSize, &inputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(outputAddr, recvBytes, &outputToken));
    CHK_RET_CCU(HcommCcuGetMemToken(cclAddr, resCtx.localBuffer.size, &cclToken));

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
} // namespace ops_hccl
