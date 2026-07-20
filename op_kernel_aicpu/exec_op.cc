/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "custom.h"
#include "log.h"
#include "exec_op.h"

namespace ops_hccl {
namespace {
constexpr uint32_t ACK_NOTIFY_IDX = 0;
constexpr uint32_t DATA_NOTIFY_IDX = 1;

const ChannelInfo *GetChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const auto &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}
} // namespace

HcclResult ExecOp(const OpParam &param, const AlgResourceCtx &resCtx)
{
    CHK_PRT_RET(resCtx.threads.empty(), HCCL_ERROR("No AICPU TS thread"), HCCL_E_INTERNAL);
    CHK_PRT_RET(param.dataType != HCCL_DATA_TYPE_FP32 || param.reduceType != HCCL_REDUCE_SUM,
        HCCL_ERROR("Unsupported datatype or reduction"), HCCL_E_NOT_SUPPORT);
    const ThreadHandle thread = resCtx.threads[0];
    constexpr uint64_t typeSize = sizeof(float);
    const uint64_t recvBytes = param.count * typeSize;
    if (recvBytes == 0) {
        return HCCL_SUCCESS;
    }
    const char *input = static_cast<const char *>(param.inputPtr);
    char *output = static_cast<char *>(param.outputPtr);
    const char *localPart = input + static_cast<uint64_t>(param.myRank) * recvBytes;
    CHK_RET(HcommLocalCopyOnThread(thread, output, localPart, recvBytes));
    if (param.rankSize == 1) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < typeSize * param.rankSize,
        HCCL_ERROR("Local HCCL buffer is invalid"), HCCL_E_INTERNAL);
    // Each source owns a disjoint slot. Different sources can progress independently across channels.
    const uint64_t slotCapacity = resCtx.localBuffer.size / param.rankSize;
    const uint64_t chunkCapacity = slotCapacity - slotCapacity % typeSize;

    // source rank 固定按 0..rankSize-1 排序；每块收到所有 ACK 后才复用工作区。
    for (uint32_t sourceRank = 0; sourceRank < param.rankSize; ++sourceRank) {
        if (sourceRank == param.myRank) {
            for (uint64_t offset = 0; offset < recvBytes; offset += chunkCapacity) {
                const uint64_t chunkBytes = (recvBytes - offset < chunkCapacity) ? recvBytes - offset : chunkCapacity;
                for (const auto &channel : resCtx.channels) {
                    const uint64_t remoteSlotCapacity = channel.remoteCclMem.size / param.rankSize;
                    CHK_PRT_RET(channel.remoteCclMem.addr == nullptr || remoteSlotCapacity < chunkBytes,
                        HCCL_ERROR("Remote HCCL buffer is too small"), HCCL_E_INTERNAL);
                    const char *src = input + static_cast<uint64_t>(channel.remoteRank) * recvBytes + offset;
                    char *remoteSlot = static_cast<char *>(channel.remoteCclMem.addr) +
                        static_cast<uint64_t>(sourceRank) * remoteSlotCapacity;
                    CHK_RET(HcommWriteWithNotifyOnThread(
                        thread, channel.handle, remoteSlot, src, chunkBytes, DATA_NOTIFY_IDX));
                }
                for (const auto &channel : resCtx.channels) {
                    CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel.handle, ACK_NOTIFY_IDX, CUSTOM_TIMEOUT));
                }
            }
            continue;
        }
        const ChannelInfo *channel = GetChannel(resCtx, sourceRank);
        CHK_PRT_RET(channel == nullptr, HCCL_ERROR("Missing channel to rank %u", sourceRank), HCCL_E_INTERNAL);
        const char *localSlot = static_cast<const char *>(resCtx.localBuffer.addr) +
            static_cast<uint64_t>(sourceRank) * slotCapacity;
        for (uint64_t offset = 0; offset < recvBytes; offset += chunkCapacity) {
            const uint64_t chunkBytes = (recvBytes - offset < chunkCapacity) ? recvBytes - offset : chunkCapacity;
            CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));
            CHK_RET(HcommLocalReduceOnThread(thread, output + offset, localSlot,
                chunkBytes / typeSize, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, ACK_NOTIFY_IDX));
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
