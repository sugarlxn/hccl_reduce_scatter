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
constexpr uint32_t LOCAL_RANK_SIZE = 8;
constexpr uint32_t GLOBAL_RANK_SIZE = 16;
constexpr uint32_t ROUND_NUM = 3;
constexpr uint32_t DATA_NOTIFY_IDX = 0;
constexpr uint32_t READY_NOTIFY_IDX = 1;

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
    CHK_PRT_RET(param.rankSize != GLOBAL_RANK_SIZE || resCtx.localRanks.size() != LOCAL_RANK_SIZE ||
            resCtx.localRankIndex >= LOCAL_RANK_SIZE || resCtx.partnerRank >= GLOBAL_RANK_SIZE,
        HCCL_ERROR("Invalid hierarchical topology"), HCCL_E_INTERNAL);

    const uint64_t halfInputBytes = recvBytes * (GLOBAL_RANK_SIZE / 2);
    const uint64_t quarterInputBytes = recvBytes * (GLOBAL_RANK_SIZE / 4);
    const uint64_t workspaceBytes = halfInputBytes + quarterInputBytes;
    CHK_PRT_RET(resCtx.localBuffer.addr == nullptr || resCtx.localBuffer.size < workspaceBytes,
        HCCL_ERROR("HCCL buffer is too small, need %lu bytes", workspaceBytes), HCCL_E_INTERNAL);

    const char *input = static_cast<const char *>(param.inputPtr);
    char *output = static_cast<char *>(param.outputPtr);
    char *bufferA = static_cast<char *>(resCtx.localBuffer.addr);
    char *bufferB = bufferA + halfInputBytes;
    // First reduce corresponding source ranks across servers. Each side sends the target half owned by its peer.
    const ChannelInfo *crossChannel = GetChannel(resCtx, resCtx.partnerRank);
    CHK_PRT_RET(crossChannel == nullptr || crossChannel->remoteCclMem.addr == nullptr ||
            crossChannel->remoteCclMem.size < workspaceBytes,
        HCCL_ERROR("Invalid cross-server partner channel"), HCCL_E_INTERNAL);
    const bool firstServer = param.myRank < LOCAL_RANK_SIZE;
    const uint64_t ownedOffset = firstServer ? 0 : halfInputBytes;
    const uint64_t sendOffset = firstServer ? halfInputBytes : 0;
    CHK_RET(HcommWriteOnThread(thread, crossChannel->handle, crossChannel->remoteCclMem.addr,
        input + sendOffset, halfInputBytes));
    CHK_RET(HcommChannelNotifyRecordOnThread(thread, crossChannel->handle, DATA_NOTIFY_IDX));
    CHK_RET(HcommChannelNotifyWaitOnThread(thread, crossChannel->handle, DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));
    CHK_RET(HcommLocalReduceOnThread(thread, bufferA, input + ownedOffset, halfInputBytes / typeSize,
        HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));

    // Recursive halving inside the server. Round 0 uses bufferB. Later rounds reuse discarded bufferA regions;
    // a READY handshake orders each remote write after all earlier work on the destination rank.
    const char *activeSrc = bufferA;
    uint32_t activeCount = LOCAL_RANK_SIZE;
    uint64_t activeOffset = 0;
    for (uint32_t round = 0; round < ROUND_NUM; ++round) {
        const uint32_t mask = LOCAL_RANK_SIZE >> (round + 1);
        const uint32_t halfCount = activeCount / 2;
        const bool keepUpper = (resCtx.localRankIndex & mask) != 0;
        const uint32_t keepOffset = keepUpper ? halfCount : 0;
        const uint32_t sendOffsetInChunk = keepUpper ? 0 : halfCount;
        const uint32_t partnerIndex = resCtx.localRankIndex ^ mask;
        const ChannelInfo *channel = GetChannel(resCtx, resCtx.localRanks[partnerIndex]);
        CHK_PRT_RET(channel == nullptr || channel->remoteCclMem.addr == nullptr ||
                channel->remoteCclMem.size < workspaceBytes,
            HCCL_ERROR("Invalid local partner channel"), HCCL_E_INTERNAL);

        char *recvDst = nullptr;
        uint64_t recvOffset = 0;
        if (round == 0) {
            recvDst = bufferB;
            recvOffset = halfInputBytes;
        } else if (round == 1) {
            activeOffset = (resCtx.localRankIndex & (LOCAL_RANK_SIZE / 2)) != 0 ? 0 : quarterInputBytes;
            recvDst = bufferA + activeOffset;
            recvOffset = activeOffset;
        } else {
            recvDst = bufferA + activeOffset + 2 * recvBytes;
            recvOffset = activeOffset + 2 * recvBytes;
        }

        const uint64_t halfBytes = static_cast<uint64_t>(halfCount) * recvBytes;
        const char *sendSrc = activeSrc + static_cast<uint64_t>(sendOffsetInChunk) * recvBytes;
        char *remoteRecvDst = static_cast<char *>(channel->remoteCclMem.addr) + recvOffset;
        if (round > 0) {
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, READY_NOTIFY_IDX));
            CHK_RET(HcommChannelNotifyWaitOnThread(
                thread, channel->handle, READY_NOTIFY_IDX, CUSTOM_TIMEOUT));
        }
        CHK_RET(HcommWriteOnThread(thread, channel->handle, remoteRecvDst, sendSrc, halfBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(thread, channel->handle, DATA_NOTIFY_IDX));
        CHK_RET(HcommChannelNotifyWaitOnThread(thread, channel->handle, DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));

        const char *keepSrc = activeSrc + static_cast<uint64_t>(keepOffset) * recvBytes;
        if (round == ROUND_NUM - 1) {
            CHK_RET(HcommLocalCopyOnThread(thread, output, keepSrc, recvBytes));
            CHK_RET(HcommLocalReduceOnThread(
                thread, output, recvDst, param.count, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        } else {
            CHK_RET(HcommLocalReduceOnThread(
                thread, recvDst, keepSrc, halfBytes / typeSize, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
            activeSrc = recvDst;
            activeCount = halfCount;
        }
    }
    return HCCL_SUCCESS;
}
} // namespace ops_hccl
