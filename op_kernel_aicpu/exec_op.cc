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
constexpr uint64_t LARGE_MESSAGE_RECV_BYTES = 1024 * 1024;
constexpr uint32_t LARGE_MESSAGE_TILE_NUM = 2;

const ChannelInfo *GetChannel(const AlgResourceCtx &resCtx, uint32_t remoteRank)
{
    for (const auto &channel : resCtx.channels) {
        if (channel.remoteRank == remoteRank) {
            return &channel;
        }
    }
    return nullptr;
}

// Split K8 into a degree-4 and a degree-3 graph. Peers in the first graph receive into the four spare chunks;
// after that wave their source chunks are dead and provide the receive area for the second graph.
bool IsFirstWavePeer(uint32_t localIndex, uint32_t peerIndex)
{
    const uint32_t distance = (peerIndex + LOCAL_RANK_SIZE - localIndex) % LOCAL_RANK_SIZE;
    return distance == 1 || distance == 2 || distance == LOCAL_RANK_SIZE - 1 ||
        distance == LOCAL_RANK_SIZE - 2;
}

uint32_t FirstWaveSlot(uint32_t ownerIndex, uint32_t peerIndex)
{
    uint32_t slot = 0;
    for (uint32_t peer = 0; peer < peerIndex; ++peer) {
        if (peer != ownerIndex && IsFirstWavePeer(ownerIndex, peer)) {
            ++slot;
        }
    }
    return slot;
}

uint32_t SecondWaveSlot(uint32_t ownerIndex, uint32_t peerIndex)
{
    uint32_t ordinal = 0;
    for (uint32_t peer = 0; peer < peerIndex; ++peer) {
        if (peer != ownerIndex && !IsFirstWavePeer(ownerIndex, peer)) {
            ++ordinal;
        }
    }
    for (uint32_t slot = 0; slot < LOCAL_RANK_SIZE; ++slot) {
        if (slot != ownerIndex && IsFirstWavePeer(ownerIndex, slot)) {
            if (ordinal == 0) {
                return slot;
            }
            --ordinal;
        }
    }
    return LOCAL_RANK_SIZE;
}

HcclResult ExecLargeMessage(const AlgResourceCtx &resCtx, uint64_t recvBytes, char *output, char *bufferA,
    char *bufferB, uint64_t halfInputBytes)
{
    CHK_PRT_RET(resCtx.threads.size() < LOCAL_RANK_SIZE,
        HCCL_ERROR("Large-message path needs %u AICPU TS threads", LOCAL_RANK_SIZE), HCCL_E_INTERNAL);
    const ThreadHandle mainThread = resCtx.threads[0];

    // bufferA contains eight cross-server pair reductions. All output reductions remain on mainThread and use
    // a fixed peer order, preserving deterministic floating-point results.
    CHK_RET(HcommLocalCopyOnThread(
        mainThread, output, bufferA + static_cast<uint64_t>(resCtx.localRankIndex) * recvBytes, recvBytes));

    uint32_t workerForPeer[LOCAL_RANK_SIZE] = {};
    uint32_t nextWorker = 1;
    for (uint32_t peer = 0; peer < LOCAL_RANK_SIZE; ++peer) {
        if (peer != resCtx.localRankIndex) {
            workerForPeer[peer] = nextWorker++;
        }
    }

    // Wave 1 exchanges four peers into the spare quarter-input region.
    uint32_t firstWaveCount = 0;
    for (uint32_t peer = 0; peer < LOCAL_RANK_SIZE; ++peer) {
        if (peer == resCtx.localRankIndex || !IsFirstWavePeer(resCtx.localRankIndex, peer)) {
            continue;
        }
        ++firstWaveCount;
        const uint32_t workerIdx = workerForPeer[peer];
        const ThreadHandle worker = resCtx.threads[workerIdx];
        const ChannelInfo *channel = GetChannel(resCtx, resCtx.localRanks[peer]);
        CHK_PRT_RET(channel == nullptr || channel->remoteCclMem.addr == nullptr ||
                channel->remoteCclMem.size < halfInputBytes + 4 * recvBytes,
            HCCL_ERROR("Invalid first-wave local channel"), HCCL_E_INTERNAL);
        const uint32_t remoteSlot = FirstWaveSlot(peer, resCtx.localRankIndex);
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        CHK_RET(HcommWriteOnThread(worker, channel->handle,
            static_cast<char *>(channel->remoteCclMem.addr) + halfInputBytes + remoteSlot * recvBytes,
            bufferA + static_cast<uint64_t>(peer) * recvBytes, recvBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel->handle, DATA_NOTIFY_IDX));
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel->handle, DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, workerIdx));
    }
    CHK_PRT_RET(firstWaveCount != 4, HCCL_ERROR("Invalid first-wave peer count %u", firstWaveCount),
        HCCL_E_INTERNAL);
    for (uint32_t peer = 0; peer < LOCAL_RANK_SIZE; ++peer) {
        if (peer != resCtx.localRankIndex && IsFirstWavePeer(resCtx.localRankIndex, peer)) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, workerForPeer[peer], CUSTOM_TIMEOUT));
        }
    }

    // Wave 2 receives into three source chunks freed by wave 1. While it runs, mainThread reduces wave-1 data.
    uint32_t secondWaveCount = 0;
    for (uint32_t peer = 0; peer < LOCAL_RANK_SIZE; ++peer) {
        if (peer == resCtx.localRankIndex || IsFirstWavePeer(resCtx.localRankIndex, peer)) {
            continue;
        }
        ++secondWaveCount;
        const uint32_t workerIdx = workerForPeer[peer];
        const ThreadHandle worker = resCtx.threads[workerIdx];
        const ChannelInfo *channel = GetChannel(resCtx, resCtx.localRanks[peer]);
        const uint32_t remoteSlot = SecondWaveSlot(peer, resCtx.localRankIndex);
        CHK_PRT_RET(channel == nullptr || channel->remoteCclMem.addr == nullptr ||
                channel->remoteCclMem.size < halfInputBytes || remoteSlot >= LOCAL_RANK_SIZE,
            HCCL_ERROR("Invalid second-wave local channel"), HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        // A local wave-1 barrier is insufficient: the destination rank may still be reading the chunk that
        // this write is about to reuse. The symmetric READY handshake orders both writes after both ranks have
        // completed all wave-1 transfers, making the discarded source chunks safe remote receive slots.
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel->handle, READY_NOTIFY_IDX));
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel->handle, READY_NOTIFY_IDX, CUSTOM_TIMEOUT));
        CHK_RET(HcommWriteOnThread(worker, channel->handle,
            static_cast<char *>(channel->remoteCclMem.addr) + static_cast<uint64_t>(remoteSlot) * recvBytes,
            bufferA + static_cast<uint64_t>(peer) * recvBytes, recvBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel->handle, DATA_NOTIFY_IDX));
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel->handle, DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, workerIdx));
    }
    CHK_PRT_RET(secondWaveCount != 3, HCCL_ERROR("Invalid second-wave peer count %u", secondWaveCount),
        HCCL_E_INTERNAL);

    for (uint32_t peer = 0; peer < LOCAL_RANK_SIZE; ++peer) {
        if (peer != resCtx.localRankIndex && IsFirstWavePeer(resCtx.localRankIndex, peer)) {
            CHK_RET(HcommLocalReduceOnThread(mainThread, output,
                bufferB + static_cast<uint64_t>(FirstWaveSlot(resCtx.localRankIndex, peer)) * recvBytes,
                recvBytes / sizeof(float), HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        }
    }
    for (uint32_t peer = 0; peer < LOCAL_RANK_SIZE; ++peer) {
        if (peer != resCtx.localRankIndex && !IsFirstWavePeer(resCtx.localRankIndex, peer)) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, workerForPeer[peer], CUSTOM_TIMEOUT));
        }
    }
    for (uint32_t peer = 0; peer < LOCAL_RANK_SIZE; ++peer) {
        if (peer != resCtx.localRankIndex && !IsFirstWavePeer(resCtx.localRankIndex, peer)) {
            const uint32_t localSlot = SecondWaveSlot(resCtx.localRankIndex, peer);
            CHK_RET(HcommLocalReduceOnThread(mainThread, output,
                bufferA + static_cast<uint64_t>(localSlot) * recvBytes, recvBytes / sizeof(float),
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        }
    }
    return HCCL_SUCCESS;
}

uint32_t PeerSlot(uint32_t ownerIndex, uint32_t peerIndex)
{
    return peerIndex < ownerIndex ? peerIndex : peerIndex - 1;
}

// Two half-output tiles let bufferB hold one receive slot for every local peer. Tile 1's Clos work overlaps
// tile 0's seven-link full-mesh exchange; all reductions remain ordered by peer rank for determinism.
HcclResult ExecLargeMessagePipelined(const OpParam &param, const AlgResourceCtx &resCtx,
    const ChannelInfo &crossChannel, const char *input, char *output, char *bufferA, char *bufferB,
    uint64_t recvBytes, uint64_t halfInputBytes, uint64_t ownedOffset, uint64_t sendOffset)
{
    CHK_PRT_RET(resCtx.threads.size() < LOCAL_RANK_SIZE + 1, HCCL_ERROR("Large-message path needs %u AICPU TS threads", LOCAL_RANK_SIZE + 1), HCCL_E_INTERNAL);
    const ThreadHandle mainThread = resCtx.threads[0];
    const ThreadHandle crossThread = resCtx.threads[1];
    const uint64_t firstTileCount = param.count / LARGE_MESSAGE_TILE_NUM;
    const uint64_t secondTileCount = param.count - firstTileCount;
    const uint64_t firstTileBytes = firstTileCount * sizeof(float);
    const uint64_t secondTileBytes = secondTileCount * sizeof(float);
    const uint64_t secondTileOffset = firstTileBytes;
    const uint64_t slotStride = secondTileBytes;
    CHK_PRT_RET(firstTileBytes == 0 || halfInputBytes + 7 * slotStride > resCtx.localBuffer.size, HCCL_ERROR("Invalid large-message tile layout"), HCCL_E_INTERNAL);
    CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, crossThread, 0));
    CHK_RET(HcommThreadNotifyWaitOnThread(crossThread, 0, CUSTOM_TIMEOUT));
    for (uint32_t tile = 0; tile < LARGE_MESSAGE_TILE_NUM; ++tile) {
        const uint64_t tileOffset = tile == 0 ? 0 : secondTileOffset;
        const uint64_t tileBytes = tile == 0 ? firstTileBytes : secondTileBytes;
        for (uint32_t chunk = 0; chunk < LOCAL_RANK_SIZE; ++chunk) {
            const uint64_t chunkOffset = static_cast<uint64_t>(chunk) * recvBytes + tileOffset;
            CHK_RET(HcommWriteOnThread(crossThread, crossChannel.handle, static_cast<char *>(crossChannel.remoteCclMem.addr) + chunkOffset, input + sendOffset + chunkOffset, tileBytes));
        }
        CHK_RET(HcommChannelNotifyRecordOnThread(crossThread, crossChannel.handle, DATA_NOTIFY_IDX));
    }
    CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, crossChannel.handle, DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));
    for (uint32_t chunk = 0; chunk < LOCAL_RANK_SIZE; ++chunk) {
        const uint64_t chunkOffset = static_cast<uint64_t>(chunk) * recvBytes;
        if (chunk == resCtx.localRankIndex) {
            CHK_RET(HcommLocalReduceOnThread(mainThread, bufferA + chunkOffset,
                input + ownedOffset + chunkOffset, firstTileCount, HCOMM_DATA_TYPE_FP32,
                HCOMM_REDUCE_SUM));
            CHK_RET(HcommLocalCopyOnThread(mainThread, output, bufferA + chunkOffset, firstTileBytes));
            continue;
        }
        const uint32_t workerOrdinal = PeerSlot(resCtx.localRankIndex, chunk);
        const ThreadHandle worker = resCtx.threads[workerOrdinal + 2];
        const ChannelInfo *channel = GetChannel(resCtx, resCtx.localRanks[chunk]);
        const uint32_t remoteSlot = PeerSlot(chunk, resCtx.localRankIndex);
        const uint64_t remoteOffset = halfInputBytes + static_cast<uint64_t>(remoteSlot) * slotStride;
        CHK_PRT_RET(channel == nullptr || channel->remoteCclMem.addr == nullptr || channel->remoteCclMem.size < remoteOffset + firstTileBytes, HCCL_ERROR("Invalid tile-0 local channel"), HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalReduceOnThread(worker, bufferA + chunkOffset, input + ownedOffset + chunkOffset,
            firstTileCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        CHK_RET(HcommWriteOnThread(worker, channel->handle, static_cast<char *>(channel->remoteCclMem.addr) + remoteOffset, bufferA + static_cast<uint64_t>(chunk) * recvBytes, firstTileBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel->handle, DATA_NOTIFY_IDX));
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel->handle, DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, workerOrdinal + 1));
    }
    CHK_RET(HcommChannelNotifyWaitOnThread(mainThread, crossChannel.handle, DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));
    for (uint32_t chunk = 0; chunk < LOCAL_RANK_SIZE; ++chunk) {
        const uint64_t chunkOffset = static_cast<uint64_t>(chunk) * recvBytes + secondTileOffset;
        if (chunk == resCtx.localRankIndex) {
            CHK_RET(HcommLocalReduceOnThread(mainThread, bufferA + chunkOffset,
                input + ownedOffset + chunkOffset, secondTileCount, HCOMM_DATA_TYPE_FP32,
                HCOMM_REDUCE_SUM));
            CHK_RET(HcommLocalCopyOnThread(
                mainThread, output + secondTileOffset, bufferA + chunkOffset, secondTileBytes));
            continue;
        }
        const uint32_t workerOrdinal = PeerSlot(resCtx.localRankIndex, chunk);
        const ThreadHandle worker = resCtx.threads[workerOrdinal + 2];
        // Queue this independent pair reduction behind the worker's tile-0 transfer. Its completion notify is
        // consumed before the later tile-1 transfer completion notify.
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        CHK_RET(HcommLocalReduceOnThread(worker, bufferA + chunkOffset, input + ownedOffset + chunkOffset,
            secondTileCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, workerOrdinal + 1));
    }
    for (uint32_t workerOrdinal = 0; workerOrdinal < LOCAL_RANK_SIZE - 1; ++workerOrdinal) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, workerOrdinal + 1, CUSTOM_TIMEOUT));
    }
    for (uint32_t peer = 0; peer < LOCAL_RANK_SIZE; ++peer) {
        if (peer == resCtx.localRankIndex) {
            continue;
        }
        const uint32_t workerOrdinal = PeerSlot(resCtx.localRankIndex, peer);
        const ThreadHandle worker = resCtx.threads[workerOrdinal + 2];
        const ChannelInfo *channel = GetChannel(resCtx, resCtx.localRanks[peer]);
        const uint32_t remoteSlot = PeerSlot(peer, resCtx.localRankIndex);
        const uint32_t localSlot = PeerSlot(resCtx.localRankIndex, peer);
        // Consume this peer's tile-0 slot before allowing its worker to reuse the slot for tile 1. Peers are
        // still accumulated in ascending order, while later tile-0 reductions overlap earlier tile-1 writes.
        CHK_RET(HcommLocalReduceOnThread(mainThread, output, bufferB +
            static_cast<uint64_t>(localSlot) * slotStride, firstTileCount, HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        const uint64_t remoteOffset = halfInputBytes + static_cast<uint64_t>(remoteSlot) * slotStride;
        CHK_PRT_RET(channel == nullptr || channel->remoteCclMem.addr == nullptr || channel->remoteCclMem.size < remoteOffset + secondTileBytes, HCCL_ERROR("Invalid tile-1 local channel"), HCCL_E_INTERNAL);
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel->handle, READY_NOTIFY_IDX));
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel->handle, READY_NOTIFY_IDX, CUSTOM_TIMEOUT));
        CHK_RET(HcommWriteOnThread(worker, channel->handle, static_cast<char *>(channel->remoteCclMem.addr) + remoteOffset, bufferA + static_cast<uint64_t>(peer) * recvBytes + secondTileOffset, secondTileBytes));
        CHK_RET(HcommChannelNotifyRecordOnThread(worker, channel->handle, DATA_NOTIFY_IDX));
        CHK_RET(HcommChannelNotifyWaitOnThread(worker, channel->handle, DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, workerOrdinal + 1));
    }
    for (uint32_t completion = 0; completion < 2; ++completion) {
        for (uint32_t workerOrdinal = 0; workerOrdinal < LOCAL_RANK_SIZE - 1; ++workerOrdinal) {
            CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, workerOrdinal + 1, CUSTOM_TIMEOUT));
        }
    }
    // The second tile is the unhidden tail. Partition its output across all eight threads; within every
    // partition peers are still accumulated in ascending order, preserving each element's FP32 operation order.
    for (uint32_t partition = 1; partition < LOCAL_RANK_SIZE; ++partition) {
        const uint32_t workerOrdinal = partition - 1;
        const ThreadHandle worker = resCtx.threads[workerOrdinal + 2];
        const uint64_t begin = secondTileCount * partition / LOCAL_RANK_SIZE;
        const uint64_t end = secondTileCount * (partition + 1) / LOCAL_RANK_SIZE;
        CHK_RET(HcommThreadNotifyRecordOnThread(mainThread, worker, 0));
        CHK_RET(HcommThreadNotifyWaitOnThread(worker, 0, CUSTOM_TIMEOUT));
        for (uint32_t peer = 0; peer < LOCAL_RANK_SIZE; ++peer) {
            if (peer == resCtx.localRankIndex) {
                continue;
            }
            const uint32_t localSlot = PeerSlot(resCtx.localRankIndex, peer);
            CHK_RET(HcommLocalReduceOnThread(worker, output + secondTileOffset + begin * sizeof(float),
                bufferB + static_cast<uint64_t>(localSlot) * slotStride + begin * sizeof(float), end - begin,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
        }
        CHK_RET(HcommThreadNotifyRecordOnThread(worker, mainThread, workerOrdinal + 1));
    }
    const uint64_t mainEnd = secondTileCount / LOCAL_RANK_SIZE;
    for (uint32_t peer = 0; peer < LOCAL_RANK_SIZE; ++peer) {
        if (peer == resCtx.localRankIndex) {
            continue;
        }
        const uint32_t localSlot = PeerSlot(resCtx.localRankIndex, peer);
        CHK_RET(HcommLocalReduceOnThread(mainThread, output + secondTileOffset,
            bufferB + static_cast<uint64_t>(localSlot) * slotStride, mainEnd, HCOMM_DATA_TYPE_FP32,
            HCOMM_REDUCE_SUM));
    }
    for (uint32_t workerOrdinal = 0; workerOrdinal < LOCAL_RANK_SIZE - 1; ++workerOrdinal) {
        CHK_RET(HcommThreadNotifyWaitOnThread(mainThread, workerOrdinal + 1, CUSTOM_TIMEOUT));
    }
    return HCCL_SUCCESS;
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
    if (recvBytes >= LARGE_MESSAGE_RECV_BYTES) {
        if (resCtx.threads.size() < LOCAL_RANK_SIZE + 1) {
            CHK_RET(HcommWriteOnThread(thread, crossChannel->handle, crossChannel->remoteCclMem.addr,
                input + sendOffset, halfInputBytes));
            CHK_RET(HcommChannelNotifyRecordOnThread(thread, crossChannel->handle, DATA_NOTIFY_IDX));
            CHK_RET(HcommChannelNotifyWaitOnThread(
                thread, crossChannel->handle, DATA_NOTIFY_IDX, CUSTOM_TIMEOUT));
            CHK_RET(HcommLocalReduceOnThread(thread, bufferA, input + ownedOffset, halfInputBytes / typeSize,
                HCOMM_DATA_TYPE_FP32, HCOMM_REDUCE_SUM));
            return ExecLargeMessage(resCtx, recvBytes, output, bufferA, bufferB, halfInputBytes);
        }
        return ExecLargeMessagePipelined(param, resCtx, *crossChannel, input, output, bufferA, bufferB, recvBytes,
            halfInputBytes, ownedOffset, sendOffset);
    }

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
