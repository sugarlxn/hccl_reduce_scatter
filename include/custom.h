/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OPS_HCCL_CUSTOM_H
#define OPS_HCCL_CUSTOM_H

#include <memory>
#include <hccl/hccl_types.h>
#include <hccl/hccl_res.h>

#include "binary_stream.h"
#include "common.h"

typedef struct {
    void *addr;
    uint64_t size;
} CommBuffer;

struct CcuKernelArgBase {
    ChannelHandle channels[MAX_RANK_SIZE];
    uint32_t channelCount;
};

// channels are ordered by increasing remote rank, with the local rank skipped.
struct CcuReduceScatterKernelArg : public CcuKernelArgBase {
    uint32_t rankSize;
    uint32_t rankId;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
};

constexpr uint32_t MAX_LOCAL_RANK_SIZE = 8;
constexpr uint32_t MAX_HIERARCHICAL_TARGETS = 3;
constexpr uint32_t MAX_CROSS_PEERS = 2;

struct CcuLocalReduceKernelArg : public CcuKernelArgBase {
    uint32_t groupSize;
    uint32_t groupRankId;
    uint32_t targetCount;
    bool useStaging;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
};

struct CcuCrossReduceKernelArg : public CcuKernelArgBase {
    uint32_t sendCount;
    uint32_t sendChannelIndices[MAX_CROSS_PEERS];
    HcclDataType dataType;
    HcclReduceOp reduceOp;
};

struct CcuPartialReduceKernelArg : public CcuKernelArgBase {
    uint32_t sourceCount;
    uint32_t localSourceIndex;
    bool includeLocalSource;
    HcclDataType dataType;
    HcclReduceOp reduceOp;
};

struct CcuMergePartialKernelArg : public CcuKernelArgBase {
    HcclDataType dataType;
    HcclReduceOp reduceOp;
};

enum class ReduceScatterAlgorithm : uint32_t {
    DIRECT_MESH = 0,
    STAGING_MESH = 1,
    HIERARCHICAL = 2,
    HIERARCHICAL_STAGING = 3,
    DUAL_DIE_PARTIAL = 4,
    STRIPED_SINGLE_DIE = 5,
};

// ccu kernel register所需信息
struct CcuKernelInfo {
    // kernel名称
    char kernelFuncName[64];
    // kernel函数
    void *kernelFunc;
    // KernelArg实例指针
    void *kernelArg;

private:
    std::shared_ptr<CcuKernelArgBase> kernelArgSmartPtr;

public:
    template <typename T> void setKernelArg(std::shared_ptr<T> arg)
    {
        kernelArgSmartPtr = std::static_pointer_cast<CcuKernelArgBase>(arg);
        kernelArg = static_cast<void *>(arg.get());
    }
};

struct AlgResourceCtx {
    ReduceScatterAlgorithm algorithm = ReduceScatterAlgorithm::DIRECT_MESH;
    ThreadHandle ccuThread;            ///< CCU通信引擎上的thread资源
    CommBuffer localBuffer;            ///< 本端HCCL通信内存
    std::vector<ThreadHandle> threads; ///< CCU通信引擎上的thread资源
    std::vector<CcuKernelHandle> ccuKernels;
    std::vector<uint32_t> localRanks;
    std::vector<uint32_t> targetRanks;
    std::vector<uint32_t> crossPeers;
    std::vector<uint32_t> crossSendTargetIndices;

    // 序列化
    std::vector<char> Serialize()
    {
        BinaryStream binaryStream;
        binaryStream << algorithm;
        binaryStream << ccuThread;
        binaryStream << localBuffer;
        binaryStream << threads;
        binaryStream << ccuKernels;
        binaryStream << localRanks;
        binaryStream << targetRanks;
        binaryStream << crossPeers;
        binaryStream << crossSendTargetIndices;
        std::vector<char> result;
        binaryStream.Dump(result);
        return result;
    }

    // 反序列化
    void DeSerialize(std::vector<char> &data)
    {
        BinaryStream binaryStream(data);
        binaryStream >> algorithm;
        binaryStream >> ccuThread;
        binaryStream >> localBuffer;
        binaryStream >> threads;
        binaryStream >> ccuKernels;
        binaryStream >> localRanks;
        binaryStream >> targetRanks;
        binaryStream >> crossPeers;
        binaryStream >> crossSendTargetIndices;
    }
};

#endif // OPS_HCCL_CUSTOM_H
