/**
 * @file upload.hpp
 * @brief orchestration layer turning sharding metadata into Vulkan uploads
 *
 * after sharding gives us a deterministic descriptor-buffer layout, this header
 * calculates staging-friendly upload chunks. it slices packed CPU buffers into
 * ring-sized copy commands, respecting alignment so vkCmdCopyBuffer never
 * complains. everything is built around pure functions + std::expected because
 * error handling deserves algebraic data types uwu ✨
 *
 * @note pairs with build_packed_buffers + plan_shards in Phase 7 roadmap
 * @note requires C++26 standard library (std::expected, std::span shenanigans)
 * @note documented with Doxygen 1.15 beta for maximal comment serotonin
 */
#pragma once

#include "cwf/gpu/sharding.hpp"

#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace cwf::gpu::upload
{

/**
 * @brief structured error describing why scheduling failed (contextual receipts)
 */
struct UploadError
{
    std::string              message;
    std::vector<std::string> context;
};

/**
 * @brief immutable view into packed CPU data referenced by name
 */
struct BufferView
{
    std::string                      name;
    std::span<const std::byte>       bytes;
};

/**
 * @brief describes the capabilities of the staging ring buffer
 *
 * @note chunk_bytes should be sized to the GPU-visible staging allocation
 */
struct StagingConfig
{
    std::size_t chunk_bytes; ///< maximum bytes per staging copy
    std::size_t alignment;   ///< staging buffer alignment (power of two)
};

/**
 * @brief single vkCmdCopyBuffer-style operation
 */
struct UploadChunk
{
    std::uint32_t              device_buffer_index;
    std::size_t                destination_offset;
    std::span<const std::byte> bytes;
};

/**
 * @brief ordered list of upload commands + stats for telemetry
 */
struct UploadSchedule
{
    std::vector<UploadChunk> commands;
    std::size_t              total_bytes;
};

/**
 * @brief compute upload commands respecting staging constraints (pure planner)
 *
 * ✨ PURE FUNCTION ✨
 *
 * consumes a sharded layout and the actual CPU buffer views, splitting each
 * shard into staging-sized UploadChunk entries. all validation errors are
 * surfaced through std::expected without throwing.
 *
 * @param[in] layout descriptor-buffer sharding metadata (from plan_shards)
 * @param[in] buffers concrete CPU views keyed by name (must cover all shards)
 * @param[in] staging staging ring limits (chunk size + alignment)
 * @return UploadSchedule enumerating copy operations, or UploadError uwu
 *
 * @pre staging.alignment must be a power of two > 0
 * @pre chunk_bytes > 0 and large enough for at least one command
 * @post schedule.total_bytes equals the sum of chunk byte counts
 * @post commands sorted in segment order for deterministic replays
 *
 * @complexity O(n) where n == layout.segments.size()
 * @warning providing duplicate BufferView names results in unspecified segment selection
 */
[[nodiscard]] auto build_upload_schedule(const shard::ShardedLayout &layout,
                                         const std::vector<BufferView> &buffers,
                                         StagingConfig staging)
    -> std::expected<UploadSchedule, UploadError>;

} // namespace cwf::gpu::upload
