/**
 * @file upload.cpp
 * @brief concrete implementation of upload scheduling (staging ring ballet uwu)
 *
 * translates sharding metadata into bite-sized uploads that respect staging
 * memory constraints. everything stays pure, so the returned schedule can be
 * re-evaluated deterministically in tests and during frame build.
 */

#include "cwf/gpu/upload.hpp"

#include <algorithm>
#include <bit>
#include <unordered_map>

namespace cwf::gpu::upload
{

namespace
{
/**
 * @brief rounds byte counts up to the next alignment boundary (pure + constexpr)
 *
 * ✨ PURE FUNCTION ✨
 *
 * @param value source byte count
 * @param alignment power-of-two target alignment
 * @return aligned byte count
 */
[[nodiscard]] constexpr auto align_up(std::size_t value, std::size_t alignment) noexcept -> std::size_t
{
    const auto mask = alignment - 1U;
    return (value + mask) & ~mask;
}
} // namespace

/**
 * @details implementation walks the segment list in order, looks up the
 * matching BufferView by name, and slices it into staging-sized spans. each
 * UploadChunk references the original CPU span so callers can record the copy
 * without extra allocations. no mutation of @p buffers occurs because purity is
 * law uwu.
 */
[[nodiscard]] auto build_upload_schedule(const shard::ShardedLayout &layout,
                                         const std::vector<BufferView> &buffers,
                                         const StagingConfig staging) -> std::expected<UploadSchedule, UploadError>
{
    if ((staging.chunk_bytes == 0U) || (staging.alignment == 0U) || !std::has_single_bit(staging.alignment))
    {
        return std::unexpected(UploadError{"invalid staging configuration",
                             {"chunk_bytes=" + std::to_string(staging.chunk_bytes),
                              "alignment=" + std::to_string(staging.alignment)}});
    }

    UploadSchedule schedule{};
    schedule.total_bytes = 0U;

    if (layout.segments.empty())
    {
        return schedule;
    }

    std::unordered_map<std::string_view, std::span<const std::byte>> lookup{};
    lookup.reserve(buffers.size());
    for (const auto &view : buffers)
    {
        lookup.emplace(view.name, view.bytes);
    }

    std::vector<std::byte> scratch{};
    scratch.reserve(staging.chunk_bytes);

    for (const auto &segment : layout.segments)
    {
        const auto found = lookup.find(segment.name);
        if (found == lookup.end())
        {
            return std::unexpected(UploadError{"missing buffer for segment", {"segment=" + segment.name}});
        }

        const auto &buffer_span = found->second;
        if (segment.source_offset + segment.size_bytes > buffer_span.size())
        {
            return std::unexpected(UploadError{"segment exceeds buffer size",
                                               {"segment=" + segment.name,
                                                "source_offset=" + std::to_string(segment.source_offset),
                                                "segment_size=" + std::to_string(segment.size_bytes),
                                                "buffer_size=" + std::to_string(buffer_span.size())}});
        }

        const auto data_begin = buffer_span.subspan(segment.source_offset, segment.size_bytes);
        std::size_t consumed = 0U;

        while (consumed < segment.size_bytes)
        {
            const auto remaining = segment.size_bytes - consumed;
            const auto aligned_chunk = align_up(std::min(remaining, staging.chunk_bytes), staging.alignment);
            const auto chunk_size = std::min(aligned_chunk, remaining);

            const auto chunk_span = data_begin.subspan(consumed, chunk_size);

            schedule.commands.push_back({
                segment.device_buffer_index,
                segment.device_offset + consumed,
                chunk_span,
            });

            consumed += chunk_size;
            schedule.total_bytes += chunk_size;
        }
    }

    return schedule;
}

} // namespace cwf::gpu::upload
