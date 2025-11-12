/**
 * @file sharding.cpp
 * @brief implementation of descriptor-buffer sharding math (memory choreography)
 *
 * this translation unit delivers the deterministic layout planner backing
 * Phase 7's descriptor-buffer uploads. it keeps logic side-effect-free so we
 * can unit test aggressively and even constexpr-evaluate small plans. the
 * algorithm greedily fills VkBuffers while honoring alignment and surfacing
 * rich errors via std::expected. functional vibes only uwu ✨
 */

#include "cwf/gpu/sharding.hpp"

#include <algorithm>
#include <bit>
#include <numeric>

namespace cwf::gpu::shard
{

namespace
{
/**
 * @brief rounds value up to the next multiple of alignment (pure + constexpr)
 *
 * ✨ PURE FUNCTION ✨
 *
 * @param value base number of bytes
 * @param alignment power-of-two alignment target
 * @return aligned byte count (>= value)
 */
[[nodiscard]] constexpr auto align_up(std::size_t value, std::size_t alignment) noexcept -> std::size_t
{
    const auto mask = alignment - 1U;
    return (value + mask) & ~mask;
}
}

[[nodiscard]] auto plan_shards(const std::vector<BufferSpecification> &specs,
                               const std::size_t                       max_buffer_bytes,
                               const std::size_t                       alignment)
    -> std::expected<ShardedLayout, ShardError>
{
    if ((alignment == 0U) || !std::has_single_bit(alignment))
    {
        return std::unexpected(ShardError{"alignment must be a non-zero power of two",
                           {"alignment=" + std::to_string(alignment)}});
    }

    if (max_buffer_bytes == 0U)
    {
        return std::unexpected(ShardError{"max buffer bytes must be positive", {}});
    }

    if (specs.empty())
    {
        return ShardedLayout{{}, {}, max_buffer_bytes, alignment};
    }

    ShardedLayout layout{};
    layout.max_buffer_bytes = max_buffer_bytes;
    layout.alignment = alignment;

    std::size_t current_buffer_size = 0U;
    std::uint32_t current_buffer_index = 0U;

    layout.device_buffer_sizes.push_back(0U);

    for (const auto &spec : specs)
    {
        if (spec.size_bytes == 0U)
        {
            return std::unexpected(ShardError{"buffer has zero size",
                                               {"name=" + spec.name}});
        }

        if ((spec.alignment == 0U) || !std::has_single_bit(spec.alignment))
        {
            return std::unexpected(ShardError{"buffer alignment must be power of two",
                                               {"name=" + spec.name,
                                                "alignment=" + std::to_string(spec.alignment)}});
        }

        const auto effective_alignment = std::max(alignment, spec.alignment);

        std::size_t remaining = spec.size_bytes;
        std::size_t logical_offset = 0U;

        while (remaining > 0U)
        {
            if (layout.device_buffer_sizes.size() <= current_buffer_index)
            {
                layout.device_buffer_sizes.push_back(0U);
            }

            current_buffer_size = layout.device_buffer_sizes.at(current_buffer_index);
            auto aligned_offset = align_up(current_buffer_size, effective_alignment);

            if (aligned_offset >= max_buffer_bytes)
            {
                current_buffer_index += 1U;
                layout.device_buffer_sizes.push_back(0U);
                continue;
            }

            const auto space_remaining = max_buffer_bytes - aligned_offset;
            if (space_remaining == 0U)
            {
                current_buffer_index += 1U;
                layout.device_buffer_sizes.push_back(0U);
                continue;
            }

            const auto slice_size = std::min(space_remaining, remaining);

            layout.segments.push_back({
                spec.name,
                current_buffer_index,
                aligned_offset,
                logical_offset,
                slice_size,
            });

            aligned_offset += slice_size;
            layout.device_buffer_sizes.at(current_buffer_index) = aligned_offset;

            remaining -= slice_size;
            logical_offset += slice_size;

            if ((remaining > 0U) && (aligned_offset == max_buffer_bytes))
            {
                current_buffer_index += 1U;
            }
        }
    }

    // trim trailing zero-sized buffers
    while (!layout.device_buffer_sizes.empty() && layout.device_buffer_sizes.back() == 0U)
    {
        layout.device_buffer_sizes.pop_back();
    }

    return layout;
}

} // namespace cwf::gpu::shard
