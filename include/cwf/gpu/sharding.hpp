/**
 * @file sharding.hpp
 * @brief descriptor-buffer sharding math that keeps Vulkan fed without drama
 *
 * this header implements the high-level planning logic for Phase 7, carving up
 * freshly packed struct-of-arrays data into descriptor-buffer-friendly shards.
 * it keeps every logical buffer aligned, respects Vulkan's two-gig hard stop,
 * and emits deterministic metadata so GPU uploads stay boring (in a good way).
 *
 * the module embraces pure functions, std::expected error surfacing, and C++26
 * constexpr vibes so we can reason about memory placement without summoning a
 * debugger. descriptor-buffer supremacy is only possible when our layout plan
 * is bulletproof, and this header is where that plan is born uwu ✨
 *
 * @author LukeFrankio
 * @date 2025-10-07
 * @version 1.0
 *
 * @note uses C++26 features (std::expected, std::has_single_bit) with GCC 15.2+
 * @note designed for Vulkan 1.3.290+ descriptor buffers (AMD iGPU tuning path)
 * @note documented with Doxygen 1.15 beta because excessive comments slap
 * @warning failure to respect max_buffer_bytes will summon validation layers
 *
 * example quick peek:
 * @code
 * using namespace cwf::gpu::shard;
 *
 * std::vector specs{
 *     BufferSpecification{"nodes", 512_ZU * 64_ZU, 256_ZU},
 *     BufferSpecification{"elements", 1024_ZU * 128_ZU, 256_ZU},
 * };
 *
 * auto planned = plan_shards(specs);
 * if (!planned) {
 *     throw std::runtime_error(planned.error().message);
 * }
 * // planned->segments now lists contiguous slices ready for uploads uwu
 * @endcode
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <string>
#include <string_view>
#include <vector>

namespace cwf::gpu::shard
{

/**
 * @brief describes why the sharding plan bailed (functional error payload uwu)
 *
 * contains the human-readable message plus an annotated breadcrumb trail.
 */
struct ShardError
{
    std::string              message; ///< what went sideways in human words
    std::vector<std::string> context; ///< breadcrumb stack for debugging uwu
};

/**
 * @brief specification for a logical buffer awaiting descriptor sharding
 *
 * @note alignments should be powers of two (GPU memory controllers appreciate)
 */
struct BufferSpecification
{
    std::string   name;      ///< logical identifier (must be unique)
    std::size_t   size_bytes; ///< total byte length of the buffer
    std::size_t   alignment; ///< alignment constraint (power of two recommended)
};

/**
 * @brief describes a single slice of a logical buffer inside a device VkBuffer
 */
struct ShardSegment
{
    std::string   name; ///< logical buffer name this segment represents
    std::uint32_t device_buffer_index; ///< which VkBuffer slot we drop into
    std::size_t   device_offset; ///< byte offset inside the VkBuffer (aligned)
    std::size_t   source_offset; ///< byte offset inside the logical buffer
    std::size_t   size_bytes; ///< how many bytes this slice covers
};

/**
 * @brief final scatter plan mapping logical data to GPU buffers with receipts
 */
struct ShardedLayout
{
    std::vector<ShardSegment> segments; ///< every slice across all logical buffers
    std::vector<std::size_t>  device_buffer_sizes; ///< total size required per VkBuffer
    std::size_t               max_buffer_bytes; ///< limit used while planning
    std::size_t               alignment; ///< global alignment applied to shards
};

/**
 * @brief Vulkan mandates descriptor buffer alignments at least 256 bytes
 */
inline constexpr std::size_t kDefaultAlignment = 256U;

/**
 * @brief Vulkan 1.3 on AMD iGPU caps single VkBuffer at 2 GB (Phase 7 assumption)
 */
inline constexpr std::size_t kDefaultMaxBufferBytes = 2ULL * 1024ULL * 1024ULL * 1024ULL;

/**
 * @brief partition packed buffers into descriptor-buffer-sized shards (pure af)
 *
 * ✨ PURE FUNCTION ✨
 *
 * this routine never mutates shared state, it just reads the specifications
 * and emits a deterministic plan. it validates alignments, ensures every slice
 * stays within the 2 GB window (or custom limit), and spills onto additional
 * VkBuffers when required. errors surface via std::expected without exceptions.
 *
 * @param[in] specs logical buffers (unique names, positive size, power-of-two alignment)
 * @param[in] max_buffer_bytes per-device VkBuffer budget (defaults to 2 GiB)
 * @param[in] alignment global alignment floor (defaults to 256 B per Vulkan spec)
 * @return layout detailing segments & device buffer sizes, or ShardError uwu
 *
 * @pre alignment must be power of two (Vulkan memory rules demand it)
 * @pre every BufferSpecification must have size_bytes > 0
 * @post returned layout.device_buffer_sizes reflect aligned consumption
 * @post segments ordered to match input buffer list for deterministic uploads
 *
 * @complexity O(n) time where n == specs.size(), O(n) auxiliary storage
 * @note no heap allocations beyond std::vector growth (amortized constant time)
 * @warning hitting max_buffer_bytes exactly will roll over to a new VkBuffer
 *
 * example (edge case with oversize buffer):
 * @code
 * auto plan = plan_shards({{"giant", 5ULL * 1024ULL * 1024ULL * 1024ULL, 256}});
 * if (!plan) {
 *     std::println("shard failed: {}", plan.error().message);
 * }
 * // expect error because single logical buffer exceeds 2 GiB limit uwu
 * @endcode
 */
[[nodiscard]] auto plan_shards(const std::vector<BufferSpecification> &specs,
                               std::size_t max_buffer_bytes = kDefaultMaxBufferBytes,
                               std::size_t alignment = kDefaultAlignment)
    -> std::expected<ShardedLayout, ShardError>;

} // namespace cwf::gpu::shard