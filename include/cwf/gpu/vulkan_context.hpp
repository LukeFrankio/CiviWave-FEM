/**
 * @file vulkan_context.hpp
 * @brief zero-cost RAII Vulkan context + memory allocators that make GPU FEM go brrr uwu
 *
 * this header declares the high-level Vulkan bootstrapper used throughout Phase 5 of the
 * roadmap. it owns instance creation, device selection (favoring the AMD iGPU), queue
 * plumbing, VMA allocator wiring, descriptor buffer enablement, and timeline semaphore /
 * barrier helpers. the goal is to centralize all the gnarly Vulkan ceremony behind a
 * functional, well-documented interface so later phases can focus purely on math kernels.
 *
 * design vibes:
 * - RAII all the things (no leaks, no manual destroy calls)
 * - constexpr-friendly data structs where practical (summaries, configs)
 * - <span style="color:MediumPurple">✨ excessive Doxygen ✨</span> so future contributors know exactly
 *   which features we flip on and why
 * - zero warnings policy respected (headers compile cleanly with -Wall -Wextra -Wpedantic)
 *
 * this header pairs with `src/gpu/vulkan_context.cpp` for the implementation. consumers grab
 * a `cwf::gpu::VulkanContext`, ask for descriptors/memory helpers, and profit.
 *
 * @author LukeFrankio
 * @date 2025-11-10
 * @version 1.0
 *
 * @note requires Vulkan SDK 1.4.328.1+ (API 1.3 core, descriptor buffer extension) and GCC 15.2+
 * @note documented with Doxygen 1.15 beta (latest) because documentation supremacy is mandatory
 */
#pragma once

#include <array>
#include <cstdint>
#include <expected>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

struct VmaAllocator_T;
struct VmaAllocation_T;

using VmaAllocator  = VmaAllocator_T *;
using VmaAllocation = VmaAllocation_T *;

namespace cwf::gpu
{

/**
 * @brief structured error payload for Vulkan bootstrap failures uwu
 *
 * ⚠️ IMPURE FUNCTION (conceptually) ⚠️ — while this is a passive data structure, every place
 * that constructs it corresponds to a side-effectful Vulkan API call gone wrong. we surface
 * contextual breadcrumbs (messages + VkResult) so logs stay actionable.
 */
struct VulkanError
{
    std::string              message;  ///< human readable summary with gen-z energy
    std::vector<std::string> context;  ///< breadcrumb trail (API, stage, etc.)
    VkResult                 result{VK_SUCCESS}; ///< raw VkResult for debugging / RGP correlation
};

/**
 * @brief descriptor buffer + indexing capability snapshot (extracted during device creation)
 *
 * ✨ PURE FUNCTION ✨ — this struct is filled using deterministic queries against the
 * selected physical device; no mutation of global state occurs.
 */
struct DescriptorSupport
{
    bool         descriptor_buffer{false};              ///< whether VK_EXT_descriptor_buffer is enabled
    bool         descriptor_buffer_push_descriptors{false}; ///< push descriptor writes via buffer
    bool         descriptor_indexing{false};            ///< descriptor indexing feature availability
    std::uint32_t max_descriptor_buffer_bindings{0U};   ///< number of descriptor buffer bindings supported
    std::uint32_t max_resource_descriptor_buffer_bindings{0U}; ///< resource descriptor buffer binding slots
    std::uint32_t max_sampler_descriptor_buffer_bindings{0U};  ///< sampler descriptor buffer binding slots
    VkDeviceSize  descriptor_buffer_address_space_size{0U};    ///< total address space exposed for descriptor buffers
    VkDeviceSize  descriptor_buffer_offset_alignment{0U};      ///< alignment requirement for descriptor buffers
};

/**
 * @brief queue configuration + runtime metadata for the compute queue we own
 *
 * ✨ PURE FUNCTION ✨ — data captured once device is created. describes queue family indices
 * and timestamp precision so scheduling code can stay deterministic.
 */
struct QueueInfo
{
    std::uint32_t family_index{0U};      ///< queue family chosen for compute/timestamps
    std::uint32_t queue_index{0U};       ///< queue index inside the family (always 0 for now)
    std::uint32_t timestamp_bits{0U};    ///< number of valid timestamp bits (hardware property)
    VkQueue       queue{VK_NULL_HANDLE}; ///< actual VkQueue handle (impure handle, use thoughtfully)
};

/**
 * @brief physical device capability summary so logs/tests can assert expectations
 *
 * ✨ PURE FUNCTION ✨ — derived exclusively from Vulkan queries; immutable snapshot of the
 * selected GPU.
 */
struct DeviceSummary
{
    std::string   name;               ///< UTF-8 device name from VkPhysicalDeviceProperties
    std::uint32_t api_version{0U};    ///< `VK_MAKE_API_VERSION` encoded API level supported
    std::uint32_t driver_version{0U}; ///< driver version straight from the driver
    VkPhysicalDeviceType type{VK_PHYSICAL_DEVICE_TYPE_OTHER}; ///< GPU type classification
    std::uint32_t vendor_id{0U};      ///< PCI vendor ID (expect 0x1002 for AMD iGPU)
    std::uint32_t device_id{0U};      ///< PCI device id, handy for logs
    VkPhysicalDevice physical_device{VK_NULL_HANDLE}; ///< raw handle for advanced queries
};

/**
 * @brief tiny RAII wrapper around a timeline semaphore with convenience helpers
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — timeline semaphores interact with the driver timeline. methods perform
 * Vulkan API calls that mutate GPU synchronization state.
 */
class TimelineSemaphore
{
public:
    TimelineSemaphore() = default;

    TimelineSemaphore(VkDevice device, VkSemaphore handle) noexcept;

    TimelineSemaphore(const TimelineSemaphore &)            = delete;
    auto operator=(const TimelineSemaphore &) -> TimelineSemaphore & = delete;

    TimelineSemaphore(TimelineSemaphore &&other) noexcept;
    auto operator=(TimelineSemaphore &&other) noexcept -> TimelineSemaphore &;

    ~TimelineSemaphore();

    [[nodiscard]] auto handle() const noexcept -> VkSemaphore { return handle_; }

    /**
     * @brief signals the semaphore to the provided timeline value
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — forwards to vkSignalSemaphore, mutating GPU-side timeline state.
     *
     * @param value timeline value to signal
     */
    void signal(std::uint64_t value) const;

    /**
     * @brief waits on the semaphore from the host side until it reaches the specified value
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — uses vkWaitSemaphores to block the host thread.
     *
     * @param value timeline value to wait for
     * @param timeout_ns timeout in nanoseconds (default: UINT64_MAX means forever)
     */
    void wait(std::uint64_t value, std::uint64_t timeout_ns = std::numeric_limits<std::uint64_t>::max()) const;

private:
    VkDevice    device_{VK_NULL_HANDLE};
    VkSemaphore handle_{VK_NULL_HANDLE};
};

/**
 * @brief persistent staging buffer ring used for uploads (host visible, persistently mapped)
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — owns GPU memory and exposes a CPU pointer for memcpy.
 */
struct StagingRing
{
    VmaAllocation allocation{VK_NULL_HANDLE}; ///< VMA allocation token
    VkBuffer      buffer{VK_NULL_HANDLE};     ///< Vulkan buffer handle
    std::byte *   mapped{nullptr};            ///< persistently mapped pointer
    std::uint64_t size{0U};                   ///< total capacity in bytes
    std::uint64_t head{0U};                   ///< simple monotonic offset; wrap manually
};

/**
 * @brief creation parameters for `VulkanContext::create`
 *
 * ✨ PURE FUNCTION ✨ — acts as a declarative spec for the runtime we want.
 */
struct ContextCreateInfo
{
    bool enable_validation{true};                  ///< request VK_LAYER_KHRONOS_validation if available
    std::optional<std::uint32_t> device_index{};   ///< force a specific physical device index
    std::string_view preferred_device_substring{"AMD"}; ///< fuzzy match hint for default selection
    std::uint64_t staging_buffer_bytes{64ULL * 1024ULL * 1024ULL}; ///< persistent staging ring size
    bool require_descriptor_buffer{true};          ///< hard fail if VK_EXT_descriptor_buffer absent
};

/**
 * @brief RAII container owning Vulkan instance/device/allocators per Phase 5 deliverables
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — constructors perform Vulkan initialization, allocate memory, and interact
 * with system drivers. the accessors themselves remain pure.
 */
class VulkanContext
{
public:
    VulkanContext() = default;

    static auto create(const ContextCreateInfo &info = {}) -> std::expected<VulkanContext, VulkanError>;

    VulkanContext(const VulkanContext &)            = delete;
    auto operator=(const VulkanContext &) -> VulkanContext & = delete;

    VulkanContext(VulkanContext &&other) noexcept;
    auto operator=(VulkanContext &&other) noexcept -> VulkanContext &;

    ~VulkanContext();

    [[nodiscard]] auto instance() const noexcept -> VkInstance { return instance_; }
    [[nodiscard]] auto device() const noexcept -> VkDevice { return device_; }
    [[nodiscard]] auto allocator() const noexcept -> VmaAllocator { return allocator_; }
    [[nodiscard]] auto queue_info() const noexcept -> const QueueInfo & { return queue_info_; }
    [[nodiscard]] auto descriptor_support() const noexcept -> const DescriptorSupport & { return descriptor_support_; }
    [[nodiscard]] auto device_summary() const noexcept -> const DeviceSummary & { return summary_; }
    [[nodiscard]] auto staging_ring() const noexcept -> const StagingRing & { return staging_ring_; }

    /**
     * @brief allocates a new timeline semaphore wrapped in RAII helper
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — Vulkan API calls allocate kernel objects.
     *
     * @param initial_value starting timeline value
     * @return timeline semaphore or error payload
     */
    [[nodiscard]] auto make_timeline_semaphore(std::uint64_t initial_value = 0U) const
        -> std::expected<TimelineSemaphore, VulkanError>;

    /**
     * @brief attaches a human-friendly name to a Vulkan object (debug markers / RGP)
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — interacts with VK_EXT_debug_utils.
     *
     * @param handle raw Vulkan object handle (cast to uint64_t per spec)
     * @param type Vulkan object type enumeration
     * @param name UTF-8 label to set (must persist until driver copies)
     */
    void set_object_name(std::uint64_t handle, VkObjectType type, std::string_view name) const;

    /**
     * @brief begins a debug label region on the provided command buffer (RGP markers!)
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — feeds VK_EXT_debug_utils entry points.
     *
     * @param cmd command buffer to annotate
     * @param name label text logged in tools
     * @param color RGBA debug color (each 0..1)
     */
    void push_debug_label(VkCommandBuffer cmd, std::string_view name, std::array<float, 4U> color = {0.3F, 0.4F, 0.9F, 1.0F}) const;

    /**
     * @brief ends the most recent debug label region on the command buffer
     */
    void pop_debug_label(VkCommandBuffer cmd) const;

    /**
     * @brief builds a simple memory barrier helper for common compute-to-compute transitions
     *
     * ✨ PURE FUNCTION ✨ — returns a POD barrier description; caller submits it via sync2 API.
     *
     * @param src_usage pipeline stage mask
     * @param dst_usage pipeline stage mask
     * @param src_access source access flags
     * @param dst_access destination access flags
     * @return VkMemoryBarrier2 struct ready for vkCmdPipelineBarrier2
     */
    [[nodiscard]] static auto make_memory_barrier(VkPipelineStageFlags2 src_usage, VkPipelineStageFlags2 dst_usage,
                                                  VkAccessFlags2 src_access, VkAccessFlags2 dst_access) noexcept -> VkMemoryBarrier2;

private:
    VkInstance           instance_{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
    VkPhysicalDevice     physical_device_{VK_NULL_HANDLE};
    VkDevice             device_{VK_NULL_HANDLE};
    VmaAllocator         allocator_{VK_NULL_HANDLE};
    QueueInfo            queue_info_{};
    DescriptorSupport    descriptor_support_{};
    DeviceSummary        summary_{};
    StagingRing          staging_ring_{};

    PFN_vkSetDebugUtilsObjectNameEXT  set_object_name_fn_{nullptr};
    PFN_vkCmdBeginDebugUtilsLabelEXT  begin_label_fn_{nullptr};
    PFN_vkCmdEndDebugUtilsLabelEXT    end_label_fn_{nullptr};

    void destroy();
};

} // namespace cwf::gpu