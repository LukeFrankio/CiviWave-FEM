/**
 * @file device_buffers.hpp
 * @brief device-local buffer arena that mirrors logical Phase 7 buffers on Vulkan
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <functional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "cwf/gpu/buffers.hpp"
#include "cwf/gpu/sharding.hpp"
#include "cwf/gpu/upload.hpp"
#include "cwf/gpu/vulkan_context.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/physics/materials.hpp"

namespace cwf::gpu
{

struct BufferSlice
{
    VkBuffer     buffer{VK_NULL_HANDLE};
    VkDeviceSize offset{0U};
    VkDeviceSize size{0U};
};

class DeviceBufferArena
{
public:
    DeviceBufferArena() = default;

    static auto create(const VulkanContext &context, const mesh::pack::PackingResult &packing,
                       std::span<const physics::materials::ElasticProperties> materials)
        -> std::expected<DeviceBufferArena, VulkanError>;

    DeviceBufferArena(const DeviceBufferArena &)            = delete;
    auto operator=(const DeviceBufferArena &) -> DeviceBufferArena & = delete;

    DeviceBufferArena(DeviceBufferArena &&other) noexcept;
    auto operator=(DeviceBufferArena &&other) noexcept -> DeviceBufferArena &;

    ~DeviceBufferArena();

    [[nodiscard]] auto has_slice(std::string_view name) const noexcept -> bool;
    [[nodiscard]] auto slice(std::string_view name) const -> const BufferSlice &;

    [[nodiscard]] auto context() const noexcept -> const VulkanContext & { return *context_; }

private:
    struct TransparentHash
    {
        using is_transparent = void;

        [[nodiscard]] auto operator()(std::string_view value) const noexcept -> std::size_t
        {
            return std::hash<std::string_view>{}(value);
        }

        [[nodiscard]] auto operator()(const std::string &value) const noexcept -> std::size_t
        {
            return std::hash<std::string>{}(value);
        }

        [[nodiscard]] auto operator()(const char *value) const noexcept -> std::size_t
        {
            return std::hash<std::string_view>{}(value);
        }
    };

    struct TransparentEqual
    {
        using is_transparent = void;

        [[nodiscard]] auto operator()(std::string_view lhs, std::string_view rhs) const noexcept -> bool
        {
            return lhs == rhs;
        }

        [[nodiscard]] auto operator()(const std::string &lhs, const std::string &rhs) const noexcept -> bool
        {
            return lhs == rhs;
        }

        [[nodiscard]] auto operator()(const std::string &lhs, std::string_view rhs) const noexcept -> bool
        {
            return lhs == rhs;
        }

        [[nodiscard]] auto operator()(std::string_view lhs, const std::string &rhs) const noexcept -> bool
        {
            return lhs == rhs;
        }

        [[nodiscard]] auto operator()(const char *lhs, std::string_view rhs) const noexcept -> bool
        {
            return std::string_view{lhs} == rhs;
        }

        [[nodiscard]] auto operator()(std::string_view lhs, const char *rhs) const noexcept -> bool
        {
            return lhs == std::string_view{rhs};
        }
    };

    struct DeviceBuffer
    {
        VkBuffer     buffer{VK_NULL_HANDLE};
        VmaAllocation allocation{nullptr};
        VkDeviceSize size{0U};
    };

    const VulkanContext *context_{nullptr};
    std::vector<DeviceBuffer> device_buffers_;
    std::unordered_map<std::string, BufferSlice, TransparentHash, TransparentEqual> slices_;

    [[nodiscard]] static auto upload_schedule(const VulkanContext &context, VkCommandBuffer cmd,
                                              const upload::UploadSchedule &schedule,
                                              std::span<const DeviceBuffer> device_buffers)
        -> std::expected<void, VulkanError>;

    void destroy();
};

} // namespace cwf::gpu
