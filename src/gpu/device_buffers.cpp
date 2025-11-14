/**
 * @file device_buffers.cpp
 * @brief implementation of device-local buffer arena used by GPU solver runtime
 */

#include "cwf/gpu/device_buffers.hpp"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <format>
#include <print>
#include <stdexcept>
#include <utility>

namespace cwf::gpu
{
namespace
{

[[nodiscard]] auto make_error(std::string message, VkResult result, std::initializer_list<std::string> ctx)
    -> VulkanError
{
    VulkanError err{};
    err.message = std::move(message);
    err.context.assign(ctx.begin(), ctx.end());
    err.result = result;
    return err;
}

[[nodiscard]] auto make_error(std::string message, std::initializer_list<std::string> ctx) -> VulkanError
{
    VulkanError err{};
    err.message = std::move(message);
    err.context.assign(ctx.begin(), ctx.end());
    err.result = VK_ERROR_UNKNOWN;
    return err;
}

[[nodiscard]] auto make_error(std::string message, const std::vector<std::string> &ctx) -> VulkanError
{
    VulkanError err{};
    err.message = std::move(message);
    err.context = ctx;
    err.result = VK_ERROR_UNKNOWN;
    return err;
}

[[nodiscard]] auto create_command_pool(const VulkanContext &context) -> std::expected<VkCommandPool, VulkanError>
{
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = context.queue_info().family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool pool = VK_NULL_HANDLE;
    const VkResult result = vkCreateCommandPool(context.device(), &pool_info, nullptr, &pool);
    if(result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkCreateCommandPool failed", result, {"device_buffers"}));
    }
    return pool;
}

[[nodiscard]] auto allocate_command_buffer(VkDevice device, VkCommandPool pool) -> std::expected<VkCommandBuffer, VulkanError>
{
    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = pool;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1U;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    const VkResult result = vkAllocateCommandBuffers(device, &alloc_info, &cmd);
    if(result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkAllocateCommandBuffers failed", result, {"device_buffers"}));
    }
    return cmd;
}

[[nodiscard]] auto begin_one_time_commands(VkCommandBuffer cmd) -> std::expected<void, VulkanError>
{
    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    const VkResult result = vkBeginCommandBuffer(cmd, &begin_info);
    if(result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkBeginCommandBuffer failed", result, {"device_buffers"}));
    }
    return {};
}

[[nodiscard]] auto submit_and_wait(const VulkanContext &context, VkCommandBuffer cmd) -> std::expected<void, VulkanError>
{
    const VkResult end_result = vkEndCommandBuffer(cmd);
    if(end_result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkEndCommandBuffer failed", end_result, {"device_buffers"}));
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1U;
    submit_info.pCommandBuffers = &cmd;

    const VkQueue queue = context.queue_info().queue;
    const VkResult submit_result = vkQueueSubmit(queue, 1U, &submit_info, VK_NULL_HANDLE);
    if(submit_result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkQueueSubmit failed", submit_result, {"device_buffers"}));
    }

    const VkResult wait_result = vkQueueWaitIdle(queue);
    if(wait_result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkQueueWaitIdle failed", wait_result, {"device_buffers"}));
    }

    const VkResult reset_result = vkResetCommandBuffer(cmd, 0U);
    if(reset_result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkResetCommandBuffer failed", reset_result, {"device_buffers"}));
    }

    return {};
}

} // namespace

auto DeviceBufferArena::create(const VulkanContext &context, const mesh::pack::PackingResult &packing,
                               std::span<const physics::materials::ElasticProperties> materials)
    -> std::expected<DeviceBufferArena, VulkanError>
{
    buffers::PreparedGpuBuffers prepared{};
    const auto logical_buffers = buffers::build_logical_buffers(packing, materials, prepared);
    const auto specs = buffers::make_shard_specs(logical_buffers);

    const auto layout_expected = shard::plan_shards(specs);
    if(!layout_expected)
    {
        return std::unexpected(make_error(layout_expected.error().message, layout_expected.error().context));
    }
    const auto layout = layout_expected.value();

    std::unordered_map<std::string, shard::ShardSegment> segments_by_name;
    for(const auto &segment : layout.segments)
    {
        if(segments_by_name.contains(segment.name))
        {
            return std::unexpected(make_error("logical buffer spans multiple device buffers (sharding not yet supported)",
                                              {segment.name}));
        }
        segments_by_name.emplace(segment.name, segment);
    }

    DeviceBufferArena arena{};
    arena.context_ = &context;

    arena.device_buffers_.reserve(layout.device_buffer_sizes.size());
    for(std::size_t i = 0; i < layout.device_buffer_sizes.size(); ++i)
    {
        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = layout.device_buffer_sizes[i];
        buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        const VkResult result = vmaCreateBuffer(context.allocator(), &buffer_info, &alloc_info, &buffer, &allocation, nullptr);
        if(result != VK_SUCCESS)
        {
            arena.destroy();
            return std::unexpected(make_error("vmaCreateBuffer failed", result, {"device buffer", std::to_string(i)}));
        }

        arena.device_buffers_.push_back(DeviceBuffer{buffer, allocation, buffer_info.size});
        context.set_object_name(reinterpret_cast<std::uint64_t>(buffer), VK_OBJECT_TYPE_BUFFER,
                                std::format("cwf_device_buffer_{}", i));
    }

    const auto buffer_views = buffers::make_upload_views(logical_buffers);
    upload::StagingConfig staging_config{};
    staging_config.chunk_bytes = static_cast<std::size_t>(context.staging_ring().size);
    staging_config.alignment = shard::kDefaultAlignment;

    const auto schedule_expected = upload::build_upload_schedule(layout, buffer_views, staging_config);
    if(!schedule_expected)
    {
        arena.destroy();
        return std::unexpected(make_error(schedule_expected.error().message, schedule_expected.error().context));
    }

    auto command_pool_expected = create_command_pool(context);
    if(!command_pool_expected)
    {
        arena.destroy();
        return std::unexpected(command_pool_expected.error());
    }
    VkCommandPool command_pool = command_pool_expected.value();

    auto command_buffer_expected = allocate_command_buffer(context.device(), command_pool);
    if(!command_buffer_expected)
    {
        vkDestroyCommandPool(context.device(), command_pool, nullptr);
        arena.destroy();
        return std::unexpected(command_buffer_expected.error());
    }
    VkCommandBuffer cmd = command_buffer_expected.value();

    const auto upload_status = upload_schedule(context, cmd, schedule_expected.value(), arena.device_buffers_);
    vkDestroyCommandPool(context.device(), command_pool, nullptr);
    if(!upload_status)
    {
        arena.destroy();
        return std::unexpected(upload_status.error());
    }

    for(const auto &logical : logical_buffers)
    {
        const auto segment_it = segments_by_name.find(logical.name);
        if(segment_it == segments_by_name.end())
        {
            arena.destroy();
            return std::unexpected(make_error("missing shard segment for logical buffer", {logical.name}));
        }
        const auto &segment = segment_it->second;
        BufferSlice slice{};
        slice.buffer = arena.device_buffers_.at(segment.device_buffer_index).buffer;
        slice.offset = segment.device_offset;
        slice.size = segment.size_bytes;
        arena.slices_.emplace(logical.name, slice);
    }

    return arena;
}

DeviceBufferArena::DeviceBufferArena(DeviceBufferArena &&other) noexcept
    : context_(other.context_), device_buffers_(std::move(other.device_buffers_)), slices_(std::move(other.slices_))
{
    other.context_ = nullptr;
}

auto DeviceBufferArena::operator=(DeviceBufferArena &&other) noexcept -> DeviceBufferArena &
{
    if(this != &other)
    {
        destroy();
        context_ = other.context_;
        device_buffers_ = std::move(other.device_buffers_);
        slices_ = std::move(other.slices_);
        other.context_ = nullptr;
    }
    return *this;
}

DeviceBufferArena::~DeviceBufferArena()
{
    destroy();
}

void DeviceBufferArena::destroy()
{
    if(context_ == nullptr)
    {
        return;
    }

    for(auto &buffer : device_buffers_)
    {
        if(buffer.buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(context_->allocator(), buffer.buffer, buffer.allocation);
            buffer.buffer = VK_NULL_HANDLE;
            buffer.allocation = nullptr;
        }
    }
    device_buffers_.clear();
    slices_.clear();
    context_ = nullptr;
}

auto DeviceBufferArena::has_slice(std::string_view name) const noexcept -> bool
{
    return slices_.find(name) != slices_.end();
}

auto DeviceBufferArena::slice(std::string_view name) const -> const BufferSlice &
{
    if(const auto it = slices_.find(name); it != slices_.end())
    {
        return it->second;
    }
    throw std::out_of_range(std::format("buffer slice '{}' not found", name));
}

auto DeviceBufferArena::upload_schedule(const VulkanContext &context, VkCommandBuffer cmd,
                                        const upload::UploadSchedule &schedule,
                                        std::span<const DeviceBuffer> device_buffers)
    -> std::expected<void, VulkanError>
{
    const auto &ring = context.staging_ring();
    if(ring.buffer == VK_NULL_HANDLE || ring.mapped == nullptr)
    {
        return std::unexpected(make_error("staging ring not initialized", {"device_buffers"}));
    }

    for(const auto &command : schedule.commands)
    {
        if(command.device_buffer_index >= device_buffers.size())
        {
            return std::unexpected(make_error("upload chunk references invalid device buffer",
                                              {"index=" + std::to_string(command.device_buffer_index)}));
        }

        if(command.bytes.size() > ring.size)
        {
            return std::unexpected(make_error("upload chunk exceeds staging ring capacity",
                                              {"chunk_bytes=" + std::to_string(command.bytes.size()),
                                               "ring_bytes=" + std::to_string(ring.size)}));
        }

        std::memcpy(ring.mapped, command.bytes.data(), command.bytes.size());

        if(auto begin_status = begin_one_time_commands(cmd); !begin_status)
        {
            return begin_status;
        }

        VkBufferCopy region{};
        region.srcOffset = 0U;
        region.dstOffset = static_cast<VkDeviceSize>(command.destination_offset);
        region.size = static_cast<VkDeviceSize>(command.bytes.size());
        vkCmdCopyBuffer(cmd, ring.buffer, device_buffers[command.device_buffer_index].buffer, 1U, &region);

        if(auto submitted = submit_and_wait(context, cmd); !submitted)
        {
            return submitted;
        }
    }

    return {};
}

} // namespace cwf::gpu
