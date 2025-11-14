/**
 * @file newmark_stepper.cpp
 * @brief CPU reference implementation that mirrors the upcoming GPU Newmark orchestration uwu
 *
 * this file binds the packed mesh buffers, material properties, Rayleigh coefficients, and the
 * matrix-free PCG solver into a single cohesive stepping routine. everything mirrors the Vulkan
 * Stage 9 plan (predictor shader → RHS assembly → PCG solve → update shader) but executes on the
 * CPU so we can lock down correctness well before dispatching to GPU. when the GPU plumbing lands,
 * we can swap out the helper methods (`write_predictor`, `apply_state_update`, etc.) with command
 * buffer submissions without rewriting orchestration logic or adaptive heuristics.
 *
 * @note documented with Doxygen 1.15 beta because we crave excessive commentary ✨
 */

#include "cwf/gpu/newmark_stepper.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <cstring>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>

#include "cwf/gpu/pcg.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/physics/newmark.hpp"

namespace cwf::gpu::newmark
{
namespace
{

constexpr auto bytes_per_dof = sizeof(float);
constexpr auto kWaveSize = 64U;

struct alignas(16) GlobalUniform
{
    std::uint32_t element_count{0U};
    std::uint32_t node_count{0U};
    std::uint32_t dof_count{0U};
    std::uint32_t reduction_block{0U};
    float stiffness_scale{0.0F};
    float mass_factor{0.0F};
    float time_step{0.0F};
    std::uint32_t iteration_index{0U};
    std::uint32_t material_count{0U};
    std::uint32_t max_local_nodes{0U};
    std::uint32_t padding0{0U};
    std::uint32_t padding1{0U};
};

struct alignas(16) PredictorUniform
{
    float predictor_gamma{0.0F};
    float predictor_beta{0.0F};
    float padding0{0.0F};
    float padding1{0.0F};
};

struct alignas(16) UpdateUniform
{
    float gamma_over_beta_dt{0.0F};
    float inv_beta_dt2{0.0F};
    float padding0{0.0F};
    float padding1{0.0F};
};

struct MappedBuffer
{
    VkBuffer buffer{VK_NULL_HANDLE};
    VmaAllocation allocation{VK_NULL_HANDLE};
    VkDeviceSize size{0U};
    void *mapped{nullptr};
};

inline void pack_float3_soa(const mesh::pack::Float3SoA &soa, float *destination, std::size_t count) noexcept
{
    for (std::size_t node = 0; node < count; ++node)
    {
        const auto base = node * 3U;
        destination[base + 0U] = soa.x[node];
        destination[base + 1U] = soa.y[node];
        destination[base + 2U] = soa.z[node];
    }
}

inline void unpack_float3_soa(const float *source, mesh::pack::Float3SoA &soa, std::size_t count) noexcept
{
    for (std::size_t node = 0; node < count; ++node)
    {
        const auto base = node * 3U;
        soa.x[node] = source[base + 0U];
        soa.y[node] = source[base + 1U];
        soa.z[node] = source[base + 2U];
    }
}

inline auto ceil_divide(std::uint32_t numerator, std::uint32_t denominator) noexcept -> std::uint32_t
{
    return static_cast<std::uint32_t>((numerator + denominator - 1U) / denominator);
}

} // namespace

class Stepper::GpuRuntime
{
public:
    static auto create(Stepper &stepper, const gpu::VulkanContext &context, gpu::DeviceBufferArena &arena,
                       const std::filesystem::path &shader_directory) -> std::expected<std::unique_ptr<GpuRuntime>, StepError>
    {
        auto ptr = std::unique_ptr<GpuRuntime>(new GpuRuntime(stepper, context, arena));
        if (auto init = ptr->initialize(shader_directory); !init)
        {
            return std::unexpected(init.error());
        }
        return ptr;
    }

    GpuRuntime(const GpuRuntime &) = delete;
    auto operator=(const GpuRuntime &) -> GpuRuntime & = delete;

    ~GpuRuntime()
    {
        destroy();
    }

    auto run_predictor(Stepper &stepper) -> std::expected<void, StepError>;
    auto run_update(Stepper &stepper) -> std::expected<void, StepError>;

private:
    GpuRuntime(Stepper &stepper, const gpu::VulkanContext &context, gpu::DeviceBufferArena &arena)
        : stepper_{&stepper}, context_{&context}, arena_{&arena}
    {
    }

    auto initialize(const std::filesystem::path &shader_directory) -> std::expected<void, StepError>;
    void destroy();

    auto create_command_resources() -> std::expected<void, StepError>;
    auto create_descriptor_layouts() -> std::expected<void, StepError>;
    auto allocate_buffers() -> std::expected<void, StepError>;
    auto create_descriptor_pool_and_sets() -> std::expected<void, StepError>;
    auto create_pipelines(const std::filesystem::path &shader_directory) -> std::expected<void, StepError>;
    [[nodiscard]] auto dispatch(VkPipeline pipeline, VkDescriptorSet set0, VkDescriptorSet set1, VkDescriptorSet set2,
                                std::uint32_t group_count) -> std::expected<void, StepError>;
    [[nodiscard]] auto load_shader_module(const std::filesystem::path &path, VkShaderModule *out_module)
        -> std::expected<void, StepError>;
    [[nodiscard]] auto flush(const MappedBuffer &buffer, VkDeviceSize offset, VkDeviceSize size)
        -> std::expected<void, StepError>;
    [[nodiscard]] auto invalidate(const MappedBuffer &buffer, VkDeviceSize offset, VkDeviceSize size)
        -> std::expected<void, StepError>;

    Stepper *stepper_{};
    const gpu::VulkanContext *context_{};
    gpu::DeviceBufferArena *arena_{};

    VkCommandPool command_pool_{VK_NULL_HANDLE};
    VkCommandBuffer command_buffer_{VK_NULL_HANDLE};

    VkDescriptorSetLayout set0_layout_{VK_NULL_HANDLE};
    VkDescriptorSetLayout set_storage_layout_{VK_NULL_HANDLE};
    VkPipelineLayout pipeline_layout_{VK_NULL_HANDLE};

    VkDescriptorPool descriptor_pool_{VK_NULL_HANDLE};
    VkDescriptorSet descriptor_set0_{VK_NULL_HANDLE};
    VkDescriptorSet predictor_set1_{VK_NULL_HANDLE};
    VkDescriptorSet predictor_set2_{VK_NULL_HANDLE};
    VkDescriptorSet update_set1_{VK_NULL_HANDLE};
    VkDescriptorSet update_set2_{VK_NULL_HANDLE};

    VkShaderModule predictor_module_{VK_NULL_HANDLE};
    VkShaderModule update_module_{VK_NULL_HANDLE};
    VkPipeline predictor_pipeline_{VK_NULL_HANDLE};
    VkPipeline update_pipeline_{VK_NULL_HANDLE};

    MappedBuffer global_uniform_{};
    MappedBuffer predictor_uniform_{};
    MappedBuffer update_uniform_{};

    MappedBuffer current_displacement_{};
    MappedBuffer current_velocity_{};
    MappedBuffer current_acceleration_{};
    MappedBuffer predicted_displacement_{};
    MappedBuffer predicted_velocity_{};
    MappedBuffer correction_{};
    MappedBuffer updated_displacement_{};
    MappedBuffer updated_velocity_{};
    MappedBuffer updated_acceleration_{};
};

auto Stepper::GpuRuntime::initialize(const std::filesystem::path &shader_directory) -> std::expected<void, StepError>
{
    if (auto status = create_command_resources(); !status)
    {
        return status;
    }

    if (auto status = allocate_buffers(); !status)
    {
        return status;
    }

    if (auto status = create_descriptor_layouts(); !status)
    {
        return status;
    }

    if (auto status = create_descriptor_pool_and_sets(); !status)
    {
        return status;
    }

    if (auto status = create_pipelines(shader_directory); !status)
    {
        return status;
    }

    return {};
}

void Stepper::GpuRuntime::destroy()
{
    if (context_ == nullptr)
    {
        return;
    }

    const VkDevice device = context_->device();
    const VmaAllocator allocator = context_->allocator();

    if (command_buffer_ != VK_NULL_HANDLE && command_pool_ != VK_NULL_HANDLE)
    {
        vkFreeCommandBuffers(device, command_pool_, 1U, &command_buffer_);
        command_buffer_ = VK_NULL_HANDLE;
    }

    if (command_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyCommandPool(device, command_pool_, nullptr);
        command_pool_ = VK_NULL_HANDLE;
    }

    if (predictor_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, predictor_pipeline_, nullptr);
        predictor_pipeline_ = VK_NULL_HANDLE;
    }

    if (update_pipeline_ != VK_NULL_HANDLE)
    {
        vkDestroyPipeline(device, update_pipeline_, nullptr);
        update_pipeline_ = VK_NULL_HANDLE;
    }

    if (predictor_module_ != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(device, predictor_module_, nullptr);
        predictor_module_ = VK_NULL_HANDLE;
    }

    if (update_module_ != VK_NULL_HANDLE)
    {
        vkDestroyShaderModule(device, update_module_, nullptr);
        update_module_ = VK_NULL_HANDLE;
    }

    if (pipeline_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyPipelineLayout(device, pipeline_layout_, nullptr);
        pipeline_layout_ = VK_NULL_HANDLE;
    }

    if (set0_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, set0_layout_, nullptr);
        set0_layout_ = VK_NULL_HANDLE;
    }

    if (set_storage_layout_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorSetLayout(device, set_storage_layout_, nullptr);
        set_storage_layout_ = VK_NULL_HANDLE;
    }

    if (descriptor_pool_ != VK_NULL_HANDLE)
    {
        vkDestroyDescriptorPool(device, descriptor_pool_, nullptr);
        descriptor_pool_ = VK_NULL_HANDLE;
    }

    const auto destroy_buffer = [allocator](MappedBuffer &buffer) {
        if (buffer.buffer != VK_NULL_HANDLE)
        {
            vmaDestroyBuffer(allocator, buffer.buffer, buffer.allocation);
            buffer.buffer = VK_NULL_HANDLE;
            buffer.allocation = VK_NULL_HANDLE;
            buffer.mapped = nullptr;
            buffer.size = 0U;
        }
    };

    destroy_buffer(global_uniform_);
    destroy_buffer(predictor_uniform_);
    destroy_buffer(update_uniform_);
    destroy_buffer(current_displacement_);
    destroy_buffer(current_velocity_);
    destroy_buffer(current_acceleration_);
    destroy_buffer(predicted_displacement_);
    destroy_buffer(predicted_velocity_);
    destroy_buffer(correction_);
    destroy_buffer(updated_displacement_);
    destroy_buffer(updated_velocity_);
    destroy_buffer(updated_acceleration_);

    stepper_ = nullptr;
    context_ = nullptr;
    arena_ = nullptr;
}

auto Stepper::GpuRuntime::create_command_resources() -> std::expected<void, StepError>
{
    VkCommandPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    pool_info.queueFamilyIndex = context_->queue_info().family_index;
    pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    if (const VkResult pool_result = vkCreateCommandPool(context_->device(), &pool_info, nullptr, &command_pool_);
        pool_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkCreateCommandPool failed", {"gpu_runtime"}));
    }

    VkCommandBufferAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    alloc_info.commandPool = command_pool_;
    alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    alloc_info.commandBufferCount = 1U;

    if (const VkResult alloc_result = vkAllocateCommandBuffers(context_->device(), &alloc_info, &command_buffer_);
        alloc_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkAllocateCommandBuffers failed", {"gpu_runtime"}));
    }

    context_->set_object_name(reinterpret_cast<std::uint64_t>(command_pool_), VK_OBJECT_TYPE_COMMAND_POOL,
                              "cwf_newmark_command_pool");
    context_->set_object_name(reinterpret_cast<std::uint64_t>(command_buffer_), VK_OBJECT_TYPE_COMMAND_BUFFER,
                              "cwf_newmark_command_buffer");

    return {};
}

auto Stepper::GpuRuntime::create_descriptor_layouts() -> std::expected<void, StepError>
{
    std::array<VkDescriptorSetLayoutBinding, 3U> set0_bindings{};
    set0_bindings[0] = VkDescriptorSetLayoutBinding{
        .binding = 0U,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1U,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };
    set0_bindings[1] = VkDescriptorSetLayoutBinding{
        .binding = 2U,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1U,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };
    set0_bindings[2] = VkDescriptorSetLayoutBinding{
        .binding = 3U,
        .descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 1U,
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .pImmutableSamplers = nullptr,
    };

    VkDescriptorSetLayoutCreateInfo set0_info{};
    set0_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    set0_info.bindingCount = static_cast<std::uint32_t>(set0_bindings.size());
    set0_info.pBindings = set0_bindings.data();

    if (const VkResult layout_result = vkCreateDescriptorSetLayout(context_->device(), &set0_info, nullptr, &set0_layout_);
        layout_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkCreateDescriptorSetLayout failed (set0)", {"gpu_runtime"}));
    }

    std::array<VkDescriptorSetLayoutBinding, 3U> storage_bindings{};
    for (std::uint32_t index = 0U; index < storage_bindings.size(); ++index)
    {
        storage_bindings[index] = VkDescriptorSetLayoutBinding{
            .binding = index,
            .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1U,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = nullptr,
        };
    }

    VkDescriptorSetLayoutCreateInfo storage_info{};
    storage_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    storage_info.bindingCount = static_cast<std::uint32_t>(storage_bindings.size());
    storage_info.pBindings = storage_bindings.data();

    if (const VkResult storage_result = vkCreateDescriptorSetLayout(context_->device(), &storage_info, nullptr,
                                                                    &set_storage_layout_);
        storage_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkCreateDescriptorSetLayout failed (storage)", {"gpu_runtime"}));
    }

    std::array<VkDescriptorSetLayout, 3U> layouts{set0_layout_, set_storage_layout_, set_storage_layout_};

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = static_cast<std::uint32_t>(layouts.size());
    pipeline_layout_info.pSetLayouts = layouts.data();

    if (const VkResult pipeline_layout_result =
            vkCreatePipelineLayout(context_->device(), &pipeline_layout_info, nullptr, &pipeline_layout_);
        pipeline_layout_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkCreatePipelineLayout failed", {"gpu_runtime"}));
    }

    context_->set_object_name(reinterpret_cast<std::uint64_t>(pipeline_layout_), VK_OBJECT_TYPE_PIPELINE_LAYOUT,
                              "cwf_newmark_pipeline_layout");

    return {};
}

auto Stepper::GpuRuntime::allocate_buffers() -> std::expected<void, StepError>
{
    const auto create_buffer = [this](VkDeviceSize size, VkBufferUsageFlags usage, std::string_view name,
                                      MappedBuffer &out) -> std::expected<void, StepError> {
        if (size == 0U)
        {
            return {};
        }

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;
        alloc_info.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

        VmaAllocationInfo allocation_info{};
        const VkResult result =
            vmaCreateBuffer(context_->allocator(), &buffer_info, &alloc_info, &out.buffer, &out.allocation, &allocation_info);
        if (result != VK_SUCCESS)
        {
            return std::unexpected(Stepper::make_error("vmaCreateBuffer failed", {std::string{name}, "gpu_runtime"}));
        }

        out.size = size;
        out.mapped = allocation_info.pMappedData;
        context_->set_object_name(reinterpret_cast<std::uint64_t>(out.buffer), VK_OBJECT_TYPE_BUFFER, name);
        return {};
    };

    const VkDeviceSize node_bytes = static_cast<VkDeviceSize>(stepper_->node_count_) * 3U * sizeof(float);

    if (auto status = create_buffer(sizeof(GlobalUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    "cwf_newmark_uniform_globals", global_uniform_);
        !status)
    {
        return status;
    }

    if (auto status = create_buffer(sizeof(PredictorUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    "cwf_newmark_uniform_predictor", predictor_uniform_);
        !status)
    {
        return status;
    }

    if (auto status = create_buffer(sizeof(UpdateUniform), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                                    "cwf_newmark_uniform_update", update_uniform_);
        !status)
    {
        return status;
    }

    const VkBufferUsageFlags storage_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    if (auto status = create_buffer(node_bytes, storage_usage, "cwf_newmark_current_displacement", current_displacement_);
        !status)
    {
        return status;
    }
    if (auto status = create_buffer(node_bytes, storage_usage, "cwf_newmark_current_velocity", current_velocity_);
        !status)
    {
        return status;
    }
    if (auto status = create_buffer(node_bytes, storage_usage, "cwf_newmark_current_acceleration", current_acceleration_);
        !status)
    {
        return status;
    }
    if (auto status = create_buffer(node_bytes, storage_usage, "cwf_newmark_predicted_displacement", predicted_displacement_);
        !status)
    {
        return status;
    }
    if (auto status = create_buffer(node_bytes, storage_usage, "cwf_newmark_predicted_velocity", predicted_velocity_);
        !status)
    {
        return status;
    }
    if (auto status = create_buffer(node_bytes, storage_usage, "cwf_newmark_correction", correction_);
        !status)
    {
        return status;
    }
    if (auto status = create_buffer(node_bytes, storage_usage, "cwf_newmark_updated_displacement", updated_displacement_);
        !status)
    {
        return status;
    }
    if (auto status = create_buffer(node_bytes, storage_usage, "cwf_newmark_updated_velocity", updated_velocity_);
        !status)
    {
        return status;
    }
    if (auto status = create_buffer(node_bytes, storage_usage, "cwf_newmark_updated_acceleration", updated_acceleration_);
        !status)
    {
        return status;
    }

    return {};
}

auto Stepper::GpuRuntime::create_descriptor_pool_and_sets() -> std::expected<void, StepError>
{
    std::array<VkDescriptorPoolSize, 2U> pool_sizes{};
    pool_sizes[0] = VkDescriptorPoolSize{
        .type = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER,
        .descriptorCount = 3U,
    };
    pool_sizes[1] = VkDescriptorPoolSize{
        .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
        .descriptorCount = 12U,
    };

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.maxSets = 5U;
    pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes = pool_sizes.data();

    if (const VkResult pool_result = vkCreateDescriptorPool(context_->device(), &pool_info, nullptr, &descriptor_pool_);
        pool_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkCreateDescriptorPool failed", {"gpu_runtime"}));
    }

    std::array<VkDescriptorSetLayout, 5U> layouts{set0_layout_, set_storage_layout_, set_storage_layout_,
                                                  set_storage_layout_, set_storage_layout_};

    VkDescriptorSetAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc_info.descriptorPool = descriptor_pool_;
    alloc_info.descriptorSetCount = static_cast<std::uint32_t>(layouts.size());
    alloc_info.pSetLayouts = layouts.data();

    std::array<VkDescriptorSet, 5U> sets{};
    if (const VkResult alloc_result = vkAllocateDescriptorSets(context_->device(), &alloc_info, sets.data());
        alloc_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkAllocateDescriptorSets failed", {"gpu_runtime"}));
    }

    descriptor_set0_ = sets[0];
    predictor_set1_ = sets[1];
    predictor_set2_ = sets[2];
    update_set1_ = sets[3];
    update_set2_ = sets[4];

    const auto buffer_info = [](const MappedBuffer &buffer) {
        return VkDescriptorBufferInfo{
            .buffer = buffer.buffer,
            .offset = 0U,
            .range = buffer.size,
        };
    };

    std::vector<VkWriteDescriptorSet> writes{};
    writes.reserve(15U);

    const auto add_write = [&writes](VkDescriptorSet set, std::uint32_t binding, VkDescriptorType type,
                                     const VkDescriptorBufferInfo &info) {
        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = set;
        write.dstBinding = binding;
        write.dstArrayElement = 0U;
        write.descriptorCount = 1U;
        write.descriptorType = type;
        write.pBufferInfo = &info;
        writes.emplace_back(write);
    };

    const auto globals_info = buffer_info(global_uniform_);
    const auto predictor_uniform_info = buffer_info(predictor_uniform_);
    const auto update_uniform_info = buffer_info(update_uniform_);
    add_write(descriptor_set0_, 0U, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, globals_info);
    add_write(descriptor_set0_, 2U, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, predictor_uniform_info);
    add_write(descriptor_set0_, 3U, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, update_uniform_info);

    const auto current_disp_info = buffer_info(current_displacement_);
    const auto current_vel_info = buffer_info(current_velocity_);
    const auto current_acc_info = buffer_info(current_acceleration_);
    add_write(predictor_set1_, 0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, current_disp_info);
    add_write(predictor_set1_, 1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, current_vel_info);
    add_write(predictor_set1_, 2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, current_acc_info);

    const auto predicted_disp_info = buffer_info(predicted_displacement_);
    const auto predicted_vel_info = buffer_info(predicted_velocity_);
    add_write(predictor_set2_, 0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, predicted_disp_info);
    add_write(predictor_set2_, 1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, predicted_vel_info);
    add_write(predictor_set2_, 2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, predicted_vel_info);

    const auto correction_info = buffer_info(correction_);
    add_write(update_set1_, 0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, predicted_disp_info);
    add_write(update_set1_, 1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, predicted_vel_info);
    add_write(update_set1_, 2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, correction_info);

    const auto updated_disp_info = buffer_info(updated_displacement_);
    const auto updated_vel_info = buffer_info(updated_velocity_);
    const auto updated_acc_info = buffer_info(updated_acceleration_);
    add_write(update_set2_, 0U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, updated_disp_info);
    add_write(update_set2_, 1U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, updated_vel_info);
    add_write(update_set2_, 2U, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, updated_acc_info);

    vkUpdateDescriptorSets(context_->device(), static_cast<std::uint32_t>(writes.size()), writes.data(), 0U, nullptr);

    return {};
}

auto Stepper::GpuRuntime::load_shader_module(const std::filesystem::path &path, VkShaderModule *out_module)
    -> std::expected<void, StepError>
{
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file)
    {
        return std::unexpected(Stepper::make_error("failed to open shader", {path.string()}));
    }

    const auto size = static_cast<std::streamsize>(file.tellg());
    if (size <= 0)
    {
        return std::unexpected(Stepper::make_error("shader file empty", {path.string()}));
    }

    file.seekg(0, std::ios::beg);
    const std::size_t byte_count = static_cast<std::size_t>(size);
    if ((byte_count % sizeof(std::uint32_t)) != 0U)
    {
        return std::unexpected(Stepper::make_error("shader byte size misaligned", {path.string()}));
    }

    std::vector<std::uint32_t> words(byte_count / sizeof(std::uint32_t));
    if (!file.read(reinterpret_cast<char *>(words.data()), size))
    {
        return std::unexpected(Stepper::make_error("failed to read shader", {path.string()}));
    }

    VkShaderModuleCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    create_info.codeSize = byte_count;
    create_info.pCode = words.data();

    if (const VkResult result = vkCreateShaderModule(context_->device(), &create_info, nullptr, out_module);
        result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkCreateShaderModule failed", {path.string()}));
    }

    return {};
}

auto Stepper::GpuRuntime::create_pipelines(const std::filesystem::path &shader_directory)
    -> std::expected<void, StepError>
{
    const auto predictor_path = shader_directory / "newmark_predictor.spv";
    const auto update_path = shader_directory / "newmark_update.spv";

    if (auto status = load_shader_module(predictor_path, &predictor_module_); !status)
    {
        return status;
    }

    if (auto status = load_shader_module(update_path, &update_module_); !status)
    {
        return status;
    }

    auto make_info = [this](VkShaderModule module) {
        VkComputePipelineCreateInfo info{};
        info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;

        VkPipelineShaderStageCreateInfo stage_info{};
        stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage_info.module = module;
        stage_info.pName = "main";

        info.stage = stage_info;
        info.layout = pipeline_layout_;
        return info;
    };

    const auto predictor_info = make_info(predictor_module_);
    const auto update_info = make_info(update_module_);

    if (const VkResult predictor_result = vkCreateComputePipelines(context_->device(), VK_NULL_HANDLE, 1U, &predictor_info,
                                                                   nullptr, &predictor_pipeline_);
        predictor_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkCreateComputePipelines failed (predictor)", {"gpu_runtime"}));
    }

    if (const VkResult update_result = vkCreateComputePipelines(context_->device(), VK_NULL_HANDLE, 1U, &update_info,
                                                                nullptr, &update_pipeline_);
        update_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkCreateComputePipelines failed (update)", {"gpu_runtime"}));
    }

    context_->set_object_name(reinterpret_cast<std::uint64_t>(predictor_pipeline_), VK_OBJECT_TYPE_PIPELINE,
                              "cwf_newmark_predictor");
    context_->set_object_name(reinterpret_cast<std::uint64_t>(update_pipeline_), VK_OBJECT_TYPE_PIPELINE,
                              "cwf_newmark_update");

    return {};
}

auto Stepper::GpuRuntime::flush(const MappedBuffer &buffer, VkDeviceSize offset, VkDeviceSize size)
    -> std::expected<void, StepError>
{
    if (buffer.allocation == VK_NULL_HANDLE)
    {
        return {};
    }
    const VkResult result = vmaFlushAllocation(context_->allocator(), buffer.allocation, offset, size);
    if (result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vmaFlushAllocation failed", {"gpu_runtime"}));
    }
    return {};
}

auto Stepper::GpuRuntime::invalidate(const MappedBuffer &buffer, VkDeviceSize offset, VkDeviceSize size)
    -> std::expected<void, StepError>
{
    if (buffer.allocation == VK_NULL_HANDLE)
    {
        return {};
    }
    const VkResult result = vmaInvalidateAllocation(context_->allocator(), buffer.allocation, offset, size);
    if (result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vmaInvalidateAllocation failed", {"gpu_runtime"}));
    }
    return {};
}

auto Stepper::GpuRuntime::dispatch(VkPipeline pipeline, VkDescriptorSet set0, VkDescriptorSet set1, VkDescriptorSet set2,
                                   std::uint32_t group_count) -> std::expected<void, StepError>
{
    if (group_count == 0U)
    {
        return {};
    }

    if (const VkResult reset_result = vkResetCommandBuffer(command_buffer_, 0U); reset_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkResetCommandBuffer failed", {"gpu_runtime"}));
    }

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

    if (const VkResult begin_result = vkBeginCommandBuffer(command_buffer_, &begin_info); begin_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkBeginCommandBuffer failed", {"gpu_runtime"}));
    }

    context_->push_debug_label(command_buffer_, "Newmark Dispatch", {0.3F, 0.6F, 0.9F, 1.0F});

    vkCmdBindPipeline(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    std::array<VkDescriptorSet, 3U> sets{set0, set1, set2};
    vkCmdBindDescriptorSets(command_buffer_, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout_, 0U,
                            static_cast<std::uint32_t>(sets.size()), sets.data(), 0U, nullptr);
    vkCmdDispatch(command_buffer_, group_count, 1U, 1U);

    context_->pop_debug_label(command_buffer_);

    if (const VkResult end_result = vkEndCommandBuffer(command_buffer_); end_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkEndCommandBuffer failed", {"gpu_runtime"}));
    }

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1U;
    submit_info.pCommandBuffers = &command_buffer_;

    const VkQueue queue = context_->queue_info().queue;
    if (const VkResult submit_result = vkQueueSubmit(queue, 1U, &submit_info, VK_NULL_HANDLE); submit_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkQueueSubmit failed", {"gpu_runtime"}));
    }

    if (const VkResult wait_result = vkQueueWaitIdle(queue); wait_result != VK_SUCCESS)
    {
        return std::unexpected(Stepper::make_error("vkQueueWaitIdle failed", {"gpu_runtime"}));
    }

    return {};
}

auto Stepper::GpuRuntime::run_predictor(Stepper &stepper) -> std::expected<void, StepError>
{
    const auto node_count = static_cast<std::size_t>(stepper.node_count_);
    if (node_count == 0U)
    {
        return {};
    }

    GlobalUniform globals{};
    globals.element_count = static_cast<std::uint32_t>(stepper.packing_->metadata.element_count);
    globals.node_count = static_cast<std::uint32_t>(node_count);
    globals.dof_count = static_cast<std::uint32_t>(stepper.dof_count_);
    globals.reduction_block = static_cast<std::uint32_t>(stepper.matrix_system_.reduction_block);
    globals.stiffness_scale = static_cast<float>(stepper.matrix_system_.stiffness_scale);
    globals.mass_factor = static_cast<float>(stepper.matrix_system_.mass_factor);
    globals.time_step = static_cast<float>(stepper.current_dt_);
    globals.iteration_index = static_cast<std::uint32_t>(stepper.frame_index_);
    globals.material_count = static_cast<std::uint32_t>(stepper.materials_storage_.size());
    globals.max_local_nodes = 8U;

    PredictorUniform predictor{};
    predictor.predictor_gamma = static_cast<float>(stepper.gamma_);
    predictor.predictor_beta = static_cast<float>(stepper.beta_);

    std::memcpy(global_uniform_.mapped, &globals, sizeof(globals));
    std::memcpy(predictor_uniform_.mapped, &predictor, sizeof(predictor));

    auto &nodes = stepper.node_buffers();
    pack_float3_soa(nodes.displacement, static_cast<float *>(current_displacement_.mapped), node_count);
    pack_float3_soa(nodes.velocity, static_cast<float *>(current_velocity_.mapped), node_count);
    pack_float3_soa(nodes.acceleration, static_cast<float *>(current_acceleration_.mapped), node_count);

    if (auto status = flush(global_uniform_, 0U, sizeof(globals)); !status)
    {
        return status;
    }
    if (auto status = flush(predictor_uniform_, 0U, sizeof(predictor)); !status)
    {
        return status;
    }
    if (auto status = flush(current_displacement_, 0U, current_displacement_.size); !status)
    {
        return status;
    }
    if (auto status = flush(current_velocity_, 0U, current_velocity_.size); !status)
    {
        return status;
    }
    if (auto status = flush(current_acceleration_, 0U, current_acceleration_.size); !status)
    {
        return status;
    }

    const auto group_count = ceil_divide(globals.node_count, kWaveSize);
    if (auto status = dispatch(predictor_pipeline_, descriptor_set0_, predictor_set1_, predictor_set2_, group_count);
        !status)
    {
        return status;
    }

    if (auto status = invalidate(predicted_displacement_, 0U, predicted_displacement_.size); !status)
    {
        return status;
    }
    if (auto status = invalidate(predicted_velocity_, 0U, predicted_velocity_.size); !status)
    {
        return status;
    }

    unpack_float3_soa(static_cast<const float *>(predicted_displacement_.mapped), stepper.predicted_displacement_, node_count);
    unpack_float3_soa(static_cast<const float *>(predicted_velocity_.mapped), stepper.predicted_velocity_, node_count);

    return {};
}

auto Stepper::GpuRuntime::run_update(Stepper &stepper) -> std::expected<void, StepError>
{
    const auto node_count = static_cast<std::size_t>(stepper.node_count_);
    if (node_count == 0U)
    {
        return {};
    }

    GlobalUniform globals{};
    globals.element_count = static_cast<std::uint32_t>(stepper.packing_->metadata.element_count);
    globals.node_count = static_cast<std::uint32_t>(node_count);
    globals.dof_count = static_cast<std::uint32_t>(stepper.dof_count_);
    globals.reduction_block = static_cast<std::uint32_t>(stepper.matrix_system_.reduction_block);
    globals.stiffness_scale = static_cast<float>(stepper.matrix_system_.stiffness_scale);
    globals.mass_factor = static_cast<float>(stepper.matrix_system_.mass_factor);
    globals.time_step = static_cast<float>(stepper.current_dt_);
    globals.iteration_index = static_cast<std::uint32_t>(stepper.frame_index_);
    globals.material_count = static_cast<std::uint32_t>(stepper.materials_storage_.size());
    globals.max_local_nodes = 8U;

    UpdateUniform update{};
    update.gamma_over_beta_dt = static_cast<float>(stepper.update_scalars_.gamma_over_beta_dt);
    update.inv_beta_dt2 = static_cast<float>(stepper.update_scalars_.inv_beta_dt2);

    std::memcpy(global_uniform_.mapped, &globals, sizeof(globals));
    std::memcpy(update_uniform_.mapped, &update, sizeof(update));

    pack_float3_soa(stepper.predicted_displacement_, static_cast<float *>(predicted_displacement_.mapped), node_count);
    pack_float3_soa(stepper.predicted_velocity_, static_cast<float *>(predicted_velocity_.mapped), node_count);

    const auto correction_span = stepper.solver_vectors_.solution;
    auto *correction_ptr = static_cast<float *>(correction_.mapped);
    for (std::size_t node = 0; node < node_count; ++node)
    {
        const auto base = node * 3U;
        correction_ptr[base + 0U] = correction_span[base + 0U];
        correction_ptr[base + 1U] = correction_span[base + 1U];
        correction_ptr[base + 2U] = correction_span[base + 2U];
    }

    if (auto status = flush(global_uniform_, 0U, sizeof(globals)); !status)
    {
        return status;
    }
    if (auto status = flush(update_uniform_, 0U, sizeof(update)); !status)
    {
        return status;
    }
    if (auto status = flush(predicted_displacement_, 0U, predicted_displacement_.size); !status)
    {
        return status;
    }
    if (auto status = flush(predicted_velocity_, 0U, predicted_velocity_.size); !status)
    {
        return status;
    }
    if (auto status = flush(correction_, 0U, correction_.size); !status)
    {
        return status;
    }

    const auto group_count = ceil_divide(globals.node_count, kWaveSize);
    if (auto status = dispatch(update_pipeline_, descriptor_set0_, update_set1_, update_set2_, group_count); !status)
    {
        return status;
    }

    if (auto status = invalidate(updated_displacement_, 0U, updated_displacement_.size); !status)
    {
        return status;
    }
    if (auto status = invalidate(updated_velocity_, 0U, updated_velocity_.size); !status)
    {
        return status;
    }
    if (auto status = invalidate(updated_acceleration_, 0U, updated_acceleration_.size); !status)
    {
        return status;
    }

    auto &nodes = stepper.node_buffers();
    unpack_float3_soa(static_cast<const float *>(updated_displacement_.mapped), nodes.displacement, node_count);
    unpack_float3_soa(static_cast<const float *>(updated_velocity_.mapped), nodes.velocity, node_count);
    unpack_float3_soa(static_cast<const float *>(updated_acceleration_.mapped), nodes.acceleration, node_count);

    return {};
}
Stepper::Stepper(mesh::pack::PackingResult &packing,
                 std::span<const physics::materials::ElasticProperties> materials,
                 physics::materials::RayleighCoefficients rayleigh,
                 const config::SolverSettings &solver_settings,
                 const config::TimeSettings &time_settings,
                 AdaptivePolicy adaptive_policy)
    : packing_{&packing},
      materials_storage_{materials.begin(), materials.end()},
      rayleigh_{rayleigh},
      solver_settings_{solver_settings},
      time_settings_{time_settings},
      adaptive_policy_{adaptive_policy},
      node_count_{packing.metadata.node_count},
      dof_count_{packing.metadata.dof_count}
{
    current_dt_ = time_settings_.initial_dt > 0.0 ? time_settings_.initial_dt : 1.0e-3;
    coeffs_ = physics::newmark::make_coefficients(current_dt_, beta_, gamma_);
    update_scalars_ = physics::newmark::compute_update_scalars(coeffs_);

    rhs_.assign(dof_count_, 0.0F);
    damping_rhs_.assign(dof_count_, 0.0F);
    damping_output_.assign(dof_count_, 0.0F);
    external_force_.assign(dof_count_, 0.0F);

    predicted_displacement_.resize(node_count_);
    predicted_velocity_.resize(node_count_);

    auto &buffers = packing.buffers;

    matrix_system_ = pcg::MatrixFreeSystem{
        .element_connectivity = std::span<const std::uint32_t>{buffers.elements.connectivity},
        .element_gradients = std::span<const float>{buffers.elements.gradients},
        .element_volume = std::span<const float>{buffers.elements.volume},
        .element_material_index = std::span<const std::uint32_t>{buffers.elements.material_index},
        .materials = std::span<const physics::materials::ElasticProperties>{materials_storage_},
        .lumped_mass = std::span<const float>{buffers.nodes.lumped_mass},
        .bc_mask = std::span<const std::uint32_t>{buffers.nodes.bc_mask},
        .node_count = node_count_,
        .element_count = packing.metadata.element_count,
        .dof_count = dof_count_,
        .stiffness_scale = 1.0,
        .mass_factor = 0.0,
        .reduction_block = packing.metadata.reduction_block,
        .reduction_partials = packing.metadata.reduction_partials,
    };

    stiffness_only_system_ = matrix_system_;
    stiffness_only_system_.stiffness_scale = 1.0;
    stiffness_only_system_.mass_factor = 0.0;

    auto &solver_buffers = buffers.solver;
    solver_vectors_ = pcg::PcgVectors{
        .solution = std::span<float>(solver_buffers.x.data(), solver_buffers.x.size()),
        .residual = std::span<float>(solver_buffers.r.data(), solver_buffers.r.size()),
        .search_direction = std::span<float>(solver_buffers.p.data(), solver_buffers.p.size()),
        .preconditioned = std::span<float>(solver_buffers.z.data(), solver_buffers.z.size()),
        .matvec = std::span<float>(solver_buffers.Ap.data(), solver_buffers.Ap.size()),
        .partials = std::span<double>(solver_buffers.partials.data(), solver_buffers.partials.size()),
    };

    refresh_coefficients();
    update_matrix_free_scalars();
}

Stepper::~Stepper() = default;

auto Stepper::enable_gpu(const gpu::VulkanContext &context, gpu::DeviceBufferArena &arena,
                         const std::filesystem::path &shader_directory) -> std::expected<void, StepError>
{
    if (gpu_runtime_)
    {
        return {};
    }

    if (!std::filesystem::is_directory(shader_directory))
    {
        return std::unexpected(make_error("shader directory not found", {shader_directory.string()}));
    }

    auto runtime_expected = GpuRuntime::create(*this, context, arena, shader_directory);
    if (!runtime_expected)
    {
        return std::unexpected(runtime_expected.error());
    }

    gpu_runtime_ = std::move(runtime_expected.value());
    return {};
}

auto Stepper::step(double simulation_time_seconds, bool paused_mode) -> std::expected<StepTelemetry, StepError>
{
    accumulated_time_ = simulation_time_seconds;

    refresh_coefficients();
    update_matrix_free_scalars();
    if (gpu_runtime_)
    {
        if (auto status = gpu_runtime_->run_predictor(*this); !status)
        {
            return std::unexpected(status.error());
        }
    }
    else
    {
        write_predictor();
    }
    flatten_external_force();

    if (auto rhs_status = assemble_rhs(); !rhs_status)
    {
        return std::unexpected(rhs_status.error());
    }

    clamp_dirichlet_rhs();

    const double tolerance = paused_mode ? solver_settings_.pause_tolerance : solver_settings_.runtime_tolerance;
    const pcg::PcgSettings pcg_settings{
        .max_iterations = static_cast<std::size_t>(solver_settings_.max_iterations),
        .relative_tolerance = tolerance,
        .warm_start = warm_start_enabled_,
    };

    const auto result =
        pcg::solve_pcg(matrix_system_, std::span<const float>{rhs_.data(), rhs_.size()}, pcg_settings, solver_vectors_, matrix_workspace_);

    if (!result)
    {
        return std::unexpected(make_error("pcg solve failed", {result.error().message}));
    }

    if (gpu_runtime_)
    {
        if (auto status = gpu_runtime_->run_update(*this); !status)
        {
            return std::unexpected(status.error());
        }
    }
    else
    {
        apply_state_update();
    }

    StepTelemetry telemetry{
        .simulation_time = simulation_time_seconds,
        .time_step = current_dt_,
        .applied_tolerance = tolerance,
        .paused_mode = paused_mode,
        .pcg = result.value(),
    };

    adapt_timestep(result.value(), telemetry);
    ++frame_index_;
    accumulated_time_ = simulation_time_seconds + current_dt_;

    return telemetry;
}

auto Stepper::assemble_rhs() -> std::expected<void, StepError>
{
    auto &nodes = node_buffers();
    const auto mass = matrix_system_.lumped_mass;

    const double a0 = coeffs_.a0;
    const double a1 = coeffs_.a1;
    const double a2 = coeffs_.a2;
    const double a3 = coeffs_.a3;
    const double a4 = coeffs_.a4;
    const double a5 = coeffs_.a5;

    for (std::size_t node = 0; node < node_count_; ++node)
    {
        const double mass_value = static_cast<double>(mass[node]);
        const auto base = node * 3U;

        const auto u = std::array<double, 3>{static_cast<double>(nodes.displacement.x[node]),
                                             static_cast<double>(nodes.displacement.y[node]),
                                             static_cast<double>(nodes.displacement.z[node])};
        const auto v = std::array<double, 3>{static_cast<double>(nodes.velocity.x[node]),
                                             static_cast<double>(nodes.velocity.y[node]),
                                             static_cast<double>(nodes.velocity.z[node])};
        const auto acc = std::array<double, 3>{static_cast<double>(nodes.acceleration.x[node]),
                                               static_cast<double>(nodes.acceleration.y[node]),
                                               static_cast<double>(nodes.acceleration.z[node])};

        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            const double mass_term = mass_value * (a0 * u[axis] + a2 * v[axis] + a3 * acc[axis]);
            const double damping_term = a1 * u[axis] + a4 * v[axis] + a5 * acc[axis];
            const double force = static_cast<double>(external_force_[base + axis]);
            const double total = force + mass_term + rayleigh_.alpha * mass_value * damping_term;
            rhs_[base + axis] = static_cast<float>(total);
            damping_rhs_[base + axis] = static_cast<float>(damping_term);
        }
    }

    if (std::abs(rayleigh_.beta) > std::numeric_limits<double>::epsilon())
    {
        auto apply_status = pcg::apply_keff(stiffness_only_system_,
                                            std::span<const float>{damping_rhs_.data(), damping_rhs_.size()},
                                            std::span<float>{damping_output_.data(), damping_output_.size()},
                                            matrix_workspace_);
        if (!apply_status)
        {
            return std::unexpected(make_error("failed to apply stiffness to damping term", {apply_status.error().message}));
        }
        for (std::size_t dof = 0; dof < dof_count_; ++dof)
        {
            rhs_[dof] += static_cast<float>(rayleigh_.beta) * damping_output_[dof];
        }
    }

    return {};
}

void Stepper::clamp_dirichlet_rhs()
{
    auto &nodes = node_buffers();
    for (std::size_t node = 0; node < node_count_; ++node)
    {
        const auto mask = nodes.bc_mask[node];
        if (mask == 0U)
        {
            continue;
        }
        const auto base = node * 3U;
        if ((mask & axis_bit(0U)) != 0U)
        {
            rhs_[base + 0U] = static_cast<float>(nodes.bc_value.x[node] - nodes.displacement.x[node]);
        }
        if ((mask & axis_bit(1U)) != 0U)
        {
            rhs_[base + 1U] = static_cast<float>(nodes.bc_value.y[node] - nodes.displacement.y[node]);
        }
        if ((mask & axis_bit(2U)) != 0U)
        {
            rhs_[base + 2U] = static_cast<float>(nodes.bc_value.z[node] - nodes.displacement.z[node]);
        }
    }
}

void Stepper::write_predictor()
{
    auto &nodes = node_buffers();
    const double dt = current_dt_;
    const double dt_sq = dt * dt;
    const double disp_factor = 0.5 - beta_;
    const double vel_factor = 1.0 - gamma_;

    for (std::size_t node = 0; node < node_count_; ++node)
    {
        const auto u = std::array<double, 3>{static_cast<double>(nodes.displacement.x[node]),
                                             static_cast<double>(nodes.displacement.y[node]),
                                             static_cast<double>(nodes.displacement.z[node])};
        const auto v = std::array<double, 3>{static_cast<double>(nodes.velocity.x[node]),
                                             static_cast<double>(nodes.velocity.y[node]),
                                             static_cast<double>(nodes.velocity.z[node])};
        const auto acc = std::array<double, 3>{static_cast<double>(nodes.acceleration.x[node]),
                                               static_cast<double>(nodes.acceleration.y[node]),
                                               static_cast<double>(nodes.acceleration.z[node])};

        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            const double u_pred = u[axis] + dt * v[axis] + disp_factor * dt_sq * acc[axis];
            const double v_pred = v[axis] + vel_factor * dt * acc[axis];
            switch (axis)
            {
            case 0:
                predicted_displacement_.x[node] = static_cast<float>(u_pred);
                predicted_velocity_.x[node] = static_cast<float>(v_pred);
                break;
            case 1:
                predicted_displacement_.y[node] = static_cast<float>(u_pred);
                predicted_velocity_.y[node] = static_cast<float>(v_pred);
                break;
            default:
                predicted_displacement_.z[node] = static_cast<float>(u_pred);
                predicted_velocity_.z[node] = static_cast<float>(v_pred);
                break;
            }
        }
    }
}

void Stepper::apply_state_update()
{
    auto &nodes = node_buffers();
    const auto delta = solver_vectors_.solution;
    const float gamma_over_beta_dt = static_cast<float>(update_scalars_.gamma_over_beta_dt);
    const float inv_beta_dt2 = static_cast<float>(update_scalars_.inv_beta_dt2);

    for (std::size_t node = 0; node < node_count_; ++node)
    {
        const auto base = node * 3U;
        const float dx = delta[base + 0U];
        const float dy = delta[base + 1U];
        const float dz = delta[base + 2U];

        nodes.displacement.x[node] = predicted_displacement_.x[node] + dx;
        nodes.displacement.y[node] = predicted_displacement_.y[node] + dy;
        nodes.displacement.z[node] = predicted_displacement_.z[node] + dz;

        nodes.acceleration.x[node] = inv_beta_dt2 * dx;
        nodes.acceleration.y[node] = inv_beta_dt2 * dy;
        nodes.acceleration.z[node] = inv_beta_dt2 * dz;

        nodes.velocity.x[node] = predicted_velocity_.x[node] + gamma_over_beta_dt * dx;
        nodes.velocity.y[node] = predicted_velocity_.y[node] + gamma_over_beta_dt * dy;
        nodes.velocity.z[node] = predicted_velocity_.z[node] + gamma_over_beta_dt * dz;
    }
}

void Stepper::refresh_coefficients()
{
    coeffs_ = physics::newmark::make_coefficients(current_dt_, beta_, gamma_);
    update_scalars_ = physics::newmark::compute_update_scalars(coeffs_);
}

void Stepper::update_matrix_free_scalars()
{
    matrix_system_.stiffness_scale = 1.0 + coeffs_.a1 * rayleigh_.beta;
    matrix_system_.mass_factor = coeffs_.a0 + coeffs_.a1 * rayleigh_.alpha;
}

void Stepper::adapt_timestep(const pcg::PcgTelemetry &pcg_stats, StepTelemetry &telemetry)
{
    telemetry.dt_increased = false;
    telemetry.dt_decreased = false;
    telemetry.dt_clamped_min = false;
    telemetry.dt_clamped_max = false;

    if (!time_settings_.adaptive)
    {
        return;
    }

    const double low_iteration_threshold =
        adaptive_policy_.low_iteration_ratio * static_cast<double>(solver_settings_.max_iterations);

    const double iteration_count = static_cast<double>(pcg_stats.iterations);

    if (iteration_count <= low_iteration_threshold)
    {
        current_dt_ *= adaptive_policy_.increase_factor;
        telemetry.dt_increased = true;
    }
    else if (!pcg_stats.converged)
    {
        current_dt_ *= adaptive_policy_.decrease_factor;
        telemetry.dt_decreased = true;
    }

    if (time_settings_.min_dt > 0.0 && current_dt_ <= time_settings_.min_dt)
    {
        current_dt_ = time_settings_.min_dt;
        telemetry.dt_clamped_min = true;
    }

    if (time_settings_.max_dt > 0.0 && current_dt_ >= time_settings_.max_dt)
    {
        current_dt_ = time_settings_.max_dt;
        telemetry.dt_clamped_max = true;
    }
}

void Stepper::flatten_external_force()
{
    const auto &forces = node_buffers().external_force;
    for (std::size_t node = 0; node < node_count_; ++node)
    {
        const auto base = node * 3U;
        external_force_[base + 0U] = forces.x[node];
        external_force_[base + 1U] = forces.y[node];
        external_force_[base + 2U] = forces.z[node];
    }
}

auto Stepper::node_buffers() -> mesh::pack::NodeBuffers &
{
    return packing_->buffers.nodes;
}

auto Stepper::make_error(std::string message, std::initializer_list<std::string> ctx) -> StepError
{
    StepError error{};
    error.message = std::move(message);
    error.context.assign(ctx.begin(), ctx.end());
    return error;
}

constexpr auto Stepper::axis_bit(std::size_t axis) noexcept -> std::uint32_t
{
    return static_cast<std::uint32_t>(1U << static_cast<unsigned int>(axis));
}

} // namespace cwf::gpu::newmark
