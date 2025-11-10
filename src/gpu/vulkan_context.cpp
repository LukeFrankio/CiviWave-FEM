#include "cwf/gpu/vulkan_context.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <format>
#include <initializer_list>
#include <limits>
#include <print>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

namespace cwf::gpu
{
namespace
{

constexpr std::array required_instance_extensions{
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
    VK_EXT_DEBUG_UTILS_EXTENSION_NAME
};

constexpr std::array required_device_extensions{
    VK_EXT_DESCRIPTOR_BUFFER_EXTENSION_NAME
};

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

[[nodiscard]] auto version_to_string(std::uint32_t version) -> std::string
{
    const auto major = VK_API_VERSION_MAJOR(version);
    const auto minor = VK_API_VERSION_MINOR(version);
    const auto patch = VK_API_VERSION_PATCH(version);
    return std::format("{}.{}.{}", major, minor, patch);
}

[[nodiscard]] auto enumerate_instance_layers() -> std::vector<VkLayerProperties>
{
    std::uint32_t count = 0U;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    std::vector<VkLayerProperties> layers(count);
    if(count != 0U)
    {
        vkEnumerateInstanceLayerProperties(&count, layers.data());
    }
    return layers;
}

[[nodiscard]] auto has_layer(std::span<const VkLayerProperties> layers, std::string_view name) -> bool
{
    return std::ranges::any_of(layers, [name](const VkLayerProperties &layer) {
        return std::string_view{layer.layerName} == name;
    });
}

[[nodiscard]] auto enumerate_instance_extensions() -> std::vector<VkExtensionProperties>
{
    std::uint32_t count = 0U;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> extensions(count);
    if(count != 0U)
    {
        vkEnumerateInstanceExtensionProperties(nullptr, &count, extensions.data());
    }
    return extensions;
}

[[nodiscard]] auto has_extension(std::span<const VkExtensionProperties> extensions, std::string_view name) -> bool
{
    return std::ranges::any_of(extensions, [name](const VkExtensionProperties &ext) {
        return std::string_view{ext.extensionName} == name;
    });
}

[[nodiscard]] auto enumerate_device_extensions(VkPhysicalDevice device) -> std::vector<VkExtensionProperties>
{
    std::uint32_t count = 0U;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> exts(count);
    if(count != 0U)
    {
        vkEnumerateDeviceExtensionProperties(device, nullptr, &count, exts.data());
    }
    return exts;
}

[[nodiscard]] auto device_type_to_string(VkPhysicalDeviceType type) -> std::string_view
{
    switch(type)
    {
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
        return "integrated";
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
        return "discrete";
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:
        return "virtual";
    case VK_PHYSICAL_DEVICE_TYPE_CPU:
        return "cpu";
    default:
        return "other";
    }
}

[[nodiscard]] auto pick_compute_queue_family(VkPhysicalDevice device) -> std::optional<QueueInfo>
{
    std::uint32_t family_count = 0U;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());

    for(std::uint32_t i = 0U; i < family_count; ++i)
    {
        const auto &props = families[i];
        const bool  compute = (props.queueFlags & VK_QUEUE_COMPUTE_BIT) != 0U;
        const bool  timestamps = props.timestampValidBits > 0U;
        if(compute && timestamps)
        {
            QueueInfo info{};
            info.family_index  = i;
            info.queue_index   = 0U;
            info.timestamp_bits = props.timestampValidBits;
            info.queue         = VK_NULL_HANDLE; // filled after logical device creation
            return info;
        }
    }

    return std::nullopt;
}

[[nodiscard]] auto score_device(const VkPhysicalDeviceProperties &props) -> std::uint32_t
{
    std::uint32_t score = 0U;
    if(props.vendorID == 0x1002U)
    {
        score += 1000U; // AMD vendor bonus
    }
    if(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU)
    {
        score += 500U;
    }
    if(props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU)
    {
        score += 250U; // discrete fallback preference over CPU
    }
    score += std::min<std::uint32_t>(props.limits.maxComputeWorkGroupInvocations, 2048U);
    return score;
}

[[nodiscard]] auto build_instance(const ContextCreateInfo &info)
    -> std::expected<std::tuple<VkInstance, VkDebugUtilsMessengerEXT>, VulkanError>
{
    const auto layers = enumerate_instance_layers();
    const auto extensions = enumerate_instance_extensions();

    std::vector<const char *> enabled_layers;
    if(info.enable_validation && has_layer(layers, "VK_LAYER_KHRONOS_validation"))
    {
        enabled_layers.push_back("VK_LAYER_KHRONOS_validation");
    }

    std::vector<const char *> enabled_extensions;
    enabled_extensions.reserve(required_instance_extensions.size());
    for(const auto &ext_name : required_instance_extensions)
    {
        if(!has_extension(extensions, ext_name))
        {
            return std::unexpected(make_error(std::format("missing required instance extension {}", ext_name), {"vkCreateInstance"}));
        }
        enabled_extensions.push_back(ext_name);
    }

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "CiviWave-FEM";
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.pEngineName = "cwf";
    app_info.engineVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &app_info;
    create_info.enabledLayerCount = static_cast<std::uint32_t>(enabled_layers.size());
    create_info.ppEnabledLayerNames = enabled_layers.empty() ? nullptr : enabled_layers.data();
    create_info.enabledExtensionCount = static_cast<std::uint32_t>(enabled_extensions.size());
    create_info.ppEnabledExtensionNames = enabled_extensions.data();

    VkInstance instance = VK_NULL_HANDLE;
    const VkResult result = vkCreateInstance(&create_info, nullptr, &instance);
    if(result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkCreateInstance failed", result, {"vkCreateInstance"}));
    }

    VkDebugUtilsMessengerEXT debug_messenger = VK_NULL_HANDLE;

    if(info.enable_validation)
    {
        VkDebugUtilsMessengerCreateInfoEXT messenger_info{};
        messenger_info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        messenger_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                          VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        messenger_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                                      VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        messenger_info.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                            VkDebugUtilsMessageTypeFlagsEXT,
                                            const VkDebugUtilsMessengerCallbackDataEXT *callback_data,
                                            void *) -> VkBool32 {
            if(severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
            {
                std::println("[vulkan-validation] {}", callback_data->pMessage);
            }
            return VK_FALSE;
        };

        const auto create_messenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"));
        if(create_messenger != nullptr)
        {
            create_messenger(instance, &messenger_info, nullptr, &debug_messenger);
        }
    }

    return std::tuple{instance, debug_messenger};
}

struct DeviceSelection
{
    VkPhysicalDevice device{VK_NULL_HANDLE};
    VkDevice         logical_device{VK_NULL_HANDLE};
    DeviceSummary    summary;
    DescriptorSupport descriptor_support;
    QueueInfo        queue_info;
};

[[nodiscard]] auto build_device(const ContextCreateInfo &info, VkInstance instance)
    -> std::expected<DeviceSelection, VulkanError>
{
    std::uint32_t physical_count = 0U;
    vkEnumeratePhysicalDevices(instance, &physical_count, nullptr);
    if(physical_count == 0U)
    {
        return std::unexpected(make_error("vkEnumeratePhysicalDevices returned zero devices", {"vkEnumeratePhysicalDevices"}));
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_count);
    vkEnumeratePhysicalDevices(instance, &physical_count, physical_devices.data());

    struct Candidate
    {
        VkPhysicalDevice             physical{VK_NULL_HANDLE};
        VkPhysicalDeviceProperties   properties{};
        QueueInfo                    queue;
        DescriptorSupport            descriptor_support;
        std::vector<const char *>    enabled_extensions;
        std::uint32_t                score{0U};
    };

    std::vector<Candidate> candidates;
    candidates.reserve(physical_devices.size());

    for(std::uint32_t idx = 0U; idx < physical_devices.size(); ++idx)
    {
        VkPhysicalDevice device = physical_devices[idx];
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(device, &props);

        if(info.device_index.has_value() && idx != info.device_index.value())
        {
            continue;
        }

        auto queue_info_opt = pick_compute_queue_family(device);
        if(!queue_info_opt)
        {
            continue;
        }

        const auto extensions = enumerate_device_extensions(device);
        const bool has_all_required = std::ranges::all_of(required_device_extensions, [&](const char *ext) {
            return has_extension(extensions, ext);
        });
        if(info.require_descriptor_buffer && !has_all_required)
        {
            continue;
        }

        VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer{};
        descriptor_buffer.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;

        VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing{};
        descriptor_indexing.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
        descriptor_indexing.pNext = &descriptor_buffer;

        VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
        timeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
        timeline_features.pNext = &descriptor_indexing;

        VkPhysicalDeviceFeatures2 features2{};
        features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
        features2.pNext = &timeline_features;
        vkGetPhysicalDeviceFeatures2(device, &features2);

        if(timeline_features.timelineSemaphore == VK_FALSE)
        {
            continue;
        }

        DescriptorSupport descriptor_support{};
        descriptor_support.descriptor_buffer = descriptor_buffer.descriptorBuffer == VK_TRUE;
        descriptor_support.descriptor_buffer_push_descriptors = descriptor_buffer.descriptorBufferPushDescriptors == VK_TRUE;
        descriptor_support.descriptor_indexing = descriptor_indexing.descriptorBindingPartiallyBound == VK_TRUE ||
                                                 descriptor_indexing.runtimeDescriptorArray == VK_TRUE;

        if(info.require_descriptor_buffer && !descriptor_support.descriptor_buffer)
        {
            continue;
        }

        VkPhysicalDeviceDescriptorBufferPropertiesEXT descriptor_props{};
        descriptor_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_PROPERTIES_EXT;

        VkPhysicalDeviceProperties2 props2{};
        props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
        props2.pNext = &descriptor_props;
        vkGetPhysicalDeviceProperties2(device, &props2);
        descriptor_support.max_descriptor_buffer_bindings = static_cast<std::uint32_t>(descriptor_props.maxDescriptorBufferBindings);
        descriptor_support.max_resource_descriptor_buffer_bindings = static_cast<std::uint32_t>(descriptor_props.maxResourceDescriptorBufferBindings);
        descriptor_support.max_sampler_descriptor_buffer_bindings = static_cast<std::uint32_t>(descriptor_props.maxSamplerDescriptorBufferBindings);
        descriptor_support.descriptor_buffer_address_space_size = descriptor_props.descriptorBufferAddressSpaceSize;
        descriptor_support.descriptor_buffer_offset_alignment = descriptor_props.descriptorBufferOffsetAlignment;

        std::vector<const char *> enabled_extensions;
        enabled_extensions.reserve(required_device_extensions.size());
        for(const auto &ext : required_device_extensions)
        {
            if(has_extension(extensions, ext))
            {
                enabled_extensions.push_back(ext);
            }
        }
#ifdef VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME
        if(has_extension(extensions, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
        {
            enabled_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
        }
#endif
#ifdef VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
        if(has_extension(extensions, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME))
        {
            enabled_extensions.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
        }
#endif

        Candidate candidate{};
        candidate.physical = device;
        candidate.properties = props;
        candidate.queue = *queue_info_opt;
        candidate.descriptor_support = descriptor_support;
        candidate.enabled_extensions = std::move(enabled_extensions);
        candidate.score = score_device(props);

        candidates.push_back(std::move(candidate));
    }

    if(candidates.empty())
    {
        return std::unexpected(make_error("no physical device satisfies Phase 5 requirements", {"device enumeration"}));
    }

    auto preferred_it = candidates.end();
    if(!info.device_index.has_value() && !info.preferred_device_substring.empty())
    {
        preferred_it = std::ranges::find_if(candidates, [&](const Candidate &candidate) {
            return std::string_view{candidate.properties.deviceName}.find(info.preferred_device_substring) != std::string_view::npos;
        });
    }

    auto best_it = candidates.begin();
    if(preferred_it != candidates.end())
    {
        best_it = preferred_it;
    }
    else
    {
        best_it = std::ranges::max_element(candidates, {}, &Candidate::score);
    }

    const Candidate &chosen = *best_it;

    const float queue_priority = 1.0F;
    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = chosen.queue.family_index;
    queue_info.queueCount = 1U;
    queue_info.pQueuePriorities = &queue_priority;

    VkPhysicalDeviceDescriptorBufferFeaturesEXT descriptor_buffer_features{};
    descriptor_buffer_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_BUFFER_FEATURES_EXT;
    descriptor_buffer_features.descriptorBuffer = chosen.descriptor_support.descriptor_buffer ? VK_TRUE : VK_FALSE;
    descriptor_buffer_features.descriptorBufferPushDescriptors = chosen.descriptor_support.descriptor_buffer_push_descriptors ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceDescriptorIndexingFeatures descriptor_indexing_features{};
    descriptor_indexing_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES;
    descriptor_indexing_features.pNext = &descriptor_buffer_features;
    descriptor_indexing_features.descriptorBindingPartiallyBound = VK_TRUE;
    descriptor_indexing_features.descriptorBindingVariableDescriptorCount = VK_TRUE;
    descriptor_indexing_features.runtimeDescriptorArray = VK_TRUE;

    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features{};
    timeline_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    timeline_features.pNext = &descriptor_indexing_features;
    timeline_features.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceFeatures2 features2{};
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    features2.pNext = &timeline_features;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &features2;
    device_info.queueCreateInfoCount = 1U;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = static_cast<std::uint32_t>(chosen.enabled_extensions.size());
    device_info.ppEnabledExtensionNames = chosen.enabled_extensions.empty() ? nullptr : chosen.enabled_extensions.data();

    VkDevice device = VK_NULL_HANDLE;
    const VkResult create_result = vkCreateDevice(chosen.physical, &device_info, nullptr, &device);
    if(create_result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkCreateDevice failed", create_result, {"vkCreateDevice"}));
    }

    QueueInfo queue = chosen.queue;
    vkGetDeviceQueue(device, queue.family_index, queue.queue_index, &queue.queue);

    DeviceSummary summary{};
    summary.name = chosen.properties.deviceName;
    summary.api_version = chosen.properties.apiVersion;
    summary.driver_version = chosen.properties.driverVersion;
    summary.type = chosen.properties.deviceType;
    summary.vendor_id = chosen.properties.vendorID;
    summary.device_id = chosen.properties.deviceID;
    summary.physical_device = chosen.physical;

    DeviceSelection selection{};
    selection.device = chosen.physical;
    selection.logical_device = device;
    selection.summary = std::move(summary);
    selection.descriptor_support = chosen.descriptor_support;
    selection.queue_info = queue;

    return selection;
}

[[nodiscard]] auto create_allocator(VkInstance instance, VkPhysicalDevice physical_device, VkDevice device)
    -> std::expected<VmaAllocator, VulkanError>
{
    VmaVulkanFunctions functions{};
    functions.vkGetInstanceProcAddr = &vkGetInstanceProcAddr;
    functions.vkGetDeviceProcAddr   = &vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo create_info{};
    create_info.instance = instance;
    create_info.physicalDevice = physical_device;
    create_info.device = device;
    create_info.vulkanApiVersion = VK_API_VERSION_1_3;
    create_info.pVulkanFunctions = &functions;

    VmaAllocator allocator = VK_NULL_HANDLE;
    const VkResult result = vmaCreateAllocator(&create_info, &allocator);
    if(result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vmaCreateAllocator failed", result, {"vmaCreateAllocator"}));
    }
    return allocator;
}

[[nodiscard]] auto create_staging_ring(VmaAllocator allocator, std::uint64_t size)
    -> std::expected<StagingRing, VulkanError>
{
    if(size == 0U)
    {
        return std::unexpected(make_error("staging buffer size cannot be zero", {"staging buffer"}));
    }

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo alloc_info{};
    alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    alloc_info.usage = VMA_MEMORY_USAGE_AUTO_PREFER_HOST;

    VkBuffer buffer = VK_NULL_HANDLE;
    VmaAllocation allocation = VK_NULL_HANDLE;
    VmaAllocationInfo allocation_info{};

    const VkResult result = vmaCreateBuffer(allocator, &buffer_info, &alloc_info, &buffer, &allocation, &allocation_info);
    if(result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vmaCreateBuffer for staging ring failed", result, {"staging buffer"}));
    }

    auto ring = StagingRing{
        .allocation = allocation,
        .buffer = buffer,
        .mapped = static_cast<std::byte *>(allocation_info.pMappedData),
        .size = size,
        .head = 0U
    };

    if(ring.mapped == nullptr)
    {
        return std::unexpected(make_error("staging buffer mapping returned null pointer", {"staging buffer"}));
    }

    return ring;
}

} // namespace

TimelineSemaphore::TimelineSemaphore(VkDevice device, VkSemaphore handle) noexcept
    : device_(device), handle_(handle)
{}

TimelineSemaphore::TimelineSemaphore(TimelineSemaphore &&other) noexcept
    : device_(std::exchange(other.device_, VK_NULL_HANDLE)),
      handle_(std::exchange(other.handle_, VK_NULL_HANDLE))
{}

auto TimelineSemaphore::operator=(TimelineSemaphore &&other) noexcept -> TimelineSemaphore &
{
    if(this != &other)
    {
        if(handle_ != VK_NULL_HANDLE)
        {
            vkDestroySemaphore(device_, handle_, nullptr);
        }
        device_ = std::exchange(other.device_, VK_NULL_HANDLE);
        handle_ = std::exchange(other.handle_, VK_NULL_HANDLE);
    }
    return *this;
}

TimelineSemaphore::~TimelineSemaphore()
{
    if(handle_ != VK_NULL_HANDLE)
    {
        vkDestroySemaphore(device_, handle_, nullptr);
    }
}

void TimelineSemaphore::signal(std::uint64_t value) const
{
    VkSemaphoreSignalInfo signal_info{};
    signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
    signal_info.semaphore = handle_;
    signal_info.value = value;
    const VkResult result = vkSignalSemaphore(device_, &signal_info);
    if(result != VK_SUCCESS)
    {
        throw std::runtime_error(std::format("vkSignalSemaphore failed with {}", static_cast<int>(result)));
    }
}

void TimelineSemaphore::wait(std::uint64_t value, std::uint64_t timeout_ns) const
{
    VkSemaphoreWaitInfo wait_info{};
    wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
    wait_info.semaphoreCount = 1U;
    wait_info.pSemaphores = &handle_;
    wait_info.pValues = &value;
    const VkResult result = vkWaitSemaphores(device_, &wait_info, timeout_ns);
    if(result != VK_SUCCESS)
    {
        throw std::runtime_error(std::format("vkWaitSemaphores failed with {}", static_cast<int>(result)));
    }
}

auto VulkanContext::create(const ContextCreateInfo &info) -> std::expected<VulkanContext, VulkanError>
{
    auto instance_result = build_instance(info);
    if(!instance_result)
    {
        return std::unexpected(instance_result.error());
    }

    auto [instance, messenger] = *instance_result;

    auto device_result = build_device(info, instance);
    if(!device_result)
    {
        if(messenger != VK_NULL_HANDLE)
        {
            const auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if(destroy_messenger != nullptr)
            {
                destroy_messenger(instance, messenger, nullptr);
            }
        }
        vkDestroyInstance(instance, nullptr);
        return std::unexpected(device_result.error());
    }

    auto selection = *device_result;

    auto allocator_result = create_allocator(instance, selection.device, selection.logical_device);
    if(!allocator_result)
    {
        vkDestroyDevice(selection.logical_device, nullptr);
        if(messenger != VK_NULL_HANDLE)
        {
            const auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if(destroy_messenger != nullptr)
            {
                destroy_messenger(instance, messenger, nullptr);
            }
        }
        vkDestroyInstance(instance, nullptr);
        return std::unexpected(allocator_result.error());
    }

    auto allocator = *allocator_result;

    auto staging_result = create_staging_ring(allocator, info.staging_buffer_bytes);
    if(!staging_result)
    {
        vmaDestroyAllocator(allocator);
        vkDestroyDevice(selection.logical_device, nullptr);
        if(messenger != VK_NULL_HANDLE)
        {
            const auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
                vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT"));
            if(destroy_messenger != nullptr)
            {
                destroy_messenger(instance, messenger, nullptr);
            }
        }
        vkDestroyInstance(instance, nullptr);
        return std::unexpected(staging_result.error());
    }

    VulkanContext context;
    context.instance_ = instance;
    context.debug_messenger_ = messenger;
    context.physical_device_ = selection.device;
    context.device_ = selection.logical_device;
    context.allocator_ = allocator;
    context.queue_info_ = selection.queue_info;
    context.descriptor_support_ = selection.descriptor_support;
    context.summary_ = selection.summary;
    context.staging_ring_ = *staging_result;

    context.set_object_name_fn_ = reinterpret_cast<PFN_vkSetDebugUtilsObjectNameEXT>(
        vkGetDeviceProcAddr(context.device_, "vkSetDebugUtilsObjectNameEXT"));
    context.begin_label_fn_ = reinterpret_cast<PFN_vkCmdBeginDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(context.device_, "vkCmdBeginDebugUtilsLabelEXT"));
    context.end_label_fn_ = reinterpret_cast<PFN_vkCmdEndDebugUtilsLabelEXT>(
        vkGetDeviceProcAddr(context.device_, "vkCmdEndDebugUtilsLabelEXT"));

    context.set_object_name(reinterpret_cast<std::uint64_t>(context.queue_info_.queue), VK_OBJECT_TYPE_QUEUE,
                            "cwf/compute-queue");
    context.set_object_name(reinterpret_cast<std::uint64_t>(context.staging_ring_.buffer), VK_OBJECT_TYPE_BUFFER,
                            "cwf/staging-ring");

    std::println("[cwf::gpu] selected device: {} (vendor 0x{:04x}, type {}, api {} / driver {})",
                 context.summary_.name,
                 context.summary_.vendor_id,
                 device_type_to_string(context.summary_.type),
                 version_to_string(context.summary_.api_version),
                 version_to_string(context.summary_.driver_version));
    std::println("[cwf::gpu] queue family {} supports {} timestamp bits", context.queue_info_.family_index,
                 context.queue_info_.timestamp_bits);
    std::println("[cwf::gpu] descriptor buffer: {}, indexing: {}, max bindings (descriptor/resource/sampler) = {}/{}/{}, alignment {} bytes, address space {} bytes",
                 context.descriptor_support_.descriptor_buffer ? "enabled" : "disabled",
                 context.descriptor_support_.descriptor_indexing ? "enabled" : "disabled",
                 context.descriptor_support_.max_descriptor_buffer_bindings,
                 context.descriptor_support_.max_resource_descriptor_buffer_bindings,
                 context.descriptor_support_.max_sampler_descriptor_buffer_bindings,
                 context.descriptor_support_.descriptor_buffer_offset_alignment,
                 context.descriptor_support_.descriptor_buffer_address_space_size);

    return context;
}

VulkanContext::VulkanContext(VulkanContext &&other) noexcept
{
    *this = std::move(other);
}

auto VulkanContext::operator=(VulkanContext &&other) noexcept -> VulkanContext &
{
    if(this != &other)
    {
        destroy();
        instance_ = std::exchange(other.instance_, VK_NULL_HANDLE);
        debug_messenger_ = std::exchange(other.debug_messenger_, VK_NULL_HANDLE);
        physical_device_ = std::exchange(other.physical_device_, VK_NULL_HANDLE);
        device_ = std::exchange(other.device_, VK_NULL_HANDLE);
        allocator_ = std::exchange(other.allocator_, VK_NULL_HANDLE);
        queue_info_ = other.queue_info_;
        descriptor_support_ = other.descriptor_support_;
        summary_ = other.summary_;
        staging_ring_ = other.staging_ring_;
        other.queue_info_ = {};
        other.descriptor_support_ = {};
        other.summary_ = {};
        other.staging_ring_ = {};
        set_object_name_fn_ = std::exchange(other.set_object_name_fn_, nullptr);
        begin_label_fn_ = std::exchange(other.begin_label_fn_, nullptr);
        end_label_fn_ = std::exchange(other.end_label_fn_, nullptr);
    }
    return *this;
}

VulkanContext::~VulkanContext()
{
    destroy();
}

auto VulkanContext::make_timeline_semaphore(std::uint64_t initial_value) const
    -> std::expected<TimelineSemaphore, VulkanError>
{
    VkSemaphoreTypeCreateInfo timeline_info{};
    timeline_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_info.initialValue = initial_value;

    VkSemaphoreCreateInfo create_info{};
    create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    create_info.pNext = &timeline_info;

    VkSemaphore semaphore = VK_NULL_HANDLE;
    const VkResult result = vkCreateSemaphore(device_, &create_info, nullptr, &semaphore);
    if(result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkCreateSemaphore (timeline) failed", result, {"timeline semaphore"}));
    }

    return TimelineSemaphore{device_, semaphore};
}

void VulkanContext::set_object_name(std::uint64_t handle, VkObjectType type, std::string_view name) const
{
    if(set_object_name_fn_ == nullptr || handle == 0U)
    {
        return;
    }

    const std::string owned{name};
    VkDebugUtilsObjectNameInfoEXT info{};
    info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_OBJECT_NAME_INFO_EXT;
    info.objectType = type;
    info.objectHandle = handle;
    info.pObjectName = owned.c_str();
    set_object_name_fn_(device_, &info);
}

void VulkanContext::push_debug_label(VkCommandBuffer cmd, std::string_view name, std::array<float, 4U> color) const
{
    if(begin_label_fn_ == nullptr)
    {
        return;
    }
    VkDebugUtilsLabelEXT label{};
    label.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT;
    label.pLabelName = name.data();
    std::copy(color.begin(), color.end(), label.color);
    begin_label_fn_(cmd, &label);
}

void VulkanContext::pop_debug_label(VkCommandBuffer cmd) const
{
    if(end_label_fn_ == nullptr)
    {
        return;
    }
    end_label_fn_(cmd);
}

auto VulkanContext::make_memory_barrier(VkPipelineStageFlags2 src_usage, VkPipelineStageFlags2 dst_usage,
                                        VkAccessFlags2 src_access, VkAccessFlags2 dst_access) noexcept -> VkMemoryBarrier2
{
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = src_usage;
    barrier.srcAccessMask = src_access;
    barrier.dstStageMask = dst_usage;
    barrier.dstAccessMask = dst_access;
    return barrier;
}

void VulkanContext::destroy()
{
    if(allocator_ != VK_NULL_HANDLE && staging_ring_.buffer != VK_NULL_HANDLE)
    {
        vmaDestroyBuffer(allocator_, staging_ring_.buffer, staging_ring_.allocation);
        staging_ring_ = {};
    }

    if(allocator_ != VK_NULL_HANDLE)
    {
        vmaDestroyAllocator(allocator_);
        allocator_ = VK_NULL_HANDLE;
    }

    if(device_ != VK_NULL_HANDLE)
    {
        vkDestroyDevice(device_, nullptr);
        device_ = VK_NULL_HANDLE;
    }

    if(debug_messenger_ != VK_NULL_HANDLE)
    {
        const auto destroy_messenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if(destroy_messenger != nullptr)
        {
            destroy_messenger(instance_, debug_messenger_, nullptr);
        }
        debug_messenger_ = VK_NULL_HANDLE;
    }

    if(instance_ != VK_NULL_HANDLE)
    {
        vkDestroyInstance(instance_, nullptr);
        instance_ = VK_NULL_HANDLE;
    }
}

} // namespace cwf::gpu
