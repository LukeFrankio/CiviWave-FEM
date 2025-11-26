/**
 * @file instrumentation.cpp
 * @brief Phase 11 GPU profiling implementation that makes performance measurable uwu
 *
 * this file implements the instrumentation subsystem declared in instrumentation.hpp. it wires
 * Vulkan timestamp queries, YAML logging, and optional Tracy integration into a cohesive
 * profiling layer. the implementation favors correctness and clarity over micro-optimization —
 * instrumentation overhead should be negligible compared to actual compute work.
 *
 * highlights:
 * - GpuTimestamps uses VK_KHR_synchronization2 for vkCmdWriteTimestamp2
 * - timestamp resolution respects device timestampPeriod and timestampValidBits
 * - YAML output uses the yaml-cpp library for clean, validated serialization
 * - Tracy integration is compile-time gated behind CWF_ENABLE_TRACY
 *
 * @author LukeFrankio
 * @date 2025-11-25
 * @version 1.0
 *
 * @note documented with Doxygen 1.15 beta because excessive comments are self-care ✨
 */

#include "cwf/gpu/instrumentation.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <fstream>
#include <initializer_list>
#include <ios>
#include <optional>
#include <print>
#include <ratio>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>
#include <vulkan/vulkan_core.h>

#include "cwf/gpu/pcg.hpp"
#include "cwf/gpu/vulkan_context.hpp"
#include "yaml-cpp/emitter.h"
#include "yaml-cpp/emittermanip.h"

#if defined(CWF_ENABLE_TRACY) && CWF_ENABLE_TRACY
#include <Tracy.hpp>
#include <TracyVulkan.hpp>
#endif

namespace cwf::gpu::instrumentation
{
namespace
{

/**
 * @brief helper that fabricates an InstrumentationError with context breadcrumbs
 */
[[nodiscard]] auto make_error(std::string message, std::initializer_list<std::string> ctx = {},
                              VkResult result = VK_SUCCESS) -> InstrumentationError
{
    InstrumentationError error{};
    error.message = std::move(message);
    error.context.assign(ctx.begin(), ctx.end());
    error.result = result;
    return error;
}

/**
 * @brief computes overflow-safe timestamp difference respecting valid bits
 *
 * ✨ PURE FUNCTION ✨ — math only, no side effects.
 *
 * @param start start timestamp
 * @param end end timestamp
 * @param valid_bits number of valid bits in the timestamp
 * @return tick difference accounting for wraparound
 */
[[nodiscard]] constexpr auto safe_timestamp_diff(std::uint64_t start, std::uint64_t end,
                                                 std::uint32_t valid_bits) noexcept -> std::uint64_t
{
    // Handle the case where valid_bits >= 64 (no masking needed)
    if (valid_bits >= 64U)
    {
        return end - start;
    }

    // Create mask for valid bits
    const std::uint64_t mask         = (1ULL << valid_bits) - 1ULL;
    const std::uint64_t masked_start = start & mask;
    const std::uint64_t masked_end   = end & mask;

    // Handle wraparound: if end < start, the timer wrapped
    if (masked_end >= masked_start)
    {
        return masked_end - masked_start;
    }

    // Wraparound case: add the max value + 1 to compensate
    return (mask - masked_start) + masked_end + 1ULL;
}

/**
 * @brief converts timestamp ticks to milliseconds using the device's timestamp period
 *
 * ✨ PURE FUNCTION ✨ — just arithmetic vibes.
 *
 * @param ticks raw timestamp tick count
 * @param period_ns nanoseconds per tick
 * @return duration in milliseconds
 */
[[nodiscard]] constexpr auto ticks_to_ms(std::uint64_t ticks, float period_ns) noexcept -> double
{
    // period_ns is nanoseconds per tick
    // ticks * period_ns = total nanoseconds
    // total_ns / 1,000,000 = milliseconds
    return static_cast<double>(ticks) * static_cast<double>(period_ns) / 1.0e6;
}

} // namespace

// -----------------------------------------------------------------------------
// GpuTimestamps implementation
// -----------------------------------------------------------------------------

auto GpuTimestamps::create(const VulkanContext &context, std::size_t max_passes)
    -> std::expected<GpuTimestamps, InstrumentationError>
{
    if (max_passes == 0U)
    {
        return std::unexpected(make_error("max_passes must be > 0", {"GpuTimestamps::create"}));
    }

    GpuTimestamps timestamps;
    timestamps.device_     = context.device();
    timestamps.max_passes_ = max_passes;
    timestamps.pass_count_ = 0U;
    timestamps.pass_names_.reserve(max_passes);
    timestamps.raw_timestamps_.resize(max_passes * 2U, 0U); // start + end per pass

    // Get timestamp properties from the device
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(context.device_summary().physical_device, &props);
    timestamps.timestamp_period_ns_  = props.limits.timestampPeriod;
    timestamps.timestamp_valid_bits_ = context.queue_info().timestamp_bits;

    if (timestamps.timestamp_valid_bits_ == 0U)
    {
        return std::unexpected(make_error("device queue does not support timestamps",
                                          {"GpuTimestamps::create", "timestampValidBits=0"}));
    }

    // Create query pool with 2 slots per pass (start + end)
    VkQueryPoolCreateInfo pool_info{};
    pool_info.sType      = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    pool_info.queryType  = VK_QUERY_TYPE_TIMESTAMP;
    pool_info.queryCount = static_cast<std::uint32_t>(max_passes * 2U);

    const VkResult result = vkCreateQueryPool(context.device(), &pool_info, nullptr, &timestamps.query_pool_);
    if (result != VK_SUCCESS)
    {
        return std::unexpected(make_error("vkCreateQueryPool failed", {"GpuTimestamps::create"}, result));
    }

    // Name the query pool for RGP/debug tools
    context.set_object_name(reinterpret_cast<std::uint64_t>(timestamps.query_pool_),
                            VK_OBJECT_TYPE_QUERY_POOL, "cwf_instrumentation_timestamps");

    std::println(
        "[cwf::instrumentation] GpuTimestamps created: max_passes={}, period_ns={:.2f}, valid_bits={}",
        max_passes, timestamps.timestamp_period_ns_, timestamps.timestamp_valid_bits_);

    return timestamps;
}

GpuTimestamps::GpuTimestamps(GpuTimestamps &&other) noexcept
    : device_{other.device_}, query_pool_{other.query_pool_}, max_passes_{other.max_passes_},
      pass_count_{other.pass_count_}, timestamp_period_ns_{other.timestamp_period_ns_},
      timestamp_valid_bits_{other.timestamp_valid_bits_}, pass_names_{std::move(other.pass_names_)},
      raw_timestamps_{std::move(other.raw_timestamps_)}
{
    other.device_     = VK_NULL_HANDLE;
    other.query_pool_ = VK_NULL_HANDLE;
    other.max_passes_ = 0U;
    other.pass_count_ = 0U;
}

auto GpuTimestamps::operator=(GpuTimestamps &&other) noexcept -> GpuTimestamps &
{
    if (this != &other)
    {
        destroy();
        device_               = other.device_;
        query_pool_           = other.query_pool_;
        max_passes_           = other.max_passes_;
        pass_count_           = other.pass_count_;
        timestamp_period_ns_  = other.timestamp_period_ns_;
        timestamp_valid_bits_ = other.timestamp_valid_bits_;
        pass_names_           = std::move(other.pass_names_);
        raw_timestamps_       = std::move(other.raw_timestamps_);

        other.device_     = VK_NULL_HANDLE;
        other.query_pool_ = VK_NULL_HANDLE;
        other.max_passes_ = 0U;
        other.pass_count_ = 0U;
    }
    return *this;
}

GpuTimestamps::~GpuTimestamps()
{
    destroy();
}

void GpuTimestamps::destroy()
{
    if (query_pool_ != VK_NULL_HANDLE && device_ != VK_NULL_HANDLE)
    {
        vkDestroyQueryPool(device_, query_pool_, nullptr);
        query_pool_ = VK_NULL_HANDLE;
    }
    device_ = VK_NULL_HANDLE;
}

void GpuTimestamps::reset(VkCommandBuffer cmd)
{
    if (query_pool_ == VK_NULL_HANDLE)
    {
        return;
    }

    pass_count_ = 0U;
    pass_names_.clear();

    // Reset all queries in the pool
    const auto query_count = static_cast<std::uint32_t>(max_passes_ * 2U);
    vkCmdResetQueryPool(cmd, query_pool_, 0U, query_count);
}

auto GpuTimestamps::mark_start(VkCommandBuffer cmd, std::string_view name, VkPipelineStageFlags2 stage)
    -> std::optional<std::size_t>
{
    if (query_pool_ == VK_NULL_HANDLE || pass_count_ >= max_passes_)
    {
        return std::nullopt;
    }

    const std::size_t pass_index  = pass_count_;
    const auto        query_index = static_cast<std::uint32_t>(pass_index * 2U); // start query

    pass_names_.emplace_back(name);
    ++pass_count_;

    // Use synchronization2 timestamp command
    vkCmdWriteTimestamp2(cmd, stage, query_pool_, query_index);

    return pass_index;
}

void GpuTimestamps::mark_end(VkCommandBuffer cmd, VkPipelineStageFlags2 stage)
{
    if (query_pool_ == VK_NULL_HANDLE || pass_count_ == 0U)
    {
        return;
    }

    const std::size_t pass_index  = pass_count_ - 1U;
    const auto        query_index = static_cast<std::uint32_t>(pass_index * 2U + 1U); // end query

    vkCmdWriteTimestamp2(cmd, stage, query_pool_, query_index);
}

auto GpuTimestamps::resolve() -> std::expected<std::vector<PassTiming>, InstrumentationError>
{
    if (query_pool_ == VK_NULL_HANDLE)
    {
        return std::unexpected(make_error("query pool not initialized", {"GpuTimestamps::resolve"}));
    }

    if (pass_count_ == 0U)
    {
        return std::vector<PassTiming>{};
    }

    // Fetch all recorded timestamps
    const auto         query_count = static_cast<std::uint32_t>(pass_count_ * 2U);
    const VkDeviceSize stride      = sizeof(std::uint64_t);
    const VkDeviceSize data_size   = static_cast<VkDeviceSize>(query_count) * stride;

    const VkResult result = vkGetQueryPoolResults(device_, query_pool_, 0U, query_count,
                                                  static_cast<size_t>(data_size), raw_timestamps_.data(),
                                                  stride, VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);

    if (result != VK_SUCCESS && result != VK_NOT_READY)
    {
        return std::unexpected(
            make_error("vkGetQueryPoolResults failed", {"GpuTimestamps::resolve"}, result));
    }

    // Convert to PassTiming structs
    std::vector<PassTiming> timings;
    timings.reserve(pass_count_);

    for (std::size_t i = 0U; i < pass_count_; ++i)
    {
        const std::uint64_t start_tick  = raw_timestamps_[i * 2U];
        const std::uint64_t end_tick    = raw_timestamps_[i * 2U + 1U];
        const std::uint64_t diff        = safe_timestamp_diff(start_tick, end_tick, timestamp_valid_bits_);
        const double        duration_ms = ticks_to_ms(diff, timestamp_period_ns_);

        timings.push_back(PassTiming{
            .name        = pass_names_[i],
            .duration_ms = duration_ms,
            .start_tick  = start_tick,
            .end_tick    = end_tick,
        });
    }

    return timings;
}

// -----------------------------------------------------------------------------
// FrameLogger implementation
// -----------------------------------------------------------------------------

auto FrameLogger::create(const InstrumentationConfig &config) -> FrameLogger
{
    FrameLogger logger;
    logger.config_  = config;
    logger.enabled_ = config.enable_yaml_logging;
    return logger;
}

void FrameLogger::begin_frame(std::uint64_t frame_index, double simulation_time, double time_step,
                              double tolerance, bool paused)
{
    frame_start_ = std::chrono::steady_clock::now();

    current_                   = FrameLog{};
    current_.frame_index       = frame_index;
    current_.simulation_time_s = simulation_time;
    current_.time_step_s       = time_step;
    current_.solver_tolerance  = tolerance;
    current_.paused_mode       = paused;
    current_.pass_timings.clear();
}

void FrameLogger::record_pass(PassTiming timing)
{
    current_.pass_timings.push_back(std::move(timing));
}

void FrameLogger::record_passes(std::span<const PassTiming> timings)
{
    current_.pass_timings.insert(current_.pass_timings.end(), timings.begin(), timings.end());
}

void FrameLogger::record_pcg(const pcg::PcgTelemetry &telemetry)
{
    current_.pcg_iterations    = telemetry.iterations;
    current_.pcg_residual_norm = telemetry.residual_norm;
    current_.pcg_rhs_norm      = telemetry.rhs_norm;
    current_.pcg_converged     = telemetry.converged;
}

void FrameLogger::record_adaptive(bool increased, bool decreased, bool clamped_min, bool clamped_max)
{
    current_.dt_increased   = increased;
    current_.dt_decreased   = decreased;
    current_.dt_clamped_min = clamped_min;
    current_.dt_clamped_max = clamped_max;
}

auto FrameLogger::end_frame() -> FrameLog
{
    // Compute wall clock time
    const auto frame_end   = std::chrono::steady_clock::now();
    const auto elapsed     = std::chrono::duration<double, std::milli>(frame_end - frame_start_);
    current_.wall_clock_ms = elapsed.count();

    // Write YAML log if enabled and stride is met
    if (enabled_ && config_.enable_yaml_logging && !config_.log_directory.empty())
    {
        if (config_.log_stride > 0U && (current_.frame_index % config_.log_stride) == 0U)
        {
            write_yaml();
        }
    }

    return current_;
}

void FrameLogger::write_yaml() const
{
    if (config_.log_directory.empty())
    {
        return;
    }

    // Ensure log directory exists
    std::error_code ec;
    std::filesystem::create_directories(config_.log_directory, ec);
    if (ec)
    {
        std::println(stderr, "[cwf::instrumentation] failed to create log directory: {}", ec.message());
        return;
    }

    // Build filename
    const std::string filename = std::format("{}{:06d}.yaml", config_.log_prefix, current_.frame_index);
    const auto        path     = config_.log_directory / filename;

    auto result = write_frame_log(current_, path);
    if (!result)
    {
        std::println(stderr, "[cwf::instrumentation] failed to write frame log: {}", result.error().message);
    }
}

// -----------------------------------------------------------------------------
// ScopedGpuPass implementation
// -----------------------------------------------------------------------------

ScopedGpuPass::ScopedGpuPass(GpuTimestamps &timestamps, VkCommandBuffer cmd, std::string_view name,
                             VkPipelineStageFlags2 stage)
    : timestamps_{&timestamps}, cmd_{cmd}, stage_{stage}
{
    timestamps_->mark_start(cmd_, name, stage_);
}

ScopedGpuPass::~ScopedGpuPass()
{
    timestamps_->mark_end(cmd_, stage_);
}

// -----------------------------------------------------------------------------
// ScopedRgpLabel implementation
// -----------------------------------------------------------------------------

ScopedRgpLabel::ScopedRgpLabel(const VulkanContext &context, VkCommandBuffer cmd, std::string_view name,
                               std::array<float, 4U> color)
    : context_{&context}, cmd_{cmd}
{
    context_->push_debug_label(cmd_, name, color);
}

ScopedRgpLabel::~ScopedRgpLabel()
{
    context_->pop_debug_label(cmd_);
}

// -----------------------------------------------------------------------------
// Tracy integration (compile-time gated)
// -----------------------------------------------------------------------------

#if defined(CWF_ENABLE_TRACY) && CWF_ENABLE_TRACY

TracyCpuZone::TracyCpuZone(std::string_view name)
{
    // Tracy uses its own zone macros; this is a simplified wrapper
    // In practice, use ZoneScoped or ZoneScopedN macros directly
    ZoneTransient(zone, true);
    ZoneName(name.data(), name.size());
    zone_ctx_ = nullptr; // Tracy manages context internally
}

TracyCpuZone::~TracyCpuZone()
{
    // Tracy automatically ends zones on scope exit via its macros
}

void tracy_submit_frame_log(const FrameLog &log)
{
    // Submit frame marker
    FrameMark;

    // Report pass timings as plot values
    for (const auto &pass : log.pass_timings)
    {
        TracyPlot(pass.name.c_str(), pass.duration_ms);
    }

    // Report PCG statistics
    TracyPlot("pcg_iterations", static_cast<double>(log.pcg_iterations));
    TracyPlot("pcg_residual", log.pcg_residual_norm);
    TracyPlot("wall_clock_ms", log.wall_clock_ms);
}

#endif // CWF_ENABLE_TRACY

// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------

auto frame_log_to_yaml(const FrameLog &log) -> std::string
{
    YAML::Emitter out;
    out << YAML::BeginMap;

    out << YAML::Key << "frame" << YAML::Value << log.frame_index;
    out << YAML::Key << "simulation_time_s" << YAML::Value << YAML::Precision(9) << log.simulation_time_s;
    out << YAML::Key << "time_step_s" << YAML::Value << YAML::Precision(9) << log.time_step_s;
    out << YAML::Key << "solver_tolerance" << YAML::Value << YAML::Precision(9) << log.solver_tolerance;
    out << YAML::Key << "paused_mode" << YAML::Value << log.paused_mode;
    out << YAML::Key << "wall_clock_ms" << YAML::Value << YAML::Precision(3) << log.wall_clock_ms;

    // Pass timings
    out << YAML::Key << "timings" << YAML::Value << YAML::BeginMap;
    for (const auto &pass : log.pass_timings)
    {
        out << YAML::Key << pass.name << YAML::Value << YAML::Precision(4) << pass.duration_ms;
    }
    out << YAML::EndMap;

    // PCG statistics
    out << YAML::Key << "pcg" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "iterations" << YAML::Value << log.pcg_iterations;
    out << YAML::Key << "residual_norm" << YAML::Value << YAML::Precision(9) << log.pcg_residual_norm;
    out << YAML::Key << "rhs_norm" << YAML::Value << YAML::Precision(9) << log.pcg_rhs_norm;
    out << YAML::Key << "converged" << YAML::Value << log.pcg_converged;
    out << YAML::EndMap;

    // Adaptive timestep flags
    out << YAML::Key << "adaptive" << YAML::Value << YAML::BeginMap;
    out << YAML::Key << "dt_increased" << YAML::Value << log.dt_increased;
    out << YAML::Key << "dt_decreased" << YAML::Value << log.dt_decreased;
    out << YAML::Key << "dt_clamped_min" << YAML::Value << log.dt_clamped_min;
    out << YAML::Key << "dt_clamped_max" << YAML::Value << log.dt_clamped_max;
    out << YAML::EndMap;

    out << YAML::EndMap;

    return out.c_str();
}

auto write_frame_log(const FrameLog &log, const std::filesystem::path &path)
    -> std::expected<void, InstrumentationError>
{
    std::ofstream file{path, std::ios::out | std::ios::trunc};
    if (!file)
    {
        return std::unexpected(
            make_error("failed to open file for writing", {path.string()}, VK_ERROR_UNKNOWN));
    }

    file << frame_log_to_yaml(log);
    file.close();

    if (file.fail())
    {
        return std::unexpected(make_error("failed to write frame log", {path.string()}, VK_ERROR_UNKNOWN));
    }

    return {};
}

} // namespace cwf::gpu::instrumentation
