/**
 * @file instrumentation.hpp
 * @brief Phase 11 GPU profiling and telemetry infrastructure that makes performance visible uwu
 *
 * this header introduces the instrumentation subsystem for CiviWave-FEM. it provides:
 * - Vulkan timestamp queries wrapped in RAII (GpuTimestamps)
 * - per-frame YAML logging with pass timings, PCG stats, dt, and tol
 * - RGP marker helpers via VK_EXT_debug_utils
 * - optional Tracy integration (compile-time gated by CWF_ENABLE_TRACY)
 *
 * the design follows functional principles where possible — pure builders for configuration
 * structs, explicit side-effect annotations, and excessive comments to keep future developers
 * sane. telemetry flows through a single FrameLog struct emitted at frame end, which can be
 * serialized to YAML or forwarded to Tracy/RGP.
 *
 * highlights:
 * - GpuTimestamps wraps vkCmdWriteTimestamp2 and resolves to milliseconds via device properties
 * - FrameLogger accumulates pass timings and PCG telemetry, then flushes to YAML
 * - TracyAdapter connects CPU zones and GPU contexts when ENABLE_TRACY=ON
 * - all markers use VK_EXT_debug_utils for RGP capture compatibility
 *
 * @author LukeFrankio
 * @date 2025-11-25
 * @version 1.0
 *
 * @note requires Vulkan 1.3 with VK_EXT_debug_utils and timeline semaphores
 * @note documented with Doxygen 1.15 beta because excessive comments are self-care ✨
 */
#pragma once

#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <vulkan/vulkan.h>

#include "cwf/gpu/pcg.hpp"

namespace cwf::gpu
{
class VulkanContext;
} // namespace cwf::gpu

namespace cwf::gpu::instrumentation
{

/**
 * @brief error payload for instrumentation failures (functional diagnostics uwu)
 */
struct InstrumentationError
{
    std::string              message;  ///< human-readable summary with gen-z vibes
    std::vector<std::string> context;  ///< breadcrumb trail for debugging
    VkResult                 result{VK_SUCCESS}; ///< underlying Vulkan error if applicable
};

/**
 * @brief named timing interval captured via Vulkan timestamp queries
 *
 * ✨ PURE FUNCTION ✨ — this is a passive data container; no mutation of external state.
 */
struct PassTiming
{
    std::string name;           ///< pass identifier (e.g., "predictor", "pcg_solve", "update")
    double      duration_ms{0.0}; ///< elapsed time in milliseconds (FP64 for precision)
    std::uint64_t start_tick{0U}; ///< raw GPU timestamp at pass start
    std::uint64_t end_tick{0U};   ///< raw GPU timestamp at pass end
};

/**
 * @brief complete telemetry snapshot for a single simulation frame
 *
 * ✨ PURE FUNCTION ✨ — immutable after construction; just vibes and data.
 */
struct FrameLog
{
    std::uint64_t frame_index{0U};          ///< monotonic frame counter
    double        simulation_time_s{0.0};   ///< current simulation time in seconds
    double        time_step_s{0.0};         ///< dt used for this frame
    double        solver_tolerance{0.0};    ///< applied PCG tolerance
    bool          paused_mode{false};       ///< whether pause tolerance was used

    std::vector<PassTiming> pass_timings;   ///< per-pass GPU timings

    // PCG solver statistics (mirrors pcg::PcgTelemetry)
    std::size_t pcg_iterations{0U};         ///< iterations executed
    double      pcg_residual_norm{0.0};     ///< ||r||_2 at exit
    double      pcg_rhs_norm{0.0};          ///< ||rhs||_2 baseline
    bool        pcg_converged{false};       ///< whether solver converged

    // adaptive timestep flags
    bool        dt_increased{false};        ///< timestep was increased this frame
    bool        dt_decreased{false};        ///< timestep was decreased this frame
    bool        dt_clamped_min{false};      ///< timestep hit minimum bound
    bool        dt_clamped_max{false};      ///< timestep hit maximum bound

    // wall clock timing
    double      wall_clock_ms{0.0};         ///< total frame time measured on CPU
};

/**
 * @brief configuration for the instrumentation subsystem
 *
 * ✨ PURE FUNCTION ✨ — declarative spec for what instrumentation features to enable.
 */
struct InstrumentationConfig
{
    bool        enable_gpu_timestamps{true};    ///< record Vulkan timestamp queries
    bool        enable_yaml_logging{true};      ///< write per-frame YAML logs
    bool        enable_rgp_markers{true};       ///< emit VK_EXT_debug_utils labels
    bool        enable_tracy{false};            ///< connect to Tracy profiler (compile-time gated)
    std::size_t max_passes{32U};                ///< maximum number of passes to timestamp per frame
    std::filesystem::path log_directory{};      ///< where to write YAML logs (empty = disabled)
    std::string_view log_prefix{"frame_"};      ///< prefix for log filenames
    std::size_t log_stride{1U};                 ///< write logs every N frames (1 = every frame)
};

/**
 * @class GpuTimestamps
 * @brief RAII wrapper for Vulkan timestamp query pool with automatic resolution uwu
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — allocates and manages GPU resources. methods perform Vulkan API calls.
 *
 * this class owns a VkQueryPool with 2 * max_passes slots (start + end per pass). callers record
 * timestamps via mark_start/mark_end, then call resolve() after GPU work completes to convert
 * raw ticks to milliseconds using the device's timestamp period.
 *
 * design:
 * - RAII destruction cleans up the query pool
 * - mark_start/mark_end insert vkCmdWriteTimestamp2 commands into the provided command buffer
 * - resolve() uses vkGetQueryPoolResults to fetch timestamps and compute durations
 * - overflow-safe subtraction respects the device's timestampValidBits
 */
class GpuTimestamps
{
public:
    GpuTimestamps() = default;

    /**
     * @brief creates a timestamp query pool bound to the provided context
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — allocates Vulkan resources.
     *
     * @param context Vulkan context providing device and queue info
     * @param max_passes maximum number of passes to timestamp per frame
     * @return GpuTimestamps instance or error
     */
    [[nodiscard]] static auto create(const VulkanContext &context, std::size_t max_passes = 32U)
        -> std::expected<GpuTimestamps, InstrumentationError>;

    GpuTimestamps(const GpuTimestamps &) = delete;
    auto operator=(const GpuTimestamps &) -> GpuTimestamps & = delete;

    GpuTimestamps(GpuTimestamps &&other) noexcept;
    auto operator=(GpuTimestamps &&other) noexcept -> GpuTimestamps &;

    ~GpuTimestamps();

    /**
     * @brief resets the query pool for a new frame
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — records vkCmdResetQueryPool into the command buffer.
     *
     * @param cmd command buffer to record reset into
     */
    void reset(VkCommandBuffer cmd);

    /**
     * @brief marks the start of a named pass with a timestamp query
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — records vkCmdWriteTimestamp2 into the command buffer.
     *
     * @param cmd command buffer to record timestamp into
     * @param name pass identifier (stored for later resolution)
     * @param stage pipeline stage for the timestamp (default: all commands)
     * @return pass index (used internally) or nullopt if max_passes exceeded
     */
    auto mark_start(VkCommandBuffer cmd, std::string_view name,
                    VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT)
        -> std::optional<std::size_t>;

    /**
     * @brief marks the end of the most recently started pass
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — records vkCmdWriteTimestamp2 into the command buffer.
     *
     * @param cmd command buffer to record timestamp into
     * @param stage pipeline stage for the timestamp
     */
    void mark_end(VkCommandBuffer cmd, VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    /**
     * @brief resolves recorded timestamps to milliseconds after GPU work completes
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — calls vkGetQueryPoolResults (blocking).
     *
     * @return vector of pass timings with names and durations
     */
    [[nodiscard]] auto resolve() -> std::expected<std::vector<PassTiming>, InstrumentationError>;

    /**
     * @brief returns the maximum number of passes this pool supports
     */
    [[nodiscard]] auto max_passes() const noexcept -> std::size_t { return max_passes_; }

    /**
     * @brief returns the number of passes recorded this frame
     */
    [[nodiscard]] auto pass_count() const noexcept -> std::size_t { return pass_count_; }

private:
    VkDevice        device_{VK_NULL_HANDLE};
    VkQueryPool     query_pool_{VK_NULL_HANDLE};
    std::size_t     max_passes_{0U};
    std::size_t     pass_count_{0U};
    float           timestamp_period_ns_{1.0F}; ///< nanoseconds per tick
    std::uint32_t   timestamp_valid_bits_{64U}; ///< valid bits in timestamp

    std::vector<std::string> pass_names_;       ///< recorded pass names (indexed by pass)
    std::vector<std::uint64_t> raw_timestamps_; ///< staging buffer for query results

    void destroy();
};

/**
 * @class FrameLogger
 * @brief accumulates per-frame telemetry and writes YAML logs on flush
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — performs file I/O when flush() is called.
 *
 * typical usage:
 * 1. call begin_frame() at frame start
 * 2. record pass timings via record_pass() or let GpuTimestamps populate
 * 3. call record_pcg() after solver completes
 * 4. call end_frame() to finalize wall clock time and optionally flush to YAML
 */
class FrameLogger
{
public:
    FrameLogger() = default;

    /**
     * @brief creates a logger with the provided configuration
     *
     * @param config instrumentation settings (log directory, stride, etc.)
     * @return FrameLogger instance
     */
    [[nodiscard]] static auto create(const InstrumentationConfig &config) -> FrameLogger;

    /**
     * @brief marks the beginning of a new frame
     *
     * @param frame_index monotonic frame counter
     * @param simulation_time current simulation time in seconds
     * @param time_step dt for this frame
     * @param tolerance applied solver tolerance
     * @param paused whether pause mode is active
     */
    void begin_frame(std::uint64_t frame_index, double simulation_time,
                     double time_step, double tolerance, bool paused);

    /**
     * @brief records a single pass timing
     *
     * @param timing pass timing data (name + duration)
     */
    void record_pass(PassTiming timing);

    /**
     * @brief records all pass timings from a GpuTimestamps instance
     *
     * @param timings vector of resolved pass timings
     */
    void record_passes(std::span<const PassTiming> timings);

    /**
     * @brief records PCG solver telemetry
     *
     * @param telemetry PCG statistics from the solver
     */
    void record_pcg(const pcg::PcgTelemetry &telemetry);

    /**
     * @brief records adaptive timestep decisions
     *
     * @param increased whether dt was increased
     * @param decreased whether dt was decreased
     * @param clamped_min whether dt hit minimum
     * @param clamped_max whether dt hit maximum
     */
    void record_adaptive(bool increased, bool decreased, bool clamped_min, bool clamped_max);

    /**
     * @brief finalizes the frame and optionally writes YAML log
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — performs file I/O if log stride is met.
     *
     * @return completed frame log (can be used for Tracy/RGP forwarding)
     */
    [[nodiscard]] auto end_frame() -> FrameLog;

    /**
     * @brief returns the current frame log (read-only)
     */
    [[nodiscard]] auto current_log() const noexcept -> const FrameLog & { return current_; }

    /**
     * @brief enables or disables YAML output
     */
    void set_enabled(bool enabled) noexcept { enabled_ = enabled; }

    /**
     * @brief checks if logging is currently enabled
     */
    [[nodiscard]] auto enabled() const noexcept -> bool { return enabled_; }

private:
    InstrumentationConfig config_{};
    FrameLog              current_{};
    bool                  enabled_{true};
    std::chrono::steady_clock::time_point frame_start_{};

    void write_yaml() const;
};

/**
 * @class ScopedGpuPass
 * @brief RAII helper for automatic start/end timestamp marking around a scope
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — records commands into the command buffer.
 *
 * usage:
 * @code
 * {
 *     ScopedGpuPass pass{timestamps, cmd, "predictor"};
 *     // dispatch predictor compute work
 * } // automatically marks end timestamp on scope exit
 * @endcode
 */
class ScopedGpuPass
{
public:
    /**
     * @brief begins a timestamped pass
     *
     * @param timestamps GpuTimestamps instance to record into
     * @param cmd command buffer for timestamp commands
     * @param name pass identifier
     * @param stage pipeline stage for timestamps
     */
    ScopedGpuPass(GpuTimestamps &timestamps, VkCommandBuffer cmd, std::string_view name,
                  VkPipelineStageFlags2 stage = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);

    ~ScopedGpuPass();

    ScopedGpuPass(const ScopedGpuPass &) = delete;
    auto operator=(const ScopedGpuPass &) -> ScopedGpuPass & = delete;
    ScopedGpuPass(ScopedGpuPass &&) = delete;
    auto operator=(ScopedGpuPass &&) -> ScopedGpuPass & = delete;

private:
    GpuTimestamps *timestamps_;
    VkCommandBuffer cmd_;
    VkPipelineStageFlags2 stage_;
};

/**
 * @class ScopedRgpLabel
 * @brief RAII helper for automatic push/pop of RGP debug labels
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — calls VK_EXT_debug_utils entry points.
 *
 * usage:
 * @code
 * {
 *     ScopedRgpLabel label{context, cmd, "PCG Iteration", {0.4F, 0.7F, 0.2F, 1.0F}};
 *     // dispatch work annotated in RGP
 * } // automatically pops label on scope exit
 * @endcode
 */
class ScopedRgpLabel
{
public:
    /**
     * @brief pushes a debug label onto the command buffer
     *
     * @param context Vulkan context with debug utils function pointers
     * @param cmd command buffer to annotate
     * @param name label text
     * @param color RGBA debug color (0..1 per component)
     */
    ScopedRgpLabel(const VulkanContext &context, VkCommandBuffer cmd, std::string_view name,
                   std::array<float, 4U> color = {0.3F, 0.4F, 0.9F, 1.0F});

    ~ScopedRgpLabel();

    ScopedRgpLabel(const ScopedRgpLabel &) = delete;
    auto operator=(const ScopedRgpLabel &) -> ScopedRgpLabel & = delete;
    ScopedRgpLabel(ScopedRgpLabel &&) = delete;
    auto operator=(ScopedRgpLabel &&) -> ScopedRgpLabel & = delete;

private:
    const VulkanContext *context_;
    VkCommandBuffer cmd_;
};

// -----------------------------------------------------------------------------
// Tracy integration (compile-time gated)
// -----------------------------------------------------------------------------

#if defined(CWF_ENABLE_TRACY) && CWF_ENABLE_TRACY

/**
 * @brief Tracy CPU zone that forwards pass names and durations
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — interacts with Tracy's global state.
 *
 * @note only available when CWF_ENABLE_TRACY=1
 */
class TracyCpuZone
{
public:
    explicit TracyCpuZone(std::string_view name);
    ~TracyCpuZone();

    TracyCpuZone(const TracyCpuZone &) = delete;
    auto operator=(const TracyCpuZone &) -> TracyCpuZone & = delete;
    TracyCpuZone(TracyCpuZone &&) = delete;
    auto operator=(TracyCpuZone &&) -> TracyCpuZone & = delete;

private:
    void *zone_ctx_{nullptr}; // opaque Tracy context
};

/**
 * @brief submits a completed frame log to Tracy for visualization
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — forwards data to Tracy.
 *
 * @param log completed frame log
 */
void tracy_submit_frame_log(const FrameLog &log);

/// Convenience macro for Tracy CPU zone in the current scope
#define CWF_TRACY_ZONE(name) ::cwf::gpu::instrumentation::TracyCpuZone _tracy_zone_##__LINE__{name}

#else // !CWF_ENABLE_TRACY

/// No-op when Tracy is disabled
#define CWF_TRACY_ZONE(name) (void)0

inline void tracy_submit_frame_log(const FrameLog &) {}

#endif // CWF_ENABLE_TRACY

// -----------------------------------------------------------------------------
// Utility functions
// -----------------------------------------------------------------------------

/**
 * @brief serializes a FrameLog to YAML string
 *
 * ✨ PURE FUNCTION ✨ — no side effects, just string generation.
 *
 * @param log frame telemetry to serialize
 * @return YAML-formatted string
 */
[[nodiscard]] auto frame_log_to_yaml(const FrameLog &log) -> std::string;

/**
 * @brief writes a FrameLog to a file in YAML format
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — performs file I/O.
 *
 * @param log frame telemetry to write
 * @param path destination file path
 * @return success or error
 */
[[nodiscard]] auto write_frame_log(const FrameLog &log, const std::filesystem::path &path)
    -> std::expected<void, InstrumentationError>;

} // namespace cwf::gpu::instrumentation
