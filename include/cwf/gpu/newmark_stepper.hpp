/**
 * @file newmark_stepper.hpp
 * @brief Phase 9 implicit Newmark frame orchestration that mirrors the planned GPU pipeline uwu
 *
 * this header introduces a fully documented stepper that ties together the predictor shaders,
 * matrix-free PCG core, and update kernels into a single ergonomic interface. while the current
 * implementation still executes on the CPU (so tests can run headless), the API mirrors the
 * Vulkan flow described in RefDocs/TODO Phase 9: predictor → effective RHS → PCG solve → state
 * update → adaptive timestep. swapping in real GPU dispatch later only requires redirecting the
 * helper hooks that currently crunch SoA buffers on the host.
 *
 * highlights:
 * - exposes a `Stepper` class that owns spans into packed buffers + solver scratch vectors
 * - builds RHS terms using the same math as the CPU reference solver (Rayleigh damping included)
 * - reuses `cwf::gpu::pcg` for matrix-free solves so convergence metrics stay identical
 * - emits telemetry struct with PCG stats, applied tolerance, and adaptive dt decisions
 * - optional warm-start and pause-time tolerance scheduling per the YAML schema 🌀
 *
 * @note targets C++26/GCC 15.2+ with std::expected + span everywhere
 * @note documented with Doxygen 1.15 beta because excessive comments are self-care ✨
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <expected>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>
#include <filesystem>

#include "cwf/config/config.hpp"
#include "cwf/gpu/device_buffers.hpp"
#include "cwf/gpu/vulkan_context.hpp"
#include "cwf/gpu/pcg.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/physics/newmark.hpp"

namespace cwf::gpu::newmark
{

/**
 * @brief structured error payload for Newmark step failures (functional diagnostics uwu)
 */
struct StepError
{
    std::string              message;  ///< spicy summary of what went sideways
    std::vector<std::string> context;  ///< breadcrumb trail (stage, value, etc.)
};

/**
 * @brief adaptive timestep knobs inspired by Phase 9 spec (increase/decrease heuristics)
 */
struct AdaptivePolicy
{
    double low_iteration_ratio{0.3};   ///< boost dt when iterations <= ratio * max_iters
    double increase_factor{1.1};       ///< multiplicative dt growth when convergence is easy
    double decrease_factor{0.5};       ///< multiplicative dt shrink when solver stagnates
};

/**
 * @brief telemetry emitted after each implicit step so callers can log + visualize convergence
 */
struct StepTelemetry
{
    double              simulation_time{};  ///< time at the beginning of the step [s]
    double              time_step{};        ///< dt used for the solve (before adaptation)
    double              applied_tolerance{};///< solver tolerance for this frame
    bool                paused_mode{};      ///< true when pause tolerance was selected
    bool                dt_increased{};     ///< true when adaptive logic bumped dt up
    bool                dt_decreased{};     ///< true when adaptive logic shrank dt
    bool                dt_clamped_min{};   ///< true when dt hit the configured min bound
    bool                dt_clamped_max{};   ///< true when dt hit the configured max bound
    pcg::PcgTelemetry   pcg{};              ///< raw PCG statistics (iterations, norms, etc.)
};

/**
 * @class Stepper
 * @brief functional wrapper that runs predictor → RHS → PCG → update just like the GPU plan
 *
 * the stepper binds to packed SoA buffers (Phase 7), keeps spans into solver scratch memory,
 * and feeds everything through the Phase 8 matrix-free PCG implementation. predictor/update
 * math matches the Slang kernels so unit tests can assert CPU vs GPU parity all day. adaptive
 * policies follow the spec: nudge dt up when iterations plummet, slash dt when convergence stalls.
 * future Vulkan plumbing will reuse the same entry points but swap out the CPU loops with actual
 * command-buffer submissions.
 */
class Stepper
{
public:
    /**
     * @brief glue packed buffers + Rayleigh damping + solver knobs into a ready-to-run stepper
     *
     * ⚠️ IMPURE FUNCTION ⚠️ because it captures non-owning spans into @p packing, but keeps
     * ownership semantics crystal clear (caller stays owner). the ctor validates counts,
     * seeds predictor buffers, and wires @ref pcg::MatrixFreeSystem views so each call to
     * step() can blast through predictor → RHS → PCG → update without extra setup.
     */
    Stepper(mesh::pack::PackingResult &packing,
        std::span<const physics::materials::ElasticProperties> materials,
        physics::materials::RayleighCoefficients rayleigh,
        const config::SolverSettings &solver_settings,
        const config::TimeSettings &time_settings,
        AdaptivePolicy adaptive_policy = {});

    ~Stepper();

    /**
     * @brief execute one implicit Newmark frame (predictor → RHS → PCG → update)
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — mutates packed node buffers in-place and updates the solver
     * scratch vectors. returns telemetry for adaptive policies + UI overlays. paused-mode
     * switches tolerance per spec (tighten when user is scrubbing), otherwise runtime tol.
     */
    [[nodiscard]] auto step(double simulation_time_seconds, bool paused_mode = false)
    -> std::expected<StepTelemetry, StepError>;

    [[nodiscard]] auto current_time() const noexcept -> double { return accumulated_time_; }
    [[nodiscard]] auto time_step() const noexcept -> double { return current_dt_; }
    [[nodiscard]] auto node_count() const noexcept -> std::size_t { return node_count_; }
    [[nodiscard]] auto dof_count() const noexcept -> std::size_t { return dof_count_; }

    void set_warm_start(bool enabled) noexcept { warm_start_enabled_ = enabled; }
    [[nodiscard]] auto warm_start_enabled() const noexcept -> bool { return warm_start_enabled_; }

    /**
     * @brief wires the stepper to a Vulkan runtime so the implicit solve can run on GPU uwu
     *
     * ⚠️ IMPURE FUNCTION ⚠️ — allocates Vulkan resources (pipelines, descriptor sets, staging helpers).
     * Must be called before the first `step()` if you want GPU acceleration. Passing a null shader
     * directory keeps the method chill and reuses the bundled shaders from the build tree.
     */
    [[nodiscard]] auto enable_gpu(const gpu::VulkanContext &context, gpu::DeviceBufferArena &arena,
                                  const std::filesystem::path &shader_directory)
        -> std::expected<void, StepError>;

private:
    [[nodiscard]] static auto make_error(std::string message, std::initializer_list<std::string> ctx = {}) -> StepError;
    [[nodiscard]] static constexpr auto axis_bit(std::size_t axis) noexcept -> std::uint32_t;

    mesh::pack::PackingResult *packing_{}; ///< non-owning pointer (caller owns buffers)
    std::vector<physics::materials::ElasticProperties> materials_storage_{};
    physics::materials::RayleighCoefficients           rayleigh_{};
    config::SolverSettings                             solver_settings_{};
    config::TimeSettings                               time_settings_{};
    AdaptivePolicy                                     adaptive_policy_{};

    pcg::MatrixFreeSystem    matrix_system_{};
    pcg::MatrixFreeSystem    stiffness_only_system_{};
    pcg::MatrixFreeWorkspace matrix_workspace_{};
    pcg::PcgVectors          solver_vectors_{};

    std::vector<float> rhs_{};
    std::vector<float> damping_rhs_{};
    std::vector<float> damping_output_{};
    std::vector<float> external_force_{};

    mesh::pack::Float3SoA predicted_displacement_{};
    mesh::pack::Float3SoA predicted_velocity_{};

    double current_dt_{};
    double accumulated_time_{0.0};
    std::size_t frame_index_{0};
    bool warm_start_enabled_{true};
    double beta_{0.25};
    double gamma_{0.5};

    physics::newmark::Coefficients  coeffs_{};
    physics::newmark::UpdateScalars update_scalars_{};

    std::size_t node_count_{0};
    std::size_t dof_count_{0};

    class GpuRuntime;
    std::unique_ptr<GpuRuntime> gpu_runtime_{};

    [[nodiscard]] auto assemble_rhs() -> std::expected<void, StepError>;
    void clamp_dirichlet_rhs();
    void write_predictor();
    void apply_state_update();
    void refresh_coefficients();
    void update_matrix_free_scalars();
    void adapt_timestep(const pcg::PcgTelemetry &pcg_stats, StepTelemetry &telemetry);
    void flatten_external_force();
    [[nodiscard]] auto node_buffers() -> mesh::pack::NodeBuffers &;
};

} // namespace cwf::gpu::newmark
