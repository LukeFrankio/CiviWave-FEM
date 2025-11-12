/**
 * @file pcg.hpp
 * @brief Phase 8 matrix-free K_eff apply + PCG orchestration that makes Vulkan kernels vibe in lockstep
 *
 * this header crowns the Phase 8 milestone with a fully documented, matrix-free conjugate-gradient core. it wires
 * together packed GPU buffers, Rayleigh damping coefficients, and Newmark magic to deliver the exact operator that the
 * roadmap calls for. matrix-free apply, FP64-powered reductions, block-jacobi preconditioning, warm-start awareness,
 * and excessive doc comments? yeah, all of that lives here uwu ✨
 *
 * design goals:
 * - zero dynamic allocations inside hot loops (workspaces own reusable buffers)
 * - ✨ PURE FUNCTION ✨ semantics wherever possible so testing stays fearless
 * - Dirichlet clamping mirrors the CPU reference solver (identity rows + zeroed columns)
 * - per-workgroup (metadata-driven) FP64 reduction partials ready for Vulkan shader parity
 *
 * this module feeds both the CPU reference path and the upcoming Slang/Vulkan kernels. the tests compare outputs
 * against the dense CPU solver to guarantee numerical alignment before we unleash the GPU implementation.
 *
 * @author LukeFrankio
 * @date 2025-11-12
 * @version 1.0
 *
 * @note uses C++26 features (std::span, std::expected) compiled with GCC 15.2+ on Windows 11 24H2 uwu
 * @note documented with Doxygen 1.15 beta because documentation supremacy never sleeps
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>
#include <expected>

#include "cwf/mesh/pack.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/physics/newmark.hpp"

namespace cwf::gpu::pcg
{

/**
 * @brief structured error payload for PCG setup/runtime mishaps (functional diagnostics uwu)
 */
struct PcgError
{
    std::string              message;  ///< human readable summary of what went sideways
    std::vector<std::string> context;  ///< breadcrumb stack (parameter/value crumbs)
};

/**
 * @brief immutable view describing the matrix-free operator ingredients
 *
 * ✨ PURE FUNCTION ✨ metadata container — no ownership, just spans. everything here points into packed buffers that
 * were produced during Phase 7. the same structure feeds both CPU validation and Vulkan descriptors. stick to the
 * documented invariants and life stays chill.
 *
 * invariants:
 * - dof_count == node_count * 3 (struct-of-arrays ordering)
 * - element_connectivity.size() == element_count * 8 (tet support uses first four slots)
 * - element_gradients.size() == element_count * 8 * 3 (gradNi packed as xyz floats)
 * - element_material_index references materials.size() entries (Phase 3 assignments)
 * - bc_mask.size() == node_count (bitmask per node: bit0→X, bit1→Y, bit2→Z)
 */
struct MatrixFreeSystem
{
    std::span<const std::uint32_t> element_connectivity; ///< flattened connectivity (8 entries per element)
    std::span<const float>         element_gradients;    ///< gradNi (element-major, 8*3 floats per element)
    std::span<const float>         element_volume;       ///< element volume [m^3]
    std::span<const std::uint32_t> element_material_index; ///< material index per element
    std::span<const physics::materials::ElasticProperties> materials; ///< isotropic elasticity stash
    std::span<const float>         lumped_mass;          ///< lumped mass per node (kg)
    std::span<const std::uint32_t> bc_mask;              ///< Dirichlet bitfield per node

    std::size_t node_count{}; ///< number of mesh nodes
    std::size_t element_count{}; ///< number of tetrahedral elements (Phase 8 assumption)
    std::size_t dof_count{}; ///< total DOFs == node_count * 3

    double stiffness_scale{}; ///< (1 + a1 * beta) from effective stiffness derivation
    double mass_factor{};     ///< (a0 + a1 * alpha) mass-diagonal contribution

    std::size_t reduction_block{};   ///< workgroup width mirrored from PackedMetadata (e.g., 256)
    std::size_t reduction_partials{}; ///< number of FP64 partial slots available (>= ceil(dofs / block))
};

/**
 * @brief reusable buffers for matrix-free apply + block-jacobi computation (no per-iteration allocations allowed)
 */
struct MatrixFreeWorkspace
{
    std::vector<double> sanitized_input; ///< DOF vector with Dirichlet DOFs zeroed (double precision)
    std::vector<double> accumulation;    ///< FP64 accumulator for apply results
    std::vector<double> block_buffer;    ///< temporary 3x3 blocks during block-jacobi assembly
    std::vector<float>  block_inverse;   ///< final 3x3 inverse blocks (row-major, FP32 per spec)
};

/**
 * @brief spans into solver scratch buffers allocated by mesh::pack (p, r, Ap, z, x, partials)
 */
struct PcgVectors
{
    std::span<float> solution; ///< Δu accumulator (warm-start aware)
    std::span<float> residual; ///< r_k
    std::span<float> search_direction; ///< p_k
    std::span<float> preconditioned; ///< z_k = M^{-1} r_k
    std::span<float> matvec; ///< Ap_k
    std::span<double> partials; ///< FP64 reduction workspace (length == MatrixFreeSystem::reduction_partials)
};

/**
 * @brief solver knobs for the GPU-style PCG loop
 */
struct PcgSettings
{
    std::size_t max_iterations{128U}; ///< hard cap per solve (>= 1)
    double      relative_tolerance{3.0e-4}; ///< target ||r|| / ||rhs|| (spec: 1e-4 .. 3e-4)
    bool        warm_start{false}; ///< reuse incoming solution vector instead of zeroing
};

/**
 * @brief telemetry emitted after a PCG run so diagnostics and tests can flex
 */
struct PcgTelemetry
{
    std::size_t iterations{}; ///< iterations executed (<= max_iterations)
    double      residual_norm{}; ///< ||r||_2 at exit (FP64)
    double      rhs_norm{}; ///< ||rhs||_2 baseline (FP64)
    double      alpha_last{}; ///< alpha from final iteration (0 when no iters)
    double      beta_last{}; ///< beta from final iteration (0 when no iters)
    bool        converged{}; ///< true when residual <= tol * rhs_norm
};

/**
 * @brief applies the effective stiffness operator in matrix-free form
 *
 * ✨ PURE FUNCTION ✨ — reads inputs, writes @p output, and bounces. no global state, no hidden allocations (workspace
 * owns the scratch). Dirichlet rows become identity, Dirichlet columns vanish thanks to sanitized input.
 *
 * @param[in] system matrix-free metadata forged from packed buffers + materials
 * @param[in] input DOF vector (FP32) to multiply (Δu candidate)
 * @param[out] output destination for K_eff * input (FP32, same span length as input)
 * @param[in,out] workspace reusable scratch buffers (sanitized_input & accumulation sized to dof_count)
 * @return std::expected<void, PcgError> success flag with contextual diagnostics on failure uwu
 *
 * @pre input.size() == output.size() == system.dof_count
 * @pre workspace.sanitized_input.size() >= system.dof_count (will resize if smaller)
 * @pre system only contains tetrahedral elements (Phase 8 scope)
 *
 * @complexity O(elements) for element loops + O(dofs) for mass/Dirichlet passes
 *
 * example (edge case with constrained DOFs):
 * @code
 * MatrixFreeWorkspace scratch{};
 * auto status = apply_keff(system, input_view, output_view, scratch);
 * ASSERT_TRUE(status.has_value());
 * // constrained rows now mirror the identity contribution
 * @endcode
 */
[[nodiscard]] auto apply_keff(const MatrixFreeSystem &system, std::span<const float> input,
                              std::span<float> output, MatrixFreeWorkspace &workspace)
    -> std::expected<void, PcgError>;

/**
 * @brief solves K_eff * x = rhs using block-jacobi PCG with FP64 reductions (Phase 8 spec compliant)
 *
 * ⚠️ IMPURE FUNCTION (mutates solver buffers) ⚠️ — while the math stays deterministic, this routine updates the
 * provided spans in-place, tracks telemetry, and performs iterative refinement with side effects confined to @p vectors
 * and @p workspace. external state remains untouched so tests can compare inputs/outputs cleanly.
 *
 * algorithm sketch (mirrors Vulkan plan):
 * 1. sanitize warm start (or zero) respecting Dirichlet DOFs (set to rhs instantly)
 * 2. build block-jacobi inverse (per-node 3x3) with Rayleigh/mass scaling
 * 3. compute residual, reduce via FP64 partials, decide convergence
 * 4. iterate CG with matrix-free matvec + block preconditioner + FP64 alpha/beta
 * 5. emit telemetry for diagnostics + timeline semaphore simulation if desired later
 *
 * @param[in] system matrix-free metadata (same as @ref apply_keff)
 * @param[in] rhs right-hand side vector (FP32, already Dirichlet-conditioned like CPU solver)
 * @param[in] settings solver limits + tolerance + warm-start toggle
 * @param[in,out] vectors solver scratch spans from packed buffers (p, r, Ap, z, x, partials)
 * @param[in,out] workspace reusable matrix-free + block-jacobi buffers
 * @return std::expected<PcgTelemetry, PcgError> telemetry on success, or diagnostic payload on failure
 *
 * @pre rhs.size() == system.dof_count
 * @pre all spans inside @p vectors have size system.dof_count (partials >= reduction_partials)
 * @pre vectors.solution contains warm-start guess when settings.warm_start == true (otherwise zeroed here)
 *
 * @warning relative tolerance uses rhs_norm fallback of 1.0 when rhs is near-zero to avoid division-by-zero crimes
 * @note timeline semaphores per iteration will live here in Phase 9+ when GPU orchestration lands
 *
 * example (happy path):
 * @code
 * MatrixFreeWorkspace scratch{};
 * PcgVectors vecs{
 *     .solution = solver_buffers.x,
 *     .residual = solver_buffers.r,
 *     .search_direction = solver_buffers.p,
 *     .preconditioned = solver_buffers.z,
 *     .matvec = solver_buffers.Ap,
 *     .partials = solver_partials
 * };
 * PcgSettings settings{.max_iterations = 64, .relative_tolerance = 2.5e-4, .warm_start = false};
 * auto result = solve_pcg(system, rhs_span, settings, vecs, scratch);
 * ASSERT_TRUE(result.has_value());
 * EXPECT_TRUE(result->converged);
 * @endcode
 */
[[nodiscard]] auto solve_pcg(const MatrixFreeSystem &system, std::span<const float> rhs,
                             const PcgSettings &settings, PcgVectors vectors, MatrixFreeWorkspace &workspace)
    -> std::expected<PcgTelemetry, PcgError>;

} // namespace cwf::gpu::pcg