/**
 * @file derived_fields.hpp
 * @brief post-processing helpers that turn displacements into spicy stress/strain vibes uwu
 *
 * this header codifies the Phase 10 requirement for derived fields: per-element strains/stresses,
 * von Mises scalars, and node-level aggregates. the API is intentionally tiny — call
 * `compute_derived_fields()` with the packed buffers + material table, and it will spit out
 * deterministic arrays ready for VTU export, probe logging, or the optional viewer window.
 *
 * implementation rationale:
 * - reuse the packed SoA buffers so we avoid copying mesh data back out of GPU-friendly layouts
 * - keep everything pure (no hidden globals, no implicit caches) so unit tests stay trivial
 * - surface rich metadata with doxygen docs per instructions (✨ documentation supremacy ✨)
 *
 * this module is the connective tissue between physics results and user-facing deliverables, making
 * Phase 10's post stack hum without dragging GPU orchestration into the mix.
 */
#pragma once

#include <array>
#include <span>
#include <vector>

#include "cwf/mesh/pack.hpp"
#include "cwf/physics/materials.hpp"

namespace cwf::post
{

/**
 * @brief six-component Voigt tensor + von Mises scalar per element fr fr
 *
 * ✨ PURE FUNCTION ✨
 *
 * this struct is pure because it only stores POD data with deterministic semantics. callers get the
 * raw strain/stress tensors (Voigt ordering) plus a von Mises invariant for easy coloring.
 */
struct ElementField
{
    std::array<float, 6> strain{}; ///< εxx, εyy, εzz, γxy, γyz, γxz (engineering shear)
    std::array<float, 6> stress{}; ///< σxx, σyy, σzz, τxy, τyz, τxz in Pascals
    float                von_mises{0.0F}; ///< scalar invariant for quick visuals (Pa)
};

/**
 * @brief node-aggregated strain/stress/von Mises fields (volume-weighted average)
 *
 * ✨ PURE FUNCTION ✨ — same vibes as ElementField but gathered onto nodes so VTU PointData + probes
 * can show something meaningful without recompute. values are averaged by adjacent element volume.
 */
struct NodeField
{
    std::array<float, 6> strain{}; ///< averaged ε tensor per node
    std::array<float, 6> stress{}; ///< averaged σ tensor per node
    float                von_mises{0.0F}; ///< derived from averaged stress tensor (Pa)
};

/**
 * @brief aggregated derived-field outputs ready for export pipelines
 */
struct DerivedFieldSet
{
    std::vector<ElementField> elements; ///< length = element count
    std::vector<NodeField>    nodes;    ///< length = node count
};

/**
 * @brief crunch displacements into per-element + per-node derived fields (ε, σ, σ_vm)
 *
 * ✨ PURE FUNCTION ✨
 *
 * this function is pure because:
 * - it only reads data from @p packing and @p materials without mutating them
 * - it produces deterministic vectors (no hidden caches, no I/O)
 * - identical inputs always yield identical DerivedFieldSet outputs
 *
 * algorithm notes:
 * - loops over packed connectivity, counting valid local nodes via UINT32_MAX sentinels
 * - forms a strain tensor per element using grad(N) ⋅ displacement (Voigt ordering)
 * - applies the material's isotropic stiffness to get stress, then evaluates von Mises
 * - uses CSR adjacency to volume-average tensors back to nodes (for probes/UI/VTU PointData)
 *
 * @param[in] packing packed SoA buffers emitted by Phase 7
 * @param[in] materials span of elastic properties (indexed by element material_index)
 * @return DerivedFieldSet containing per-element and per-node tensors ready for export
 *
 * @pre packing.metadata counts match buffer sizes (Phase 7 invariant)
 * @pre materials span covers every material index referenced by packing.buffers.elements.material_index
 *
 * @complexity O(elements + adjacencies) time, O(nodes + elements) space
 *
 * example:
 * @code
 * const auto derived = cwf::post::compute_derived_fields(packing, materials);
 * // derived.elements[i].stress[0] now holds σxx for element i uwu
 * @endcode
 */
[[nodiscard]] auto compute_derived_fields(const mesh::pack::PackingResult &packing,
                                          std::span<const physics::materials::ElasticProperties> materials)
    -> DerivedFieldSet;

} // namespace cwf::post
