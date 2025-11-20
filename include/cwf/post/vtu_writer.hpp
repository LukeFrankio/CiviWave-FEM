/**
 * @file vtu_writer.hpp
 * @brief binary VTU exporter that satisfies Phase 10's deliverables uwu
 */
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/post/derived_fields.hpp"

namespace cwf::post
{

/**
 * @brief contextual error payload for VTU export mishaps
 */
struct VtuError
{
    std::string              message; ///< spicy human-readable reason
    std::vector<std::string> context; ///< breadcrumbs (path, stage, etc.)
};

/**
 * @brief deterministic binary VTU dump (appended raw) covering Phase 10 fields
 *
 * ⚠️ IMPURE FUNCTION ⚠️ — touches the filesystem, so we keep the effects explicit and return
 * std::expected for ergonomic error handling.
 *
 * exported fields:
 * - PointData: displacement, velocity, acceleration, nodal strain/stress tensors, nodal von Mises
 * - CellData: element strain/stress tensors, element von Mises
 * - Geometry: deformed positions (x0 + u), unstructured connectivity from packed buffers
 *
 * @param[in] path output `.vtu` path (parent directories auto-created)
 * @param[in] mesh original mesh (geometry/type metadata)
 * @param[in] packing packed buffers containing positions + kinematics
 * @param[in] derived previously computed derived fields (per-node + per-element)
 * @param[in] simulation_time current simulation time (seconds)
 * @param[in] frame_index zero-based frame counter (for metadata only)
 * @return std::expected<void, VtuError> success or contextual failure
 */
[[nodiscard]] auto write_vtu(const std::filesystem::path &path,
                             const mesh::Mesh &mesh,
                             const mesh::pack::PackingResult &packing,
                             const DerivedFieldSet &derived,
                             double simulation_time,
                             std::uint32_t frame_index) -> std::expected<void, VtuError>;

} // namespace cwf::post
