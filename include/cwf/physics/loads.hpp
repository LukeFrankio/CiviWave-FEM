/**
 * @file loads.hpp
 * @brief CPU-side load assembly (gravity, tractions, point loads) uwu
 *
 * provides pure helpers to evaluate YAML time curves and assemble the nodal
 * force vector for the CPU reference solver. gravity uses lumped masses from
 * preprocessing, surface tractions integrate over tagged faces, and point
 * loads blast directly onto node groups. everything is deterministic, fully
 * documented, and matches the Vulkan path so validation hits differentials.
 */
#pragma once

#include <vector>

#include "cwf/config/config.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/preprocess.hpp"

namespace cwf::physics::loads
{

/**
 * @brief evaluate piecewise-linear curve at specific time (with clamping)
 *
 * ✨ PURE FUNCTION ✨
 *
 * @param curve YAML curve definition (sorted time,value pairs)
 * @param time current simulation time [s]
 * @return interpolated scale factor
 */
[[nodiscard]] auto evaluate_curve(const config::Curve &curve, double time) -> double;

/**
 * @brief assemble global nodal force vector from gravity, tractions, points
 *
 * ⚠️ IMPURE FUNCTION (depends on mesh + config state)
 *
 * @param mesh parsed mesh with surfaces + node groups
 * @param cfg validated configuration (loads, curves)
 * @param preprocess outputs from preprocessing stage (volumes, masses)
 * @param time current simulation time [s]
 * @return nodal force vector sized (nodes * 3)
 */
[[nodiscard]] auto assemble_load_vector(const mesh::Mesh &mesh, const config::Config &cfg,
                                        const mesh::pre::Outputs &preprocess, double time)
    -> std::vector<double>;

} // namespace cwf::physics::loads
