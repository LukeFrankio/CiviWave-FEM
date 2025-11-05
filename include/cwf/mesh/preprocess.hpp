/**
 * @file preprocess.hpp
 * @brief mesh preprocessing sorcery: gradients, volumes, adjacency uwu
 *
 * this header wires together mesh + config to produce CPU-side data ready for
 * GPU upload. it computes tetrahedral gradients, element volumes, node-element
 * adjacency, and lumped mass vectors using config::Material densities. results
 * slot directly into Vulkan buffers later.
 *
 * @author LukeFrankio
 * @date 2025-11-05
 * @version 1.0
 *
 * @note built for GCC 15.2+, C++26, and AMD iGPU-friendly workflows
 */
#pragma once

#include <expected>
#include <vector>

#include "cwf/common/math.hpp"
#include "cwf/config/config.hpp"
#include "cwf/mesh/mesh.hpp"

namespace cwf::mesh::pre {

/**
 * @brief preprocessing error info with context breadcrumbs
 */
struct PreprocessError {
    std::string message;
    std::vector<std::string> context;
};

/**
 * @brief CSR-style node adjacency (node → elements, local ids)
 */
struct NodeAdjacency {
    std::vector<std::uint32_t> offsets;          ///< size = nodes + 1
    std::vector<std::uint32_t> element_indices;  ///< flattened element indices per node
    std::vector<std::uint8_t> local_indices;     ///< corresponding local node slot (0-7)
};

/**
 * @brief bundle of FEM preprocessing outputs ready for GPU packaging
 */
struct Outputs {
    NodeAdjacency adjacency;                                           ///< node-element incidence
    std::vector<double> element_volumes;                               ///< volume per element [m^3]
    std::vector<std::array<common::Vec3, 8>> shape_gradients;          ///< grad Ni per element (tet uses first 4)
    std::vector<double> lumped_mass;                                   ///< mass per node [kg]
    std::vector<std::size_t> element_material_index;                   ///< index into config.materials per element
};

/**
 * @brief preprocess mesh with config to obtain gradients + adjacency
 *
 * ⚠️ IMPURE FUNCTION (depends on config contents and numeric stability)
 *
 * @param[in] mesh parsed mesh model from mesh::load_gmsh_file
 * @param[in] cfg validated configuration data (materials, assignments)
 * @return std::expected of Outputs or PreprocessError on invalid geometry/config
 */
[[nodiscard]] auto run(const mesh::Mesh& mesh, const config::Config& cfg)
    -> std::expected<Outputs, PreprocessError>;

}  // namespace cwf::mesh::pre
