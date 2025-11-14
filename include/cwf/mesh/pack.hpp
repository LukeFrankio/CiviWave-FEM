/**
 * @file pack.hpp
 * @brief Phase 7 CPU-side SoA packing that preps FEM data for GPU uploads uwu
 *
 * this header declares the struct-of-arrays data layout used to bridge the
 * reference CPU preprocessing pipeline with the Vulkan upload path.
 * positions, velocities, accelerations, forces, constraints, and solver
 * scratch buffers all live in lovingly documented vectors so descriptor
 * buffers can go absolutely feral later.
 *
 * this module is the functional core of Phase 7: it consumes mesh + config
 * inputs, respects dirichlet vibes, assembles external loads, and emits
 * alignment-friendly buffers ready for descriptor-buffer sharding. everything
 * stays pure, deterministic, and C++26-tier modern because we are living on
 * the bleeding edge fr fr.
 *
 * @author LukeFrankio
 * @date 2025-11-12
 * @version 1.0
 *
 * @note Targets GCC 15.2+ with -std=c++2c (C++26), compiled with zero warnings.
 * @note Documented with Doxygen 1.15 beta because documentation supremacy ftw.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <expected>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cwf/config/config.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/preprocess.hpp"

namespace cwf::mesh::pack
{

/**
 * @brief struct-of-arrays triple that keeps xyz channels crispy clean
 *
 * ✨ PURE FUNCTION ✨
 *
 * this struct is pure because:
 * - it only stores value semantics (std::vector) with no global side effects
 * - resizing/filling is deterministic and scoped to the owning instance
 * - no hidden state, no callbacks, no spooky action at a distance
 *
 * @note arrays use 32-bit floats to align with GPU expectations
 */
struct Float3SoA
{
    std::vector<float> x; ///< x-components in degrees-of-freedom order
    std::vector<float> y; ///< y-components in degrees-of-freedom order
    std::vector<float> z; ///< z-components in degrees-of-freedom order

    /**
     * @brief resize all component arrays to the requested count
     *
     * ✨ PURE FUNCTION ✨
     *
     * @param count number of entries each component should own
     *
     * @pre count is finite (no overflow shenanigans)
     * @post x.size() == y.size() == z.size() == count
     */
    void resize(std::size_t count)
    {
        x.resize(count);
        y.resize(count);
        z.resize(count);
    }

    /**
     * @brief fill every component lane with the provided value
     *
     * ✨ PURE FUNCTION ✨
     *
     * @param value scalar assigned to each xyz lane
     */
    void fill(float value) noexcept
    {
        std::fill(x.begin(), x.end(), value);
        std::fill(y.begin(), y.end(), value);
        std::fill(z.begin(), z.end(), value);
    }
};

/**
 * @brief packed node-centric buffers that capture kinematics + constraints
 */
struct NodeBuffers
{
    Float3SoA              position0; ///< reference positions (mesh nodes)
    Float3SoA              displacement; ///< u vector (initially zero)
    Float3SoA              velocity; ///< v vector (initially zero)
    Float3SoA              acceleration; ///< a vector (initially zero)
    Float3SoA              external_force; ///< assembled loads from YAML
    std::vector<std::uint32_t> bc_mask; ///< bitmask per node (xyz bits)
    Float3SoA              bc_value; ///< optional prescribed displacement values
    std::vector<float>     lumped_mass; ///< per-node lumped mass (kg)
};

/**
 * @brief packed element data (connectivity, gradients, volumes, materials)
 */
struct ElementBuffers
{
    std::vector<std::uint32_t> connectivity; ///< flattened node indices (8 entries per element)
    std::vector<float>         gradients; ///< grad Ni (element-major, 8*3 floats per element)
    std::vector<float>         volume; ///< element volume [m^3]
    std::vector<std::uint32_t> material_index; ///< index into config::materials
};

/**
 * @brief CSR adjacency buffers copied straight from preprocessing outputs
 */
struct AdjacencyBuffers
{
    std::vector<std::uint32_t> offsets; ///< node offset table (size nodes+1)
    std::vector<std::uint32_t> element_indices; ///< incident element indices
    std::vector<std::uint8_t>  local_indices; ///< local node slot per adjacency entry
};

/**
 * @brief solver scratch buffers (matrix-free PCG state + FP64 partials)
 */
struct SolverBuffers
{
    std::vector<float> p;   ///< search direction (FP32)
    std::vector<float> r;   ///< residual (FP32)
    std::vector<float> Ap;  ///< operator application (FP32)
    std::vector<float> z;   ///< preconditioned residual (FP32)
    std::vector<float> x;   ///< solution accumulator / delta-u (FP32)
    std::vector<double> partials; ///< FP64 reduction partial sums (per workgroup)
    std::vector<float> block_inverse; ///< cached block-jacobi inverse (3x3 per node, row-major)
};

/**
 * @brief aggregate container for every SoA buffer group Phase 7 expects
 */
struct PackedBuffers
{
    NodeBuffers      nodes; ///< node-centric data (pos0, kinematics, constraints)
    ElementBuffers   elements; ///< element-centric data
    AdjacencyBuffers adjacency; ///< node-element adjacency (CSR)
    SolverBuffers    solver; ///< solver scratch vectors and FP64 partials
};

/**
 * @brief metadata describing packed counts + reduction sizing
 */
struct PackedMetadata
{
    std::size_t node_count{}; ///< number of mesh nodes
    std::size_t element_count{}; ///< number of solid elements
    std::size_t dof_count{}; ///< degrees of freedom (node_count * 3)
    std::size_t reduction_block{}; ///< reduction workgroup width (usually 256)
    std::size_t reduction_partials{}; ///< length of solver.partials
};

/**
 * @brief all-in-one result returned by the packing pipeline
 */
struct PackingResult
{
    PackedBuffers buffers; ///< struct-of-arrays outputs
    PackedMetadata metadata; ///< counts and reduction info
};

/**
 * @brief error payload surfaced when packing detects invalid inputs
 */
struct PackError
{
    std::string              message; ///< spicy human-readable explanation
    std::vector<std::string> context; ///< breadcrumb trail for debugging
};

/**
 * @brief tunable parameters for packing (load evaluation + reduction sizing)
 */
struct PackingParameters
{
    double      load_time_seconds{0.0}; ///< time instant for assembling loads
    std::size_t reduction_block_size{256U}; ///< nodes per reduction workgroup (>= 1)
};

/**
 * @brief compute struct-of-arrays buffers ready for GPU upload
 *
 * ✨ PURE FUNCTION ✨
 *
 * this function is pure because:
 * - identical mesh/config/preprocess inputs always produce the same outputs
 * - it does not mutate global state or perform I/O
 * - it allocates deterministic STL containers and returns them by value
 *
 * @param[in] mesh source mesh (must match preprocess outputs)
 * @param[in] preprocess outputs from cwf::mesh::pre::run
 * @param[in] cfg validated YAML configuration (materials, BCs, loads)
 * @param[in] params packing knobs (load evaluation time, reduction width)
 * @return packed buffers on success or PackError describing what broke uwu
 *
 * @pre mesh.nodes.size() == preprocess.lumped_mass.size()
 * @pre preprocess vectors cover every element present in mesh.elements
 * @post result.buffers vectors respect node_count/element_count sizing
 *
 * @complexity O(nodes + elements) time, O(nodes + elements) space
 */
[[nodiscard]] auto build_packed_buffers(const mesh::Mesh &mesh, const mesh::pre::Outputs &preprocess,
                                        const config::Config &cfg, const PackingParameters &params = {})
    -> std::expected<PackingResult, PackError>;

} // namespace cwf::mesh::pack