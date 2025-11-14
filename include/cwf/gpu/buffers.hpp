/**
 * @file buffers.hpp
 * @brief logical GPU buffer descriptions + staging helpers for Phase 7 uploads uwu
 *
 * this header translates mesh::pack outputs plus material metadata into the logical
 * buffers consumed by the descriptor-buffer sharding + upload planners. it keeps the
 * mapping deterministic, handles the tiny data conversions we need (like promoting
 * uint8 adjacency indices to uint32 and squeezing constitutive tensors down to fp32),
 * and surfaces the result as spans so staging code never copies unless absolutely
 * necessary.
 *
 * the goal: feed Phase 7 sharding + Phase 8 GPU solvers with pristine buffer names
 * and byte views so Vulkan uploads stay boring (in the best way). every helper here
 * is pure, fully documented, and drenched in gen-z commentary per house style ✨
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "cwf/gpu/sharding.hpp"
#include "cwf/gpu/upload.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/physics/materials.hpp"

namespace cwf::gpu::buffers
{

/**
 * @brief owning scratch buffers that back the logical views we hand to uploads
 */
struct PreparedGpuBuffers
{
    std::vector<float>        material_stiffness_fp32; ///< converted 6x6 tensors (materials * 36 floats)
    std::vector<std::uint32_t> adjacency_local_indices; ///< CSR local indices widened to uint32
};

/**
 * @brief lightweight description of a logical GPU buffer baked by packing
 */
struct LogicalBuffer
{
    std::string              name;      ///< stable identifier ("elements.connectivity", etc.)
    std::span<const std::byte> bytes;   ///< read-only view into CPU storage
    std::size_t              alignment; ///< alignment requirement (power-of-two, usually 256)
};

/**
 * @brief build logical buffer views that mirror packed CPU data
 *
 * ✨ PURE FUNCTION ✨ — never mutates the packing inputs, only writes into @p prepared.
 *
 * @param[in] packing packed struct-of-arrays output from mesh::pack
 * @param[in] materials elasticity metadata referenced by elements
 * @param[in,out] prepared scratch buffers that own converted data
 * @param[in] alignment alignment to request for each buffer (defaults to descriptor-friendly 256 bytes)
 * @return vector of logical buffer descriptors ready for sharding/upload planning
 */
[[nodiscard]] auto build_logical_buffers(const mesh::pack::PackingResult &packing,
                                         std::span<const physics::materials::ElasticProperties> materials,
                                         PreparedGpuBuffers &prepared,
                                         std::size_t alignment = shard::kDefaultAlignment)
    -> std::vector<LogicalBuffer>;

/**
 * @brief convert logical buffer descriptors into sharding specs (size+alignment only)
 */
[[nodiscard]] auto make_shard_specs(const std::vector<LogicalBuffer> &buffers)
    -> std::vector<shard::BufferSpecification>;

/**
 * @brief convert logical buffers into upload::BufferView entries (name + bytes)
 */
[[nodiscard]] auto make_upload_views(const std::vector<LogicalBuffer> &buffers)
    -> std::vector<upload::BufferView>;

} // namespace cwf::gpu::buffers
