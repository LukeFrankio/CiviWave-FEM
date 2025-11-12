/**
 * @file pack.cpp
 * @brief implementation for Phase 7 CPU-side SoA packing uwu
 *
 * this translation unit materializes the struct-of-arrays buffers declared in
 * pack.hpp. it fuses mesh geometry, preprocessing outputs, config-driven loads,
 * and dirichlet sauce into deterministic vectors that Vulkan descriptor buffers
 * can chew on without choking. all calculations stay pure, allocations stay
 * scoped, and the vibes stay immaculate.
 */
#include "cwf/mesh/pack.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <limits>
#include <string>

#include "cwf/physics/loads.hpp"
#include "cwf/physics/solver.hpp"

namespace cwf::mesh::pack
{
namespace
{

[[nodiscard]] auto make_error(std::string message, std::initializer_list<std::string> ctx) -> PackError
{
    PackError err{};
    err.message = std::move(message);
    err.context.assign(ctx.begin(), ctx.end());
    return err;
}

[[nodiscard]] auto axis_bit(std::size_t axis) noexcept -> std::uint32_t
{
    return static_cast<std::uint32_t>(1U << static_cast<unsigned>(axis));
}

[[nodiscard]] auto safe_cast_double_to_float(double value) noexcept -> float
{
    if (!std::isfinite(value))
    {
        return value > 0 ? std::numeric_limits<float>::infinity()
                          : (value < 0 ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::quiet_NaN());
    }
    if (value > static_cast<double>(std::numeric_limits<float>::max()))
    {
        return std::numeric_limits<float>::max();
    }
    if (value < -static_cast<double>(std::numeric_limits<float>::max()))
    {
        return -std::numeric_limits<float>::max();
    }
    return static_cast<float>(value);
}

} // namespace

auto build_packed_buffers(const mesh::Mesh &mesh, const mesh::pre::Outputs &preprocess, const config::Config &cfg,
                          const PackingParameters &params) -> std::expected<PackingResult, PackError>
{
    if (params.reduction_block_size == 0U)
    {
        return std::unexpected(make_error("reduction block size must be >= 1", {"PackingParameters", "reduction_block_size"}));
    }

    const std::size_t node_count = mesh.nodes.size();
    if (node_count != preprocess.lumped_mass.size())
    {
        return std::unexpected(make_error("preprocess lumped mass count mismatches mesh nodes",
                                          {"nodes", std::to_string(node_count),
                                           "lumped_mass", std::to_string(preprocess.lumped_mass.size())}));
    }

    const std::size_t element_count = mesh.elements.size();
    if (element_count != preprocess.element_volumes.size() || element_count != preprocess.shape_gradients.size() ||
        element_count != preprocess.element_material_index.size())
    {
        return std::unexpected(make_error("preprocess element buffers mismatched element count",
                                          {"elements", std::to_string(element_count)}));
    }

    if (node_count > 0 && preprocess.adjacency.offsets.size() != node_count + 1U)
    {
        return std::unexpected(make_error("adjacency offsets length invalid",
                                          {"adjacency", std::to_string(preprocess.adjacency.offsets.size())}));
    }

    if (node_count > (std::numeric_limits<std::size_t>::max() / 3U))
    {
        return std::unexpected(make_error("node count overflow when computing DOFs", {"node_count", std::to_string(node_count)}));
    }

    const std::size_t dof_count = node_count * 3U;

    const auto dirichlet = physics::solver::build_dirichlet_conditions(mesh, cfg);
    if (dirichlet.mask.size() != dof_count || dirichlet.targets.size() != dof_count)
    {
        return std::unexpected(make_error("dirichlet mask size mismatch",
                                          {"dof_count", std::to_string(dof_count),
                                           "mask", std::to_string(dirichlet.mask.size()),
                                           "targets", std::to_string(dirichlet.targets.size())}));
    }

    const auto load_vector = physics::loads::assemble_load_vector(mesh, cfg, preprocess, params.load_time_seconds);
    if (load_vector.size() != dof_count)
    {
        return std::unexpected(make_error("load vector size mismatch",
                                          {"dof_count", std::to_string(dof_count),
                                           "loads", std::to_string(load_vector.size())}));
    }

    PackingResult result{};
    auto &        buffers = result.buffers;

    // nodes --------------------------------------------------------------
    buffers.nodes.position0.resize(node_count);
    buffers.nodes.displacement.resize(node_count);
    buffers.nodes.velocity.resize(node_count);
    buffers.nodes.acceleration.resize(node_count);
    buffers.nodes.external_force.resize(node_count);
    buffers.nodes.bc_mask.assign(node_count, 0U);
    buffers.nodes.bc_value.resize(node_count);
    buffers.nodes.lumped_mass.resize(node_count);

    buffers.nodes.displacement.fill(0.0F);
    buffers.nodes.velocity.fill(0.0F);
    buffers.nodes.acceleration.fill(0.0F);
    buffers.nodes.bc_value.fill(0.0F);

    for (std::size_t node = 0; node < node_count; ++node)
    {
        const auto &pos = mesh.nodes[node].position;
        buffers.nodes.position0.x[node] = safe_cast_double_to_float(pos[0]);
        buffers.nodes.position0.y[node] = safe_cast_double_to_float(pos[1]);
        buffers.nodes.position0.z[node] = safe_cast_double_to_float(pos[2]);

        buffers.nodes.lumped_mass[node] = safe_cast_double_to_float(preprocess.lumped_mass[node]);

        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            const std::size_t dof = node * 3U + axis;
            const float       load_value = safe_cast_double_to_float(load_vector[dof]);
            switch (axis)
            {
            case 0:
                buffers.nodes.external_force.x[node] = load_value;
                break;
            case 1:
                buffers.nodes.external_force.y[node] = load_value;
                break;
            default:
                buffers.nodes.external_force.z[node] = load_value;
                break;
            }

            if (dirichlet.mask[dof])
            {
                buffers.nodes.bc_mask[node] |= axis_bit(axis);
                const float target = safe_cast_double_to_float(dirichlet.targets[dof]);
                switch (axis)
                {
                case 0:
                    buffers.nodes.bc_value.x[node] = target;
                    break;
                case 1:
                    buffers.nodes.bc_value.y[node] = target;
                    break;
                default:
                    buffers.nodes.bc_value.z[node] = target;
                    break;
                }
            }
        }
    }

    // elements ----------------------------------------------------------
    buffers.elements.connectivity.assign(element_count * 8U, std::numeric_limits<std::uint32_t>::max());
    buffers.elements.gradients.assign(element_count * 8U * 3U, 0.0F);
    buffers.elements.volume.resize(element_count);
    buffers.elements.material_index.resize(element_count);

    for (std::size_t elem_index = 0; elem_index < element_count; ++elem_index)
    {
        const auto &element = mesh.elements[elem_index];
        const auto  local_node_count = element.geometry == mesh::ElementGeometry::Tetrahedron4 ? 4U : 8U;
        const auto  base_conn = elem_index * 8U;
        for (std::size_t local = 0; local < local_node_count; ++local)
        {
            buffers.elements.connectivity[base_conn + local] = element.nodes[local];
        }

        const auto  gradient_base = elem_index * 8U * 3U;
        const auto &gradients     = preprocess.shape_gradients[elem_index];
        for (std::size_t local = 0; local < 8U; ++local)
        {
            for (std::size_t axis = 0; axis < 3U; ++axis)
            {
                const auto idx = gradient_base + local * 3U + axis;
                buffers.elements.gradients[idx] = safe_cast_double_to_float(gradients[local][axis]);
            }
        }

        buffers.elements.volume[elem_index] = safe_cast_double_to_float(preprocess.element_volumes[elem_index]);
        buffers.elements.material_index[elem_index] = static_cast<std::uint32_t>(preprocess.element_material_index[elem_index]);
    }

    // adjacency ---------------------------------------------------------
    buffers.adjacency.offsets = preprocess.adjacency.offsets;
    buffers.adjacency.element_indices = preprocess.adjacency.element_indices;
    buffers.adjacency.local_indices = preprocess.adjacency.local_indices;

    // solver ------------------------------------------------------------
    buffers.solver.p.assign(dof_count, 0.0F);
    buffers.solver.r.assign(dof_count, 0.0F);
    buffers.solver.Ap.assign(dof_count, 0.0F);
    buffers.solver.z.assign(dof_count, 0.0F);
    buffers.solver.x.assign(dof_count, 0.0F);

    const std::size_t reduction_block = std::max<std::size_t>(1U, params.reduction_block_size);
    const std::size_t partial_count = std::max<std::size_t>(1U, (dof_count + reduction_block - 1U) / reduction_block);
    buffers.solver.partials.assign(partial_count, 0.0);

    // metadata ----------------------------------------------------------
    result.metadata.node_count          = node_count;
    result.metadata.element_count       = element_count;
    result.metadata.dof_count           = dof_count;
    result.metadata.reduction_block     = reduction_block;
    result.metadata.reduction_partials  = partial_count;

    return result;
}

} // namespace cwf::mesh::pack
