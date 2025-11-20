/**
 * @file derived_fields.cpp
 * @brief deterministic implementation of Phase 10 derived-field math uwu
 *
 * this translation unit keeps the math close to the spec: evaluate per-element strain from grad(N)
 * ⋅ displacement, apply isotropic stiffness for stress, derive von Mises, then scatter-gather onto
 * nodes via volume weighting. everything stays CPU-side for now so VTU/probe/export stacks can party
 * without waiting for GPU plumbing.
 */
#include "cwf/post/derived_fields.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <vector>

namespace cwf::post
{
namespace
{

constexpr std::size_t kMaxLocalNodes = 8U;
constexpr std::uint32_t kInvalidIndex = std::numeric_limits<std::uint32_t>::max();

[[nodiscard]] inline auto node_count_for_element(const std::vector<std::uint32_t> &connectivity,
                                                 std::size_t element_index) noexcept -> std::size_t
{
    const auto base = element_index * kMaxLocalNodes;
    std::size_t count = 0U;
    for (; count < kMaxLocalNodes; ++count)
    {
        if (connectivity[base + count] == kInvalidIndex)
        {
            break;
        }
    }
    return count;
}

[[nodiscard]] inline auto fetch_displacement(const mesh::pack::Float3SoA &soa, std::size_t node) noexcept
    -> std::array<double, 3>
{
    return {static_cast<double>(soa.x[node]), static_cast<double>(soa.y[node]), static_cast<double>(soa.z[node])};
}

[[nodiscard]] inline auto compute_von_mises(const std::array<double, 6> &stress) noexcept -> double
{
    const double sx = stress[0];
    const double sy = stress[1];
    const double sz = stress[2];
    const double txy = stress[3];
    const double tyz = stress[4];
    const double txz = stress[5];

    const double diff_xy = sx - sy;
    const double diff_yz = sy - sz;
    const double diff_zx = sz - sx;

    const double energy = 0.5 * (diff_xy * diff_xy + diff_yz * diff_yz + diff_zx * diff_zx)
                          + 3.0 * (txy * txy + tyz * tyz + txz * txz);
    return std::sqrt(std::max(energy, 0.0));
}

[[nodiscard]] inline auto stiffness_mul(const std::array<double, 36> &stiffness,
                                        const std::array<double, 6> &strain) noexcept -> std::array<double, 6>
{
    std::array<double, 6> stress{};
    for (std::size_t row = 0; row < 6U; ++row)
    {
        double accum = 0.0;
        for (std::size_t col = 0; col < 6U; ++col)
        {
            accum += stiffness[row * 6U + col] * strain[col];
        }
        stress[row] = accum;
    }
    return stress;
}

inline void accumulate_node(std::vector<std::array<double, 6>> &strain_accum,
                            std::vector<std::array<double, 6>> &stress_accum,
                            std::vector<double> &volume_accum,
                            std::uint32_t node,
                            double volume,
                            const std::array<double, 6> &strain,
                            const std::array<double, 6> &stress)
{
    volume_accum[node] += volume;
    for (std::size_t c = 0; c < 6U; ++c)
    {
        strain_accum[node][c] += strain[c] * volume;
        stress_accum[node][c] += stress[c] * volume;
    }
}

inline void store_element(ElementField &field,
                          const std::array<double, 6> &strain,
                          const std::array<double, 6> &stress)
{
    for (std::size_t c = 0; c < 6U; ++c)
    {
        field.strain[c] = static_cast<float>(strain[c]);
        field.stress[c] = static_cast<float>(stress[c]);
    }
    field.von_mises = static_cast<float>(compute_von_mises(stress));
}

inline void finalize_node(NodeField &field,
                          const std::array<double, 6> &strain,
                          const std::array<double, 6> &stress,
                          double weight)
{
    if (weight <= 0.0)
    {
        field = NodeField{};
        return;
    }

    const double inv = 1.0 / weight;
    std::array<double, 6> averaged_stress{};
    for (std::size_t c = 0; c < 6U; ++c)
    {
        const double s = strain[c] * inv;
        const double t = stress[c] * inv;
        field.strain[c] = static_cast<float>(s);
        field.stress[c] = static_cast<float>(t);
        averaged_stress[c] = t;
    }
    field.von_mises = static_cast<float>(compute_von_mises(averaged_stress));
}

} // namespace

auto compute_derived_fields(const mesh::pack::PackingResult &packing,
                            std::span<const physics::materials::ElasticProperties> materials) -> DerivedFieldSet
{
    const auto &buffers = packing.buffers;
    const auto  element_count = packing.metadata.element_count;
    const auto  node_count = packing.metadata.node_count;

    DerivedFieldSet fields{};
    fields.elements.resize(element_count);
    fields.nodes.resize(node_count);

    std::vector<std::array<double, 6>> node_strain(node_count);
    std::vector<std::array<double, 6>> node_stress(node_count);
    std::vector<double>                node_weights(node_count, 0.0);

    for (std::size_t elem = 0; elem < element_count; ++elem)
    {
        const auto local_count = node_count_for_element(buffers.elements.connectivity, elem);
        if (local_count == 0U)
        {
            continue;
        }

        const auto mat_index = buffers.elements.material_index[elem];
        assert(mat_index < materials.size());
        const auto &material = materials[mat_index];

        std::array<double, 6> strain{};
        const auto gradient_base = elem * kMaxLocalNodes * 3U;
        const auto conn_base = elem * kMaxLocalNodes;
        for (std::size_t local = 0; local < local_count; ++local)
        {
            const auto node_index = buffers.elements.connectivity[conn_base + local];
            if (node_index == kInvalidIndex)
            {
                continue;
            }

            const auto displacement = fetch_displacement(buffers.nodes.displacement, node_index);
            const double grad_x = static_cast<double>(buffers.elements.gradients[gradient_base + local * 3U + 0U]);
            const double grad_y = static_cast<double>(buffers.elements.gradients[gradient_base + local * 3U + 1U]);
            const double grad_z = static_cast<double>(buffers.elements.gradients[gradient_base + local * 3U + 2U]);

            strain[0] += grad_x * displacement[0];
            strain[1] += grad_y * displacement[1];
            strain[2] += grad_z * displacement[2];
            strain[3] += grad_y * displacement[0] + grad_x * displacement[1];
            strain[4] += grad_z * displacement[1] + grad_y * displacement[2];
            strain[5] += grad_z * displacement[0] + grad_x * displacement[2];
        }

        const auto stress = stiffness_mul(material.stiffness, strain);
        store_element(fields.elements[elem], strain, stress);

        const double volume = static_cast<double>(buffers.elements.volume[elem]);
        for (std::size_t local = 0; local < local_count; ++local)
        {
            const auto node_index = buffers.elements.connectivity[conn_base + local];
            if (node_index == kInvalidIndex)
            {
                continue;
            }
            accumulate_node(node_strain, node_stress, node_weights, node_index, volume, strain, stress);
        }
    }

    for (std::size_t node = 0; node < node_count; ++node)
    {
        finalize_node(fields.nodes[node], node_strain[node], node_stress[node], node_weights[node]);
    }

    return fields;
}

} // namespace cwf::post
