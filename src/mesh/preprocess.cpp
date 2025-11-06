/**
 * @file preprocess.cpp
 * @brief CPU preprocessing pipeline: adjacency, gradients, masses uwu
 */
#include "cwf/mesh/preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <unordered_map>

namespace cwf::mesh::pre
{
namespace
{

using common::Vec3;

[[nodiscard]] auto make_error(std::string message, std::vector<std::string> ctx)
    -> std::expected<Outputs, PreprocessError>
{
    return std::unexpected(PreprocessError{std::move(message), std::move(ctx)});
}

[[nodiscard]] auto subtract(const Vec3 &a, const Vec3 &b) noexcept -> Vec3
{
    return Vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

struct MaterialBinding
{
    std::unordered_map<std::uint32_t, std::size_t> group_to_material;
};

[[nodiscard]] auto bind_materials(const mesh::Mesh &mesh, const config::Config &cfg)
    -> std::expected<MaterialBinding, PreprocessError>
{
    MaterialBinding                                binding{};
    std::unordered_map<std::string, std::uint32_t> name_to_group;
    for (const auto &group : mesh.physical_groups)
    {
        name_to_group.emplace(group.name, group.id);
    }
    for (std::size_t i = 0; i < cfg.assignments.size(); ++i)
    {
        const auto &assignment = cfg.assignments[i];
        const auto  name_iter  = name_to_group.find(assignment.group);
        if (name_iter == name_to_group.end())
        {
            return std::unexpected(PreprocessError{
                std::format("assignment references missing physical group '{}'", assignment.group),
                {"assignments", std::format("[{}]", i)}});
        }
        std::size_t material_index{};
        bool        found_material = false;
        for (std::size_t m = 0; m < cfg.materials.size(); ++m)
        {
            if (cfg.materials[m].name == assignment.material)
            {
                material_index = m;
                found_material = true;
                break;
            }
        }
        if (!found_material)
        {
            return std::unexpected(PreprocessError{
                std::format("assignment references missing material '{}'", assignment.material),
                {"assignments", std::format("[{}]", i)}});
        }
        binding.group_to_material.emplace(name_iter->second, material_index);
    }
    return binding;
}

[[nodiscard]] auto scale_vec(const Vec3 &value, double scalar) noexcept -> Vec3
{
    return Vec3{value[0] * scalar, value[1] * scalar, value[2] * scalar};
}

[[nodiscard]] auto check_duplicate_nodes(const mesh::Mesh &mesh) -> std::expected<void, PreprocessError>
{
    constexpr double                                            kEpsilon = 1.0e-12;
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> coord_hash_to_indices;

    auto hash_coord = [](double x) -> std::uint64_t {
        const auto scaled = static_cast<std::int64_t>(x / kEpsilon);
        return static_cast<std::uint64_t>(scaled);
    };

    for (std::size_t i = 0; i < mesh.nodes.size(); ++i)
    {
        const auto &pos = mesh.nodes[i].position;
        const auto  hx  = hash_coord(pos[0]);
        const auto  hy  = hash_coord(pos[1]);
        const auto  hz  = hash_coord(pos[2]);
        const auto  key = (hx * 73856093ULL) ^ (hy * 19349663ULL) ^ (hz * 83492791ULL);
        coord_hash_to_indices[key].push_back(i);
    }

    for (const auto &[key, indices] : coord_hash_to_indices)
    {
        if (indices.size() > 1)
        {
            for (std::size_t i = 0; i < indices.size(); ++i)
            {
                for (std::size_t j = i + 1; j < indices.size(); ++j)
                {
                    const auto &pos_i   = mesh.nodes[indices[i]].position;
                    const auto &pos_j   = mesh.nodes[indices[j]].position;
                    const auto  dx      = pos_i[0] - pos_j[0];
                    const auto  dy      = pos_i[1] - pos_j[1];
                    const auto  dz      = pos_i[2] - pos_j[2];
                    const auto  dist_sq = dx * dx + dy * dy + dz * dz;
                    if (dist_sq < kEpsilon * kEpsilon)
                    {
                        return std::unexpected(PreprocessError{
                            std::format("duplicate nodes detected: node {} and node {} at same position",
                                        indices[i], indices[j]),
                            {"mesh", "nodes"}});
                    }
                }
            }
        }
    }
    return {};
}

[[nodiscard]] auto check_duplicate_elements(const mesh::Mesh &mesh) -> std::expected<void, PreprocessError>
{
    std::unordered_map<std::uint64_t, std::vector<std::size_t>> conn_hash_to_indices;

// GCC 15 false-positive array-bounds warnings when sorting variable-length arrays
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Warray-bounds"

    for (std::size_t i = 0; i < mesh.elements.size(); ++i)
    {
        const auto &elem = mesh.elements[i];
        const auto  node_count =
            static_cast<std::size_t>(elem.geometry == ElementGeometry::Tetrahedron4 ? 4U : 8U);

        std::array<std::uint32_t, 8> sorted_nodes = elem.nodes;
        std::sort(sorted_nodes.begin(), sorted_nodes.begin() + static_cast<std::ptrdiff_t>(node_count));

        std::uint64_t hash = 0;
        for (std::size_t n = 0; n < node_count; ++n)
        {
            hash ^= (static_cast<std::uint64_t>(sorted_nodes[n]) * 2654435761ULL);
        }
        conn_hash_to_indices[hash].push_back(i);
    }

    for (const auto &[hash, indices] : conn_hash_to_indices)
    {
        if (indices.size() > 1)
        {
            for (std::size_t i = 0; i < indices.size(); ++i)
            {
                for (std::size_t j = i + 1; j < indices.size(); ++j)
                {
                    const auto &elem_i = mesh.elements[indices[i]];
                    const auto &elem_j = mesh.elements[indices[j]];
                    if (elem_i.geometry != elem_j.geometry)
                        continue;

                    const auto count =
                        static_cast<std::size_t>(elem_i.geometry == ElementGeometry::Tetrahedron4 ? 4U : 8U);
                    std::array<std::uint32_t, 8> sorted_i = elem_i.nodes;
                    std::array<std::uint32_t, 8> sorted_j = elem_j.nodes;
                    std::sort(sorted_i.begin(), sorted_i.begin() + static_cast<std::ptrdiff_t>(count));
                    std::sort(sorted_j.begin(), sorted_j.begin() + static_cast<std::ptrdiff_t>(count));

                    if (std::equal(sorted_i.begin(), sorted_i.begin() + static_cast<std::ptrdiff_t>(count),
                                   sorted_j.begin()))
                    {
                        return std::unexpected(
                            PreprocessError{std::format("duplicate elements detected: element {} and element "
                                                        "{} have same connectivity",
                                                        indices[i], indices[j]),
                                            {"mesh", "elements"}});
                    }
                }
            }
        }
    }

#pragma GCC diagnostic pop

    return {};
}

[[nodiscard]] auto validate_config_groups(const mesh::Mesh &mesh, const config::Config &cfg)
    -> std::expected<void, PreprocessError>
{
    std::unordered_map<std::string, std::uint32_t> name_to_group;
    for (const auto &group : mesh.physical_groups)
    {
        name_to_group.emplace(group.name, group.id);
    }

    for (std::size_t i = 0; i < cfg.dirichlet.size(); ++i)
    {
        const auto &fix = cfg.dirichlet[i];
        if (name_to_group.find(fix.group) == name_to_group.end())
        {
            return std::unexpected(PreprocessError{
                std::format("dirichlet fix references missing physical group '{}'", fix.group),
                {"dirichlet", "fixes", std::format("[{}]", i)}});
        }
        const auto group_id = name_to_group.at(fix.group);
        const auto surf_it  = mesh.surface_groups.find(group_id);
        if (surf_it == mesh.surface_groups.end() || surf_it->second.empty())
        {
            return std::unexpected(
                PreprocessError{std::format("dirichlet group '{}' lacks surface elements", fix.group),
                                {"dirichlet", "fixes", std::format("[{}]", i)}});
        }
    }

    for (std::size_t i = 0; i < cfg.loads.tractions.size(); ++i)
    {
        const auto &traction   = cfg.loads.tractions[i];
        const auto  group_iter = name_to_group.find(traction.group);
        if (group_iter == name_to_group.end())
        {
            return std::unexpected(PreprocessError{
                std::format("traction load references missing physical group '{}'", traction.group),
                {"loads", "tractions", std::format("[{}]", i)}});
        }
        const auto surf_it = mesh.surface_groups.find(group_iter->second);
        if (surf_it == mesh.surface_groups.end() || surf_it->second.empty())
        {
            return std::unexpected(
                PreprocessError{std::format("traction group '{}' has no discretized faces", traction.group),
                                {"loads", "tractions", std::format("[{}]", i)}});
        }
    }

    for (std::size_t i = 0; i < cfg.loads.points.size(); ++i)
    {
        const auto &load       = cfg.loads.points[i];
        const auto  group_iter = name_to_group.find(load.group);
        if (group_iter == name_to_group.end())
        {
            return std::unexpected(
                PreprocessError{std::format("point load references missing physical group '{}'", load.group),
                                {"loads", "points", std::format("[{}]", i)}});
        }
        const auto nodes_it = mesh.node_groups.find(group_iter->second);
        if (nodes_it == mesh.node_groups.end() || nodes_it->second.empty())
        {
            return std::unexpected(
                PreprocessError{std::format("point load group '{}' has no tagged nodes", load.group),
                                {"loads", "points", std::format("[{}]", i)}});
        }
    }

    return {};
}

[[nodiscard]] auto compute_tet_gradients(const std::array<Vec3, 4> &positions, double volume6)
    -> std::array<Vec3, 4>
{
    const Vec3  &p0   = positions[0];
    const Vec3  &p1   = positions[1];
    const Vec3  &p2   = positions[2];
    const Vec3  &p3   = positions[3];
    const double inv6 = -1.0 / volume6;
    return {scale_vec(common::cross(subtract(p2, p1), subtract(p3, p1)), inv6),
            scale_vec(common::cross(subtract(p3, p0), subtract(p2, p0)), inv6),
            scale_vec(common::cross(subtract(p1, p0), subtract(p3, p0)), inv6),
            scale_vec(common::cross(subtract(p2, p0), subtract(p1, p0)), inv6)};
}

} // namespace

auto run(const mesh::Mesh &mesh, const config::Config &cfg) -> std::expected<Outputs, PreprocessError>
{
    if (mesh.nodes.empty())
    {
        return make_error("mesh has zero nodes", {"mesh"});
    }
    if (mesh.elements.empty())
    {
        return make_error("mesh has zero elements", {"mesh"});
    }

    if (auto dup_nodes = check_duplicate_nodes(mesh); !dup_nodes)
    {
        return std::unexpected(dup_nodes.error());
    }
    if (auto dup_elems = check_duplicate_elements(mesh); !dup_elems)
    {
        return std::unexpected(dup_elems.error());
    }
    if (auto group_check = validate_config_groups(mesh, cfg); !group_check)
    {
        return std::unexpected(group_check.error());
    }

    auto binding_result = bind_materials(mesh, cfg);
    if (!binding_result)
    {
        return std::unexpected(binding_result.error());
    }
    const auto &binding = binding_result.value();

    Outputs outputs{};
    outputs.element_volumes.resize(mesh.elements.size());
    outputs.shape_gradients.resize(mesh.elements.size());
    outputs.element_material_index.resize(mesh.elements.size());
    outputs.lumped_mass.assign(mesh.nodes.size(), 0.0);

    std::vector<std::uint32_t> incident_counts(mesh.nodes.size(), 0U);

    for (std::size_t elem_index = 0; elem_index < mesh.elements.size(); ++elem_index)
    {
        const auto &element = mesh.elements[elem_index];
        if (element.geometry != ElementGeometry::Tetrahedron4)
        {
            return make_error("only tetrahedron elements supported in Phase 3",
                              {"elements", std::format("[{}]", elem_index)});
        }
        std::array<Vec3, 4> positions{};
        for (std::size_t local = 0; local < 4; ++local)
        {
            const auto node_idx = element.nodes[local];
            if (node_idx >= mesh.nodes.size())
            {
                return make_error("element references node out of range",
                                  {"elements", std::format("[{}]", elem_index)});
            }
            positions[local] = mesh.nodes[node_idx].position;
            ++incident_counts[node_idx];
        }
        const Vec3   e0      = subtract(positions[1], positions[0]);
        const Vec3   e1      = subtract(positions[2], positions[0]);
        const Vec3   e2      = subtract(positions[3], positions[0]);
        const double volume6 = common::dot(e0, common::cross(e1, e2));
        const double volume  = std::abs(volume6) / 6.0;
        if (volume <= std::numeric_limits<double>::epsilon())
        {
            return make_error("tetrahedron volume non-positive",
                              {"elements", std::format("[{}]", elem_index)});
        }
        outputs.element_volumes[elem_index] = volume;

        auto gradients = compute_tet_gradients(positions, volume6);
        outputs.shape_gradients[elem_index].fill(Vec3{0.0, 0.0, 0.0});
        for (std::size_t i = 0; i < 4; ++i)
        {
            outputs.shape_gradients[elem_index][i] = gradients[i];
        }

        const auto material_iter = binding.group_to_material.find(element.physical_group);
        if (material_iter == binding.group_to_material.end())
        {
            return make_error("element physical group missing assignment",
                              {"elements", std::format("[{}]", elem_index)});
        }
        const auto material_index                  = material_iter->second;
        outputs.element_material_index[elem_index] = material_index;
        const auto   density                       = cfg.materials[material_index].density;
        const double lump                          = density * volume / 4.0;
        for (std::size_t local = 0; local < 4; ++local)
        {
            outputs.lumped_mass[element.nodes[local]] += lump;
        }
    }

    outputs.adjacency.offsets.resize(mesh.nodes.size() + 1U, 0U);
    std::uint32_t accumulator = 0U;
    for (std::size_t node = 0; node < mesh.nodes.size(); ++node)
    {
        outputs.adjacency.offsets[node] = accumulator;
        accumulator += incident_counts[node];
    }
    outputs.adjacency.offsets.back() = accumulator;
    outputs.adjacency.element_indices.resize(accumulator, 0U);
    outputs.adjacency.local_indices.resize(accumulator, 0U);

    std::vector<std::uint32_t> cursor(mesh.nodes.size(), 0U);
    for (std::size_t elem_index = 0; elem_index < mesh.elements.size(); ++elem_index)
    {
        const auto &element = mesh.elements[elem_index];
        for (std::size_t local = 0; local < 4; ++local)
        {
            const auto node_index = element.nodes[local];
            const auto write_base = outputs.adjacency.offsets[node_index] + cursor[node_index];
            outputs.adjacency.element_indices[write_base] = static_cast<std::uint32_t>(elem_index);
            outputs.adjacency.local_indices[write_base]   = static_cast<std::uint8_t>(local);
            ++cursor[node_index];
        }
    }

    return outputs;
}

} // namespace cwf::mesh::pre
