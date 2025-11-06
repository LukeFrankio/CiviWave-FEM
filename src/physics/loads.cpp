/**
 * @file loads.cpp
 * @brief implementation of nodal load assembly for CPU reference solver
 */
#include "cwf/physics/loads.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "cwf/common/math.hpp"

namespace cwf::physics::loads
{
namespace
{

[[nodiscard]] auto make_group_lookup(const mesh::Mesh &mesh) -> std::unordered_map<std::string, std::uint32_t>
{
    std::unordered_map<std::string, std::uint32_t> lookup;
    lookup.reserve(mesh.physical_groups.size());
    for (const auto &group : mesh.physical_groups)
    {
        lookup.emplace(group.name, group.id);
    }
    return lookup;
}

[[nodiscard]] auto get_curve_factor(const config::Config &cfg, const std::string &name, double time) -> double
{
    if (name.empty())
    {
        return 1.0;
    }
    const auto iter = cfg.curves.find(name);
    if (iter == cfg.curves.end())
    {
        return 1.0;
    }
    return evaluate_curve(iter->second, time);
}

[[nodiscard]] auto diff(const common::Vec3 &a, const common::Vec3 &b) noexcept -> common::Vec3
{
    return common::Vec3{a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

[[nodiscard]] auto triangle_area(const mesh::Mesh &mesh, std::uint32_t i0, std::uint32_t i1, std::uint32_t i2)
    -> double
{
    const auto  &p0  = mesh.nodes.at(i0).position;
    const auto  &p1  = mesh.nodes.at(i1).position;
    const auto  &p2  = mesh.nodes.at(i2).position;
    const auto   v1  = diff(p1, p0);
    const auto   v2  = diff(p2, p0);
    const auto   cr  = common::cross(v1, v2);
    const double mag = 0.5 * std::sqrt(common::dot(cr, cr));
    return mag;
}

} // namespace

auto evaluate_curve(const config::Curve &curve, double time) -> double
{
    if (curve.points.empty())
    {
        return 1.0;
    }
    if (time <= curve.points.front().first)
    {
        return curve.points.front().second;
    }
    for (std::size_t i = 1; i < curve.points.size(); ++i)
    {
        const auto &previous = curve.points[i - 1];
        const auto &current  = curve.points[i];
        if (time <= current.first)
        {
            const double span   = current.first - previous.first;
            const double weight = span > 0.0 ? (time - previous.first) / span : 0.0;
            return std::lerp(previous.second, current.second, weight);
        }
    }
    return curve.points.back().second;
}

auto assemble_load_vector(const mesh::Mesh &mesh, const config::Config &cfg,
                          const mesh::pre::Outputs &preprocess, double time) -> std::vector<double>
{
    const std::size_t   dof_count = mesh.nodes.size() * 3U;
    std::vector<double> loads(dof_count, 0.0);

    for (std::size_t node = 0; node < mesh.nodes.size(); ++node)
    {
        const double mass = preprocess.lumped_mass.at(node);
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            loads[node * 3U + axis] += mass * cfg.loads.gravity[axis];
        }
    }

    const auto group_lookup = make_group_lookup(mesh);

    for (const auto &traction : cfg.loads.tractions)
    {
        const auto group_iter = group_lookup.find(traction.group);
        if (group_iter == group_lookup.end())
        {
            continue;
        }
        const auto surf_iter = mesh.surface_groups.find(group_iter->second);
        if (surf_iter == mesh.surface_groups.end())
        {
            continue;
        }
        const double scale = get_curve_factor(cfg, traction.scale_curve, time);
        for (const auto surface_index : surf_iter->second)
        {
            const auto &surface    = mesh.surfaces.at(surface_index);
            double      area       = 0.0;
            std::size_t node_count = 0U;
            switch (surface.geometry)
            {
            case mesh::SurfaceGeometry::Triangle3:
                area       = triangle_area(mesh, surface.nodes[0], surface.nodes[1], surface.nodes[2]);
                node_count = 3U;
                break;
            case mesh::SurfaceGeometry::Quadrilateral4: {
                area = triangle_area(mesh, surface.nodes[0], surface.nodes[1], surface.nodes[2]) +
                       triangle_area(mesh, surface.nodes[0], surface.nodes[2], surface.nodes[3]);
                node_count = 4U;
                break;
            }
            }
            if (node_count == 0U)
            {
                continue;
            }
            const double nodal_share = (area * scale) / static_cast<double>(node_count);
            for (std::size_t node_slot = 0; node_slot < node_count; ++node_slot)
            {
                const auto node_index = surface.nodes[node_slot];
                for (std::size_t axis = 0; axis < 3U; ++axis)
                {
                    loads[node_index * 3U + axis] += nodal_share * traction.value[axis];
                }
            }
        }
    }

    for (const auto &point : cfg.loads.points)
    {
        const auto group_iter = group_lookup.find(point.group);
        if (group_iter == group_lookup.end())
        {
            continue;
        }
        const auto nodes_iter = mesh.node_groups.find(group_iter->second);
        if (nodes_iter == mesh.node_groups.end())
        {
            continue;
        }
        const double scale = get_curve_factor(cfg, point.scale_curve, time);
        for (const auto node_index : nodes_iter->second)
        {
            for (std::size_t axis = 0; axis < 3U; ++axis)
            {
                loads[node_index * 3U + axis] += scale * point.value[axis];
            }
        }
    }

    return loads;
}

} // namespace cwf::physics::loads
