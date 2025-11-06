/**
 * @file solver.cpp
 * @brief CPU reference assembly + CG solver implementation
 */
#include "cwf/physics/solver.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <unordered_map>
#include <unordered_set>

#include "cwf/common/math.hpp"
#include "cwf/physics/loads.hpp"

namespace cwf::physics::solver
{
namespace
{

[[nodiscard]] auto dof_count(const mesh::Mesh &mesh) -> std::size_t
{
    return mesh.nodes.size() * 3U;
}

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

[[nodiscard]] auto build_element_stiffness(const std::array<common::Vec3, 8> &gradients, double volume,
                                           const std::array<double, 36> &stiffness) -> std::array<double, 144>
{
    constexpr std::size_t   kDofs   = 12U;
    constexpr std::size_t   kStrain = 6U;
    std::array<double, 72>  B{};
    std::array<double, 72>  DB{};
    std::array<double, 144> Ke{};

    for (std::size_t node = 0; node < 4U; ++node)
    {
        const auto       &grad   = gradients[node];
        const std::size_t col    = node * 3U;
        B[0U * kDofs + col + 0U] = grad[0];
        B[1U * kDofs + col + 1U] = grad[1];
        B[2U * kDofs + col + 2U] = grad[2];
        B[3U * kDofs + col + 0U] = grad[1];
        B[3U * kDofs + col + 1U] = grad[0];
        B[4U * kDofs + col + 1U] = grad[2];
        B[4U * kDofs + col + 2U] = grad[1];
        B[5U * kDofs + col + 0U] = grad[2];
        B[5U * kDofs + col + 2U] = grad[0];
    }

    for (std::size_t row = 0; row < kStrain; ++row)
    {
        for (std::size_t col = 0; col < kDofs; ++col)
        {
            double sum = 0.0;
            for (std::size_t mid = 0; mid < kStrain; ++mid)
            {
                sum += stiffness[row * kStrain + mid] * B[mid * kDofs + col];
            }
            DB[row * kDofs + col] = sum;
        }
    }

    for (std::size_t i = 0; i < kDofs; ++i)
    {
        for (std::size_t j = 0; j < kDofs; ++j)
        {
            double sum = 0.0;
            for (std::size_t row = 0; row < kStrain; ++row)
            {
                sum += B[row * kDofs + i] * DB[row * kDofs + j];
            }
            Ke[i * kDofs + j] = sum * volume;
        }
    }

    return Ke;
}

[[nodiscard]] auto gather_surface_nodes(const mesh::Mesh &mesh, std::uint32_t group_id)
    -> std::unordered_set<std::uint32_t>
{
    std::unordered_set<std::uint32_t> nodes;
    const auto                        surf_iter = mesh.surface_groups.find(group_id);
    if (surf_iter == mesh.surface_groups.end())
    {
        return nodes;
    }
    for (const auto surface_index : surf_iter->second)
    {
        const auto &surface = mesh.surfaces.at(surface_index);
        const auto  limit   = surface.geometry == mesh::SurfaceGeometry::Quadrilateral4 ? 4U : 3U;
        for (std::size_t i = 0; i < limit; ++i)
        {
            nodes.insert(surface.nodes[i]);
        }
    }
    return nodes;
}

struct CgResult
{
    std::vector<double> solution;
    SolveStats          stats;
};

[[nodiscard]] auto dot(const std::vector<double> &a, const std::vector<double> &b) -> double
{
    double sum = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        sum += a[i] * b[i];
    }
    return sum;
}

[[nodiscard]] auto apply_matrix(const std::vector<double> &matrix, const std::vector<double> &vector)
    -> std::vector<double>
{
    const std::size_t   n = vector.size();
    std::vector<double> result(n, 0.0);
    for (std::size_t row = 0; row < n; ++row)
    {
        double        sum     = 0.0;
        const double *row_ptr = matrix.data() + (row * n);
        for (std::size_t col = 0; col < n; ++col)
        {
            sum += row_ptr[col] * vector[col];
        }
        result[row] = sum;
    }
    return result;
}

[[nodiscard]] auto conjugate_gradient(const std::vector<double> &matrix, const std::vector<double> &rhs,
                                      std::size_t max_iterations, double tolerance) -> CgResult
{
    const std::size_t   n = rhs.size();
    std::vector<double> x(n, 0.0);
    std::vector<double> r = rhs;
    std::vector<double> diag(n, 1.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        const double value = matrix[i * n + i];
        diag[i]            = std::abs(value) > std::numeric_limits<double>::epsilon() ? value : 1.0;
    }
    std::vector<double> z(n, 0.0);
    for (std::size_t i = 0; i < n; ++i)
    {
        z[i] = r[i] / diag[i];
    }
    std::vector<double> p             = z;
    double              rho           = dot(r, z);
    double              residual_norm = std::sqrt(dot(r, r));
    SolveStats          stats{};
    if (residual_norm <= tolerance)
    {
        stats.converged     = true;
        stats.residual_norm = residual_norm;
        stats.iterations    = 0U;
        return CgResult{std::move(x), stats};
    }

    for (std::size_t iter = 0; iter < max_iterations; ++iter)
    {
        const auto   Ap    = apply_matrix(matrix, p);
        const double denom = dot(p, Ap);
        if (std::abs(denom) < std::numeric_limits<double>::epsilon())
        {
            break;
        }
        const double alpha = rho / denom;
        for (std::size_t i = 0; i < n; ++i)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        residual_norm    = std::sqrt(dot(r, r));
        stats.iterations = iter + 1U;
        if (residual_norm <= tolerance)
        {
            stats.converged     = true;
            stats.residual_norm = residual_norm;
            return CgResult{std::move(x), stats};
        }
        for (std::size_t i = 0; i < n; ++i)
        {
            z[i] = r[i] / diag[i];
        }
        const double rho_new = dot(r, z);
        const double beta    = rho_new / rho;
        rho                  = rho_new;
        for (std::size_t i = 0; i < n; ++i)
        {
            p[i] = z[i] + beta * p[i];
        }
    }
    stats.converged     = false;
    stats.residual_norm = residual_norm;
    return CgResult{std::move(x), stats};
}

[[nodiscard]] auto assemble_lumped_mass(const mesh::pre::Outputs &preprocess, std::size_t dofs)
    -> std::vector<double>
{
    std::vector<double> mass(dofs, 0.0);
    for (std::size_t node = 0; node < preprocess.lumped_mass.size(); ++node)
    {
        const double value = preprocess.lumped_mass[node];
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            mass[node * 3U + axis] = value;
        }
    }
    return mass;
}

inline void apply_dirichlet(std::vector<double> &matrix, std::vector<double> &rhs,
                            const DirichletConditions &conditions, const newmark::State &state)
{
    const std::size_t n = rhs.size();
    for (std::size_t dof = 0; dof < n; ++dof)
    {
        if (!conditions.mask[dof])
        {
            continue;
        }
        for (std::size_t col = 0; col < n; ++col)
        {
            matrix[dof * n + col] = 0.0;
        }
        for (std::size_t row = 0; row < n; ++row)
        {
            matrix[row * n + dof] = 0.0;
        }
        matrix[dof * n + dof] = 1.0;
        rhs[dof]              = conditions.targets[dof] - state.displacement[dof];
    }
}

} // namespace

auto assemble_linear_system(const mesh::Mesh &mesh, const mesh::pre::Outputs &preprocess,
                            const std::vector<materials::ElasticProperties> &materials) -> Assembly
{
    const std::size_t n = dof_count(mesh);
    Assembly          assembly{};
    assembly.stiffness.assign(n * n, 0.0);
    assembly.mass_diag = assemble_lumped_mass(preprocess, n);

    for (std::size_t elem = 0; elem < mesh.elements.size(); ++elem)
    {
        const auto &element = mesh.elements[elem];
        if (element.geometry != mesh::ElementGeometry::Tetrahedron4)
        {
            continue;
        }
        const auto   material_index = preprocess.element_material_index[elem];
        const auto  &props          = materials.at(material_index);
        const auto  &grads          = preprocess.shape_gradients[elem];
        const double volume         = preprocess.element_volumes[elem];
        const auto   ke             = build_element_stiffness(grads, volume, props.stiffness);

        for (std::size_t a = 0; a < 4U; ++a)
        {
            const auto node_a = element.nodes[a];
            for (std::size_t axis_a = 0; axis_a < 3U; ++axis_a)
            {
                const std::size_t global_i = node_a * 3U + axis_a;
                const std::size_t local_i  = a * 3U + axis_a;
                for (std::size_t b = 0; b < 4U; ++b)
                {
                    const auto node_b = element.nodes[b];
                    for (std::size_t axis_b = 0; axis_b < 3U; ++axis_b)
                    {
                        const std::size_t global_j = node_b * 3U + axis_b;
                        const std::size_t local_j  = b * 3U + axis_b;
                        assembly.stiffness[global_i * n + global_j] += ke[local_i * 12U + local_j];
                    }
                }
            }
        }
    }

    return assembly;
}

auto build_dirichlet_conditions(const mesh::Mesh &mesh, const config::Config &cfg) -> DirichletConditions
{
    const std::size_t   n = dof_count(mesh);
    DirichletConditions conditions{};
    conditions.mask.assign(n, false);
    conditions.targets.assign(n, 0.0);

    const auto                         group_lookup = make_group_lookup(mesh);
    std::vector<std::optional<double>> existing(n);

    for (const auto &fix : cfg.dirichlet)
    {
        const auto group_iter = group_lookup.find(fix.group);
        if (group_iter == group_lookup.end())
        {
            continue;
        }
        const auto nodes = gather_surface_nodes(mesh, group_iter->second);
        for (const auto node : nodes)
        {
            for (std::size_t axis = 0; axis < 3U; ++axis)
            {
                if (!fix.constrain_axis[axis])
                {
                    continue;
                }
                const double      value = fix.value[axis].value_or(0.0);
                const std::size_t dof   = node * 3U + axis;
                conditions.mask[dof]    = true;
                conditions.targets[dof] = value;
                if (existing[dof].has_value() && std::abs(existing[dof].value() - value) > 1.0e-12)
                {
                    conditions.targets[dof] = value;
                }
                existing[dof] = value;
            }
        }
    }

    return conditions;
}

auto solve_newmark_step(const Assembly &assembly, const materials::RayleighCoefficients &rayleigh,
                        const DirichletConditions &dirichlet, const mesh::Mesh &mesh,
                        const config::Config &cfg, const mesh::pre::Outputs &preprocess,
                        const newmark::Coefficients &coeffs, const newmark::State &previous_state,
                        double time, double tolerance, std::size_t max_iterations) -> StepResult
{
    const auto load = physics::loads::assemble_load_vector(mesh, cfg, preprocess, time);
    auto rhs  = newmark::build_effective_rhs(load, assembly.stiffness, assembly.mass_diag, rayleigh, coeffs,
                                             previous_state);
    auto keff = newmark::build_effective_stiffness(assembly.stiffness, assembly.mass_diag, rayleigh, coeffs);

    apply_dirichlet(keff, rhs, dirichlet, previous_state);
    const auto cg         = conjugate_gradient(keff, rhs, max_iterations, tolerance);
    auto       next_state = newmark::update_state(coeffs, previous_state, cg.solution);

    for (std::size_t dof = 0; dof < dirichlet.mask.size(); ++dof)
    {
        if (dirichlet.mask[dof])
        {
            next_state.displacement[dof] = dirichlet.targets[dof];
        }
    }

    return StepResult{std::move(next_state), cg.stats};
}

} // namespace cwf::physics::solver
