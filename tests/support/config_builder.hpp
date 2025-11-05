/**
 * @file config_builder.hpp
 * @brief test-only YAML config generator so suites stay DRY uwu
 *
 * this header lives inside the tests/ tree and fabricates YAML docs matching
 * the RefDocs spec. test cases tweak the options struct to dial in weird edge
 * cases (missing sections, invalid scalars, you name it) without duplicating
 * 80+ lines of literal strings. builders emit strings that feed directly into
 * cwf::config::load_config_from_string so validation logic exercises the real
 * parser every time.
 */
#pragma once

#include <array>
#include <cstdint>
#include <iomanip>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cwf/config/config.hpp"

namespace cwf::test_support
{

struct MaterialSpec
{
    std::string name{"concrete"};
    double      youngs_modulus{3.0e10};
    double      poisson_ratio{0.2};
    double      density{2500.0};
};

struct AssignmentSpec
{
    std::string group{"SOLID"};
    std::string material{"concrete"};
};

struct CurveSpec
{
    std::string                            name{"load_curve1"};
    std::vector<std::pair<double, double>> points{{0.0, 0.0}, {0.5, 0.75}, {1.0, 1.0}};
};

struct TractionSpec
{
    std::string           group{"LOAD_FACE"};
    std::array<double, 3> value{0.0, 0.0, -1.0e5};
    std::string           scale_curve{"load_curve1"};
};

struct DirichletSpec
{
    std::string                          group{"FIXED_BASE"};
    std::array<bool, 3>                  constrain{true, true, true};
    std::array<std::optional<double>, 3> values{std::nullopt, std::nullopt, std::nullopt};
};

struct ConfigBuilderOptions
{
    bool        include_mesh{true};
    std::string mesh_path{"tests/data/cantilever.msh"};

    bool                      include_materials{true};
    std::vector<MaterialSpec> materials{{}};

    bool                        include_assignments{true};
    std::vector<AssignmentSpec> assignments{{}};

    bool   include_damping{true};
    double damping_xi{0.02};
    double damping_w1{10.0};
    double damping_w2{100.0};

    bool   include_time{true};
    double time_dt{0.01111};
    bool   time_adaptive{true};
    bool   include_time_min_dt{true};
    double time_min_dt{0.005};
    bool   include_time_max_dt{true};
    double time_max_dt{0.02};

    bool          include_solver{true};
    std::string   solver_type{"pcg"};
    std::string   solver_preconditioner{"block_jacobi"};
    double        solver_runtime_tol{2.0e-4};
    double        solver_pause_tol{1.0e-5};
    std::uint32_t solver_max_iters{120U};

    bool        include_precision{true};
    std::string vector_precision{"fp32"};
    std::string reduction_precision{"fp64"};

    bool                   include_curves{true};
    std::vector<CurveSpec> curves{{}};

    bool                      include_loads{true};
    std::array<double, 3>     gravity{0.0, 0.0, -9.81};
    std::vector<TractionSpec> tractions{{}};

    bool                       include_dirichlet{true};
    std::vector<DirichletSpec> dirichlet_fixes{{}};

    bool                       include_output{true};
    std::uint32_t              output_stride{10U};
    std::vector<std::uint32_t> output_probes{1U, 2U};
};

namespace detail
{

inline void write_vec3(std::ostringstream &oss, std::string_view indent, const std::array<double, 3> &vec)
{
    oss << indent << "[" << vec[0] << ", " << vec[1] << ", " << vec[2] << "]\n";
}

inline void write_optional_vec3(std::ostringstream &oss, std::string_view indent,
                                const std::array<std::optional<double>, 3> &vec)
{
    oss << indent << "[";
    for (std::size_t i = 0; i < vec.size(); ++i)
    {
        if (i != 0U)
        {
            oss << ", ";
        }
        if (vec[i].has_value())
        {
            oss << vec[i].value();
        }
        else
        {
            oss << "null";
        }
    }
    oss << "]\n";
}

inline void write_dof_list(std::ostringstream &oss, std::string_view indent,
                           const std::array<bool, 3> &dof_mask)
{
    oss << indent << "[";
    bool                                  first = true;
    const std::array<std::string_view, 3> labels{"x", "y", "z"};
    for (std::size_t axis = 0; axis < dof_mask.size(); ++axis)
    {
        if (!dof_mask[axis])
        {
            continue;
        }
        if (!first)
        {
            oss << ", ";
        }
        first = false;
        oss << labels[axis];
    }
    oss << "]\n";
}

} // namespace detail

inline auto make_config_yaml(const ConfigBuilderOptions &options = {}) -> std::string
{
    std::ostringstream oss;
    oss << std::setprecision(12) << std::boolalpha;

    if (options.include_mesh)
    {
        oss << "mesh:\n";
        oss << "  path: " << options.mesh_path << "\n";
    }

    if (options.include_materials)
    {
        oss << "materials:\n";
        if (options.materials.empty())
        {
            oss << "  []\n";
        }
        else
        {
            for (const auto &material : options.materials)
            {
                oss << "  - name: " << material.name << "\n";
                oss << "    E: " << material.youngs_modulus << "\n";
                oss << "    nu: " << material.poisson_ratio << "\n";
                oss << "    rho: " << material.density << "\n";
            }
        }
    }

    if (options.include_assignments)
    {
        oss << "assignments:\n";
        if (options.assignments.empty())
        {
            oss << "  []\n";
        }
        else
        {
            for (const auto &assignment : options.assignments)
            {
                oss << "  - group: " << assignment.group << "\n";
                oss << "    material: " << assignment.material << "\n";
            }
        }
    }

    if (options.include_damping)
    {
        oss << "damping:\n";
        oss << "  xi: " << options.damping_xi << "\n";
        oss << "  w1: " << options.damping_w1 << "\n";
        oss << "  w2: " << options.damping_w2 << "\n";
    }

    if (options.include_time)
    {
        oss << "time:\n";
        oss << "  dt: " << options.time_dt << "\n";
        oss << "  adaptive: " << options.time_adaptive << "\n";
        if (options.include_time_min_dt)
        {
            oss << "  min_dt: " << options.time_min_dt << "\n";
        }
        if (options.include_time_max_dt)
        {
            oss << "  max_dt: " << options.time_max_dt << "\n";
        }
    }

    if (options.include_solver)
    {
        oss << "solver:\n";
        oss << "  type: " << options.solver_type << "\n";
        oss << "  preconditioner: " << options.solver_preconditioner << "\n";
        oss << "  tol_runtime: " << options.solver_runtime_tol << "\n";
        oss << "  tol_pause: " << options.solver_pause_tol << "\n";
        oss << "  max_iters: " << options.solver_max_iters << "\n";
    }

    if (options.include_precision)
    {
        oss << "precision:\n";
        oss << "  vectors: " << options.vector_precision << "\n";
        oss << "  reductions: " << options.reduction_precision << "\n";
    }

    if (options.include_curves)
    {
        oss << "curves:\n";
        if (options.curves.empty())
        {
            oss << "  {}\n";
        }
        else
        {
            for (const auto &curve : options.curves)
            {
                oss << "  " << curve.name << ":\n";
                if (curve.points.empty())
                {
                    oss << "    []\n";
                }
                else
                {
                    for (const auto &[time, value] : curve.points)
                    {
                        oss << "    - [" << time << ", " << value << "]\n";
                    }
                }
            }
        }
    }

    if (options.include_loads)
    {
        oss << "loads:\n";
        oss << "  gravity: ";
        detail::write_vec3(oss, "", options.gravity);
        if (!options.tractions.empty())
        {
            oss << "  tractions:\n";
            for (const auto &traction : options.tractions)
            {
                oss << "    - group: " << traction.group << "\n";
                oss << "      value: ";
                detail::write_vec3(oss, "", traction.value);
                if (!traction.scale_curve.empty())
                {
                    oss << "      scale_curve: " << traction.scale_curve << "\n";
                }
            }
        }
    }

    if (options.include_dirichlet)
    {
        oss << "dirichlet:\n";
        oss << "  fixes:\n";
        if (options.dirichlet_fixes.empty())
        {
            oss << "    []\n";
        }
        else
        {
            for (const auto &fix : options.dirichlet_fixes)
            {
                oss << "    - group: " << fix.group << "\n";
                oss << "      dof: ";
                detail::write_dof_list(oss, "", fix.constrain);
                oss << "      value: ";
                detail::write_optional_vec3(oss, "", fix.values);
            }
        }
    }

    if (options.include_output)
    {
        oss << "output:\n";
        oss << "  vtu_stride: " << options.output_stride << "\n";
        oss << "  probes:\n";
        if (options.output_probes.empty())
        {
            oss << "    []\n";
        }
        else
        {
            for (auto probe : options.output_probes)
            {
                oss << "    - " << probe << "\n";
            }
        }
    }

    return oss.str();
}

inline auto load_config(const ConfigBuilderOptions &options = {}) -> config::ConfigResult
{
    return config::load_config_from_string(make_config_yaml(options));
}

} // namespace cwf::test_support
