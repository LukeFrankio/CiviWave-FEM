/**
 * @file pcg_test.cpp
 * @brief unit tests for Phase 8 matrix-free K_eff apply + PCG solver uwu
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <numeric>
#include <vector>

#include "cwf/config/config.hpp"
#include "cwf/gpu/pcg.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/mesh/preprocess.hpp"
#include "cwf/physics/loads.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/physics/newmark.hpp"
#include "cwf/physics/solver.hpp"

namespace
{

using ::testing::Le;

constexpr double kDt = 0.01;
constexpr double kRelativeTol = 3.0e-4;
constexpr std::size_t kMaxIterations = 64U;

[[nodiscard]] auto make_single_tet_mesh() -> cwf::mesh::Mesh
{
    cwf::mesh::Mesh mesh{};
    mesh.nodes = {
        cwf::mesh::Node{0U, cwf::common::Vec3{0.0, 0.0, 0.0}},
        cwf::mesh::Node{1U, cwf::common::Vec3{1.0, 0.0, 0.0}},
        cwf::mesh::Node{2U, cwf::common::Vec3{0.0, 1.0, 0.0}},
        cwf::mesh::Node{3U, cwf::common::Vec3{0.0, 0.0, 1.0}},
    };

    cwf::mesh::Element tet{};
    tet.original_id   = 0U;
    tet.physical_group = 1U; // SOLID
    tet.geometry       = cwf::mesh::ElementGeometry::Tetrahedron4;
    tet.nodes          = {0U, 1U, 2U, 3U, std::numeric_limits<std::uint32_t>::max(),
                          std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max(),
                          std::numeric_limits<std::uint32_t>::max()};
    mesh.elements.push_back(tet);

    cwf::mesh::Surface fixed{};
    fixed.original_id    = 0U;
    fixed.physical_group = 2U; // FIXED surface
    fixed.geometry       = cwf::mesh::SurfaceGeometry::Triangle3;
    fixed.nodes          = {0U, 1U, 2U, std::numeric_limits<std::uint32_t>::max()};
    mesh.surfaces.push_back(fixed);

    mesh.physical_groups = {
        cwf::mesh::PhysicalGroup{3U, 1U, "SOLID"},
        cwf::mesh::PhysicalGroup{2U, 2U, "FIXED"},
        cwf::mesh::PhysicalGroup{0U, 3U, "POINT"},
    };
    for (std::size_t i = 0; i < mesh.physical_groups.size(); ++i)
    {
        mesh.group_lookup.emplace(mesh.physical_groups[i].id, i);
    }

    mesh.surface_groups[2U] = {0U};
    mesh.node_groups[3U]    = {3U};

    return mesh;
}

[[nodiscard]] auto make_basic_config() -> cwf::config::Config
{
    cwf::config::Config cfg{};
    cfg.mesh_path = std::filesystem::path{"synthetic.msh"};

    cwf::config::Material mat{};
    mat.name           = "steel";
    mat.youngs_modulus = 30.0e9;
    mat.poisson_ratio  = 0.2;
    mat.density        = 2500.0;
    cfg.materials.push_back(mat);

    cwf::config::Assignment assignment{};
    assignment.group    = "SOLID";
    assignment.material = "steel";
    cfg.assignments.push_back(assignment);

    cfg.damping = cwf::config::Damping{.xi = 0.02, .w1 = 5.0, .w2 = 50.0};

    cfg.time = cwf::config::TimeSettings{.initial_dt = kDt, .adaptive = false, .min_dt = 0.0, .max_dt = 0.0};
    cfg.solver = cwf::config::SolverSettings{
        .type = "pcg",
        .preconditioner = "block_jacobi",
        .runtime_tolerance = kRelativeTol,
        .pause_tolerance   = 1.0e-5,
        .max_iterations    = static_cast<std::uint32_t>(kMaxIterations)
    };
    cfg.precision = cwf::config::PrecisionSettings{.vector_precision = "fp32", .reduction_precision = "fp64"};

    cwf::config::PointLoad point{};
    point.group = "POINT";
    point.value = {0.0, 0.0, -500.0};
    cfg.loads.points.push_back(point);
    cfg.loads.gravity = {0.0, 0.0, 0.0};

    cwf::config::DirichletFix fix{};
    fix.group          = "FIXED";
    fix.constrain_axis = {true, true, true};
    fix.value          = {0.0, 0.0, 0.0};
    cfg.dirichlet.push_back(fix);

    cfg.output = cwf::config::OutputSettings{.vtu_stride = 10U, .probes = {}};

    return cfg;
}

[[nodiscard]] auto make_materials(const cwf::config::Config &cfg)
    -> std::vector<cwf::physics::materials::ElasticProperties>
{
    std::vector<cwf::physics::materials::ElasticProperties> materials{};
    materials.reserve(cfg.materials.size());
    for (const auto &material : cfg.materials)
    {
        materials.push_back(cwf::physics::materials::make_properties(material));
    }
    return materials;
}

void apply_dirichlet_dense(std::vector<double> &matrix, const cwf::physics::solver::DirichletConditions &dirichlet)
{
    const auto n = dirichlet.mask.size();
    const auto stride = n;
    for (std::size_t dof = 0; dof < n; ++dof)
    {
        if (!dirichlet.mask[dof])
        {
            continue;
        }
        for (std::size_t col = 0; col < n; ++col)
        {
            matrix[dof * stride + col] = (col == dof) ? 1.0 : 0.0;
        }
        for (std::size_t row = 0; row < n; ++row)
        {
            if (row == dof)
            {
                continue;
            }
            matrix[row * stride + dof] = 0.0;
        }
    }
}

void apply_dirichlet_rhs(std::vector<double> &rhs, const cwf::physics::solver::DirichletConditions &dirichlet,
                         const cwf::physics::newmark::State &state)
{
    for (std::size_t dof = 0; dof < dirichlet.mask.size(); ++dof)
    {
        if (dirichlet.mask[dof])
        {
            rhs[dof] = dirichlet.targets[dof] - state.displacement[dof];
        }
    }
}

[[nodiscard]] auto dense_apply(const std::vector<double> &matrix, const std::vector<double> &vector)
    -> std::vector<double>
{
    const auto n = vector.size();
    std::vector<double> result(n, 0.0);
    for (std::size_t row = 0; row < n; ++row)
    {
        double sum = 0.0;
        const double *row_ptr = matrix.data() + row * n;
        for (std::size_t col = 0; col < n; ++col)
        {
            sum += row_ptr[col] * vector[col];
        }
        result[row] = sum;
    }
    return result;
}

} // namespace

/**
 * @test matrix-free K_eff apply matches dense CPU reference within FP32 tolerance
 */
TEST(PcgPhase8, MatrixFreeApplyMatchesDense)
{
    auto mesh = make_single_tet_mesh();
    auto cfg  = make_basic_config();

    const auto preprocess_result = cwf::mesh::pre::run(mesh, cfg);
    ASSERT_TRUE(preprocess_result.has_value());
    const auto &pre = preprocess_result.value();

    const auto pack_result = cwf::mesh::pack::build_packed_buffers(mesh, pre, cfg, {});
    ASSERT_TRUE(pack_result.has_value());
    auto pack = pack_result.value();

    const auto materials = make_materials(cfg);

    const auto assembly = cwf::physics::solver::assemble_linear_system(mesh, pre, materials);
    const auto dirichlet = cwf::physics::solver::build_dirichlet_conditions(mesh, cfg);

    const auto coeffs = cwf::physics::newmark::make_coefficients(kDt, 0.25, 0.5);
    const auto rayleigh = cwf::physics::materials::compute_rayleigh(cfg.damping);

    auto keff = cwf::physics::newmark::build_effective_stiffness(assembly.stiffness, assembly.mass_diag, rayleigh, coeffs);
    apply_dirichlet_dense(keff, dirichlet);

    const auto dof_count = pack.metadata.dof_count;
    std::vector<float> input(dof_count, 0.0F);
    for (std::size_t i = 0; i < dof_count; ++i)
    {
        input[i] = static_cast<float>(0.1 * static_cast<double>(i + 1));
    }
    std::vector<double> input_double(input.begin(), input.end());

    cwf::gpu::pcg::MatrixFreeSystem system{
        .element_connectivity = std::span<const std::uint32_t>{pack.buffers.elements.connectivity},
        .element_gradients = std::span<const float>{pack.buffers.elements.gradients},
        .element_volume = std::span<const float>{pack.buffers.elements.volume},
        .element_material_index = std::span<const std::uint32_t>{pack.buffers.elements.material_index},
        .materials = std::span<const cwf::physics::materials::ElasticProperties>{materials},
        .lumped_mass = std::span<const float>{pack.buffers.nodes.lumped_mass},
        .bc_mask = std::span<const std::uint32_t>{pack.buffers.nodes.bc_mask},
        .node_count = pack.metadata.node_count,
        .element_count = pack.metadata.element_count,
        .dof_count = pack.metadata.dof_count,
        .stiffness_scale = 1.0 + coeffs.a1 * rayleigh.beta,
        .mass_factor = coeffs.a0 + coeffs.a1 * rayleigh.alpha,
        .reduction_block = pack.metadata.reduction_block,
        .reduction_partials = pack.metadata.reduction_partials,
    };

    cwf::gpu::pcg::MatrixFreeWorkspace workspace{};
    std::vector<float> output(dof_count, 0.0F);
    const auto status = cwf::gpu::pcg::apply_keff(system, input, output, workspace);
    ASSERT_TRUE(status.has_value());

    const auto dense_output = dense_apply(keff, input_double);
    for (std::size_t dof = 0; dof < dof_count; ++dof)
    {
        const double ref = dense_output[dof];
        const double got = static_cast<double>(output[dof]);
        const double abs_diff = std::abs(ref - got);
        const double tol = std::max(1.0e-4, kRelativeTol * std::abs(ref));
        EXPECT_LE(abs_diff, tol) << "DOF mismatch at index " << dof << " (ref=" << ref << ", got=" << got << ", abs_diff=" << abs_diff << ", tol=" << tol << ")";
    }
}

/**
 * @test PCG solver matches CPU Newmark step within tolerance and converges under spec limits
 */
TEST(PcgPhase8, PcgMatchesCpuNewmark)
{
    auto mesh = make_single_tet_mesh();
    auto cfg  = make_basic_config();

    const auto preprocess_result = cwf::mesh::pre::run(mesh, cfg);
    ASSERT_TRUE(preprocess_result.has_value());
    const auto &pre = preprocess_result.value();

    const auto pack_result = cwf::mesh::pack::build_packed_buffers(mesh, pre, cfg, {});
    ASSERT_TRUE(pack_result.has_value());
    auto pack = pack_result.value();

    const auto materials = make_materials(cfg);

    const auto assembly  = cwf::physics::solver::assemble_linear_system(mesh, pre, materials);
    const auto dirichlet = cwf::physics::solver::build_dirichlet_conditions(mesh, cfg);

    const auto coeffs   = cwf::physics::newmark::make_coefficients(kDt, 0.25, 0.5);
    const auto rayleigh = cwf::physics::materials::compute_rayleigh(cfg.damping);

    auto keff = cwf::physics::newmark::build_effective_stiffness(assembly.stiffness, assembly.mass_diag, rayleigh, coeffs);
    apply_dirichlet_dense(keff, dirichlet);

    cwf::physics::newmark::State previous{};
    previous.displacement.assign(pack.metadata.dof_count, 0.0);
    previous.velocity.assign(pack.metadata.dof_count, 0.0);
    previous.acceleration.assign(pack.metadata.dof_count, 0.0);

    const auto load_vector = cwf::physics::loads::assemble_load_vector(mesh, cfg, pre, /*time=*/0.0);
    auto rhs_dense = cwf::physics::newmark::build_effective_rhs(
        load_vector,
        assembly.stiffness,
        assembly.mass_diag,
        rayleigh,
        coeffs,
        previous);
    apply_dirichlet_rhs(rhs_dense, dirichlet, previous);

    const auto reference_step = cwf::physics::solver::solve_newmark_step(assembly, rayleigh, dirichlet, mesh, cfg, pre,
                                                                         coeffs, previous, /*time=*/0.0, kRelativeTol,
                                                                         kMaxIterations);

    cwf::gpu::pcg::MatrixFreeSystem system{
        .element_connectivity = std::span<const std::uint32_t>{pack.buffers.elements.connectivity},
        .element_gradients = std::span<const float>{pack.buffers.elements.gradients},
        .element_volume = std::span<const float>{pack.buffers.elements.volume},
        .element_material_index = std::span<const std::uint32_t>{pack.buffers.elements.material_index},
        .materials = std::span<const cwf::physics::materials::ElasticProperties>{materials},
        .lumped_mass = std::span<const float>{pack.buffers.nodes.lumped_mass},
        .bc_mask = std::span<const std::uint32_t>{pack.buffers.nodes.bc_mask},
        .node_count = pack.metadata.node_count,
        .element_count = pack.metadata.element_count,
        .dof_count = pack.metadata.dof_count,
        .stiffness_scale = 1.0 + coeffs.a1 * rayleigh.beta,
        .mass_factor = coeffs.a0 + coeffs.a1 * rayleigh.alpha,
        .reduction_block = pack.metadata.reduction_block,
        .reduction_partials = pack.metadata.reduction_partials,
    };

    std::vector<float> rhs_float(rhs_dense.begin(), rhs_dense.end());

    auto &solver_buffers = pack.buffers.solver;
    auto &partials       = solver_buffers.partials;
    std::fill(solver_buffers.p.begin(), solver_buffers.p.end(), 0.0F);
    std::fill(solver_buffers.r.begin(), solver_buffers.r.end(), 0.0F);
    std::fill(solver_buffers.Ap.begin(), solver_buffers.Ap.end(), 0.0F);
    std::fill(solver_buffers.z.begin(), solver_buffers.z.end(), 0.0F);
    std::fill(solver_buffers.x.begin(), solver_buffers.x.end(), 0.0F);
    std::fill(partials.begin(), partials.end(), 0.0);

    cwf::gpu::pcg::PcgVectors vectors{
        .solution = std::span<float>(solver_buffers.x.data(), solver_buffers.x.size()),
        .residual = std::span<float>(solver_buffers.r.data(), solver_buffers.r.size()),
        .search_direction = std::span<float>(solver_buffers.p.data(), solver_buffers.p.size()),
        .preconditioned = std::span<float>(solver_buffers.z.data(), solver_buffers.z.size()),
        .matvec = std::span<float>(solver_buffers.Ap.data(), solver_buffers.Ap.size()),
        .partials = std::span<double>(partials.data(), partials.size()),
    };

    cwf::gpu::pcg::MatrixFreeWorkspace workspace{};
    const cwf::gpu::pcg::PcgSettings settings{
        .max_iterations = kMaxIterations,
        .relative_tolerance = kRelativeTol,
        .warm_start = false,
    };

    const auto result = cwf::gpu::pcg::solve_pcg(system, rhs_float, settings, vectors, workspace);
    ASSERT_TRUE(result.has_value());
    EXPECT_TRUE(result->converged);
    EXPECT_THAT(result->iterations, Le(kMaxIterations));

    const auto &pcg_solution = solver_buffers.x;
    for (std::size_t dof = 0; dof < pcg_solution.size(); ++dof)
    {
        EXPECT_NEAR(reference_step.state.displacement[dof], static_cast<double>(pcg_solution[dof]), 2.5e-4)
            << "DOF mismatch at index " << dof;
    }
}
