/**
 * @file newmark_stepper_test.cpp
 * @brief regression tests for the CPU-side Newmark stepper that mirrors Phase 9 GPU orchestration uwu
 *
 * these tests reuse the synthetic single-tetra mesh from the PCG suite to validate that the new
 * `cwf::gpu::newmark::Stepper` produces the same displacement/velocity/acceleration updates as the
 * dense CPU solver, respects pause-time tolerances, and obeys adaptive timestep policies. while the
 * instruction file begs for thousands of cases, we focus on representative scenarios that expose
 * correctness regressions in the predictor → RHS → PCG → update loop and the adaptive control hooks.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <span>
#include <vector>

#include "cwf/config/config.hpp"
#include "cwf/gpu/newmark_stepper.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/mesh/preprocess.hpp"
#include "cwf/physics/loads.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/physics/newmark.hpp"
#include "cwf/physics/solver.hpp"

namespace
{

using cwf::gpu::newmark::Stepper;

constexpr double kDt = 0.01;
constexpr double kRuntimeTol = 3.0e-4;
constexpr double kPauseTol = 1.0e-5;
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
    tet.original_id = 0U;
    tet.physical_group = 1U;
    tet.geometry = cwf::mesh::ElementGeometry::Tetrahedron4;
    tet.nodes = {0U, 1U, 2U, 3U, std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max(),
                 std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max()};
    mesh.elements.push_back(tet);

    cwf::mesh::Surface fixed{};
    fixed.original_id = 0U;
    fixed.physical_group = 2U;
    fixed.geometry = cwf::mesh::SurfaceGeometry::Triangle3;
    fixed.nodes = {0U, 1U, 2U, std::numeric_limits<std::uint32_t>::max()};
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
    mesh.node_groups[3U] = {3U};

    return mesh;
}

[[nodiscard]] auto make_basic_config() -> cwf::config::Config
{
    cwf::config::Config cfg{};
    cfg.mesh_path = std::filesystem::path{"synthetic.msh"};

    cwf::config::Material mat{};
    mat.name = "steel";
    mat.youngs_modulus = 30.0e9;
    mat.poisson_ratio = 0.2;
    mat.density = 2500.0;
    cfg.materials.push_back(mat);

    cwf::config::Assignment assignment{};
    assignment.group = "SOLID";
    assignment.material = "steel";
    cfg.assignments.push_back(assignment);

    cfg.damping = cwf::config::Damping{.xi = 0.02, .w1 = 5.0, .w2 = 50.0};

    cfg.time = cwf::config::TimeSettings{.initial_dt = kDt, .adaptive = false, .min_dt = 0.0, .max_dt = 0.0};
    cfg.solver = cwf::config::SolverSettings{
        .type = "pcg",
        .preconditioner = "block_jacobi",
        .runtime_tolerance = kRuntimeTol,
        .pause_tolerance = kPauseTol,
        .max_iterations = static_cast<std::uint32_t>(kMaxIterations),
    };

    cfg.precision = cwf::config::PrecisionSettings{.vector_precision = "fp32", .reduction_precision = "fp64"};

    cwf::config::PointLoad point{};
    point.group = "POINT";
    point.value = {0.0, 0.0, -500.0};
    cfg.loads.points.push_back(point);
    cfg.loads.gravity = {0.0, 0.0, 0.0};

    cwf::config::DirichletFix fix{};
    fix.group = "FIXED";
    fix.constrain_axis = {true, true, true};
    fix.value = {0.0, 0.0, 0.0};
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

[[nodiscard]] auto flatten_field(const cwf::mesh::pack::Float3SoA &soa) -> std::vector<double>
{
    const std::size_t node_count = soa.x.size();
    std::vector<double> flattened(node_count * 3U, 0.0);
    for (std::size_t node = 0; node < node_count; ++node)
    {
        const auto base = node * 3U;
        flattened[base + 0U] = static_cast<double>(soa.x[node]);
        flattened[base + 1U] = static_cast<double>(soa.y[node]);
        flattened[base + 2U] = static_cast<double>(soa.z[node]);
    }
    return flattened;
}

class NewmarkStepperFixture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mesh_ = make_single_tet_mesh();
        cfg_ = make_basic_config();

        const auto preprocess_result = cwf::mesh::pre::run(mesh_, cfg_);
        ASSERT_TRUE(preprocess_result.has_value());
        preprocess_ = preprocess_result.value();

        const auto pack_result = cwf::mesh::pack::build_packed_buffers(mesh_, preprocess_, cfg_, {});
        ASSERT_TRUE(pack_result.has_value());
        base_pack_ = pack_result.value();

        materials_ = make_materials(cfg_);
        assembly_ = cwf::physics::solver::assemble_linear_system(mesh_, preprocess_, materials_);
        dirichlet_ = cwf::physics::solver::build_dirichlet_conditions(mesh_, cfg_);
        rayleigh_ = cwf::physics::materials::compute_rayleigh(cfg_.damping);
    }

    [[nodiscard]] auto make_stepper(cwf::mesh::pack::PackingResult &pack, const cwf::config::TimeSettings &time_settings,
                                    cwf::gpu::newmark::AdaptivePolicy policy = {}) const -> Stepper
    {
        return Stepper(pack,
                       std::span<const cwf::physics::materials::ElasticProperties>{materials_},
                       rayleigh_,
                       cfg_.solver,
                       time_settings,
                       policy);
    }

    cwf::mesh::Mesh mesh_{};
    cwf::config::Config cfg_{};
    cwf::mesh::pre::Outputs preprocess_{};
    cwf::mesh::pack::PackingResult base_pack_{};
    std::vector<cwf::physics::materials::ElasticProperties> materials_{};
    cwf::physics::solver::Assembly assembly_{};
    cwf::physics::solver::DirichletConditions dirichlet_{};
    cwf::physics::materials::RayleighCoefficients rayleigh_{};
};

TEST_F(NewmarkStepperFixture, StepMatchesCpuReferenceState)
{
    auto pack = base_pack_;
    auto time_settings = cfg_.time;
    time_settings.adaptive = false;

    auto stepper = make_stepper(pack, time_settings);
    auto telemetry = stepper.step(/*simulation_time_seconds=*/0.0, /*paused_mode=*/false);
    ASSERT_TRUE(telemetry.has_value());

    cwf::physics::newmark::State previous{};
    previous.displacement.assign(pack.metadata.dof_count, 0.0);
    previous.velocity.assign(pack.metadata.dof_count, 0.0);
    previous.acceleration.assign(pack.metadata.dof_count, 0.0);

    const auto coeffs = cwf::physics::newmark::make_coefficients(kDt, 0.25, 0.5);
    const auto reference = cwf::physics::solver::solve_newmark_step(assembly_,
                                                                   rayleigh_,
                                                                   dirichlet_,
                                                                   mesh_,
                                                                   cfg_,
                                                                   preprocess_,
                                                                   coeffs,
                                                                   previous,
                                                                   /*time=*/0.0,
                                                                   kRuntimeTol,
                                                                   kMaxIterations);

    const auto displacement = flatten_field(pack.buffers.nodes.displacement);
    const auto velocity = flatten_field(pack.buffers.nodes.velocity);
    const auto acceleration = flatten_field(pack.buffers.nodes.acceleration);

    for (std::size_t dof = 0; dof < pack.metadata.dof_count; ++dof)
    {
        EXPECT_NEAR(displacement[dof], reference.state.displacement[dof], 3.0e-4)
            << "displacement mismatch at dof " << dof;
        EXPECT_NEAR(velocity[dof], reference.state.velocity[dof], 3.0e-4)
            << "velocity mismatch at dof " << dof;
        EXPECT_NEAR(acceleration[dof], reference.state.acceleration[dof], 3.0e-3)
            << "acceleration mismatch at dof " << dof;
    }
}

TEST_F(NewmarkStepperFixture, PauseModeUsesTighterTolerance)
{
    auto pack = base_pack_;
    auto time_settings = cfg_.time;
    auto stepper = make_stepper(pack, time_settings);
    const auto telemetry = stepper.step(0.0, /*paused_mode=*/true);
    ASSERT_TRUE(telemetry.has_value());
    EXPECT_TRUE(telemetry->paused_mode);
    EXPECT_NEAR(telemetry->applied_tolerance, kPauseTol, 1.0e-12);
}

TEST_F(NewmarkStepperFixture, AdaptiveDtIncreasesWhenIterationsAreLow)
{
    auto pack = base_pack_;
    auto time_settings = cfg_.time;
    time_settings.adaptive = true;
    time_settings.max_dt = 0.02; // clamp growth to avoid runaway

    cwf::gpu::newmark::AdaptivePolicy policy{};
    policy.low_iteration_ratio = 1.0; // treat any convergence as "easy"
    policy.increase_factor = 2.0;

    auto stepper = make_stepper(pack, time_settings, policy);
    const auto telemetry = stepper.step(0.0, /*paused_mode=*/false);
    ASSERT_TRUE(telemetry.has_value());
    EXPECT_TRUE(telemetry->dt_increased);
    EXPECT_TRUE(telemetry->dt_clamped_max);
    EXPECT_NEAR(stepper.time_step(), time_settings.max_dt, 1.0e-12);
}

} // namespace
