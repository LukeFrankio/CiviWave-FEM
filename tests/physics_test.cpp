/**
 * @file physics_test.cpp
 * @brief regression + invariants suite for physics helpers (loads/newmark/solver) uwu
 *
 * this test TU validates the freshly-added CPU reference physics stack. we cover curve
 * evaluation edge cases, load vector composition, newmark algebra, linear-system assembly,
 * dirichlet mask generation, and the coupled solver step. each test documents the
 * invariants it enforces so future refactors know exactly what gotchas will explode.
 *
 * tl;dr: exhaustive-ish coverage for the pure helpers plus smoke-tests for the coupled
 * solve path so hidden regressions get caught fast.
 */

#include <array>
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

#include "cwf/common/math.hpp"
#include "cwf/config/config.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/preprocess.hpp"
#include "cwf/physics/loads.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/physics/newmark.hpp"
#include "cwf/physics/solver.hpp"

using ::testing::Each;
using ::testing::Ge;

namespace
{

constexpr double kEpsilon = 1.0e-9;

[[nodiscard]] auto make_curve(std::initializer_list<std::pair<double, double>> points) -> cwf::config::Curve
{
    cwf::config::Curve curve{};
    curve.points.assign(points.begin(), points.end());
    return curve;
}

[[nodiscard]] auto synthetic_mesh_for_loads() -> cwf::mesh::Mesh
{
    cwf::mesh::Mesh mesh{};
    mesh.nodes = {
        cwf::mesh::Node{1U, cwf::common::Vec3{0.0, 0.0, 0.0}},
        cwf::mesh::Node{2U, cwf::common::Vec3{1.0, 0.0, 0.0}},
        cwf::mesh::Node{3U, cwf::common::Vec3{0.0, 1.0, 0.0}},
        cwf::mesh::Node{4U, cwf::common::Vec3{0.0, 0.0, 1.0}},
    };

    mesh.physical_groups = {
        cwf::mesh::PhysicalGroup{2U, 10U, "FIXED"},
        cwf::mesh::PhysicalGroup{2U, 11U, "LOAD_FACE"},
        cwf::mesh::PhysicalGroup{3U, 12U, "SOLID"},
        cwf::mesh::PhysicalGroup{0U, 13U, "POINT_LOAD"},
    };
    for (std::size_t i = 0; i < mesh.physical_groups.size(); ++i)
    {
        mesh.group_lookup.emplace(mesh.physical_groups[i].id, i);
    }

    cwf::mesh::Surface fixed{};
    fixed.original_id    = 100U;
    fixed.geometry       = cwf::mesh::SurfaceGeometry::Triangle3;
    fixed.physical_group = 10U;
    fixed.nodes          = {0U, 1U, 2U, std::numeric_limits<std::uint32_t>::max()};

    cwf::mesh::Surface load{};
    load.original_id    = 101U;
    load.geometry       = cwf::mesh::SurfaceGeometry::Triangle3;
    load.physical_group = 11U;
    load.nodes          = {1U, 2U, 3U, std::numeric_limits<std::uint32_t>::max()};

    mesh.surfaces = {fixed, load};
    mesh.surface_groups.emplace(10U, std::vector<std::size_t>{0U});
    mesh.surface_groups.emplace(11U, std::vector<std::size_t>{1U});

    mesh.node_groups.emplace(13U, std::vector<std::uint32_t>{3U});

    cwf::mesh::Element tet{};
    tet.original_id    = 200U;
    tet.geometry       = cwf::mesh::ElementGeometry::Tetrahedron4;
    tet.physical_group = 12U;
    tet.nodes          = {0U,
                          1U,
                          2U,
                          3U,
                          std::numeric_limits<std::uint32_t>::max(),
                          std::numeric_limits<std::uint32_t>::max(),
                          std::numeric_limits<std::uint32_t>::max(),
                          std::numeric_limits<std::uint32_t>::max()};
    mesh.elements      = {tet};

    return mesh;
}

[[nodiscard]] auto triangle_area(const cwf::mesh::Mesh &mesh, std::uint32_t a, std::uint32_t b,
                                 std::uint32_t c) -> double
{
    const auto &pa    = mesh.nodes.at(a).position;
    const auto &pb    = mesh.nodes.at(b).position;
    const auto &pc    = mesh.nodes.at(c).position;
    const auto  v1    = cwf::common::Vec3{pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]};
    const auto  v2    = cwf::common::Vec3{pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]};
    const auto  cross = cwf::common::cross(v1, v2);
    return 0.5 * std::sqrt(cwf::common::dot(cross, cross));
}

/**
 * @brief fixture wiring a minimal tetrahedral mesh + config for solver smoke tests
 */
class SolverFixture : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        mesh_ = synthetic_mesh_for_loads();

        cfg_.mesh_path = "synthetic.msh";
        cfg_.materials.push_back(cwf::config::Material{"test_material", 7.0e10, 0.25, 1000.0});
        cfg_.assignments.push_back(cwf::config::Assignment{"SOLID", "test_material"});
        cfg_.damping   = cwf::config::Damping{0.02, 5.0, 50.0};
        cfg_.time      = cwf::config::TimeSettings{0.01, false, 0.005, 0.02};
        cfg_.solver    = cwf::config::SolverSettings{"pcg", "diag", 1.0e-8, 1.0e-9, 128U};
        cfg_.precision = cwf::config::PrecisionSettings{"fp32", "fp64"};

        cfg_.loads.gravity = {0.0, 0.0, 0.0};
        cfg_.loads.tractions.clear();
        cfg_.loads.points.clear();

        cfg_.curves.clear();
        cfg_.dirichlet.push_back(cwf::config::DirichletFix{"FIXED", {true, true, true}, {0.0, 0.0, 0.0}});
        cfg_.output = cwf::config::OutputSettings{10U, {}};

        auto preprocess_result = cwf::mesh::pre::run(mesh_, cfg_);
        ASSERT_TRUE(preprocess_result.has_value()) << preprocess_result.error().message;
        preprocess_ = std::move(preprocess_result.value());

        materials_.clear();
        materials_.reserve(cfg_.materials.size());
        for (const auto &mat : cfg_.materials)
        {
            materials_.push_back(cwf::physics::materials::make_properties(mat));
        }
        rayleigh_ = cwf::physics::materials::compute_rayleigh(cfg_.damping);
        coeffs_   = cwf::physics::newmark::make_coefficients(cfg_.time.initial_dt);

        const std::size_t dofs = mesh_.nodes.size() * 3U;
        state_.displacement.assign(dofs, 0.0);
        state_.velocity.assign(dofs, 0.0);
        state_.acceleration.assign(dofs, 0.0);
    }

    cwf::mesh::Mesh                                         mesh_{};
    cwf::config::Config                                     cfg_{};
    cwf::mesh::pre::Outputs                                 preprocess_{};
    std::vector<cwf::physics::materials::ElasticProperties> materials_{};
    cwf::physics::materials::RayleighCoefficients           rayleigh_{};
    cwf::physics::newmark::Coefficients                     coeffs_{};
    cwf::physics::newmark::State                            state_{};
};

// -----------------------------------------------------------------------------
// curve evaluation tests
// -----------------------------------------------------------------------------

TEST(CurveEvaluation, InterpolatesLinearlyBetweenPoints)
{
    const auto   curve = make_curve({{0.0, 0.0}, {1.0, 2.0}});
    const double mid   = cwf::physics::loads::evaluate_curve(curve, 0.5);
    EXPECT_NEAR(mid, 1.0, kEpsilon);
}

TEST(CurveEvaluation, ClampsBeforeFirstAndAfterLast)
{
    const auto curve = make_curve({{1.0, -2.0}, {3.0, 4.0}});
    EXPECT_NEAR(cwf::physics::loads::evaluate_curve(curve, -10.0), -2.0, kEpsilon);
    EXPECT_NEAR(cwf::physics::loads::evaluate_curve(curve, 10.0), 4.0, kEpsilon);
}

TEST(CurveEvaluation, HandlesDegenerateSegments)
{
    const auto curve = make_curve({{0.0, 1.0}, {0.0, 3.0}, {2.0, 5.0}});
    EXPECT_NEAR(cwf::physics::loads::evaluate_curve(curve, 0.0), 1.0, kEpsilon);
    EXPECT_NEAR(cwf::physics::loads::evaluate_curve(curve, 1.0), 4.0, kEpsilon);
}

// -----------------------------------------------------------------------------
// load assembly tests
// -----------------------------------------------------------------------------

TEST(LoadAssembly, CombinesGravitySurfaceTractionAndPointLoads)
{
    auto mesh = synthetic_mesh_for_loads();

    cwf::config::Config cfg{};
    cfg.loads.gravity = {0.0, 0.0, -9.81};
    cfg.loads.tractions.push_back(cwf::config::SurfaceTraction{"LOAD_FACE", {0.0, 0.0, -5000.0}, ""});
    cfg.loads.points.push_back(cwf::config::PointLoad{"POINT_LOAD", {0.0, 0.0, -200.0}, ""});

    cwf::mesh::pre::Outputs preprocess{};
    preprocess.lumped_mass = {41.666666666666664, 41.666666666666664, 41.666666666666664, 41.666666666666664};

    const auto loads = cwf::physics::loads::assemble_load_vector(mesh, cfg, preprocess, 0.0);
    ASSERT_EQ(loads.size(), mesh.nodes.size() * 3U);

    for (std::size_t node = 0; node < mesh.nodes.size(); ++node)
    {
        EXPECT_NEAR(loads[node * 3U + 0U], 0.0, kEpsilon);
        EXPECT_NEAR(loads[node * 3U + 1U], 0.0, kEpsilon);
    }

    const double gravity        = preprocess.lumped_mass[0] * cfg.loads.gravity[2];
    const double traction_area  = triangle_area(mesh, 1U, 2U, 3U);
    const double traction_share = (traction_area / 3.0) * cfg.loads.tractions[0].value[2];

    EXPECT_NEAR(loads[0U * 3U + 2U], gravity, 1.0e-6);
    EXPECT_NEAR(loads[1U * 3U + 2U], gravity + traction_share, 1.0e-6);
    EXPECT_NEAR(loads[2U * 3U + 2U], gravity + traction_share, 1.0e-6);
    EXPECT_NEAR(loads[3U * 3U + 2U], gravity + traction_share + cfg.loads.points[0].value[2], 1.0e-6);
}

// -----------------------------------------------------------------------------
// newmark algebra tests
// -----------------------------------------------------------------------------

TEST(NewmarkCoefficients, MatchesClassicAverageAccelerationValues)
{
    const auto coeffs = cwf::physics::newmark::make_coefficients(0.02, 0.25, 0.5);
    EXPECT_NEAR(coeffs.a0, 10000.0, kEpsilon);
    EXPECT_NEAR(coeffs.a1, 100.0, kEpsilon);
    EXPECT_NEAR(coeffs.a2, 200.0, kEpsilon);
    EXPECT_NEAR(coeffs.a3, 1.0, kEpsilon);
    EXPECT_NEAR(coeffs.a4, 1.0, kEpsilon);
    EXPECT_NEAR(coeffs.a5, 0.0, kEpsilon);
}

TEST(NewmarkEffectiveStiffness, ScalesStiffnessAndAddsMass)
{
    const std::vector<double> stiffness{10.0, 2.0, 2.0, 6.0};
    const std::vector<double> mass_diag{4.0, 8.0};
    const auto                coeffs   = cwf::physics::newmark::make_coefficients(0.1, 0.25, 0.5);
    const auto                rayleigh = cwf::physics::materials::RayleighCoefficients{0.01, 0.02};

    const auto keff =
        cwf::physics::newmark::build_effective_stiffness(stiffness, mass_diag, rayleigh, coeffs);
    ASSERT_EQ(keff.size(), stiffness.size());

    const double scale = 1.0 + coeffs.a1 * rayleigh.beta;
    EXPECT_NEAR(keff[0], stiffness[0] * scale + mass_diag[0] * (coeffs.a0 + coeffs.a1 * rayleigh.alpha),
                kEpsilon);
    EXPECT_NEAR(keff[3], stiffness[3] * scale + mass_diag[1] * (coeffs.a0 + coeffs.a1 * rayleigh.alpha),
                kEpsilon);
    EXPECT_NEAR(keff[1], stiffness[1] * scale, kEpsilon);
    EXPECT_NEAR(keff[2], stiffness[2] * scale, kEpsilon);
}

TEST(NewmarkEffectiveRhs, BuildsConsistentForceVector)
{
    const std::vector<double> load{5.0, -3.0};
    const std::vector<double> stiffness{4.0, 1.0, 1.0, 2.0};
    const std::vector<double> mass_diag{2.0, 3.0};
    const auto                coeffs   = cwf::physics::newmark::make_coefficients(0.05, 0.25, 0.5);
    const auto                rayleigh = cwf::physics::materials::RayleighCoefficients{0.0, 0.1};

    cwf::physics::newmark::State state{};
    state.displacement = {0.1, -0.2};
    state.velocity     = {0.0, 0.3};
    state.acceleration = {0.5, -0.1};

    const auto rhs =
        cwf::physics::newmark::build_effective_rhs(load, stiffness, mass_diag, rayleigh, coeffs, state);
    ASSERT_EQ(rhs.size(), load.size());

    std::vector<double> expected = load;
    std::vector<double> damping_rhs(rhs.size(), 0.0);

    for (std::size_t i = 0; i < rhs.size(); ++i)
    {
        const double mass_term =
            mass_diag[i] * (coeffs.a0 * state.displacement[i] + coeffs.a2 * state.velocity[i] +
                            coeffs.a3 * state.acceleration[i]);
        const double damping_term = coeffs.a1 * state.displacement[i] + coeffs.a4 * state.velocity[i] +
                                    coeffs.a5 * state.acceleration[i];
        expected[i] += mass_term;
        expected[i] += rayleigh.alpha * mass_diag[i] * damping_term;
        damping_rhs[i] = damping_term;
    }

    if (rayleigh.beta != 0.0)
    {
        for (std::size_t row = 0; row < rhs.size(); ++row)
        {
            double accum = 0.0;
            for (std::size_t col = 0; col < rhs.size(); ++col)
            {
                accum += stiffness[row * rhs.size() + col] * damping_rhs[col];
            }
            expected[row] += rayleigh.beta * accum;
        }
    }

    for (std::size_t i = 0; i < rhs.size(); ++i)
    {
        EXPECT_NEAR(rhs[i], expected[i], 1.0e-6);
    }
}

TEST(NewmarkUpdate, ProducesConsistentKinematics)
{
    const auto                   coeffs = cwf::physics::newmark::make_coefficients(0.1, 0.25, 0.5);
    cwf::physics::newmark::State previous{};
    previous.displacement = {0.0, 0.0};
    previous.velocity     = {1.0, -1.0};
    previous.acceleration = {0.0, 0.5};

    const std::vector<double> delta{0.2, -0.1};
    const auto                next = cwf::physics::newmark::update_state(coeffs, previous, delta);

    EXPECT_NEAR(next.displacement[0], 0.2, kEpsilon);
    const auto expected_v0 =
        previous.velocity[0] +
        coeffs.dt * ((1.0 - coeffs.gamma) * previous.acceleration[0] + coeffs.gamma * next.acceleration[0]);
    EXPECT_NEAR(next.velocity[0], expected_v0, 1.0e-6);

    const auto expected_a1 =
        coeffs.a0 * delta[1] - coeffs.a2 * previous.velocity[1] - coeffs.a3 * previous.acceleration[1];
    EXPECT_NEAR(next.acceleration[1], expected_a1, 1.0e-6);

    const auto expected_v1 =
        previous.velocity[1] +
        coeffs.dt * ((1.0 - coeffs.gamma) * previous.acceleration[1] + coeffs.gamma * next.acceleration[1]);
    EXPECT_NEAR(next.velocity[1], expected_v1, 1.0e-6);
}

// -----------------------------------------------------------------------------
// solver assembly + step tests
// -----------------------------------------------------------------------------

TEST_F(SolverFixture, AssembleLinearSystemProducesSymmetricMatrix)
{
    const auto        assembly = cwf::physics::solver::assemble_linear_system(mesh_, preprocess_, materials_);
    const std::size_t n        = mesh_.nodes.size() * 3U;
    ASSERT_EQ(assembly.stiffness.size(), n * n);

    for (std::size_t row = 0; row < n; ++row)
    {
        for (std::size_t col = row; col < n; ++col)
        {
            const double a = assembly.stiffness[row * n + col];
            const double b = assembly.stiffness[col * n + row];
            EXPECT_NEAR(a, b, 1.0e-6);
        }
    }

    ASSERT_EQ(assembly.mass_diag.size(), n);
    EXPECT_THAT(assembly.mass_diag, Each(Ge(0.0)));
}

TEST_F(SolverFixture, BuildDirichletConditionsLocksSurfaceNodes)
{
    const auto        conditions = cwf::physics::solver::build_dirichlet_conditions(mesh_, cfg_);
    const std::size_t n          = mesh_.nodes.size() * 3U;
    ASSERT_EQ(conditions.mask.size(), n);
    ASSERT_EQ(conditions.targets.size(), n);

    const std::vector<std::uint32_t> fixed_nodes{0U, 1U, 2U};
    for (auto node : fixed_nodes)
    {
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            const std::size_t dof = node * 3U + axis;
            EXPECT_TRUE(conditions.mask[dof]);
            EXPECT_NEAR(conditions.targets[dof], 0.0, kEpsilon);
        }
    }

    for (std::size_t dof = 9U; dof < conditions.mask.size(); ++dof)
    {
        EXPECT_FALSE(conditions.mask[dof]);
    }
}

TEST_F(SolverFixture, SolveNewmarkStepMaintainsDirichletConstraints)
{
    const auto        assembly = cwf::physics::solver::assemble_linear_system(mesh_, preprocess_, materials_);
    const auto        dirichlet      = cwf::physics::solver::build_dirichlet_conditions(mesh_, cfg_);
    const double      time           = 0.0;
    const double      tolerance      = 1.0e-8;
    const std::size_t max_iterations = 256U;

    const auto result =
        cwf::physics::solver::solve_newmark_step(assembly, rayleigh_, dirichlet, mesh_, cfg_, preprocess_,
                                                 coeffs_, state_, time, tolerance, max_iterations);

    EXPECT_TRUE(result.stats.converged);
    EXPECT_LT(result.stats.residual_norm, 1.0);

    const std::vector<std::uint32_t> fixed_nodes{0U, 1U, 2U};
    for (auto node : fixed_nodes)
    {
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            const auto dof = node * 3U + axis;
            EXPECT_NEAR(result.state.displacement[dof], 0.0, kEpsilon);
        }
    }
}

} // namespace
