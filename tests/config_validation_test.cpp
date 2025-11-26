/**
 * @file config_validation_test.cpp
 * @brief exhaustive config loader validation because parsing bugs are cringe uwu
 */
#include <filesystem>
#include <functional>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "cwf/config/config.hpp"
#include "support/config_builder.hpp"
#include "test_config.hpp"

using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::HasSubstr;

namespace
{

[[nodiscard]] auto test_data_path(std::string_view file) -> std::filesystem::path
{
    return std::filesystem::path{CWF_TEST_DATA_DIR} / file;
}

[[nodiscard]] auto make_good_config() -> cwf::config::Config
{
    const auto result = cwf::test_support::load_config();
    if (!result)
    {
        throw std::runtime_error("expected default builder to succeed");
    }
    return result.value();
}

} // namespace

TEST(ConfigValidation, ParsesGoldenConfigFromBuilder)
{
    const auto config = make_good_config();
    EXPECT_EQ(config.mesh_path.generic_string(), "tests/data/cantilever.msh");
    ASSERT_EQ(config.materials.size(), 1U);
    EXPECT_EQ(config.materials.front().name, "concrete");
    EXPECT_DOUBLE_EQ(config.materials.front().youngs_modulus, 3.0e10);
    EXPECT_EQ(config.assignments.front().group, "SOLID");
    EXPECT_DOUBLE_EQ(config.damping.xi, 0.02);
    EXPECT_DOUBLE_EQ(config.time.initial_dt, 0.01111);
    EXPECT_TRUE(config.time.adaptive);
    EXPECT_EQ(config.solver.type, "pcg");
    EXPECT_EQ(config.precision.vector_precision, "fp32");
    ASSERT_EQ(config.curves.size(), 1U);
    EXPECT_EQ(config.curves.begin()->first, "load_curve1");
    EXPECT_THAT(config.loads.gravity, ElementsAre(0.0, 0.0, -9.81));
    EXPECT_TRUE(config.loads.points.empty());
    ASSERT_EQ(config.dirichlet.size(), 1U);
    EXPECT_TRUE(config.dirichlet.front().constrain_axis[0]);
    EXPECT_EQ(config.output.vtu_stride, 10U);
}

TEST(ConfigValidation, LoadsConfigFromFixtureOnDisk)
{
    const auto yaml_path = test_data_path("cantilever.yaml");
    const auto result    = cwf::config::load_config_from_file(yaml_path);
    ASSERT_TRUE(result.has_value()) << result.error().message;
    EXPECT_EQ(result->mesh_path.generic_string(), "tests/data/cantilever.msh");
}

TEST(ConfigValidation, ParsesPointLoadsWithOptionalCurve)
{
    cwf::test_support::ConfigBuilderOptions options;
    options.point_loads = {
        {.group = "FIXED_BASE", .value = {0.0, 0.0, -1234.5}, .scale_curve = "load_curve1"},
        {.group = "LOAD_FACE", .value = {10.0, 0.0, 0.0}, .scale_curve = {}}};
    const auto yaml     = cwf::test_support::make_config_yaml(options);
    const auto parsed   = cwf::config::load_config_from_string(yaml);
    ASSERT_TRUE(parsed.has_value()) << parsed.error().message;
    ASSERT_EQ(parsed->loads.points.size(), 2U);
    EXPECT_EQ(parsed->loads.points.front().group, "FIXED_BASE");
    EXPECT_DOUBLE_EQ(parsed->loads.points.front().value[2], -1234.5);
    EXPECT_EQ(parsed->loads.points.front().scale_curve, "load_curve1");
    EXPECT_TRUE(parsed->loads.points.back().scale_curve.empty());
}

struct InvalidConfigCase
{
    std::string                             name;
    cwf::test_support::ConfigBuilderOptions options;
    std::string                             expected_message_substring;
    std::vector<std::string>                expected_context;
    std::function<void(std::string &)>      mutate_yaml; ///< optional for bespoke tweaks
};

class ConfigInvalidTest : public ::testing::TestWithParam<InvalidConfigCase>
{};

TEST_P(ConfigInvalidTest, ReportsDetailedValidationErrors)
{
    auto yaml = cwf::test_support::make_config_yaml(GetParam().options);
    if (GetParam().mutate_yaml)
    {
        GetParam().mutate_yaml(yaml);
    }
    const auto result = cwf::config::load_config_from_string(yaml);
    ASSERT_FALSE(result.has_value()) << "expected failure for case: " << GetParam().name;
    EXPECT_THAT(result.error().message, HasSubstr(GetParam().expected_message_substring));
    if (!GetParam().expected_context.empty())
    {
        EXPECT_THAT(result.error().context, ElementsAreArray(GetParam().expected_context));
    }
}

namespace
{
auto make_invalid_cases() -> std::vector<InvalidConfigCase>
{
    using cwf::test_support::AssignmentSpec;
    using cwf::test_support::ConfigBuilderOptions;
    using cwf::test_support::CurveSpec;
    using cwf::test_support::DirichletSpec;
    using cwf::test_support::MaterialSpec;
    using cwf::test_support::TractionSpec;

    std::vector<InvalidConfigCase> cases;

    {
        ConfigBuilderOptions opts{};
        opts.include_mesh = false;
        cases.push_back({.name                       = "MissingMeshSection",
                         .options                    = opts,
                         .expected_message_substring = "missing 'mesh' section",
                         .expected_context           = {"mesh"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.materials = {MaterialSpec{
            .name = "concrete", .youngs_modulus = -1.0, .poisson_ratio = 0.2, .density = 2500.0}};
        cases.push_back({.name                       = "NegativeYoungsModulus",
                         .options                    = opts,
                         .expected_message_substring = "material.E must be > 0",
                         .expected_context           = {"materials", "[0]", "E"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.materials = {MaterialSpec{
            .name = "concrete", .youngs_modulus = 3.0e10, .poisson_ratio = 0.75, .density = 2500.0}};
        cases.push_back({.name                       = "PoissonRatioTooLarge",
                         .options                    = opts,
                         .expected_message_substring = "material.nu must be (-0.999, 0.5)",
                         .expected_context           = {"materials", "[0]", "nu"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.materials = {
            MaterialSpec{
                .name = "duplicate", .youngs_modulus = 3.0e10, .poisson_ratio = 0.2, .density = 2500.0},
            MaterialSpec{
                .name = "duplicate", .youngs_modulus = 1.0e11, .poisson_ratio = 0.3, .density = 7800.0}};
        cases.push_back({.name                       = "DuplicateMaterialNames",
                         .options                    = opts,
                         .expected_message_substring = "material names must be unique",
                         .expected_context           = {"materials", "[1]", "name"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.assignments = {AssignmentSpec{.group = "SOLID", .material = "missing"}};
        cases.push_back({.name                       = "AssignmentUnknownMaterial",
                         .options                    = opts,
                         .expected_message_substring = "assignment references unknown material",
                         .expected_context           = {"assignments", "[0]", "material"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.damping_xi = 1.2;
        cases.push_back({.name                       = "DampingXiOutOfRange",
                         .options                    = opts,
                         .expected_message_substring = "damping.xi must be (0,1)",
                         .expected_context           = {"damping", "xi"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.damping_w2 = 5.0;
        opts.damping_w1 = 10.0;
        cases.push_back({.name                       = "DampingW2TooSmall",
                         .options                    = opts,
                         .expected_message_substring = "damping.w2 must be > damping.w1",
                         .expected_context           = {"damping", "w2"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.time_dt = -0.01;
        cases.push_back({.name                       = "NegativeTimeStep",
                         .options                    = opts,
                         .expected_message_substring = "time.dt must be > 0",
                         .expected_context           = {"time", "dt"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.time_min_dt = -0.01;
        cases.push_back({.name                       = "NegativeMinDt",
                         .options                    = opts,
                         .expected_message_substring = "time.min_dt must be >= 0",
                         .expected_context           = {"time", "min_dt"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.time_max_dt = 0.001;
        cases.push_back({.name                       = "MaxDtBelowInitial",
                         .options                    = opts,
                         .expected_message_substring = "time.max_dt must be >= time.dt",
                         .expected_context           = {"time", "max_dt"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.solver_max_iters = 0U;
        cases.push_back({.name                       = "ZeroSolverIterations",
                         .options                    = opts,
                         .expected_message_substring = "solver.max_iters must be >= 1",
                         .expected_context           = {"solver", "max_iters"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.solver_runtime_tol = -1.0;
        cases.push_back({.name                       = "NegativeSolverTolerance",
                         .options                    = opts,
                         .expected_message_substring = "solver tolerances must be > 0",
                         .expected_context           = {"solver"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.include_precision = false;
        cases.push_back({.name                       = "MissingPrecisionSection",
                         .options                    = opts,
                         .expected_message_substring = "missing precision map",
                         .expected_context           = {"precision"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.curves = {CurveSpec{.name = "load_curve1", .points = {{0.0, 0.0}, {0.6, 1.0}, {0.5, 1.1}}}};
        cases.push_back({.name                       = "CurveTimesNotMonotonic",
                         .options                    = opts,
                         .expected_message_substring = "curve times must be non-decreasing",
                         .expected_context           = {"curves", "load_curve1", "[2]"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.include_curves = false;
        opts.tractions      = {
            TractionSpec{.group = "LOAD_FACE", .value = {0.0, 0.0, -1.0e5}, .scale_curve = "load_curve1"}};
        cases.push_back({.name                       = "TractionUnknownCurve",
                         .options                    = opts,
                         .expected_message_substring = "traction references unknown curve",
                         .expected_context           = {"loads", "tractions", "[0]", "scale_curve"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.dirichlet_fixes = {DirichletSpec{.group     = "FIXED_BASE",
                                              .constrain = {false, false, false},
                                              .values    = {std::nullopt, std::nullopt, std::nullopt}}};
        cases.push_back({.name                       = "EmptyDirichletDof",
                         .options                    = opts,
                         .expected_message_substring = "dirichlet.dof must not be empty",
                         .expected_context           = {"dirichlet", "fixes", "[0]", "dof"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.output_stride = 0U;
        cases.push_back({.name                       = "ZeroOutputStride",
                         .options                    = opts,
                         .expected_message_substring = "output.vtu_stride must be >= 1",
                         .expected_context           = {"output", "vtu_stride"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.include_output = false;
        cases.push_back({.name                       = "MissingOutputSection",
                         .options                    = opts,
                         .expected_message_substring = "missing output map",
                         .expected_context           = {"output"},
                         .mutate_yaml                = nullptr});
    }

    {
        ConfigBuilderOptions opts{};
        opts.dirichlet_fixes = {DirichletSpec{}};
        cases.push_back({.name                       = "DirichletInvalidAxis",
                         .options                    = opts,
                         .expected_message_substring = "dirichlet.dof must be subset of {x,y,z}",
                         .expected_context           = {"dirichlet", "fixes", "[0]", "dof"},
                         .mutate_yaml                = [](std::string &yaml) -> void {
                             const auto token = std::string{"dof: [x, y, z]"};
                             const auto pos   = yaml.find(token);
                             if (pos != std::string::npos)
                             {
                                 yaml.replace(pos, token.size(), "dof: [x, q]");
                             }
                         }});
    }

    return cases;
}
}  // namespace

INSTANTIATE_TEST_SUITE_P(ExhaustiveInvalidConfigs, ConfigInvalidTest,
                         ::testing::ValuesIn(make_invalid_cases()),
                         [](const ::testing::TestParamInfo<InvalidConfigCase> &test_info) -> std::string {
                             return test_info.param.name;
                         });
