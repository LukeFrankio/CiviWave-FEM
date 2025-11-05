/**
 * @file preprocess_test.cpp
 * @brief preprocessing pipeline regression so gradients + masses stay sane uwu
 */
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>
#include <stdexcept>
#include <cstdlib>

#include "test_config.hpp"
#include "cwf/config/config.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/preprocess.hpp"
#include "support/config_builder.hpp"

using testing::AllOf;
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::HasSubstr;

namespace {

constexpr double kTol = 1.0e-9;

[[nodiscard]] auto load_fixture_mesh() -> cwf::mesh::Mesh
{
    std::filesystem::path data_dir;

    data_dir = std::filesystem::path{CWF_TEST_DATA_DIR};

    const auto mesh_result = cwf::mesh::load_gmsh_file(data_dir / "cantilever.msh");
    if (!mesh_result) {
        throw std::runtime_error("cantilever mesh fixture failed to load");
    }
    return mesh_result.value();
}

[[nodiscard]] auto load_config(const cwf::test_support::ConfigBuilderOptions& options = {}) -> cwf::config::Config
{
    const auto config_result = cwf::test_support::load_config(options);
    if (!config_result) {
        throw std::runtime_error("builder options produced invalid config unexpectedly");
    }
    return config_result.value();
}

[[nodiscard]] auto load_mesh_from_string(const std::string& gmsh) -> cwf::mesh::Mesh
{
    const auto mesh_result = cwf::mesh::load_gmsh_from_string(gmsh);
    if (!mesh_result) {
        throw std::runtime_error(mesh_result.error().message);
    }
    return mesh_result.value();
}

}  // namespace

TEST(PreprocessPipeline, ProducesExpectedOutputsForCantileverFixture)
{
    const auto mesh = load_fixture_mesh();
    const auto config = load_config();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_TRUE(preprocess.has_value()) << preprocess.error().message;
    const auto& outputs = preprocess.value();

    ASSERT_EQ(outputs.element_volumes.size(), 1U);
    EXPECT_NEAR(outputs.element_volumes.front(), 1.0 / 6.0, kTol);

    ASSERT_EQ(outputs.shape_gradients.size(), 1U);
    const auto& grads = outputs.shape_gradients.front();
    EXPECT_THAT(grads[0], ElementsAre(-1.0, -1.0, -1.0));
    EXPECT_THAT(grads[1], ElementsAre(1.0, 0.0, 0.0));
    EXPECT_THAT(grads[2], ElementsAre(0.0, 1.0, 0.0));
    EXPECT_THAT(grads[3], ElementsAre(0.0, 0.0, 1.0));

    const double expected_mass = (2500.0 * (1.0 / 6.0)) / 4.0;
    ASSERT_EQ(outputs.lumped_mass.size(), 4U);
    for (double mass : outputs.lumped_mass) {
        EXPECT_NEAR(mass, expected_mass, kTol);
    }

    EXPECT_THAT(outputs.element_material_index, ElementsAre(0U));

    ASSERT_EQ(outputs.adjacency.offsets.size(), 5U);
    EXPECT_THAT(outputs.adjacency.offsets, ElementsAre(0U, 1U, 2U, 3U, 4U));
    EXPECT_THAT(outputs.adjacency.element_indices, ElementsAre(0U, 0U, 0U, 0U));
    EXPECT_THAT(outputs.adjacency.local_indices, ElementsAre(0U, 1U, 2U, 3U));
}

TEST(PreprocessPipeline, RejectsHexahedraUntilPhaseFour)
{
    const std::string gmsh =
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$PhysicalNames\n1\n3 3 \"SOLID\"\n$EndPhysicalNames\n"
        "$Nodes\n1 8 1 8\n3 3 0 8\n1\n2\n3\n4\n5\n6\n7\n8\n"
        "0 0 0\n1 0 0\n0 1 0\n1 1 0\n0 0 1\n1 0 1\n0 1 1\n1 1 1\n$EndNodes\n"
        "$Elements\n1 1 1 1\n3 3 5 1\n1 1 2 4 3 5 6 8 7\n$EndElements\n";

    const auto mesh = load_mesh_from_string(gmsh);
    // Use minimal config without dirichlet/traction groups to avoid validation errors
    cwf::test_support::ConfigBuilderOptions opts;
    opts.dirichlet_fixes.clear();
    opts.tractions.clear();
    const auto config_result = cwf::test_support::load_config(opts);
    ASSERT_TRUE(config_result.has_value());
    const auto config = config_result.value();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    EXPECT_THAT(preprocess.error().message, HasSubstr("only tetrahedron elements supported"));
}

TEST(PreprocessPipeline, ErrorsWhenPhysicalGroupMissingAssignment)
{
    // Create a mesh with a physical group that doesn't match any assignment in config
    // The config has assignment for "SOLID" but mesh has "UNASSIGNED_GROUP"
    const std::string gmsh =
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$PhysicalNames\n1\n3 99 \"UNASSIGNED_GROUP\"\n$EndPhysicalNames\n"
        "$Nodes\n1 4 1 4\n3 99 0 4\n1\n2\n3\n4\n"
        "0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n"
        "$Elements\n1 1 1 1\n3 99 4 1\n1 1 2 3 4\n$EndElements\n";
    
    const auto mesh = load_mesh_from_string(gmsh);
    const auto config = load_config();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    // The validation detects that the config references a physical group not in the mesh
    EXPECT_THAT(preprocess.error().message, HasSubstr("missing physical group"));
}

TEST(PreprocessPipeline, RejectsDegenerateTetrahedron)
{
    const std::string gmsh =
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$PhysicalNames\n1\n3 3 \"SOLID\"\n$EndPhysicalNames\n"
        "$Nodes\n1 4 1 4\n3 3 0 4\n1\n2\n3\n4\n"
        "0 0 0\n1 0 0\n0 1 0\n0 1 0\n$EndNodes\n"
        "$Elements\n1 1 1 1\n3 3 4 1\n1 1 2 3 4\n$EndElements\n";

    const auto mesh = load_mesh_from_string(gmsh);
    // Use minimal config without dirichlet/traction groups to avoid validation errors
    cwf::test_support::ConfigBuilderOptions opts;
    opts.dirichlet_fixes.clear();
    opts.tractions.clear();
    const auto config_result = cwf::test_support::load_config(opts);
    ASSERT_TRUE(config_result.has_value());
    const auto config = config_result.value();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    // Note: duplicate node detection runs before volume check, so we get that error first
    EXPECT_THAT(preprocess.error().message, HasSubstr("duplicate nodes"));
}

TEST(PreprocessPipeline, DetectsDuplicateNodes)
{
    const std::string gmsh =
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$PhysicalNames\n1\n3 3 \"SOLID\"\n$EndPhysicalNames\n"
        "$Nodes\n1 5 1 5\n3 3 0 5\n1\n2\n3\n4\n5\n"
        "0 0 0\n1 0 0\n0 1 0\n0 0 1\n0 0 0\n$EndNodes\n"
        "$Elements\n1 1 1 1\n3 3 4 1\n1 1 2 3 4\n$EndElements\n";

    const auto mesh = load_mesh_from_string(gmsh);
    const auto config = load_config();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    EXPECT_THAT(preprocess.error().message, HasSubstr("duplicate nodes"));
}

TEST(PreprocessPipeline, DetectsDuplicateElements)
{
    const std::string gmsh =
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$PhysicalNames\n1\n3 3 \"SOLID\"\n$EndPhysicalNames\n"
        "$Nodes\n1 4 1 4\n3 3 0 4\n1\n2\n3\n4\n"
        "0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n"
        "$Elements\n1 2 1 2\n3 3 4 2\n1 1 2 3 4\n2 1 2 3 4\n$EndElements\n";

    const auto mesh = load_mesh_from_string(gmsh);
    const auto config = load_config();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    EXPECT_THAT(preprocess.error().message, HasSubstr("duplicate elements"));
}

TEST(PreprocessPipeline, ValidatesDirichletGroupsExist)
{
    const std::string gmsh =
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$PhysicalNames\n1\n3 3 \"SOLID\"\n$EndPhysicalNames\n"
        "$Nodes\n1 4 1 4\n3 3 0 4\n1\n2\n3\n4\n"
        "0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n"
        "$Elements\n1 1 1 1\n3 3 4 1\n1 1 2 3 4\n$EndElements\n";

    const auto mesh = load_mesh_from_string(gmsh);
    
    // Create config with dirichlet fix referencing non-existent group
    cwf::test_support::ConfigBuilderOptions options;
    options.dirichlet_fixes = {{
        "NONEXISTENT_GROUP", 
        {true, true, true},
        {std::nullopt, std::nullopt, std::nullopt}
    }};
    const auto config_result = cwf::test_support::load_config(options);
    ASSERT_TRUE(config_result.has_value());
    const auto config = config_result.value();
    
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    EXPECT_THAT(preprocess.error().message, HasSubstr("dirichlet fix references missing physical group"));
}

TEST(PreprocessPipeline, ValidatesTractionGroupsExist)
{
    const std::string gmsh =
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$PhysicalNames\n1\n3 3 \"SOLID\"\n$EndPhysicalNames\n"
        "$Nodes\n1 4 1 4\n3 3 0 4\n1\n2\n3\n4\n"
        "0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n"
        "$Elements\n1 1 1 1\n3 3 4 1\n1 1 2 3 4\n$EndElements\n";

    const auto mesh = load_mesh_from_string(gmsh);
    
    // Create config with traction referencing non-existent group
    // ALSO clear default dirichlet groups to avoid validation errors before traction check
    cwf::test_support::ConfigBuilderOptions options;
    options.dirichlet_fixes.clear();  // Remove default FIXED_BASE group
    options.tractions = {{
        "NONEXISTENT_GROUP", 
        {1.0, 0.0, 0.0},
        ""
    }};
    const auto config_result = cwf::test_support::load_config(options);
    ASSERT_TRUE(config_result.has_value());
    const auto config = config_result.value();
    
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    EXPECT_THAT(preprocess.error().message, HasSubstr("traction load references missing physical group"));
}
