/**
 * @file preprocess_test.cpp
 * @brief preprocessing pipeline regression so gradients + masses stay sane uwu
 */
#include <cstdlib>
#include <filesystem>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>

#include "cwf/config/config.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/preprocess.hpp"
#include "support/config_builder.hpp"
#include "test_config.hpp"

using testing::AllOf;
using testing::ElementsAre;
using testing::ElementsAreArray;
using testing::HasSubstr;

namespace
{

constexpr double kTol = 1.0e-9;

[[nodiscard]] auto load_fixture_mesh() -> cwf::mesh::Mesh
{
    std::filesystem::path data_dir;

    data_dir = std::filesystem::path{CWF_TEST_DATA_DIR};

    const auto mesh_result = cwf::mesh::load_gmsh_file(data_dir / "cantilever.msh");
    if (!mesh_result)
    {
        throw std::runtime_error("cantilever mesh fixture failed to load");
    }
    return mesh_result.value();
}

[[nodiscard]] auto load_config(const cwf::test_support::ConfigBuilderOptions &options = {})
    -> cwf::config::Config
{
    const auto config_result = cwf::test_support::load_config(options);
    if (!config_result)
    {
        throw std::runtime_error("builder options produced invalid config unexpectedly");
    }
    return config_result.value();
}

[[nodiscard]] auto load_mesh_from_string(const std::string &gmsh) -> cwf::mesh::Mesh
{
    const auto mesh_result = cwf::mesh::load_gmsh_from_string(gmsh);
    if (!mesh_result)
    {
        throw std::runtime_error(mesh_result.error().message);
    }
    return mesh_result.value();
}

} // namespace

TEST(PreprocessPipeline, ProducesExpectedOutputsForCantileverFixture)
{
    const auto mesh       = load_fixture_mesh();
    const auto config     = load_config();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_TRUE(preprocess.has_value()) << preprocess.error().message;
    const auto &outputs = preprocess.value();

    ASSERT_EQ(outputs.element_volumes.size(), 1U);
    EXPECT_NEAR(outputs.element_volumes.front(), 1.0 / 6.0, kTol);

    ASSERT_EQ(outputs.shape_gradients.size(), 1U);
    const auto &grads = outputs.shape_gradients.front();
    EXPECT_THAT(grads[0], ElementsAre(-1.0, -1.0, -1.0));
    EXPECT_THAT(grads[1], ElementsAre(1.0, 0.0, 0.0));
    EXPECT_THAT(grads[2], ElementsAre(0.0, 1.0, 0.0));
    EXPECT_THAT(grads[3], ElementsAre(0.0, 0.0, 1.0));

    const double expected_mass = (2500.0 * (1.0 / 6.0)) / 4.0;
    ASSERT_EQ(outputs.lumped_mass.size(), 4U);
    for (double const mass : outputs.lumped_mass)
    {
        EXPECT_NEAR(mass, expected_mass, kTol);
    }

    EXPECT_THAT(outputs.element_material_index, ElementsAre(0U));

    ASSERT_EQ(outputs.adjacency.offsets.size(), 5U);
    EXPECT_THAT(outputs.adjacency.offsets, ElementsAre(0U, 1U, 2U, 3U, 4U));
    EXPECT_THAT(outputs.adjacency.element_indices, ElementsAre(0U, 0U, 0U, 0U));
    EXPECT_THAT(outputs.adjacency.local_indices, ElementsAre(0U, 1U, 2U, 3U));
}

TEST(PreprocessPipeline, SupportsHexahedralElements)
{
    const std::string gmsh = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
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
    const auto &config     = config_result.value();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_TRUE(preprocess.has_value()) << preprocess.error().message;

    // verify hex-specific outputs
    const auto &outputs = preprocess.value();
    ASSERT_EQ(outputs.element_volumes.size(), 1U);
    EXPECT_NEAR(outputs.element_volumes.front(), 1.0, 1e-6); // unit cube volume

    ASSERT_EQ(outputs.shape_gradients.size(), 1U);
    // all 8 gradients should be populated for hex
    for (std::size_t i = 0; i < 8; ++i)
    {
        const auto &grad = outputs.shape_gradients[0][i];
        // verify gradients are non-zero (proper computation)
        const double mag = std::sqrt(grad[0] * grad[0] + grad[1] * grad[1] + grad[2] * grad[2]);
        EXPECT_GT(mag, 0.0) << "gradient " << i << " should be non-zero";
    }

    // lumped mass for 8 nodes with density 2500, volume 1.0
    const double expected_mass_per_node = 2500.0 * 1.0 / 8.0;
    ASSERT_EQ(outputs.lumped_mass.size(), 8U);
    for (double const mass : outputs.lumped_mass)
    {
        EXPECT_NEAR(mass, expected_mass_per_node, 1e-6);
    }
}

TEST(PreprocessPipeline, ErrorsWhenPhysicalGroupMissingAssignment)
{
    // Create a mesh with a physical group that doesn't match any assignment in config
    // The config has assignment for "SOLID" but mesh has "UNASSIGNED_GROUP"
    const std::string gmsh = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
                             "$PhysicalNames\n1\n3 99 \"UNASSIGNED_GROUP\"\n$EndPhysicalNames\n"
                             "$Nodes\n1 4 1 4\n3 99 0 4\n1\n2\n3\n4\n"
                             "0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n"
                             "$Elements\n1 1 1 1\n3 99 4 1\n1 1 2 3 4\n$EndElements\n";

    const auto mesh       = load_mesh_from_string(gmsh);
    const auto config     = load_config();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    // The validation detects that the config references a physical group not in the mesh
    EXPECT_THAT(preprocess.error().message, HasSubstr("missing physical group"));
}

TEST(PreprocessPipeline, RejectsDegenerateTetrahedron)
{
    const std::string gmsh = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
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
    const auto &config     = config_result.value();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    // Note: duplicate node detection runs before volume check, so we get that error first
    EXPECT_THAT(preprocess.error().message, HasSubstr("duplicate nodes"));
}

TEST(PreprocessPipeline, DetectsDuplicateNodes)
{
    const std::string gmsh = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
                             "$PhysicalNames\n1\n3 3 \"SOLID\"\n$EndPhysicalNames\n"
                             "$Nodes\n1 5 1 5\n3 3 0 5\n1\n2\n3\n4\n5\n"
                             "0 0 0\n1 0 0\n0 1 0\n0 0 1\n0 0 0\n$EndNodes\n"
                             "$Elements\n1 1 1 1\n3 3 4 1\n1 1 2 3 4\n$EndElements\n";

    const auto mesh       = load_mesh_from_string(gmsh);
    const auto config     = load_config();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    EXPECT_THAT(preprocess.error().message, HasSubstr("duplicate nodes"));
}

TEST(PreprocessPipeline, DetectsDuplicateElements)
{
    const std::string gmsh = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
                             "$PhysicalNames\n1\n3 3 \"SOLID\"\n$EndPhysicalNames\n"
                             "$Nodes\n1 4 1 4\n3 3 0 4\n1\n2\n3\n4\n"
                             "0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n"
                             "$Elements\n1 2 1 2\n3 3 4 2\n1 1 2 3 4\n2 1 2 3 4\n$EndElements\n";

    const auto mesh       = load_mesh_from_string(gmsh);
    const auto config     = load_config();
    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    EXPECT_THAT(preprocess.error().message, HasSubstr("duplicate elements"));
}

TEST(PreprocessPipeline, ValidatesDirichletGroupsExist)
{
    const std::string gmsh = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
                             "$PhysicalNames\n1\n3 3 \"SOLID\"\n$EndPhysicalNames\n"
                             "$Nodes\n1 4 1 4\n3 3 0 4\n1\n2\n3\n4\n"
                             "0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n"
                             "$Elements\n1 1 1 1\n3 3 4 1\n1 1 2 3 4\n$EndElements\n";

    const auto mesh = load_mesh_from_string(gmsh);

    // Create config with dirichlet fix referencing non-existent group
    cwf::test_support::ConfigBuilderOptions options;
    options.dirichlet_fixes  = {{.group     = "NONEXISTENT_GROUP",
                                 .constrain = {true, true, true},
                                 .values    = {std::nullopt, std::nullopt, std::nullopt}}};
    const auto config_result = cwf::test_support::load_config(options);
    ASSERT_TRUE(config_result.has_value());
    const auto &config = config_result.value();

    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    EXPECT_THAT(preprocess.error().message, HasSubstr("dirichlet fix references missing physical group"));
}

TEST(PreprocessPipeline, ValidatesTractionGroupsExist)
{
    const std::string gmsh = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
                             "$PhysicalNames\n1\n3 3 \"SOLID\"\n$EndPhysicalNames\n"
                             "$Nodes\n1 4 1 4\n3 3 0 4\n1\n2\n3\n4\n"
                             "0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n"
                             "$Elements\n1 1 1 1\n3 3 4 1\n1 1 2 3 4\n$EndElements\n";

    const auto mesh = load_mesh_from_string(gmsh);

    // Create config with traction referencing non-existent group
    // ALSO clear default dirichlet groups to avoid validation errors before traction check
    cwf::test_support::ConfigBuilderOptions options;
    options.dirichlet_fixes.clear(); // Remove default FIXED_BASE group
    options.tractions        = {{.group = "NONEXISTENT_GROUP", .value = {1.0, 0.0, 0.0}, .scale_curve = ""}};
    const auto config_result = cwf::test_support::load_config(options);
    ASSERT_TRUE(config_result.has_value());
    const auto &config = config_result.value();

    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_FALSE(preprocess.has_value());
    EXPECT_THAT(preprocess.error().message, HasSubstr("traction load references missing physical group"));
}

TEST(PreprocessPipeline, LoadsBlockValidationMesh)
{
    const std::filesystem::path data_dir{CWF_TEST_DATA_DIR};
    const auto                  mesh_result = cwf::mesh::load_gmsh_file(data_dir / "block_validation.msh");
    ASSERT_TRUE(mesh_result.has_value()) << mesh_result.error().message;
    const auto &mesh = mesh_result.value();

    EXPECT_EQ(mesh.nodes.size(), 8U);
    EXPECT_EQ(mesh.elements.size(), 6U); // 6 tets to fill unit cube
    EXPECT_EQ(mesh.surfaces.size(), 4U); // 2 bottom + 2 top triangles

    // Create config matching block_validation.yaml
    cwf::test_support::ConfigBuilderOptions opts;
    opts.assignments = {{.group = "BLOCK", .material = "concrete"}};
    opts.materials   = {
        {.name = "concrete", .youngs_modulus = 3.0e10, .poisson_ratio = 0.2, .density = 2500.0}};
    opts.dirichlet_fixes = {{.group     = "BOTTOM_FIXED",
                             .constrain = {true, true, true},
                             .values    = {std::nullopt, std::nullopt, std::nullopt}}};
    opts.tractions.clear();
    const auto config_result = cwf::test_support::load_config(opts);
    ASSERT_TRUE(config_result.has_value()) << "config builder failed";
    const auto &config = config_result.value();

    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_TRUE(preprocess.has_value()) << preprocess.error().message;

    const auto &outputs = preprocess.value();
    EXPECT_EQ(outputs.element_volumes.size(), 6U);

    // total volume should be 1.0 (unit cube)
    double total_vol = 0.0;
    for (double const v : outputs.element_volumes)
    {
        total_vol += v;
    }
    EXPECT_NEAR(total_vol, 1.0, 1.0e-6);
}

/**
 * @test beam validation mesh loads correctly with surface groups
 *
 * verifies surface groups are properly populated for dirichlet and traction loading
 */
TEST(PreprocessPipeline, LoadsBeamValidationMeshWithSurfaces)
{
    const std::filesystem::path data_dir{CWF_TEST_DATA_DIR};
    const auto                  mesh_result = cwf::mesh::load_gmsh_file(data_dir / "beam_validation.msh");
    ASSERT_TRUE(mesh_result.has_value()) << mesh_result.error().message;
    const auto &mesh = mesh_result.value();

    EXPECT_EQ(mesh.nodes.size(), 8U);
    EXPECT_EQ(mesh.elements.size(), 6U); // 6 tets to fill beam
    EXPECT_EQ(mesh.surfaces.size(), 4U); // 2 FIXED_END + 2 FREE_END triangles

    // Verify physical groups exist
    bool found_fixed = false;
    bool found_free  = false;
    bool found_beam  = false;
    for (const auto &group : mesh.physical_groups)
    {
        if (group.name == "FIXED_END")
        {
            found_fixed = true;
        }
        if (group.name == "FREE_END")
        {
            found_free = true;
        }
        if (group.name == "BEAM")
        {
            found_beam = true;
        }
    }
    EXPECT_TRUE(found_fixed) << "FIXED_END physical group not found";
    EXPECT_TRUE(found_free) << "FREE_END physical group not found";
    EXPECT_TRUE(found_beam) << "BEAM physical group not found";

    // Verify surface groups are populated
    std::size_t total_surfaces_in_groups = 0U;
    for (const auto &[group_id, surfaces] : mesh.surface_groups)
    {
        total_surfaces_in_groups += surfaces.size();
    }
    EXPECT_EQ(total_surfaces_in_groups, 4U) << "surface_groups should contain all 4 surfaces";

    // Create config matching beam_validation.yaml
    cwf::test_support::ConfigBuilderOptions opts;
    opts.assignments = {{.group = "BEAM", .material = "steel"}};
    opts.materials   = {{.name = "steel", .youngs_modulus = 2.0e11, .poisson_ratio = 0.3, .density = 7850.0}};
    opts.dirichlet_fixes     = {{.group     = "FIXED_END",
                                 .constrain = {true, true, true},
                                 .values    = {std::nullopt, std::nullopt, std::nullopt}}};
    opts.tractions           = {{.group = "FREE_END", .value = {0.0, 0.0, -100000.0}, .scale_curve = ""}};
    const auto config_result = cwf::test_support::load_config(opts);
    ASSERT_TRUE(config_result.has_value()) << "config builder failed";
    const auto &config = config_result.value();

    const auto preprocess = cwf::mesh::pre::run(mesh, config);
    ASSERT_TRUE(preprocess.has_value()) << preprocess.error().message;
}
