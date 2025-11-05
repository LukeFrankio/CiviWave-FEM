/**
 * @file mesh_loader_test.cpp
 * @brief gmsh parser shakedown so malformed meshes get wrecked fast uwu
 */
#include "test_config.hpp"
#include "support/config_builder.hpp"
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <string>

#include "cwf/mesh/mesh.hpp"

using testing::ElementsAre;
using testing::HasSubstr;

namespace {

[[nodiscard]] auto test_data_path(std::string_view file) -> std::filesystem::path
{
    return std::filesystem::path{CWF_TEST_DATA_DIR} / file;
}

[[nodiscard]] auto basic_gmsh_header() -> std::string
{
    return std::string{
        "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n"
        "$Nodes\n1 4 1 4\n3 3 0 4\n1\n2\n3\n4\n"
        "0 0 0\n1 0 0\n0 1 0\n0 0 1\n$EndNodes\n"
    };
}

[[nodiscard]] auto basic_tet_elements_block(std::string element_payload) -> std::string
{
    return std::string{"$Elements\n1 1 1 1\n"} + std::move(element_payload) + "$EndElements\n";
}

[[nodiscard]] auto tet_block_for_nodes(std::initializer_list<int> node_ids,
                                       std::uint32_t element_type = 4U) -> std::string
{
    std::string line = "3 3 " + std::to_string(element_type) + " 1\n1";
    for (const auto node : node_ids) {
        line += " " + std::to_string(node);
    }
    line.push_back('\n');
    return line;
}

}  // namespace

TEST(MeshLoader, LoadsCantileverFixtureAndCreatesPhysicalLookup)
{
    const auto mesh_result = cwf::mesh::load_gmsh_file(test_data_path("cantilever.msh"));
    ASSERT_TRUE(mesh_result.has_value()) << mesh_result.error().message;
    const auto& mesh = mesh_result.value();

    ASSERT_EQ(mesh.nodes.size(), 4U);
    EXPECT_THAT(mesh.nodes[0].position, ElementsAre(0.0, 0.0, 0.0));
    EXPECT_THAT(mesh.nodes[1].position, ElementsAre(1.0, 0.0, 0.0));

    ASSERT_EQ(mesh.elements.size(), 1U);
    const auto& element = mesh.elements.front();
    EXPECT_EQ(static_cast<std::uint32_t>(element.geometry), static_cast<std::uint32_t>(cwf::mesh::ElementGeometry::Tetrahedron4));
    EXPECT_THAT(element.nodes, ElementsAre(0U, 1U, 2U, 3U, ::testing::_, ::testing::_, ::testing::_, ::testing::_));

    EXPECT_FALSE(mesh.physical_groups.empty());
    const auto lookup_iter = mesh.group_lookup.find(3U);
    ASSERT_NE(lookup_iter, mesh.group_lookup.end());
    EXPECT_EQ(mesh.physical_groups.at(lookup_iter->second).name, "SOLID");
}

TEST(MeshLoader, ReportsFileIoErrors)
{
    const auto bogus_path = test_data_path("definitely_missing.msh");
    const auto result = cwf::mesh::load_gmsh_file(bogus_path);
    ASSERT_FALSE(result.has_value());
    EXPECT_THAT(result.error().message, HasSubstr("failed to open mesh file"));
}

TEST(MeshLoader, ErrorsWhenElementsSectionMissing)
{
    const auto gmsh = basic_gmsh_header();
    const auto result = cwf::mesh::load_gmsh_from_string(gmsh);
    ASSERT_FALSE(result.has_value());
    EXPECT_THAT(result.error().message, HasSubstr("missing $Elements section"));
}

TEST(MeshLoader, ErrorsWhenNodeReferenceMissing)
{
    auto gmsh = basic_gmsh_header();
    gmsh += basic_tet_elements_block(tet_block_for_nodes({1, 2, 3, 99}));
    const auto result = cwf::mesh::load_gmsh_from_string(gmsh);
    ASSERT_FALSE(result.has_value());
    EXPECT_THAT(result.error().message, HasSubstr("element references unknown node"));
}

TEST(MeshLoader, RejectsUnsupportedElementTypes)
{
    auto gmsh = basic_gmsh_header();
    gmsh += basic_tet_elements_block(tet_block_for_nodes({1, 2, 3, 4}, 6U));
    const auto result = cwf::mesh::load_gmsh_from_string(gmsh);
    ASSERT_FALSE(result.has_value());
    EXPECT_THAT(result.error().message, HasSubstr("unsupported Gmsh element type"));
}
