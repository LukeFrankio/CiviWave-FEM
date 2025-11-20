/**
 * @file export_writer_test.cpp
 * @brief smoke tests for the VTU writer + probe logger stack
 */

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <limits>

#include "cwf/config/config.hpp"
#include "cwf/common/math.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/mesh/preprocess.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/post/derived_fields.hpp"
#include "cwf/post/output_manager.hpp"
#include "cwf/post/probe_logger.hpp"
#include "cwf/post/vtu_writer.hpp"

namespace
{

using cwf::post::compute_derived_fields;

[[nodiscard]] auto make_mesh_and_config() -> std::pair<cwf::mesh::Mesh, cwf::config::Config>
{
    cwf::mesh::Mesh mesh{};
    mesh.nodes = {
        cwf::mesh::Node{0U, cwf::common::Vec3{0.0, 0.0, 0.0}},
        cwf::mesh::Node{1U, cwf::common::Vec3{1.0, 0.0, 0.0}},
        cwf::mesh::Node{2U, cwf::common::Vec3{0.0, 1.0, 0.0}},
        cwf::mesh::Node{3U, cwf::common::Vec3{0.0, 0.0, 1.0}},
    };
    mesh.physical_groups.push_back(cwf::mesh::PhysicalGroup{.dimension = 3U, .id = 1U, .name = "SOLID"});
    mesh.group_lookup.emplace(1U, 0U);

    cwf::mesh::Element tet{};
    tet.original_id = 0U;
    tet.physical_group = 1U;
    tet.geometry = cwf::mesh::ElementGeometry::Tetrahedron4;
    tet.nodes = {0U, 1U, 2U, 3U, std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max(),
                 std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max()};
    mesh.elements.push_back(tet);

    cwf::config::Config cfg{};
    cfg.mesh_path = std::filesystem::path{"synthetic.msh"};
    cfg.materials.push_back(cwf::config::Material{.name = "steel", .youngs_modulus = 30.0e9, .poisson_ratio = 0.2, .density = 2500.0});
    cfg.assignments.push_back(cwf::config::Assignment{.group = "SOLID", .material = "steel"});
    cfg.damping = cwf::config::Damping{.xi = 0.02, .w1 = 5.0, .w2 = 50.0};
    cfg.time = cwf::config::TimeSettings{.initial_dt = 0.01, .adaptive = false, .min_dt = 0.0, .max_dt = 0.0};
    cfg.solver = cwf::config::SolverSettings{.type = "pcg",
                                             .preconditioner = "block_jacobi",
                                             .runtime_tolerance = 1.0e-4,
                                             .pause_tolerance = 1.0e-5,
                                             .max_iterations = 64U};
    cfg.precision = cwf::config::PrecisionSettings{.vector_precision = "fp32", .reduction_precision = "fp64"};
    cfg.output = cwf::config::OutputSettings{.vtu_stride = 1U, .probes = {}};

    return {mesh, cfg};
}

TEST(VtuWriter, WritesBinaryFileWithMetadata)
{
    auto [mesh, cfg] = make_mesh_and_config();
    const auto preprocess_result = cwf::mesh::pre::run(mesh, cfg);
    ASSERT_TRUE(preprocess_result.has_value());
    auto pack_result = cwf::mesh::pack::build_packed_buffers(mesh, preprocess_result.value(), cfg, {});
    ASSERT_TRUE(pack_result.has_value());
    auto pack = std::move(pack_result.value());

    const auto materials = [&cfg]() {
        std::vector<cwf::physics::materials::ElasticProperties> mats{};
        mats.reserve(cfg.materials.size());
        for (const auto &mat : cfg.materials)
        {
            mats.push_back(cwf::physics::materials::make_properties(mat));
        }
        return mats;
    }();

    const auto derived = compute_derived_fields(pack, materials);

    const auto out_dir = std::filesystem::temp_directory_path() / "cwf_vtu_test";
    const auto path = out_dir / "frame_0.vtu";
    const auto result = cwf::post::write_vtu(path, mesh, pack, derived, 0.0, 0U);
    ASSERT_TRUE(result.has_value());
    ASSERT_TRUE(std::filesystem::exists(path));

    std::ifstream file(path, std::ios::binary);
    ASSERT_TRUE(file.good());
    std::string header(128, '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    EXPECT_NE(header.find("VTKFile"), std::string::npos);
}

TEST(ProbeLoggerTests, WritesCsvRows)
{
    auto [mesh, cfg] = make_mesh_and_config();
    const auto preprocess_result = cwf::mesh::pre::run(mesh, cfg);
    ASSERT_TRUE(preprocess_result.has_value());
    auto pack_result = cwf::mesh::pack::build_packed_buffers(mesh, preprocess_result.value(), cfg, {});
    ASSERT_TRUE(pack_result.has_value());
    auto pack = std::move(pack_result.value());

    for (std::size_t node = 0; node < mesh.nodes.size(); ++node)
    {
        pack.buffers.nodes.displacement.x[node] = static_cast<float>(0.01 * mesh.nodes[node].position[0]);
    }

    const auto materials = [&cfg]() {
        std::vector<cwf::physics::materials::ElasticProperties> mats{};
        for (const auto &mat : cfg.materials)
        {
            mats.push_back(cwf::physics::materials::make_properties(mat));
        }
        return mats;
    }();

    const auto derived = compute_derived_fields(pack, materials);

    const auto csv_path = std::filesystem::temp_directory_path() / "cwf_vtu_test" / "probes.csv";
    cwf::post::ProbeLogger logger(csv_path, {0U, 1U});
    const auto status = logger.log_frame(0.0, 0U, pack, derived);
    ASSERT_TRUE(status.has_value());

    std::ifstream file(csv_path);
    ASSERT_TRUE(file.good());
    std::string header{};
    std::getline(file, header);
    EXPECT_NE(header.find("frame"), std::string::npos);
    std::string row{};
    std::getline(file, row);
    EXPECT_FALSE(row.empty());
}

TEST(OutputManagerTests, EnforcesStrideAndProbes)
{
    auto [mesh, cfg] = make_mesh_and_config();
    const auto preprocess_result = cwf::mesh::pre::run(mesh, cfg);
    ASSERT_TRUE(preprocess_result.has_value());
    auto pack_result = cwf::mesh::pack::build_packed_buffers(mesh, preprocess_result.value(), cfg, {});
    ASSERT_TRUE(pack_result.has_value());
    auto pack = std::move(pack_result.value());

    const auto materials = [&cfg]() {
        std::vector<cwf::physics::materials::ElasticProperties> mats{};
        for (const auto &mat : cfg.materials)
        {
            mats.push_back(cwf::physics::materials::make_properties(mat));
        }
        return mats;
    }();

    cwf::config::OutputSettings settings{.vtu_stride = 2U, .probes = {0U}};
    const auto out_dir = std::filesystem::temp_directory_path() / "cwf_vtu_test_manager";
    cwf::post::OutputManager manager(out_dir, mesh, pack, materials, settings);

    for (std::uint32_t frame = 0; frame < 3U; ++frame)
    {
        ASSERT_TRUE(manager.handle_frame(static_cast<double>(frame) * 0.01, frame).has_value());
    }

    EXPECT_TRUE(std::filesystem::exists(out_dir / "vtu" / "frame_00000.vtu"));
    EXPECT_FALSE(std::filesystem::exists(out_dir / "vtu" / "frame_00001.vtu"));
    EXPECT_TRUE(std::filesystem::exists(out_dir / "vtu" / "frame_00002.vtu"));
    EXPECT_TRUE(std::filesystem::exists(out_dir / "probes" / "probes.csv"));
}

} // namespace
