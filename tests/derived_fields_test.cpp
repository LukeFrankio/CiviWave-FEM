/**
 * @file derived_fields_test.cpp
 * @brief regression tests for Phase 10 derived field calculator uwu
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

#include "cwf/config/config.hpp"
#include "cwf/common/math.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/mesh/preprocess.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/post/derived_fields.hpp"

namespace
{

using cwf::post::compute_derived_fields;

[[nodiscard]] auto make_single_tet_mesh() -> cwf::mesh::Mesh
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
    cfg.time = cwf::config::TimeSettings{.initial_dt = 0.01, .adaptive = false, .min_dt = 0.0, .max_dt = 0.0};
    cfg.solver = cwf::config::SolverSettings{.type = "pcg",
                                             .preconditioner = "block_jacobi",
                                             .runtime_tolerance = 1.0e-4,
                                             .pause_tolerance = 1.0e-5,
                                             .max_iterations = 64U};
    cfg.precision = cwf::config::PrecisionSettings{.vector_precision = "fp32", .reduction_precision = "fp64"};
    cfg.output = cwf::config::OutputSettings{.vtu_stride = 1U, .probes = {}};
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

TEST(DerivedFields, ComputesUniformXStrain)
{
    auto mesh = make_single_tet_mesh();
    auto cfg = make_basic_config();

    const auto preprocess_result = cwf::mesh::pre::run(mesh, cfg);
    ASSERT_TRUE(preprocess_result.has_value());
    const auto preprocess = preprocess_result.value();

    auto pack_result = cwf::mesh::pack::build_packed_buffers(mesh, preprocess, cfg, {});
    ASSERT_TRUE(pack_result.has_value());
    auto pack = std::move(pack_result.value());

    const auto materials = make_materials(cfg);

    constexpr double kStrain = 0.01; // 1% stretch along X
    for (std::size_t node = 0; node < mesh.nodes.size(); ++node)
    {
        const double x = mesh.nodes[node].position[0];
        pack.buffers.nodes.displacement.x[node] = static_cast<float>(kStrain * x);
        pack.buffers.nodes.displacement.y[node] = 0.0F;
        pack.buffers.nodes.displacement.z[node] = 0.0F;
    }

    const auto derived = compute_derived_fields(pack, materials);
    ASSERT_EQ(derived.elements.size(), 1U);
    ASSERT_EQ(derived.nodes.size(), pack.metadata.node_count);

    const auto &elem = derived.elements.front();
    EXPECT_NEAR(elem.strain[0], kStrain, 1.0e-5F);
    EXPECT_NEAR(elem.strain[1], 0.0F, 1.0e-5F);
    EXPECT_NEAR(elem.strain[2], 0.0F, 1.0e-5F);
    EXPECT_NEAR(elem.strain[3], 0.0F, 1.0e-5F);

    const double lambda = (cfg.materials[0].poisson_ratio * cfg.materials[0].youngs_modulus)
                          / ((1.0 + cfg.materials[0].poisson_ratio) * (1.0 - 2.0 * cfg.materials[0].poisson_ratio));
    const double mu = cfg.materials[0].youngs_modulus / (2.0 * (1.0 + cfg.materials[0].poisson_ratio));
    const double expected_sx = (lambda + 2.0 * mu) * kStrain;
    const double expected_sy = lambda * kStrain;

    EXPECT_NEAR(elem.stress[0], static_cast<float>(expected_sx), 5.0e3F);
    EXPECT_NEAR(elem.stress[1], static_cast<float>(expected_sy), 5.0e3F);
    EXPECT_NEAR(elem.stress[2], static_cast<float>(expected_sy), 5.0e3F);

    for (const auto &node_field : derived.nodes)
    {
        EXPECT_NEAR(node_field.strain[0], kStrain, 1.0e-4F);
        EXPECT_NEAR(node_field.stress[0], static_cast<float>(expected_sx), 5.0e3F);
    }
}

} // namespace
