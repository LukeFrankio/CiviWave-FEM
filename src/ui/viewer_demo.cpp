/**
 * @file viewer_demo.cpp
 * @brief tiny harness that drives the Phase 10 viewer with a synthetic mesh uwu
 *
 * this executable keeps everything self-contained so developers can flip
 * `-DBUILD_UI=ON`, build once, and immediately see the von-mises overlay doing
 * its thing without waiting for later phases. it procedurally builds a single
 * tetrahedron mesh + config, runs preprocessing, packs the data, computes
 * derived fields, and finally calls `cwf::ui::run_viewer_once`.
 *
 * @note requires BUILD_UI=ON (GLFW + ImGui fetched via FetchContent)
 * @note targets the usual GCC 15.2 + C++26 toolchain per repo defaults
 */

#include "cwf/ui/viewer.hpp"

#include <array>
#include <cstdlib>
#include <expected>
#include <filesystem>
#include <limits>
#include <print>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cwf/common/math.hpp"
#include "cwf/config/config.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/mesh/preprocess.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/physics/newmark.hpp"
#include "cwf/physics/solver.hpp"
#include "cwf/post/derived_fields.hpp"

namespace
{

/**
 * @brief build a single tetrahedron mesh with SOLID physical group
 *
 * ✨ PURE FUNCTION ✨
 *
 * returns a miniature mesh that mirrors the synthetic fixtures living in the
 * test suite but adds the physical group metadata that preprocessing needs to
 * bind materials. this keeps the viewer demo light-weight while still flowing
 * through the exact same code paths as production meshes.
 */
[[nodiscard]] auto make_demo_mesh() -> cwf::mesh::Mesh
{
    cwf::mesh::Mesh mesh{};
    mesh.nodes = {
        cwf::mesh::Node{.original_id = 0U, .position = cwf::common::Vec3{0.0, 0.0, 0.0}},
        cwf::mesh::Node{.original_id = 1U, .position = cwf::common::Vec3{1.0, 0.0, 0.0}},
        cwf::mesh::Node{.original_id = 2U, .position = cwf::common::Vec3{0.0, 1.0, 0.0}},
        cwf::mesh::Node{.original_id = 3U, .position = cwf::common::Vec3{0.0, 0.0, 1.0}},
    };

    mesh.physical_groups = {
        cwf::mesh::PhysicalGroup{.dimension = 3U, .id = 1U, .name = "SOLID"},
        cwf::mesh::PhysicalGroup{.dimension = 2U, .id = 2U, .name = "FIXED"},
        cwf::mesh::PhysicalGroup{.dimension = 0U, .id = 3U, .name = "POINT"},
    };
    for (std::size_t i = 0; i < mesh.physical_groups.size(); ++i)
    {
        mesh.group_lookup.emplace(mesh.physical_groups[i].id, i);
    }

    cwf::mesh::Element tet{};
    tet.original_id    = 0U;
    tet.geometry       = cwf::mesh::ElementGeometry::Tetrahedron4;
    tet.physical_group = 1U;
    tet.nodes          = {0U, 1U, 2U, 3U, std::numeric_limits<std::uint32_t>::max(),
                 std::numeric_limits<std::uint32_t>::max(), std::numeric_limits<std::uint32_t>::max(),
                 std::numeric_limits<std::uint32_t>::max()};
    mesh.elements.push_back(tet);

    cwf::mesh::Surface base{};
    base.original_id    = 0U;
    base.geometry       = cwf::mesh::SurfaceGeometry::Triangle3;
    base.physical_group = 2U;
    base.nodes          = {0U, 1U, 2U, std::numeric_limits<std::uint32_t>::max()};
    mesh.surfaces.push_back(base);
    mesh.surface_groups[2U] = {0U};
    mesh.node_groups[3U]    = {3U};

    return mesh;
}

/**
 * @brief craft a minimal config that binds the SOLID group to steel
 *
 * ⚠️ IMPURE FUNCTION (depends on std::vector allocations)
 *
 * the config mirrors the helper used in the unit tests: one material, one
 * assignment, simple solver knobs, and default zero outputs. this is enough to
 * exercise preprocessing + packing without touching disk.
 */
[[nodiscard]] auto make_demo_config() -> cwf::config::Config
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

    cfg.damping  = cwf::config::Damping{.xi = 0.02, .w1 = 5.0, .w2 = 50.0};
    cfg.time     = cwf::config::TimeSettings{.initial_dt = 0.01, .adaptive = false, .min_dt = 0.0, .max_dt = 0.0};
    cfg.solver   = cwf::config::SolverSettings{.type = "pcg",
                                             .preconditioner   = "block_jacobi",
                                             .runtime_tolerance = 1.0e-4,
                                             .pause_tolerance   = 1.0e-5,
                                             .max_iterations    = 64U};
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

    cfg.output = cwf::config::OutputSettings{.vtu_stride = 1U, .probes = {}};

    return cfg;
}

/**
 * @brief convert YAML-ish materials into physics::ElasticProperties
 */
[[nodiscard]] auto make_materials(const cwf::config::Config &cfg)
    -> std::vector<cwf::physics::materials::ElasticProperties>
{
    std::vector<cwf::physics::materials::ElasticProperties> mats{};
    mats.reserve(cfg.materials.size());
    for (const auto &material : cfg.materials)
    {
        mats.push_back(cwf::physics::materials::make_properties(material));
    }
    return mats;
}

/**
 * @brief copy a dense Newmark state back into the SoA pack buffers
 *
 * ⚠️ IMPURE FUNCTION (mutates pack buffers)
 *
 * @param state    Newmark displacement/velocity/acceleration triplet
 * @param pack     packed buffers that drive both GPU uploads and derived fields
 */
void write_state_to_pack(const cwf::physics::newmark::State &state, cwf::mesh::pack::PackingResult &pack)
{
    const std::size_t dof_count  = pack.metadata.dof_count;
    const std::size_t node_count = pack.metadata.node_count;
    if (state.displacement.size() != dof_count || state.velocity.size() != dof_count ||
        state.acceleration.size() != dof_count)
    {
        throw std::runtime_error("newmark state size mismatch");
    }

    for (std::size_t node = 0; node < node_count; ++node)
    {
        const std::size_t base = node * 3U;
        pack.buffers.nodes.displacement.x[node] = static_cast<float>(state.displacement[base + 0U]);
        pack.buffers.nodes.displacement.y[node] = static_cast<float>(state.displacement[base + 1U]);
        pack.buffers.nodes.displacement.z[node] = static_cast<float>(state.displacement[base + 2U]);

        pack.buffers.nodes.velocity.x[node] = static_cast<float>(state.velocity[base + 0U]);
        pack.buffers.nodes.velocity.y[node] = static_cast<float>(state.velocity[base + 1U]);
        pack.buffers.nodes.velocity.z[node] = static_cast<float>(state.velocity[base + 2U]);

        pack.buffers.nodes.acceleration.x[node] = static_cast<float>(state.acceleration[base + 0U]);
        pack.buffers.nodes.acceleration.y[node] = static_cast<float>(state.acceleration[base + 1U]);
        pack.buffers.nodes.acceleration.z[node] = static_cast<float>(state.acceleration[base + 2U]);
    }
}

/**
 * @brief assemble + solve one implicit Newmark step for the demo mesh
 *
 * ⚠️ IMPURE FUNCTION (allocations + pack mutation)
 *
 * @param mesh        synthetic mesh model
 * @param cfg         demo configuration (materials, BCs, loads)
 * @param preprocess  cached preprocessing outputs (gradients, volumes, etc.)
 * @param materials   elastic properties derived from cfg.materials
 * @param pack        packed buffers that will receive solved displacements
 */
void run_physics_step(const cwf::mesh::Mesh &mesh,
                      const cwf::config::Config &cfg,
                      const cwf::mesh::pre::Outputs &preprocess,
                      const std::vector<cwf::physics::materials::ElasticProperties> &materials,
                      cwf::mesh::pack::PackingResult &pack)
{
    const auto assembly  = cwf::physics::solver::assemble_linear_system(mesh, preprocess, materials);
    const auto dirichlet = cwf::physics::solver::build_dirichlet_conditions(mesh, cfg);
    const auto rayleigh  = cwf::physics::materials::compute_rayleigh(cfg.damping);
    const auto coeffs    = cwf::physics::newmark::make_coefficients(cfg.time.initial_dt, 0.25, 0.5);

    cwf::physics::newmark::State previous{};
    previous.displacement.assign(pack.metadata.dof_count, 0.0);
    previous.velocity.assign(pack.metadata.dof_count, 0.0);
    previous.acceleration.assign(pack.metadata.dof_count, 0.0);

    const auto result = cwf::physics::solver::solve_newmark_step(assembly,
                                                                 rayleigh,
                                                                 dirichlet,
                                                                 mesh,
                                                                 cfg,
                                                                 preprocess,
                                                                 coeffs,
                                                                 previous,
                                                                 /*time=*/0.0,
                                                                 cfg.solver.runtime_tolerance,
                                                                 static_cast<std::size_t>(cfg.solver.max_iterations));
    write_state_to_pack(result.state, pack);
    std::print("Solver iterations: {} residual {:.3e} (converged: {})\n",
               result.stats.iterations,
               result.stats.residual_norm,
               result.stats.converged ? "yes" : "no");
}

} // namespace

int main()
{
#if !(defined(CWF_ENABLE_UI) && CWF_ENABLE_UI)
    std::print(stderr, "viewer_demo was built without UI support. Reconfigure with -DBUILD_UI=ON.\n");
    return EXIT_FAILURE;
#else
    auto mesh = make_demo_mesh();
    auto cfg  = make_demo_config();

    const auto preprocess_result = cwf::mesh::pre::run(mesh, cfg);
    if (!preprocess_result)
    {
        std::print(stderr, "preprocess error: {}\n", preprocess_result.error().message);
        return EXIT_FAILURE;
    }

    auto pack_result = cwf::mesh::pack::build_packed_buffers(mesh, preprocess_result.value(), cfg, {});
    if (!pack_result)
    {
        std::print(stderr, "packing error: {}\n", pack_result.error().message);
        return EXIT_FAILURE;
    }
    auto pack = std::move(pack_result.value());

    std::print("Running viewer demo with {} nodes and {} elements\n", mesh.nodes.size(), mesh.elements.size());
    for (size_t i = 0; i < mesh.nodes.size(); ++i) {
        std::print("Node {}: ({}, {}, {})\n", i, 
            pack.buffers.nodes.position0.x[i],
            pack.buffers.nodes.position0.y[i],
            pack.buffers.nodes.position0.z[i]);
    }

    auto materials = make_materials(cfg);
    run_physics_step(mesh, cfg, preprocess_result.value(), materials, pack);
    const auto derived = cwf::post::compute_derived_fields(pack, materials);

    const auto viewer_status = cwf::ui::run_viewer_once(mesh, pack, derived, 0.0);
    if (!viewer_status)
    {
        std::print(stderr, "viewer error: {}\n", viewer_status.error().message);
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
#endif
}
