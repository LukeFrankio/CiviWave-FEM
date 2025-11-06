/**
 * @file mesh.hpp
 * @brief gmsh v4 mesh ingestion + data model that keeps FEM preprocessing zen
 *
 * this header defines the mesh data structures and loader entry points for
 * Phase 3. we parse Gmsh 4.x ASCII meshes, map nodes + volume elements into
 * contiguous arrays, and capture physical group metadata so materials + BCs can
 * hook in seamlessly later. data layout leans SoA-friendly to prep for Vulkan
 * buffer packing.
 *
 * @author LukeFrankio
 * @date 2025-11-05
 * @version 1.0
 *
 * @note expects GCC 15.2+, C++26, and Gmsh 4.1+ ASCII meshes
 * @note integrates with config::Assignment to match materials per group
 *
 * example (basic usage):
 * @code
 * using namespace cwf::mesh;
 * auto mesh_result = load_gmsh_file("assets/cantilever.msh");
 * if (!mesh_result) {
 *     fmt::print(stderr, "mesh error: {}\n", mesh_result.error().message);
 *     return;
 * }
 * const Mesh& mesh = *mesh_result;
 * // mesh.elements now holds tetrahedra referencing mesh.nodes indices uwu
 * @endcode
 */
#pragma once

#include <array>
#include <expected>
#include <filesystem>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "cwf/common/math.hpp"

namespace cwf::mesh
{

/**
 * @brief mesh loader error payload with spicy breadcrumbs
 */
struct MeshError
{
    std::string              message; ///< human-friendly vibe check for what failed
    std::vector<std::string> context; ///< structured breadcrumbs (e.g., "Elements[12]")
};

/**
 * @brief supported element topologies (tet first, hex later)
 */
enum class ElementGeometry : std::uint8_t
{
    Tetrahedron4 = 4U,
    Hexahedron8  = 8U
};

/**
 * @brief supported surface topologies for boundary shenanigans
 */
enum class SurfaceGeometry : std::uint8_t
{
    Triangle3      = 3U,
    Quadrilateral4 = 4U
};

/**
 * @brief physical group metadata direct from Gmsh (dimension aware)
 */
struct PhysicalGroup
{
    std::uint32_t dimension; ///< topological dimension (2 surface, 3 volume)
    std::uint32_t id;        ///< numeric identifier from Gmsh $PhysicalNames
    std::string   name;      ///< optional user name ("SOLID", "FIXED_BASE", ...)
};

/**
 * @brief node definition with original id + position
 */
struct Node
{
    std::uint32_t original_id; ///< id from Gmsh file (1-indexed)
    common::Vec3  position;    ///< xyz coordinates in meters
};

/**
 * @brief volume element (tet/hex) referencing node indices
 */
struct Element
{
    std::uint32_t                original_id;    ///< id from Gmsh file (1-indexed)
    ElementGeometry              geometry;       ///< element topology
    std::array<std::uint32_t, 8> nodes{};        ///< node indices (unused entries set to UINT32_MAX)
    std::uint32_t                physical_group; ///< group id (links to PhysicalGroup.id)
};

/**
 * @brief surface element describing traction/BC faces (tri/quad vibes)
 */
struct Surface
{
    std::uint32_t                original_id;    ///< id from mesh file for traceability
    SurfaceGeometry              geometry;       ///< topology (Triangle3/Quadrilateral4)
    std::array<std::uint32_t, 4> nodes{};        ///< node indices (unused slots max)
    std::uint32_t                physical_group; ///< associated physical group id
};

/**
 * @brief compact adjacency-friendly mesh representation
 */
struct Mesh
{
    std::vector<Node>                              nodes;           ///< contiguous node buffer (id-sorted)
    std::vector<Element>                           elements;        ///< supported volume elements (tet/hexa)
    std::vector<PhysicalGroup>                     physical_groups; ///< physical metadata
    std::unordered_map<std::uint32_t, std::size_t> group_lookup;    ///< id → index map for fast lookup
    std::vector<Surface>                           surfaces; ///< boundary faces tagged via physical groups
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>>
        node_groups; ///< physical id -> node indices
    std::unordered_map<std::uint32_t, std::vector<std::size_t>>
        surface_groups; ///< physical id -> surface indices
};

/**
 * @brief result alias for gmsh loader (std::expected wrapper)
 */
using MeshResult = std::expected<Mesh, MeshError>;

/**
 * @brief reads a Gmsh v4 ASCII mesh from disk and returns a mesh model
 *
 * ⚠️ IMPURE FUNCTION (file I/O and yaml-cpp-like parsing side effects)
 *
 * handles nodes, tetrahedra, and hexahedra. orientation validated; zero/negative
 * volumes throw errors. tagged boundary faces (tri/quad) get captured explicitly
 * so later stages can map tractions/dirichlet constraints to concrete node sets.
 * node physical groups (dimension 0) also populate mesh.node_groups for spicy
 * point loads.
 *
 * @param[in] path filesystem path to .msh file
 * @return MeshResult containing mesh or MeshError with context breadcrumbs
 */
[[nodiscard]] auto load_gmsh_file(const std::filesystem::path &path) -> MeshResult;

/**
 * @brief loads mesh from already buffered ASCII content (tests/tooling)
 *
 * ⚠️ IMPURE FUNCTION (parsing has stateful side effects)
 *
 * @param[in] ascii_contents text contents of a Gmsh .msh file
 * @return MeshResult analogous to load_gmsh_file
 */
[[nodiscard]] auto load_gmsh_from_string(std::string_view ascii_contents) -> MeshResult;

} // namespace cwf::mesh
