/**
 * @file mesh.cpp
 * @brief gmsh v4 ASCII parser implementation w/ physical group vibes
 *
 * this TU ingests Gmsh 4.x ASCII meshes, decodes nodes/elements/physical names,
 * and produces the Mesh struct used by preprocessing. coverage includes tetra4
 * and hexa8 elements (Phase 3 scope) with robust validation + error breadcrumbs.
 */
#include "cwf/mesh/mesh.hpp"

#include <cctype>
#include <charconv>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>
#include <unordered_set>

namespace cwf::mesh
{
namespace
{

using EntityKey = std::uint64_t;

constexpr auto make_entity_key(std::uint32_t dimension, std::uint32_t tag) noexcept -> EntityKey
{
    return (static_cast<EntityKey>(dimension) << 32U) | static_cast<EntityKey>(tag);
}

struct EntitiesInfo
{
    std::unordered_map<EntityKey, std::vector<std::uint32_t>> physical_mapping;
    std::unordered_map<std::uint32_t, std::uint32_t>          physical_dimensions;
};

struct PhysicalNamesInfo
{
    std::unordered_map<EntityKey, std::string> names;
};

struct NodesParseResult
{
    std::vector<Node>                                             nodes;
    std::unordered_map<std::uint32_t, std::size_t>                id_to_index;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> nodes_by_group;
};

struct ElementsParseResult
{
    std::vector<Element>                                        volume_elements;
    std::vector<Surface>                                        surface_elements;
    std::unordered_map<std::uint32_t, std::vector<std::size_t>> surface_groups;
    std::unordered_set<std::uint32_t>                           used_physical_ids;
};

[[nodiscard]] auto trim(std::string_view value) -> std::string_view
{
    const auto start = value.find_first_not_of(" \t\r");
    if (start == std::string_view::npos)
    {
        return {};
    }
    const auto end = value.find_last_not_of(" \t\r");
    return value.substr(start, end - start + 1U);
}

[[nodiscard]] auto parse_physical_names(std::istringstream &stream)
    -> std::expected<PhysicalNamesInfo, MeshError>
{
    PhysicalNamesInfo info{};
    std::string       line;
    std::getline(stream, line);
    const auto count = static_cast<std::uint32_t>(std::stoul(std::string(trim(line))));
    for (std::uint32_t i = 0; i < count; ++i)
    {
        if (!std::getline(stream, line))
        {
            return std::unexpected(MeshError{"unexpected EOF in $PhysicalNames", {"PhysicalNames"}});
        }
        std::istringstream line_stream{line};
        std::uint32_t      dim = 0U;
        std::uint32_t      tag = 0U;
        std::string        name;
        line_stream >> dim >> tag;
        std::getline(line_stream >> std::ws, name);
        if (!name.empty() && name.front() == '"' && name.back() == '"')
        {
            name = name.substr(1, name.size() - 2U);
        }
        info.names.emplace(make_entity_key(dim, tag), std::move(name));
    }
    return info;
}

[[nodiscard]] auto parse_entities(std::istringstream &stream) -> std::expected<EntitiesInfo, MeshError>
{
    EntitiesInfo info{};
    std::string  line;
    if (!std::getline(stream, line))
    {
        return std::unexpected(MeshError{"unexpected EOF in $Entities header", {"Entities"}});
    }
    std::istringstream header_stream{line};
    std::uint32_t      num_points{}, num_curves{}, num_surfaces{}, num_volumes{};
    header_stream >> num_points >> num_curves >> num_surfaces >> num_volumes;

    auto parse_entity_block = [&](std::uint32_t dimension,
                                  std::uint32_t count) -> std::expected<void, MeshError> {
        for (std::uint32_t i = 0; i < count; ++i)
        {
            if (!std::getline(stream, line))
            {
                return std::unexpected(MeshError{"unexpected EOF inside $Entities block",
                                                 {"Entities", std::format("dim{}", dimension)}});
            }
            std::istringstream entity_stream{line};
            std::uint32_t      tag{};
            entity_stream >> tag;
            double minx{}, miny{}, minz{}, maxx{}, maxy{}, maxz{};
            entity_stream >> minx >> miny >> minz >> maxx >> maxy >> maxz;
            std::uint32_t num_phys = 0U;
            entity_stream >> num_phys;
            std::vector<std::uint32_t> phys_ids;
            phys_ids.reserve(num_phys);
            for (std::uint32_t j = 0; j < num_phys; ++j)
            {
                std::uint32_t phys{};
                entity_stream >> phys;
                phys_ids.push_back(phys);
                info.physical_dimensions.emplace(phys, dimension);
            }
            if (!phys_ids.empty())
            {
                info.physical_mapping.emplace(make_entity_key(dimension, tag), std::move(phys_ids));
            }
        }
        return {};
    };

    if (auto res = parse_entity_block(0U, num_points); !res)
    {
        return std::unexpected(res.error());
    }
    if (auto res = parse_entity_block(1U, num_curves); !res)
    {
        return std::unexpected(res.error());
    }
    if (auto res = parse_entity_block(2U, num_surfaces); !res)
    {
        return std::unexpected(res.error());
    }
    if (auto res = parse_entity_block(3U, num_volumes); !res)
    {
        return std::unexpected(res.error());
    }
    return info;
}

[[nodiscard]] auto parse_nodes(std::istringstream &stream, const EntitiesInfo &entities)
    -> std::expected<NodesParseResult, MeshError>
{
    std::vector<Node>                                             nodes;
    std::unordered_map<std::uint32_t, std::size_t>                id_to_index;
    std::unordered_map<std::uint32_t, std::vector<std::uint32_t>> nodes_by_group;
    std::string                                                   line;
    if (!std::getline(stream, line))
    {
        return std::unexpected(MeshError{"unexpected EOF in $Nodes header", {"Nodes"}});
    }
    std::istringstream header_stream{line};
    std::uint64_t      num_entity_blocks{}, num_nodes{}, min_node{}, max_node{};
    header_stream >> num_entity_blocks >> num_nodes >> min_node >> max_node;
    nodes.reserve(static_cast<std::size_t>(num_nodes));

    for (std::uint64_t block = 0; block < num_entity_blocks; ++block)
    {
        if (!std::getline(stream, line))
        {
            return std::unexpected(MeshError{"unexpected EOF in $Nodes block header", {"Nodes"}});
        }
        std::istringstream block_stream{line};
        std::uint32_t      entity_dim{}, entity_tag{};
        std::uint32_t      parametric{};
        std::uint64_t      nodes_in_block{};
        block_stream >> entity_dim >> entity_tag >> parametric >> nodes_in_block;
        (void) entity_dim;
        (void) parametric;

        const auto  entity_key = make_entity_key(entity_dim, entity_tag);
        const auto  phys_iter  = entities.physical_mapping.find(entity_key);
        const auto *phys_ids   = phys_iter != entities.physical_mapping.end() ? &phys_iter->second : nullptr;

        std::vector<std::uint32_t> node_ids(nodes_in_block);
        for (std::uint64_t i = 0; i < nodes_in_block; ++i)
        {
            if (!std::getline(stream, line))
            {
                return std::unexpected(MeshError{"unexpected EOF reading node ids", {"Nodes"}});
            }
            node_ids[static_cast<std::size_t>(i)] =
                static_cast<std::uint32_t>(std::stoul(std::string(trim(line))));
        }
        for (std::uint64_t i = 0; i < nodes_in_block; ++i)
        {
            if (!std::getline(stream, line))
            {
                return std::unexpected(MeshError{"unexpected EOF reading node coordinates", {"Nodes"}});
            }
            std::istringstream coord_stream{line};
            double             x{}, y{}, z{};
            coord_stream >> x >> y >> z;
            Node node{node_ids[static_cast<std::size_t>(i)], common::Vec3{x, y, z}};
            id_to_index[node.original_id] = nodes.size();
            nodes.push_back(std::move(node));
            const auto node_index = static_cast<std::uint32_t>(nodes.size() - 1U);
            if (phys_ids != nullptr)
            {
                for (const auto phys_id : *phys_ids)
                {
                    nodes_by_group[phys_id].push_back(node_index);
                }
            }
        }
    }

    if (nodes.size() != static_cast<std::size_t>(num_nodes))
    {
        return std::unexpected(MeshError{"node count mismatch", {"Nodes"}});
    }
    return NodesParseResult{std::move(nodes), std::move(id_to_index), std::move(nodes_by_group)};
}

[[nodiscard]] auto element_node_count(std::uint32_t gmsh_type) -> std::optional<std::size_t>
{
    switch (gmsh_type)
    {
    case 2U:
        return 3U; // Triangle
    case 3U:
        return 4U; // Quadrilateral
    case 4U:
        return 4U; // Tetrahedron
    case 5U:
        return 8U; // Hexahedron
    default:
        return std::nullopt;
    }
}

[[nodiscard]] auto to_geometry(std::uint32_t gmsh_type) -> std::optional<ElementGeometry>
{
    switch (gmsh_type)
    {
    case 4U:
        return ElementGeometry::Tetrahedron4;
    case 5U:
        return ElementGeometry::Hexahedron8;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] auto to_surface_geometry(std::uint32_t gmsh_type) -> std::optional<SurfaceGeometry>
{
    switch (gmsh_type)
    {
    case 2U:
        return SurfaceGeometry::Triangle3;
    case 3U:
        return SurfaceGeometry::Quadrilateral4;
    default:
        return std::nullopt;
    }
}

[[nodiscard]] auto parse_elements(std::istringstream                                   &stream,
                                  const std::unordered_map<std::uint32_t, std::size_t> &id_to_index,
                                  const EntitiesInfo                                   &entities)
    -> std::expected<ElementsParseResult, MeshError>
{
    ElementsParseResult result{};
    std::string         line;
    if (!std::getline(stream, line))
    {
        return std::unexpected(MeshError{"unexpected EOF in $Elements header", {"Elements"}});
    }
    std::istringstream header_stream{line};
    std::uint64_t      num_blocks{}, num_elements{}, min_tag{}, max_tag{};
    header_stream >> num_blocks >> num_elements >> min_tag >> max_tag;
    std::size_t processed_count = 0U;

    for (std::uint64_t block = 0; block < num_blocks; ++block)
    {
        if (!std::getline(stream, line))
        {
            return std::unexpected(MeshError{"unexpected EOF reading element block header", {"Elements"}});
        }
        std::istringstream block_stream{line};
        std::uint32_t      entity_dim{}, entity_tag{}, element_type{};
        std::uint64_t      elements_in_block{};
        block_stream >> entity_dim >> entity_tag >> element_type >> elements_in_block;
        const auto node_count_opt = element_node_count(element_type);
        if (!node_count_opt)
        {
            return std::unexpected(MeshError{std::format("unsupported Gmsh element type {}", element_type),
                                             {"Elements", std::format("entityTag={}", entity_tag)}});
        }
        const auto node_count = node_count_opt.value();

        const auto    entity_key        = make_entity_key(entity_dim, entity_tag);
        const auto    phys_iter         = entities.physical_mapping.find(entity_key);
        std::uint32_t physical_group_id = entity_tag;
        if (phys_iter != entities.physical_mapping.end() && !phys_iter->second.empty())
        {
            physical_group_id = phys_iter->second.front();
        }

        const bool is_volume  = entity_dim == 3U;
        const bool is_surface = entity_dim == 2U;

        for (std::uint64_t i = 0; i < elements_in_block; ++i)
        {
            if (!std::getline(stream, line))
            {
                return std::unexpected(MeshError{"unexpected EOF reading element data", {"Elements"}});
            }
            ++processed_count;
            std::istringstream elem_stream{line};
            std::uint32_t      element_tag{};
            elem_stream >> element_tag;

            if (is_volume)
            {
                const auto geom_opt = to_geometry(element_type);
                if (!geom_opt)
                {
                    return std::unexpected(
                        MeshError{std::format("unsupported volume element type {}", element_type),
                                  {"Elements", std::format("elementTag={}", element_tag)}});
                }
                Element element{};
                element.original_id    = element_tag;
                element.geometry       = geom_opt.value();
                element.physical_group = physical_group_id;
                element.nodes.fill(std::numeric_limits<std::uint32_t>::max());
                for (std::size_t node_idx = 0; node_idx < node_count; ++node_idx)
                {
                    std::uint32_t node_tag{};
                    elem_stream >> node_tag;
                    const auto map_iter = id_to_index.find(node_tag);
                    if (map_iter == id_to_index.end())
                    {
                        return std::unexpected(
                            MeshError{std::format("element references unknown node {}", node_tag),
                                      {"Elements", std::format("elementTag={}", element_tag)}});
                    }
                    element.nodes[node_idx] = static_cast<std::uint32_t>(map_iter->second);
                }
                result.used_physical_ids.insert(physical_group_id);
                result.volume_elements.push_back(std::move(element));
            }
            else if (is_surface)
            {
                const auto geom_opt = to_surface_geometry(element_type);
                if (!geom_opt)
                {
                    return std::unexpected(
                        MeshError{std::format("unsupported surface element type {}", element_type),
                                  {"Elements", std::format("elementTag={}", element_tag)}});
                }
                Surface surface{};
                surface.original_id    = element_tag;
                surface.geometry       = geom_opt.value();
                surface.physical_group = physical_group_id;
                surface.nodes.fill(std::numeric_limits<std::uint32_t>::max());
                for (std::size_t node_idx = 0; node_idx < node_count; ++node_idx)
                {
                    std::uint32_t node_tag{};
                    elem_stream >> node_tag;
                    const auto map_iter = id_to_index.find(node_tag);
                    if (map_iter == id_to_index.end())
                    {
                        return std::unexpected(
                            MeshError{std::format("surface references unknown node {}", node_tag),
                                      {"Elements", std::format("elementTag={}", element_tag)}});
                    }
                    surface.nodes[node_idx] = static_cast<std::uint32_t>(map_iter->second);
                }
                result.used_physical_ids.insert(physical_group_id);
                const auto index = result.surface_elements.size();
                result.surface_groups[physical_group_id].push_back(index);
                result.surface_elements.push_back(std::move(surface));
            }
            else
            {
                // Element belongs to dimension we currently ignore (lines, points). Consume node tags
                // quietly.
                for (std::size_t node_idx = 0; node_idx < node_count; ++node_idx)
                {
                    std::uint32_t discard{};
                    elem_stream >> discard;
                }
            }
        }
    }

    if (processed_count != static_cast<std::size_t>(num_elements))
    {
        return std::unexpected(MeshError{"element count mismatch", {"Elements"}});
    }
    return result;
}

[[nodiscard]] auto read_section(std::istringstream &stream, std::string_view expected_end)
    -> std::istringstream
{
    std::string contents;
    std::string line;
    while (std::getline(stream, line))
    {
        if (trim(line) == expected_end)
        {
            break;
        }
        contents.append(line);
        contents.push_back('\n');
    }
    return std::istringstream{contents};
}

} // namespace

auto load_gmsh_file(const std::filesystem::path &path) -> MeshResult
{
    std::ifstream file(path);
    if (!file.is_open())
    {
        return std::unexpected(
            MeshError{std::format("failed to open mesh file: {}", path.string()), {path.string()}});
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return load_gmsh_from_string(buffer.str());
}

auto load_gmsh_from_string(std::string_view ascii_contents) -> MeshResult
{
    Mesh                                           mesh{};
    std::unordered_map<std::uint32_t, std::size_t> node_lookup;
    EntitiesInfo                                   entities{};
    PhysicalNamesInfo                              physical_names{};
    std::unordered_set<EntityKey>                  seen_sections;
    std::unordered_set<std::uint32_t>              referenced_group_ids;

    std::istringstream input{std::string(ascii_contents)};
    std::string        line;
    while (std::getline(input, line))
    {
        const auto trimmed = trim(line);
        if (trimmed == "$PhysicalNames")
        {
            auto section_stream = read_section(input, "$EndPhysicalNames");
            auto result         = parse_physical_names(section_stream);
            if (!result)
            {
                return std::unexpected(result.error());
            }
            physical_names = std::move(result.value());
            seen_sections.insert(make_entity_key(9U, 9U));
        }
        else if (trimmed == "$Entities")
        {
            auto section_stream = read_section(input, "$EndEntities");
            auto result         = parse_entities(section_stream);
            if (!result)
            {
                return std::unexpected(result.error());
            }
            entities = std::move(result.value());
            seen_sections.insert(make_entity_key(8U, 8U));
        }
        else if (trimmed == "$Nodes")
        {
            auto section_stream = read_section(input, "$EndNodes");
            auto result         = parse_nodes(section_stream, entities);
            if (!result)
            {
                return std::unexpected(result.error());
            }
            mesh.nodes       = std::move(result->nodes);
            node_lookup      = std::move(result->id_to_index);
            mesh.node_groups = std::move(result->nodes_by_group);
            for (const auto &[group_id, _] : mesh.node_groups)
            {
                referenced_group_ids.insert(group_id);
            }
            seen_sections.insert(make_entity_key(7U, 7U));
        }
        else if (trimmed == "$Elements")
        {
            auto section_stream = read_section(input, "$EndElements");
            auto result         = parse_elements(section_stream, node_lookup, entities);
            if (!result)
            {
                return std::unexpected(result.error());
            }
            mesh.elements       = std::move(result->volume_elements);
            mesh.surfaces       = std::move(result->surface_elements);
            mesh.surface_groups = std::move(result->surface_groups);
            referenced_group_ids.insert(result->used_physical_ids.begin(), result->used_physical_ids.end());
            seen_sections.insert(make_entity_key(6U, 6U));
        }
    }

    if (!seen_sections.contains(make_entity_key(7U, 7U)))
    {
        return std::unexpected(MeshError{"missing $Nodes section", {}});
    }
    if (!seen_sections.contains(make_entity_key(6U, 6U)))
    {
        return std::unexpected(MeshError{"missing $Elements section", {}});
    }

    std::unordered_map<std::uint32_t, PhysicalGroup> group_map;
    for (const auto &[key, name] : physical_names.names)
    {
        const auto dimension = static_cast<std::uint32_t>(key >> 32U);
        const auto tag       = static_cast<std::uint32_t>(key & 0xFFFFFFFFU);
        group_map.emplace(tag, PhysicalGroup{dimension, tag, name});
    }
    for (const auto &[phys_id, dimension] : entities.physical_dimensions)
    {
        auto &group = group_map[phys_id];
        if (group.id == 0U)
        {
            group.id        = phys_id;
            group.dimension = dimension;
            group.name      = "";
        }
        else
        {
            group.dimension = dimension;
        }
    }
    for (const auto group_id : referenced_group_ids)
    {
        auto &group = group_map[group_id];
        if (group.id == 0U)
        {
            group.id        = group_id;
            group.dimension = entities.physical_dimensions.contains(group_id)
                                  ? entities.physical_dimensions.at(group_id)
                                  : 0U;
            group.name      = "";
        }
    }
    mesh.physical_groups.reserve(group_map.size());
    for (auto &[id, group] : group_map)
    {
        mesh.group_lookup.emplace(id, mesh.physical_groups.size());
        mesh.physical_groups.push_back(std::move(group));
    }

    return mesh;
}

} // namespace cwf::mesh
