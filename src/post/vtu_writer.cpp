/**
 * @file vtu_writer.cpp
 * @brief implementation for the binary VTU export pipeline uwu
 */
#include "cwf/post/vtu_writer.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>

namespace cwf::post
{
namespace
{

constexpr std::size_t kVoigtComponents = 6U;
constexpr std::size_t kVec3Components = 3U;
constexpr std::uint8_t kVtkTetra = 10U;
constexpr std::uint8_t kVtkHex = 12U;

struct DataArraySpec
{
    std::string name;
    std::string type;
    std::uint32_t components;
    std::size_t offset{0U};
};

[[nodiscard]] auto make_error(std::string message, std::initializer_list<std::string> ctx = {}) -> VtuError
{
    VtuError err{};
    err.message = std::move(message);
    err.context.assign(ctx.begin(), ctx.end());
    return err;
}

[[nodiscard]] auto flatten_float3(const mesh::pack::Float3SoA &soa) -> std::vector<float>
{
    std::vector<float> flat;
    flat.resize(soa.x.size() * kVec3Components);
    for (std::size_t node = 0; node < soa.x.size(); ++node)
    {
        const auto base = node * kVec3Components;
        flat[base + 0U] = soa.x[node];
        flat[base + 1U] = soa.y[node];
        flat[base + 2U] = soa.z[node];
    }
    return flat;
}

[[nodiscard]] auto flatten_deformed_points(const mesh::pack::PackingResult &packing) -> std::vector<float>
{
    const auto node_count = packing.metadata.node_count;
    std::vector<float> points(node_count * kVec3Components, 0.0F);
    for (std::size_t node = 0; node < node_count; ++node)
    {
        const auto base = node * kVec3Components;
        points[base + 0U] = packing.buffers.nodes.position0.x[node] + packing.buffers.nodes.displacement.x[node];
        points[base + 1U] = packing.buffers.nodes.position0.y[node] + packing.buffers.nodes.displacement.y[node];
        points[base + 2U] = packing.buffers.nodes.position0.z[node] + packing.buffers.nodes.displacement.z[node];
    }
    return points;
}

[[nodiscard]] auto flatten_tensor_field(const auto &source) -> std::vector<float>
{
    std::vector<float> result(source.size() * kVoigtComponents, 0.0F);
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        const auto base = index * kVoigtComponents;
        for (std::size_t comp = 0; comp < kVoigtComponents; ++comp)
        {
            result[base + comp] = source[index].strain[comp];
        }
    }
    return result;
}

[[nodiscard]] auto flatten_tensor_field_stress(const auto &source) -> std::vector<float>
{
    std::vector<float> result(source.size() * kVoigtComponents, 0.0F);
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        const auto base = index * kVoigtComponents;
        for (std::size_t comp = 0; comp < kVoigtComponents; ++comp)
        {
            result[base + comp] = source[index].stress[comp];
        }
    }
    return result;
}

[[nodiscard]] auto flatten_scalar_field(const auto &source) -> std::vector<float>
{
    std::vector<float> result(source.size(), 0.0F);
    for (std::size_t index = 0; index < source.size(); ++index)
    {
        result[index] = source[index].von_mises;
    }
    return result;
}

[[nodiscard]] auto build_connectivity(const mesh::Mesh &mesh) -> std::tuple<std::vector<std::int32_t>,
                                                                            std::vector<std::int32_t>,
                                                                            std::vector<std::uint8_t>>
{
    std::vector<std::int32_t> connectivity;
    std::vector<std::int32_t> offsets;
    std::vector<std::uint8_t> types;
    connectivity.reserve(mesh.elements.size() * 8U);
    offsets.reserve(mesh.elements.size());
    types.reserve(mesh.elements.size());

    std::int32_t running_offset = 0;
    for (const auto &element : mesh.elements)
    {
        const std::size_t local_count = element.geometry == mesh::ElementGeometry::Tetrahedron4 ? 4U : 8U;
        for (std::size_t local = 0; local < local_count; ++local)
        {
            connectivity.push_back(static_cast<std::int32_t>(element.nodes[local]));
        }
        running_offset += static_cast<std::int32_t>(local_count);
        offsets.push_back(running_offset);
        types.push_back(element.geometry == mesh::ElementGeometry::Tetrahedron4 ? kVtkTetra : kVtkHex);
    }

    return {std::move(connectivity), std::move(offsets), std::move(types)};
}

[[nodiscard]] auto append_block(std::vector<std::uint8_t> &blob, const void *data, std::size_t byte_count)
    -> std::size_t
{
    if (byte_count > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()))
    {
        throw std::runtime_error("VTU block exceeds UInt32 header limit");
    }
    const std::size_t offset = blob.size();
    const std::uint32_t payload_size = static_cast<std::uint32_t>(byte_count);
    const auto *size_ptr = reinterpret_cast<const std::uint8_t *>(&payload_size);
    blob.insert(blob.end(), size_ptr, size_ptr + sizeof(std::uint32_t));
    const auto *bytes = reinterpret_cast<const std::uint8_t *>(data);
    blob.insert(blob.end(), bytes, bytes + byte_count);
    return offset;
}

[[nodiscard]] auto append_block(std::vector<std::uint8_t> &blob, const std::vector<float> &data) -> std::size_t
{
    return append_block(blob, data.data(), data.size() * sizeof(float));
}

[[nodiscard]] auto append_block(std::vector<std::uint8_t> &blob, const std::vector<std::int32_t> &data) -> std::size_t
{
    return append_block(blob, data.data(), data.size() * sizeof(std::int32_t));
}

[[nodiscard]] auto append_block(std::vector<std::uint8_t> &blob, const std::vector<std::uint8_t> &data) -> std::size_t
{
    return append_block(blob, data.data(), data.size() * sizeof(std::uint8_t));
}

} // namespace

auto write_vtu(const std::filesystem::path &path,
               const mesh::Mesh &mesh,
               const mesh::pack::PackingResult &packing,
               const DerivedFieldSet &derived,
               double simulation_time,
               std::uint32_t frame_index) -> std::expected<void, VtuError>
{
    try
    {
        if (!path.parent_path().empty())
        {
            std::filesystem::create_directories(path.parent_path());
        }

        std::ofstream file(path, std::ios::binary);
        if (!file)
        {
            return std::unexpected(make_error("failed to open VTU file", {path.string()}));
        }

        const auto points = flatten_deformed_points(packing);
        const auto displacement = flatten_float3(packing.buffers.nodes.displacement);
        const auto velocity = flatten_float3(packing.buffers.nodes.velocity);
        const auto acceleration = flatten_float3(packing.buffers.nodes.acceleration);

        const auto node_strain = flatten_tensor_field(derived.nodes);
        const auto node_stress = flatten_tensor_field_stress(derived.nodes);
        const auto node_vm = flatten_scalar_field(derived.nodes);
        const auto elem_strain = flatten_tensor_field(derived.elements);
        const auto elem_stress = flatten_tensor_field_stress(derived.elements);
        const auto elem_vm = flatten_scalar_field(derived.elements);

        auto [connectivity, offsets, types] = build_connectivity(mesh);

        std::vector<std::uint8_t> appended;
        appended.reserve(4U * (points.size() + displacement.size()));

        DataArraySpec point_arrays[] = {
            {.name = "displacement", .type = "Float32", .components = 3U},
            {.name = "velocity", .type = "Float32", .components = 3U},
            {.name = "acceleration", .type = "Float32", .components = 3U},
            {.name = "strain_node", .type = "Float32", .components = 6U},
            {.name = "stress_node", .type = "Float32", .components = 6U},
            {.name = "von_mises_node", .type = "Float32", .components = 1U},
        };

        point_arrays[0].offset = append_block(appended, displacement);
        point_arrays[1].offset = append_block(appended, velocity);
        point_arrays[2].offset = append_block(appended, acceleration);
        point_arrays[3].offset = append_block(appended, node_strain);
        point_arrays[4].offset = append_block(appended, node_stress);
        point_arrays[5].offset = append_block(appended, node_vm);

        DataArraySpec cell_arrays[] = {
            {.name = "strain_elem", .type = "Float32", .components = 6U},
            {.name = "stress_elem", .type = "Float32", .components = 6U},
            {.name = "von_mises_elem", .type = "Float32", .components = 1U},
        };

        cell_arrays[0].offset = append_block(appended, elem_strain);
        cell_arrays[1].offset = append_block(appended, elem_stress);
        cell_arrays[2].offset = append_block(appended, elem_vm);

        const auto points_offset = append_block(appended, points);
        const auto connectivity_offset = append_block(appended, connectivity);
        const auto offsets_offset = append_block(appended, offsets);
        const auto types_offset = append_block(appended, types);

        file << "<?xml version=\"1.0\"?>\n";
        file << "<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt32\">\n";
        file << "  <UnstructuredGrid>\n";
        file << "    <FieldData>\n";
        file << "      <DataArray type=\"Float64\" Name=\"time\" NumberOfTuples=\"1\">"
             << simulation_time << "</DataArray>\n";
        file << "      <DataArray type=\"UInt32\" Name=\"frame\" NumberOfTuples=\"1\">"
             << frame_index << "</DataArray>\n";
        file << "    </FieldData>\n";
        file << "    <Piece NumberOfPoints=\"" << packing.metadata.node_count << "\" NumberOfCells=\""
             << mesh.elements.size() << "\">\n";

        file << "      <PointData Scalars=\"von_mises_node\">\n";
        for (const auto &spec : point_arrays)
        {
            file << "        <DataArray type=\"" << spec.type << "\" Name=\"" << spec.name
                 << "\" NumberOfComponents=\"" << spec.components << "\" format=\"appended\" offset=\""
                 << spec.offset << "\"/>\n";
        }
        file << "      </PointData>\n";

        file << "      <CellData Scalars=\"von_mises_elem\">\n";
        for (const auto &spec : cell_arrays)
        {
            file << "        <DataArray type=\"" << spec.type << "\" Name=\"" << spec.name
                 << "\" NumberOfComponents=\"" << spec.components << "\" format=\"appended\" offset=\""
                 << spec.offset << "\"/>\n";
        }
        file << "      </CellData>\n";

        file << "      <Points>\n";
        file << "        <DataArray type=\"Float32\" NumberOfComponents=\"3\" format=\"appended\" offset=\""
             << points_offset << "\"/>\n";
        file << "      </Points>\n";

        file << "      <Cells>\n";
        file << "        <DataArray type=\"Int32\" Name=\"connectivity\" format=\"appended\" offset=\""
             << connectivity_offset << "\"/>\n";
        file << "        <DataArray type=\"Int32\" Name=\"offsets\" format=\"appended\" offset=\""
             << offsets_offset << "\"/>\n";
        file << "        <DataArray type=\"UInt8\" Name=\"types\" format=\"appended\" offset=\""
             << types_offset << "\"/>\n";
        file << "      </Cells>\n";

        file << "    </Piece>\n";
        file << "  </UnstructuredGrid>\n";
        file << "  <AppendedData encoding=\"raw\">\n";
        file << "_";
        file.write(reinterpret_cast<const char *>(appended.data()), static_cast<std::streamsize>(appended.size()));
        file << "\n  </AppendedData>\n";
        file << "</VTKFile>\n";

        return {};
    }
    catch (const std::exception &ex)
    {
        return std::unexpected(make_error(ex.what(), {path.string()}));
    }
}

} // namespace cwf::post
