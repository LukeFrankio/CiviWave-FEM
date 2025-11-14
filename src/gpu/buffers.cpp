/**
 * @file buffers.cpp
 * @brief implementation of logical GPU buffer descriptors for Phase 7 uploads
 */

#include "cwf/gpu/buffers.hpp"

#include <algorithm>
#include <bit>
#include <span>

namespace cwf::gpu::buffers
{
namespace
{

template <typename T>
[[nodiscard]] constexpr auto as_bytes(std::span<const T> values) noexcept -> std::span<const std::byte>
{
    return std::as_bytes(values);
}

template <typename T>
[[nodiscard]] auto make_span(const std::vector<T> &values) noexcept -> std::span<const T>
{
    return {values.data(), values.size()};
}

void append_buffer(std::vector<LogicalBuffer> &buffers, std::string_view name, std::span<const std::byte> bytes,
                   std::size_t alignment)
{
    LogicalBuffer logical{};
    logical.name = std::string{name};
    logical.bytes = bytes;
    logical.alignment = alignment;
    buffers.emplace_back(logical);
}

} // namespace

auto build_logical_buffers(const mesh::pack::PackingResult &packing,
                           std::span<const physics::materials::ElasticProperties> materials,
                           PreparedGpuBuffers &prepared, const std::size_t alignment) -> std::vector<LogicalBuffer>
{
    const std::size_t effective_alignment = std::max<std::size_t>(alignment, shard::kDefaultAlignment);

    prepared.material_stiffness_fp32.resize(materials.size() * 36U);
    for(std::size_t mat = 0; mat < materials.size(); ++mat)
    {
        const auto &src = materials[mat].stiffness;
        auto       *dst = prepared.material_stiffness_fp32.data() + mat * 36U;
        for(std::size_t i = 0; i < 36U; ++i)
        {
            dst[i] = static_cast<float>(src[i]);
        }
    }

    prepared.adjacency_local_indices.resize(packing.buffers.adjacency.local_indices.size());
    std::transform(packing.buffers.adjacency.local_indices.begin(), packing.buffers.adjacency.local_indices.end(),
                   prepared.adjacency_local_indices.begin(), [](std::uint8_t value) {
                       return static_cast<std::uint32_t>(value);
                   });

    std::vector<LogicalBuffer> buffers{};
    buffers.reserve(16U);

    append_buffer(buffers, "elements.connectivity",
                  as_bytes(make_span(packing.buffers.elements.connectivity)), effective_alignment);
    append_buffer(buffers, "elements.gradients", as_bytes(make_span(packing.buffers.elements.gradients)),
                  effective_alignment);
    append_buffer(buffers, "elements.volume", as_bytes(make_span(packing.buffers.elements.volume)), effective_alignment);
    append_buffer(buffers, "elements.material_index",
                  as_bytes(make_span(packing.buffers.elements.material_index)), effective_alignment);

    append_buffer(buffers, "materials.stiffness", as_bytes(make_span(prepared.material_stiffness_fp32)),
                  effective_alignment);

    append_buffer(buffers, "nodes.bc_mask", as_bytes(make_span(packing.buffers.nodes.bc_mask)), effective_alignment);
    append_buffer(buffers, "nodes.lumped_mass", as_bytes(make_span(packing.buffers.nodes.lumped_mass)),
                  effective_alignment);

    append_buffer(buffers, "adjacency.offsets", as_bytes(make_span(packing.buffers.adjacency.offsets)),
                  effective_alignment);
    append_buffer(buffers, "adjacency.indices", as_bytes(make_span(packing.buffers.adjacency.element_indices)),
                  effective_alignment);
    append_buffer(buffers, "adjacency.local_indices", as_bytes(make_span(prepared.adjacency_local_indices)),
                  effective_alignment);

    append_buffer(buffers, "solver.vector.p", as_bytes(make_span(packing.buffers.solver.p)), effective_alignment);
    append_buffer(buffers, "solver.vector.r", as_bytes(make_span(packing.buffers.solver.r)), effective_alignment);
    append_buffer(buffers, "solver.vector.Ap", as_bytes(make_span(packing.buffers.solver.Ap)), effective_alignment);
    append_buffer(buffers, "solver.vector.z", as_bytes(make_span(packing.buffers.solver.z)), effective_alignment);
    append_buffer(buffers, "solver.vector.x", as_bytes(make_span(packing.buffers.solver.x)), effective_alignment);
    append_buffer(buffers, "solver.partials", as_bytes(make_span(packing.buffers.solver.partials)), effective_alignment);
    append_buffer(buffers, "solver.block_inverse", as_bytes(make_span(packing.buffers.solver.block_inverse)),
                  effective_alignment);

    return buffers;
}

auto make_shard_specs(const std::vector<LogicalBuffer> &buffers) -> std::vector<shard::BufferSpecification>
{
    std::vector<shard::BufferSpecification> specs;
    specs.reserve(buffers.size());
    for(const auto &logical : buffers)
    {
        shard::BufferSpecification spec{};
        spec.name = logical.name;
        spec.size_bytes = logical.bytes.size();
        spec.alignment = logical.alignment;
        specs.emplace_back(spec);
    }
    return specs;
}

auto make_upload_views(const std::vector<LogicalBuffer> &buffers) -> std::vector<upload::BufferView>
{
    std::vector<upload::BufferView> views;
    views.reserve(buffers.size());
    for(const auto &logical : buffers)
    {
        upload::BufferView view{};
        view.name = logical.name;
        view.bytes = logical.bytes;
        views.emplace_back(view);
    }
    return views;
}

} // namespace cwf::gpu::buffers
