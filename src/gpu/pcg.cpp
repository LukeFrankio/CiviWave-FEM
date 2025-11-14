/**
 * @file pcg.cpp
 * @brief implementation for Phase 8 matrix-free K_eff apply + PCG solver uwu
 */
#include "cwf/gpu/pcg.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <initializer_list>
#include <new>
#include <numeric>

namespace cwf::gpu::pcg
{
namespace
{

[[nodiscard]] auto make_error(std::string message, std::initializer_list<std::string> ctx) -> PcgError
{
    PcgError err{};
    err.message = std::move(message);
    err.context.assign(ctx.begin(), ctx.end());
    return err;
}

[[nodiscard]] constexpr auto axis_bit(std::size_t axis) noexcept -> std::uint32_t
{
    return static_cast<std::uint32_t>(1U << static_cast<unsigned int>(axis));
}

[[nodiscard]] constexpr auto is_axis_constrained(std::uint32_t mask, std::size_t axis) noexcept -> bool
{
    return (mask & axis_bit(axis)) != 0U;
}

[[nodiscard]] constexpr auto dof_index(std::size_t node, std::size_t axis) noexcept -> std::size_t
{
    return node * 3U + axis;
}

[[nodiscard]] auto ensure_workspace(MatrixFreeWorkspace &workspace, const MatrixFreeSystem &system)
    -> std::expected<void, PcgError>
{
    try
    {
        if (workspace.sanitized_input.size() < system.dof_count)
        {
            workspace.sanitized_input.resize(system.dof_count);
        }
        if (workspace.accumulation.size() < system.dof_count)
        {
            workspace.accumulation.resize(system.dof_count);
        }
        const auto block_entries = system.node_count * 9U;
        if (workspace.block_buffer.size() < block_entries)
        {
            workspace.block_buffer.resize(block_entries);
        }
        if (workspace.block_inverse.size() < block_entries)
        {
            workspace.block_inverse.resize(block_entries);
        }
    }
    catch (const std::bad_alloc &)
    {
        return std::unexpected(make_error("failed to grow matrix-free workspace buffers", {"dofs=" + std::to_string(system.dof_count)}));
    }
    return {};
}

[[nodiscard]] constexpr auto element_gradient_offset(std::size_t element_index) noexcept -> std::size_t
{
    return element_index * 8U * 3U;
}

[[nodiscard]] constexpr auto connectivity_offset(std::size_t element_index) noexcept -> std::size_t
{
    return element_index * 8U;
}

[[nodiscard]] auto validate_system(const MatrixFreeSystem &system) -> std::expected<void, PcgError>
{
    if (system.element_connectivity.size() != system.element_count * 8U)
    {
        return std::unexpected(make_error("connectivity size mismatch",
                                          {"expected=" + std::to_string(system.element_count * 8U),
                                           "actual=" + std::to_string(system.element_connectivity.size())}));
    }
    if (system.element_gradients.size() != system.element_count * 8U * 3U)
    {
        return std::unexpected(make_error("gradient table size mismatch",
                                          {"expected=" + std::to_string(system.element_count * 8U * 3U),
                                           "actual=" + std::to_string(system.element_gradients.size())}));
    }
    if (system.element_volume.size() != system.element_count)
    {
        return std::unexpected(make_error("volume table size mismatch",
                                          {"expected=" + std::to_string(system.element_count),
                                           "actual=" + std::to_string(system.element_volume.size())}));
    }
    if (system.element_material_index.size() != system.element_count)
    {
        return std::unexpected(make_error("material index table size mismatch",
                                          {"expected=" + std::to_string(system.element_count),
                                           "actual=" + std::to_string(system.element_material_index.size())}));
    }
    if (system.bc_mask.size() != system.node_count)
    {
        return std::unexpected(make_error("bc mask size mismatch",
                                          {"expected=" + std::to_string(system.node_count),
                                           "actual=" + std::to_string(system.bc_mask.size())}));
    }
    if (system.lumped_mass.size() != system.node_count)
    {
        return std::unexpected(make_error("lumped mass size mismatch",
                                          {"expected=" + std::to_string(system.node_count),
                                           "actual=" + std::to_string(system.lumped_mass.size())}));
    }
    if (system.dof_count != system.node_count * 3U)
    {
        return std::unexpected(make_error("dof count mismatch (expected node_count * 3)",
                                          {"node_count=" + std::to_string(system.node_count),
                                           "dof_count=" + std::to_string(system.dof_count)}));
    }
    if (system.materials.empty())
    {
        return std::unexpected(make_error("materials table is empty", {}));
    }
    if (system.reduction_block == 0U)
    {
        return std::unexpected(make_error("reduction block must be >= 1", {"reduction_block=0"}));
    }
    if (system.reduction_partials == 0U)
    {
        return std::unexpected(make_error("reduction partial count must be >= 1", {"reduction_partials=0"}));
    }
    return {};
}

[[nodiscard]] auto accumulator_view(MatrixFreeWorkspace &workspace, const MatrixFreeSystem &system)
    -> std::span<double>
{
    return std::span<double>{workspace.accumulation.data(), system.dof_count};
}

[[nodiscard]] auto sanitized_view(MatrixFreeWorkspace &workspace, const MatrixFreeSystem &system)
    -> std::span<double>
{
    return std::span<double>{workspace.sanitized_input.data(), system.dof_count};
}

[[nodiscard]] auto block_buffer_view(MatrixFreeWorkspace &workspace, const MatrixFreeSystem &system)
    -> std::span<double>
{
    return std::span<double>{workspace.block_buffer.data(), system.node_count * 9U};
}

[[nodiscard]] auto block_inverse_view(MatrixFreeWorkspace &workspace, const MatrixFreeSystem &system)
    -> std::span<float>
{
    return std::span<float>{workspace.block_inverse.data(), system.node_count * 9U};
}

[[nodiscard]] constexpr auto chunk_count(std::size_t count, std::size_t block) noexcept -> std::size_t
{
    return (count + block - 1U) / block;
}

[[nodiscard]] auto dot_accumulate(const MatrixFreeSystem &system, std::span<const float> a,
                                  std::span<const float> b, std::span<double> partials)
    -> std::expected<double, PcgError>
{
    if (a.size() != b.size())
    {
        return std::unexpected(make_error("dot product span size mismatch",
                                          {"lhs=" + std::to_string(a.size()),
                                           "rhs=" + std::to_string(b.size())}));
    }
    const auto block = std::max<std::size_t>(1U, system.reduction_block);
    const auto required_partials = chunk_count(a.size(), block);
    if (partials.size() < required_partials)
    {
        return std::unexpected(make_error("partials span too small for reduction",
                                          {"required=" + std::to_string(required_partials),
                                           "available=" + std::to_string(partials.size())}));
    }

    double total = 0.0;
    for (std::size_t chunk = 0; chunk < required_partials; ++chunk)
    {
        const auto begin = chunk * block;
        const auto end = std::min(begin + block, a.size());
        double     accumulator = 0.0;
        for (std::size_t idx = begin; idx < end; ++idx)
        {
            accumulator += static_cast<double>(a[idx]) * static_cast<double>(b[idx]);
        }
        partials[chunk] = accumulator;
        total += accumulator;
    }
    for (std::size_t chunk = required_partials; chunk < partials.size(); ++chunk)
    {
        partials[chunk] = 0.0;
    }
    return total;
}

[[nodiscard]] auto norm_accumulate(const MatrixFreeSystem &system, std::span<const float> v,
                                   std::span<double> partials) -> std::expected<double, PcgError>
{
    return dot_accumulate(system, v, v, partials);
}

[[nodiscard]] auto invert_spd_3x3(std::array<double, 9> matrix) -> std::array<double, 9>
{
    constexpr double kDetTol = 1.0e-12;

    auto determinant = [&]() {
        const double a = matrix[0];
        const double b = matrix[1];
        const double c = matrix[2];
        const double d = matrix[3];
        const double e = matrix[4];
        const double f = matrix[5];
        const double g = matrix[6];
        const double h = matrix[7];
        const double i = matrix[8];
        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    };

    auto add_regularization = [&]() {
        const double max_diag = std::max({matrix[0], matrix[4], matrix[8]});
        const double epsilon = std::max(1.0e-6, max_diag * 1.0e-6 + 1.0e-12);
        matrix[0] += epsilon;
        matrix[4] += epsilon;
        matrix[8] += epsilon;
    };

    double det = determinant();
    if (std::abs(det) < kDetTol)
    {
        add_regularization();
        det = determinant();
    }

    if (std::abs(det) < kDetTol)
    {
        std::array<double, 9> inverse{};
        inverse[0] = 1.0 / std::max(matrix[0], 1.0e-6);
        inverse[4] = 1.0 / std::max(matrix[4], 1.0e-6);
        inverse[8] = 1.0 / std::max(matrix[8], 1.0e-6);
        return inverse;
    }

    const double inv_det = 1.0 / det;
    std::array<double, 9> result{};
    result[0] = (matrix[4] * matrix[8] - matrix[5] * matrix[7]) * inv_det;
    result[1] = (matrix[2] * matrix[7] - matrix[1] * matrix[8]) * inv_det;
    result[2] = (matrix[1] * matrix[5] - matrix[2] * matrix[4]) * inv_det;
    result[3] = (matrix[5] * matrix[6] - matrix[3] * matrix[8]) * inv_det;
    result[4] = (matrix[0] * matrix[8] - matrix[2] * matrix[6]) * inv_det;
    result[5] = (matrix[2] * matrix[3] - matrix[0] * matrix[5]) * inv_det;
    result[6] = (matrix[3] * matrix[7] - matrix[4] * matrix[6]) * inv_det;
    result[7] = (matrix[1] * matrix[6] - matrix[0] * matrix[7]) * inv_det;
    result[8] = (matrix[0] * matrix[4] - matrix[1] * matrix[3]) * inv_det;
    return result;
}

[[nodiscard]] auto prepare_block_jacobi(const MatrixFreeSystem &system, MatrixFreeWorkspace &workspace)
    -> std::expected<void, PcgError>
{
    auto buffer = block_buffer_view(workspace, system);
    std::fill(buffer.begin(), buffer.end(), 0.0);

    constexpr std::size_t kLocalNodes = 4U;
    constexpr std::size_t kLocalDofs = 12U;
    constexpr std::size_t kStrain = 6U;

    for (std::size_t element = 0; element < system.element_count; ++element)
    {
        const auto connectivity_base = connectivity_offset(element);
        const auto gradient_base = element_gradient_offset(element);
        const auto material_index = system.element_material_index[element];
        if (material_index >= system.materials.size())
        {
            return std::unexpected(make_error("element references material out of range",
                                              {"element=" + std::to_string(element),
                                               "material_index=" + std::to_string(material_index)}));
        }

        std::array<double, kStrain * kLocalDofs> B{};
        for (std::size_t local = 0; local < kLocalNodes; ++local)
        {
            const auto grad_index = gradient_base + local * 3U;
            const double gx = static_cast<double>(system.element_gradients[grad_index + 0U]);
            const double gy = static_cast<double>(system.element_gradients[grad_index + 1U]);
            const double gz = static_cast<double>(system.element_gradients[grad_index + 2U]);
            const auto col = local * 3U;
            B[0U * kLocalDofs + col + 0U] = gx;
            B[1U * kLocalDofs + col + 1U] = gy;
            B[2U * kLocalDofs + col + 2U] = gz;
            B[3U * kLocalDofs + col + 0U] = gy;
            B[3U * kLocalDofs + col + 1U] = gx;
            B[4U * kLocalDofs + col + 1U] = gz;
            B[4U * kLocalDofs + col + 2U] = gy;
            B[5U * kLocalDofs + col + 0U] = gz;
            B[5U * kLocalDofs + col + 2U] = gx;
        }

        std::array<double, kStrain * kLocalDofs> DB{};
        const auto &stiffness = system.materials[material_index].stiffness;
        for (std::size_t row = 0; row < kStrain; ++row)
        {
            for (std::size_t col = 0; col < kLocalDofs; ++col)
            {
                double sum = 0.0;
                for (std::size_t mid = 0; mid < kStrain; ++mid)
                {
                    sum += stiffness[row * kStrain + mid] * B[mid * kLocalDofs + col];
                }
                DB[row * kLocalDofs + col] = sum;
            }
        }

        std::array<double, kLocalDofs * kLocalDofs> ke{};
        for (std::size_t i = 0; i < kLocalDofs; ++i)
        {
            for (std::size_t j = 0; j < kLocalDofs; ++j)
            {
                double sum = 0.0;
                for (std::size_t row = 0; row < kStrain; ++row)
                {
                    sum += B[row * kLocalDofs + i] * DB[row * kLocalDofs + j];
                }
                ke[i * kLocalDofs + j] = sum;
            }
        }

        const double volume = static_cast<double>(system.element_volume[element]);
        const double scaled_volume = volume * system.stiffness_scale;

        for (double &value : ke)
        {
            value *= scaled_volume;
        }

        for (std::size_t local = 0; local < kLocalNodes; ++local)
        {
            const auto node_index = system.element_connectivity[connectivity_base + local];
            if (node_index >= system.node_count)
            {
                return std::unexpected(make_error("element connectivity references node out of range",
                                                  {"element=" + std::to_string(element),
                                                   "node=" + std::to_string(node_index)}));
            }
            const auto block_base = node_index * 9U;
            for (std::size_t axis_i = 0; axis_i < 3U; ++axis_i)
            {
                const auto local_i = local * 3U + axis_i;
                for (std::size_t axis_j = 0; axis_j < 3U; ++axis_j)
                {
                    const auto local_j = local * 3U + axis_j;
                    buffer[block_base + axis_i * 3U + axis_j] += ke[local_i * kLocalDofs + local_j];
                }
            }
        }
    }

    for (std::size_t node = 0; node < system.node_count; ++node)
    {
        const double mass = static_cast<double>(system.lumped_mass[node]) * system.mass_factor;
        const auto   block_base = node * 9U;
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            buffer[block_base + axis * 3U + axis] += mass;
        }
    }

    auto inverse = block_inverse_view(workspace, system);
    for (std::size_t node = 0; node < system.node_count; ++node)
    {
        std::array<double, 9> block{};
        const auto block_base = node * 9U;
        for (std::size_t i = 0; i < 9U; ++i)
        {
            block[i] = buffer[block_base + i];
        }
        auto inv = invert_spd_3x3(block);
        const auto mask = system.bc_mask[node];
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            if (is_axis_constrained(mask, axis))
            {
                for (std::size_t column = 0; column < 3U; ++column)
                {
                    inv[axis * 3U + column] = (axis == column) ? 1.0 : 0.0;
                }
            }
        }
        for (std::size_t i = 0; i < 9U; ++i)
        {
            inverse[block_base + i] = static_cast<float>(inv[i]);
        }
    }

    return {};
}

auto apply_preconditioner(const MatrixFreeSystem &system, std::span<const float> residual, std::span<float> out,
                          MatrixFreeWorkspace &workspace) -> std::expected<void, PcgError>
{
    if (residual.size() != system.dof_count || out.size() != system.dof_count)
    {
        return std::unexpected(make_error("preconditioner span size mismatch",
                                          {"residual=" + std::to_string(residual.size()),
                                           "out=" + std::to_string(out.size()),
                                           "dofs=" + std::to_string(system.dof_count)}));
    }

    auto inverse = block_inverse_view(workspace, system);
    for (std::size_t node = 0; node < system.node_count; ++node)
    {
        const auto block_base = node * 9U;
        const auto dof_base = node * 3U;
        std::array<double, 3> r{
            static_cast<double>(residual[dof_base + 0U]),
            static_cast<double>(residual[dof_base + 1U]),
            static_cast<double>(residual[dof_base + 2U])
        };
        std::array<double, 3> z{};
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            double sum = 0.0;
            for (std::size_t column = 0; column < 3U; ++column)
            {
                sum += static_cast<double>(inverse[block_base + axis * 3U + column]) * r[column];
            }
            z[axis] = sum;
        }
        const auto mask = system.bc_mask[node];
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            const auto dof = dof_base + axis;
            if (is_axis_constrained(mask, axis))
            {
                out[dof] = 0.0F;
            }
            else
            {
                out[dof] = static_cast<float>(z[axis]);
            }
        }
    }
    return {};
}

void enforce_dirichlet_solution(const MatrixFreeSystem &system, std::span<const float> rhs,
                                std::span<float> solution, std::span<float> residual)
{
    for (std::size_t node = 0; node < system.node_count; ++node)
    {
        const auto mask = system.bc_mask[node];
        const auto base = node * 3U;
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            const auto dof = base + axis;
            if (is_axis_constrained(mask, axis))
            {
                solution[dof] = rhs[dof];
                residual[dof] = 0.0F;
            }
        }
    }
}

} // namespace

auto build_block_jacobi_inverse(const MatrixFreeSystem &system, MatrixFreeWorkspace &workspace,
                                std::span<float> out_inverse) -> std::expected<void, PcgError>
{
    const auto required = system.node_count * 9U;
    if (out_inverse.size() < required)
    {
        return std::unexpected(make_error("block inverse span too small",
                                          {"required=" + std::to_string(required),
                                           "available=" + std::to_string(out_inverse.size())}));
    }

    if (auto ensured = ensure_workspace(workspace, system); !ensured)
    {
        return ensured;
    }

    if (auto prepared = prepare_block_jacobi(system, workspace); !prepared)
    {
        return prepared;
    }

    auto inverse = block_inverse_view(workspace, system);
    std::copy_n(inverse.begin(), required, out_inverse.begin());
    return {};
}

[[nodiscard]] auto apply_keff(const MatrixFreeSystem &system, std::span<const float> input,
                              std::span<float> output, MatrixFreeWorkspace &workspace)
    -> std::expected<void, PcgError>
{
    if (input.size() != system.dof_count || output.size() != system.dof_count)
    {
        return std::unexpected(make_error("input/output span size mismatch",
                                          {"input=" + std::to_string(input.size()),
                                           "output=" + std::to_string(output.size()),
                                           "dofs=" + std::to_string(system.dof_count)}));
    }

    if (auto valid = validate_system(system); !valid)
    {
        return std::unexpected(valid.error());
    }

    if (auto ensured = ensure_workspace(workspace, system); !ensured)
    {
        return ensured;
    }

    auto sanitized = sanitized_view(workspace, system);
    auto accumulation = accumulator_view(workspace, system);

    for (std::size_t dof = 0; dof < system.dof_count; ++dof)
    {
        sanitized[dof] = static_cast<double>(input[dof]);
    }

    for (std::size_t node = 0; node < system.node_count; ++node)
    {
        const auto mask = system.bc_mask[node];
        const auto base = node * 3U;
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            if (is_axis_constrained(mask, axis))
            {
                sanitized[base + axis] = 0.0;
            }
        }
    }

    std::fill(accumulation.begin(), accumulation.end(), 0.0);

    constexpr std::size_t kLocalNodes = 4U;
    constexpr std::size_t kLocalDofs = 12U;
    constexpr std::size_t kStrain = 6U;

    std::array<double, kStrain * kLocalDofs> B{};
    std::array<double, kStrain * kLocalDofs> DB{};
    std::array<double, kLocalDofs> local_u{};
    std::array<double, kStrain>    strain{};
    std::array<double, kStrain>    stress{};
    std::array<double, kLocalDofs> local_force{};

    for (std::size_t element = 0; element < system.element_count; ++element)
    {
        const auto connectivity_base = connectivity_offset(element);
        const auto gradient_base = element_gradient_offset(element);
        const auto material_index = system.element_material_index[element];
        if (material_index >= system.materials.size())
        {
            return std::unexpected(make_error("element references material out of range",
                                              {"element=" + std::to_string(element),
                                               "material_index=" + std::to_string(material_index)}));
        }

        B.fill(0.0);
        for (std::size_t local = 0; local < kLocalNodes; ++local)
        {
            const auto grad_index = gradient_base + local * 3U;
            const double gx = static_cast<double>(system.element_gradients[grad_index + 0U]);
            const double gy = static_cast<double>(system.element_gradients[grad_index + 1U]);
            const double gz = static_cast<double>(system.element_gradients[grad_index + 2U]);
            const auto col = local * 3U;
            B[0U * kLocalDofs + col + 0U] = gx;
            B[1U * kLocalDofs + col + 1U] = gy;
            B[2U * kLocalDofs + col + 2U] = gz;
            B[3U * kLocalDofs + col + 0U] = gy;
            B[3U * kLocalDofs + col + 1U] = gx;
            B[4U * kLocalDofs + col + 1U] = gz;
            B[4U * kLocalDofs + col + 2U] = gy;
            B[5U * kLocalDofs + col + 0U] = gz;
            B[5U * kLocalDofs + col + 2U] = gx;
        }

        const auto &stiffness = system.materials[material_index].stiffness;
        for (std::size_t row = 0; row < kStrain; ++row)
        {
            for (std::size_t col = 0; col < kLocalDofs; ++col)
            {
                double sum = 0.0;
                for (std::size_t mid = 0; mid < kStrain; ++mid)
                {
                    sum += stiffness[row * kStrain + mid] * B[mid * kLocalDofs + col];
                }
                DB[row * kLocalDofs + col] = sum;
            }
        }

        for (std::size_t local = 0; local < kLocalNodes; ++local)
        {
            const auto node_index = system.element_connectivity[connectivity_base + local];
            if (node_index >= system.node_count)
            {
                return std::unexpected(make_error("element connectivity references node out of range",
                                                  {"element=" + std::to_string(element),
                                                   "node=" + std::to_string(node_index)}));
            }
            const auto dof_base = node_index * 3U;
            for (std::size_t axis = 0; axis < 3U; ++axis)
            {
                local_u[local * 3U + axis] = sanitized[dof_base + axis];
            }
        }

        for (std::size_t row = 0; row < kStrain; ++row)
        {
            double sum = 0.0;
            for (std::size_t col = 0; col < kLocalDofs; ++col)
            {
                sum += B[row * kLocalDofs + col] * local_u[col];
            }
            strain[row] = sum;
        }

        for (std::size_t row = 0; row < kStrain; ++row)
        {
            double sum = 0.0;
            for (std::size_t col = 0; col < kStrain; ++col)
            {
                sum += stiffness[row * kStrain + col] * strain[col];
            }
            stress[row] = sum;
        }

        const double volume = static_cast<double>(system.element_volume[element]) * system.stiffness_scale;
        for (std::size_t col = 0; col < kLocalDofs; ++col)
        {
            double sum = 0.0;
            for (std::size_t row = 0; row < kStrain; ++row)
            {
                sum += B[row * kLocalDofs + col] * stress[row];
            }
            local_force[col] = sum * volume;
        }

        for (std::size_t local = 0; local < kLocalNodes; ++local)
        {
            const auto node_index = system.element_connectivity[connectivity_base + local];
            const auto dof_base = node_index * 3U;
            for (std::size_t axis = 0; axis < 3U; ++axis)
            {
                accumulation[dof_base + axis] += local_force[local * 3U + axis];
            }
        }
    }

    for (std::size_t node = 0; node < system.node_count; ++node)
    {
        const double mass = static_cast<double>(system.lumped_mass[node]) * system.mass_factor;
        const auto   dof_base = node * 3U;
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            accumulation[dof_base + axis] += mass * sanitized[dof_base + axis];
        }
    }

    for (std::size_t node = 0; node < system.node_count; ++node)
    {
        const auto mask = system.bc_mask[node];
        const auto base = node * 3U;
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            const auto dof = base + axis;
            if (is_axis_constrained(mask, axis))
            {
                accumulation[dof] = static_cast<double>(input[dof]);
            }
        }
    }

    for (std::size_t dof = 0; dof < system.dof_count; ++dof)
    {
        output[dof] = static_cast<float>(accumulation[dof]);
    }

    return {};
}

[[nodiscard]] auto solve_pcg(const MatrixFreeSystem &system, std::span<const float> rhs,
                             const PcgSettings &settings, PcgVectors vectors, MatrixFreeWorkspace &workspace)
    -> std::expected<PcgTelemetry, PcgError>
{
    if (auto valid = validate_system(system); !valid)
    {
        return std::unexpected(valid.error());
    }

    if (rhs.size() != system.dof_count)
    {
        return std::unexpected(make_error("rhs span size mismatch",
                                          {"rhs=" + std::to_string(rhs.size()),
                                           "dofs=" + std::to_string(system.dof_count)}));
    }

    if (vectors.solution.size() != system.dof_count || vectors.residual.size() != system.dof_count ||
        vectors.search_direction.size() != system.dof_count || vectors.preconditioned.size() != system.dof_count ||
        vectors.matvec.size() != system.dof_count)
    {
        return std::unexpected(make_error("solver vector span size mismatch",
                                          {"solution=" + std::to_string(vectors.solution.size()),
                                           "residual=" + std::to_string(vectors.residual.size()),
                                           "search_direction=" + std::to_string(vectors.search_direction.size()),
                                           "preconditioned=" + std::to_string(vectors.preconditioned.size()),
                                           "matvec=" + std::to_string(vectors.matvec.size()),
                                           "dofs=" + std::to_string(system.dof_count)}));
    }

    if (auto ensured = ensure_workspace(workspace, system); !ensured)
    {
        return std::unexpected(ensured.error());
    }

    const auto block = std::max<std::size_t>(1U, system.reduction_block);
    const auto required_partials = chunk_count(system.dof_count, block);
    if (vectors.partials.size() < required_partials)
    {
        return std::unexpected(make_error("partials span insufficient for reductions",
                                          {"required=" + std::to_string(required_partials),
                                           "available=" + std::to_string(vectors.partials.size())}));
    }

    if (settings.max_iterations == 0U)
    {
        return std::unexpected(make_error("max_iterations must be >= 1", {"max_iterations=0"}));
    }

    if (!settings.warm_start)
    {
        std::fill(vectors.solution.begin(), vectors.solution.end(), 0.0F);
    }

    if (auto prepared = prepare_block_jacobi(system, workspace); !prepared)
    {
        return std::unexpected(prepared.error());
    }

    if (auto matvec_status = apply_keff(system, std::span<const float>{vectors.solution.data(), system.dof_count},
                                        vectors.matvec, workspace);
        !matvec_status)
    {
        return std::unexpected(matvec_status.error());
    }

    for (std::size_t dof = 0; dof < system.dof_count; ++dof)
    {
        vectors.residual[dof] = rhs[dof] - vectors.matvec[dof];
    }

    enforce_dirichlet_solution(system, rhs, vectors.solution, vectors.residual);

    auto rhs_norm_sq = norm_accumulate(system, rhs, vectors.partials);
    if (!rhs_norm_sq)
    {
        return std::unexpected(rhs_norm_sq.error());
    }
    double rhs_norm = std::sqrt(rhs_norm_sq.value());
    if (rhs_norm < 1.0e-12)
    {
        rhs_norm = 1.0;
    }

    auto residual_norm_sq = norm_accumulate(system, vectors.residual, vectors.partials);
    if (!residual_norm_sq)
    {
        return std::unexpected(residual_norm_sq.error());
    }
    double residual_norm = std::sqrt(residual_norm_sq.value());

    PcgTelemetry telemetry{};
    telemetry.residual_norm = residual_norm;
    telemetry.rhs_norm = std::sqrt(rhs_norm_sq.value());

    const double tolerance = settings.relative_tolerance * rhs_norm;
    if (residual_norm <= tolerance)
    {
        telemetry.converged = true;
        telemetry.iterations = 0U;
        return telemetry;
    }

    if (auto precond_status = apply_preconditioner(system, vectors.residual, vectors.preconditioned, workspace);
        !precond_status)
    {
        return std::unexpected(precond_status.error());
    }

    auto rho_value = dot_accumulate(system, vectors.residual, vectors.preconditioned, vectors.partials);
    if (!rho_value)
    {
        return std::unexpected(rho_value.error());
    }
    double rho = rho_value.value();
    if (std::abs(rho) < 1.0e-18)
    {
        return std::unexpected(make_error("preconditioner produced near-zero rho", {"rho~0"}));
    }

    std::copy(vectors.preconditioned.begin(), vectors.preconditioned.end(), vectors.search_direction.begin());

    for (std::size_t node = 0; node < system.node_count; ++node)
    {
        const auto mask = system.bc_mask[node];
        const auto base = node * 3U;
        for (std::size_t axis = 0; axis < 3U; ++axis)
        {
            if (is_axis_constrained(mask, axis))
            {
                vectors.search_direction[base + axis] = 0.0F;
            }
        }
    }

    for (std::size_t iteration = 0; iteration < settings.max_iterations; ++iteration)
    {
        auto matvec_result = apply_keff(system,
                                        std::span<const float>{vectors.search_direction.data(), system.dof_count},
                                        vectors.matvec, workspace);
        if (!matvec_result)
        {
            return std::unexpected(matvec_result.error());
        }

        auto denom_value = dot_accumulate(system, vectors.search_direction, vectors.matvec, vectors.partials);
        if (!denom_value)
        {
            return std::unexpected(denom_value.error());
        }
        double denom = denom_value.value();
        if (std::abs(denom) < 1.0e-18)
        {
            return std::unexpected(make_error("CG denominator approached zero", {"iteration=" + std::to_string(iteration)}));
        }

        const double alpha = rho / denom;
        telemetry.alpha_last = alpha;

        for (std::size_t dof = 0; dof < system.dof_count; ++dof)
        {
            vectors.solution[dof] += static_cast<float>(alpha * static_cast<double>(vectors.search_direction[dof]));
            vectors.residual[dof] -= static_cast<float>(alpha * static_cast<double>(vectors.matvec[dof]));
        }

        enforce_dirichlet_solution(system, rhs, vectors.solution, vectors.residual);

        auto res_norm_sq = norm_accumulate(system, vectors.residual, vectors.partials);
        if (!res_norm_sq)
        {
            return std::unexpected(res_norm_sq.error());
        }
        residual_norm = std::sqrt(res_norm_sq.value());
        telemetry.residual_norm = residual_norm;
        telemetry.iterations = iteration + 1U;

        if (residual_norm <= tolerance)
        {
            telemetry.converged = true;
            break;
        }

        if (auto precond_again = apply_preconditioner(system, vectors.residual, vectors.preconditioned, workspace);
            !precond_again)
        {
            return std::unexpected(precond_again.error());
        }

        auto rho_new_value = dot_accumulate(system, vectors.residual, vectors.preconditioned, vectors.partials);
        if (!rho_new_value)
        {
            return std::unexpected(rho_new_value.error());
        }
        const double rho_new = rho_new_value.value();
        if (std::abs(rho) < 1.0e-18)
        {
            return std::unexpected(make_error("CG rho approached zero", {"iteration=" + std::to_string(iteration)}));
        }
        const double beta = rho_new / rho;
        telemetry.beta_last = beta;
        rho = rho_new;

        for (std::size_t dof = 0; dof < system.dof_count; ++dof)
        {
            vectors.search_direction[dof] = static_cast<float>(static_cast<double>(vectors.preconditioned[dof]) +
                                                              beta * static_cast<double>(vectors.search_direction[dof]));
        }

        for (std::size_t node = 0; node < system.node_count; ++node)
        {
            const auto mask = system.bc_mask[node];
            const auto base = node * 3U;
            for (std::size_t axis = 0; axis < 3U; ++axis)
            {
                if (is_axis_constrained(mask, axis))
                {
                    vectors.search_direction[base + axis] = 0.0F;
                }
            }
        }
    }

    return telemetry;
}

} // namespace cwf::gpu::pcg
