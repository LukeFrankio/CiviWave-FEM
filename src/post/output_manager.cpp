/**
 * @file output_manager.cpp
 * @brief orchestration logic for Phase 10 exports uwu
 */
#include "cwf/post/output_manager.hpp"

#include <format>
#include <string>
#include <string_view>

namespace cwf::post
{
namespace
{

[[nodiscard]] auto make_error(std::string message, std::initializer_list<std::string> ctx = {}) -> OutputError
{
    OutputError err{};
    err.message = std::move(message);
    err.context.assign(ctx.begin(), ctx.end());
    return err;
}

template <typename ErrorPayload>
[[nodiscard]] auto wrap_error(std::string_view label, const ErrorPayload &child_error) -> OutputError
{
    OutputError err{};
    err.message = std::string(label) + ": " + child_error.message;
    err.context = child_error.context;
    return err;
}

} // namespace

OutputManager::OutputManager(std::filesystem::path root,
                             const mesh::Mesh &mesh,
                             mesh::pack::PackingResult &packing,
                             std::span<const physics::materials::ElasticProperties> materials,
                             config::OutputSettings settings)
    : root_{std::move(root)},
      mesh_{&mesh},
      packing_{&packing},
      materials_{materials},
      settings_{settings},
      probe_logger_{root_ / "probes" / "probes.csv", settings.probes}
{
}

[[nodiscard]] auto OutputManager::write_vtu_frame(const DerivedFieldSet &derived,
                                                  double simulation_time,
                                                  std::uint32_t frame_index) -> std::expected<void, OutputError>
{
    if (settings_.vtu_stride == 0U)
    {
        return {};
    }
    if (frame_index % settings_.vtu_stride != 0U)
    {
        return {};
    }

    const auto filename = std::format("frame_{:05}.vtu", frame_index);
    const auto path = root_ / "vtu" / filename;
    if (auto write = post::write_vtu(path, *mesh_, *packing_, derived, simulation_time, frame_index); !write)
    {
        return std::unexpected(wrap_error("vtu", write.error()));
    }
    return {};
}

[[nodiscard]] auto OutputManager::handle_frame(double simulation_time, std::uint32_t frame_index)
    -> std::expected<void, OutputError>
{
    const auto derived = compute_derived_fields(*packing_, materials_);

    if (auto vtu = write_vtu_frame(derived, simulation_time, frame_index); !vtu)
    {
        return vtu;
    }

    if (auto probes = probe_logger_.log_frame(simulation_time, frame_index, *packing_, derived); !probes)
    {
        return std::unexpected(wrap_error("probes", probes.error()));
    }

    return {};
}

} // namespace cwf::post
