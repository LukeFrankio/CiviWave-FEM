/**
 * @file viewer.hpp
 * @brief optional ImGui-based viewer for Phase 10 (build-time toggle) uwu
 */
#pragma once

#include <expected>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "cwf/config/config.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/post/derived_fields.hpp"

namespace cwf::ui
{

struct ViewerError
{
    std::string              message;
    std::vector<std::string> context;
};

/**
 * @brief result from viewer run indicating whether restart with new config is requested
 */
struct ViewerResult
{
    /**
     * @brief path to new config file if user requested mesh change
     *
     * when set, the caller should reload the config and restart the viewer
     */
    std::optional<std::filesystem::path> restart_with_config{};
};

#if defined(CWF_ENABLE_UI) && CWF_ENABLE_UI

[[nodiscard]] auto run_viewer_once(const mesh::Mesh &mesh, mesh::pack::PackingResult packing,
                                   post::DerivedFieldSet                              derived,
                                   std::vector<physics::materials::ElasticProperties> materials,
                                   config::SolverSettings solver_settings, config::TimeSettings time_settings,
                                   physics::materials::RayleighCoefficients rayleigh, double simulation_time,
                                   std::filesystem::path config_directory = {})
    -> std::expected<ViewerResult, ViewerError>;

#else

[[nodiscard]] inline auto
run_viewer_once(const mesh::Mesh &, mesh::pack::PackingResult, post::DerivedFieldSet,
                std::vector<physics::materials::ElasticProperties>, config::SolverSettings,
                config::TimeSettings, physics::materials::RayleighCoefficients, double,
                std::filesystem::path = {}) -> std::expected<ViewerResult, ViewerError>
{
    return std::unexpected(ViewerError{"BUILD_UI=OFF — viewer unavailable", {}});
}

#endif

} // namespace cwf::ui
