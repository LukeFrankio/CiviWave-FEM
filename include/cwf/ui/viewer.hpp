/**
 * @file viewer.hpp
 * @brief optional ImGui-based viewer for Phase 10 (build-time toggle) uwu
 */
#pragma once

#include <expected>
#include <filesystem>

#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/post/derived_fields.hpp"

namespace cwf::ui
{

struct ViewerError
{
    std::string              message;
    std::vector<std::string> context;
};

#if defined(CWF_ENABLE_UI) && CWF_ENABLE_UI

[[nodiscard]] auto run_viewer_once(const mesh::Mesh &mesh,
                                   const mesh::pack::PackingResult &packing,
                                   const post::DerivedFieldSet &derived,
                                   double simulation_time) -> std::expected<void, ViewerError>;

#else

[[nodiscard]] inline auto run_viewer_once(const mesh::Mesh &,
                                          const mesh::pack::PackingResult &,
                                          const post::DerivedFieldSet &,
                                          double) -> std::expected<void, ViewerError>
{
    return std::unexpected(ViewerError{"BUILD_UI=OFF — viewer unavailable", {}});
}

#endif

} // namespace cwf::ui
