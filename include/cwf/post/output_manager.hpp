/**
 * @file output_manager.hpp
 * @brief high-level orchestrator for Phase 10 export cadence uwu
 */
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <span>

#include "cwf/config/config.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/post/derived_fields.hpp"
#include "cwf/post/probe_logger.hpp"
#include "cwf/post/vtu_writer.hpp"

namespace cwf::post
{

struct OutputError
{
    std::string              message;
    std::vector<std::string> context;
};

/**
 * @brief coordinates derived field computation, VTU cadence, and probes
 */
class OutputManager
{
public:
    OutputManager(std::filesystem::path root,
                  const mesh::Mesh &mesh,
                  mesh::pack::PackingResult &packing,
                  std::span<const physics::materials::ElasticProperties> materials,
                  config::OutputSettings settings);

    [[nodiscard]] auto handle_frame(double simulation_time, std::uint32_t frame_index) -> std::expected<void, OutputError>;

private:
    [[nodiscard]] auto write_vtu_frame(const DerivedFieldSet &derived, double simulation_time, std::uint32_t frame_index)
        -> std::expected<void, OutputError>;

    std::filesystem::path root_{};
    const mesh::Mesh *mesh_{};
    mesh::pack::PackingResult *packing_{};
    std::span<const physics::materials::ElasticProperties> materials_{};
    config::OutputSettings settings_{};
    ProbeLogger probe_logger_;
};

} // namespace cwf::post
