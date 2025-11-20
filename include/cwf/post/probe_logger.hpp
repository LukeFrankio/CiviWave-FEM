/**
 * @file probe_logger.hpp
 * @brief CSV probe logger for node-level time histories uwu
 */
#pragma once

#include <cstdint>
#include <expected>
#include <filesystem>
#include <string>
#include <vector>

#include "cwf/config/config.hpp"
#include "cwf/mesh/pack.hpp"
#include "cwf/post/derived_fields.hpp"

namespace cwf::post
{

struct ProbeError
{
    std::string              message;
    std::vector<std::string> context;
};

/**
 * @brief thin helper that writes probe CSV rows every frame
 */
class ProbeLogger
{
public:
    ProbeLogger(std::filesystem::path path, std::vector<std::uint32_t> probes);

    [[nodiscard]] auto log_frame(double simulation_time,
                                 std::uint32_t frame_index,
                                 const mesh::pack::PackingResult &packing,
                                 const DerivedFieldSet &derived) -> std::expected<void, ProbeError>;

private:
    [[nodiscard]] auto write_header() -> std::expected<void, ProbeError>;

    std::filesystem::path path_{};
    std::vector<std::uint32_t> probes_{};
    bool header_written_{false};
};

} // namespace cwf::post
