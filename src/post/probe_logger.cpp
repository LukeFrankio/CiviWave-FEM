/**
 * @file probe_logger.cpp
 * @brief implementation of the CSV probe logger uwu
 */
#include "cwf/post/probe_logger.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>

namespace cwf::post
{
namespace
{

[[nodiscard]] auto make_error(std::string message, std::initializer_list<std::string> ctx = {}) -> ProbeError
{
    ProbeError err{};
    err.message = std::move(message);
    err.context.assign(ctx.begin(), ctx.end());
    return err;
}

[[nodiscard]] auto serialize_row(std::uint32_t frame,
                                 double time,
                                 std::uint32_t node,
                                 const mesh::pack::PackingResult &packing,
                                 const DerivedFieldSet &derived) -> std::string
{
    std::ostringstream oss;
    oss.setf(std::ios::fixed, std::ios::floatfield);
    oss.precision(9);

    const auto &disp = packing.buffers.nodes.displacement;
    const auto &vel = packing.buffers.nodes.velocity;
    const auto &acc = packing.buffers.nodes.acceleration;
    const auto &field = derived.nodes[node];

    oss << frame << ',' << time << ',' << node << ','
        << disp.x[node] << ',' << disp.y[node] << ',' << disp.z[node] << ','
        << vel.x[node] << ',' << vel.y[node] << ',' << vel.z[node] << ','
        << acc.x[node] << ',' << acc.y[node] << ',' << acc.z[node];

    for (std::size_t c = 0; c < 6U; ++c)
    {
        oss << ',' << field.strain[c];
    }
    for (std::size_t c = 0; c < 6U; ++c)
    {
        oss << ',' << field.stress[c];
    }
    oss << ',' << field.von_mises;
    oss << '\n';
    return oss.str();
}

} // namespace

ProbeLogger::ProbeLogger(std::filesystem::path path, std::vector<std::uint32_t> probes)
    : path_{std::move(path)}, probes_{std::move(probes)}
{
}

[[nodiscard]] auto ProbeLogger::write_header() -> std::expected<void, ProbeError>
{
    if (header_written_ || probes_.empty())
    {
        header_written_ = true;
        return {};
    }

    if (!path_.parent_path().empty())
    {
        std::filesystem::create_directories(path_.parent_path());
    }

    std::ofstream file(path_, std::ios::trunc);
    if (!file)
    {
        return std::unexpected(make_error("failed to open probe CSV for header", {path_.string()}));
    }

    file << "frame,time,node,ux,uy,uz,vx,vy,vz,ax,ay,az"
         << ",strain_xx,strain_yy,strain_zz,strain_xy,strain_yz,strain_xz"
         << ",stress_xx,stress_yy,stress_zz,stress_xy,stress_yz,stress_xz,von_mises\n";
    header_written_ = true;
    return {};
}

[[nodiscard]] auto ProbeLogger::log_frame(double simulation_time,
                                          std::uint32_t frame_index,
                                          const mesh::pack::PackingResult &packing,
                                          const DerivedFieldSet &derived) -> std::expected<void, ProbeError>
{
    if (probes_.empty())
    {
        return {};
    }

    if (!header_written_)
    {
        if (auto hdr = write_header(); !hdr)
        {
            return hdr;
        }
    }

    std::ofstream file(path_, std::ios::app);
    if (!file)
    {
        return std::unexpected(make_error("failed to open probe CSV", {path_.string()}));
    }

    const auto node_count = packing.metadata.node_count;
    for (const auto probe : probes_)
    {
        if (probe >= node_count)
        {
            return std::unexpected(make_error("probe index out of range", {std::to_string(probe)}));
        }
        file << serialize_row(frame_index, simulation_time, probe, packing, derived);
    }
    return {};
}

} // namespace cwf::post
