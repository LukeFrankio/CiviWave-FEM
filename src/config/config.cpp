/**
 * @file config.cpp
 * @brief implementation of the YAML config loader with bougie validation uwu
 *
 * this translation unit backs config.hpp with the full YAML parsing pipeline.
 * it leans on yaml-cpp 0.8.0+, wraps everything in std::expected, and emits
 * error breadcrumbs so humans can fix typos without doom scrolling logs.
 */
#include "cwf/config/config.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <format>
#include <stdexcept>
#include <yaml-cpp/yaml.h>

namespace cwf::config
{
namespace
{

[[nodiscard]] auto make_error(std::string message, std::vector<std::string> ctx) -> ConfigResult
{
    return std::unexpected(ConfigError{std::move(message), std::move(ctx)});
}

[[nodiscard]] auto make_scalar_error(const std::string &expectation, std::vector<std::string> ctx)
    -> ConfigResult
{
    return make_error(expectation, std::move(ctx));
}

[[nodiscard]] auto node_to_vec3(const YAML::Node &node, std::vector<std::string> ctx)
    -> std::expected<std::array<double, 3>, ConfigError>
{
    if (!node || !node.IsSequence() || node.size() != 3U)
    {
        return std::unexpected(ConfigError{"expected sequence[3] for vector", std::move(ctx)});
    }
    std::array<double, 3> values{};
    for (std::size_t i = 0; i < 3; ++i)
    {
        try
        {
            values[i] = node[i].as<double>();
        }
        catch (const YAML::Exception &ex)
        {
            auto child_ctx = ctx;
            child_ctx.emplace_back(std::format("[{}]", i));
            return std::unexpected(ConfigError{ex.what(), std::move(child_ctx)});
        }
    }
    return values;
}

[[nodiscard]] auto node_to_optional_vec3(const YAML::Node &node, std::vector<std::string> ctx)
    -> std::expected<std::array<std::optional<double>, 3>, ConfigError>
{
    std::array<std::optional<double>, 3> result{};
    if (!node || node.IsNull())
    {
        return result;
    }
    if (!node.IsSequence() || node.size() != 3U)
    {
        return std::unexpected(ConfigError{"expected sequence[3] for value override", std::move(ctx)});
    }
    for (std::size_t i = 0; i < 3; ++i)
    {
        if (node[i].IsNull())
        {
            result[i] = std::nullopt;
            continue;
        }
        try
        {
            result[i] = node[i].as<double>();
        }
        catch (const YAML::Exception &ex)
        {
            auto child_ctx = ctx;
            child_ctx.emplace_back(std::format("[{}]", i));
            return std::unexpected(ConfigError{ex.what(), std::move(child_ctx)});
        }
    }
    return result;
}

[[nodiscard]] auto node_to_string_vec(const YAML::Node &node, std::vector<std::string> ctx)
    -> std::expected<std::vector<std::string>, ConfigError>
{
    if (!node || !node.IsSequence())
    {
        return std::unexpected(ConfigError{"expected sequence for string list", std::move(ctx)});
    }
    std::vector<std::string> items;
    items.reserve(node.size());
    for (std::size_t i = 0; i < node.size(); ++i)
    {
        try
        {
            items.emplace_back(node[i].as<std::string>());
        }
        catch (const YAML::Exception &ex)
        {
            auto child_ctx = ctx;
            child_ctx.emplace_back(std::format("[{}]", i));
            return std::unexpected(ConfigError{ex.what(), std::move(child_ctx)});
        }
    }
    return items;
}

} // namespace

auto load_config_from_file(const std::filesystem::path &path) -> ConfigResult
{
    try
    {
        const auto node = YAML::LoadFile(path.string());
        return parse_config_node(node);
    }
    catch (const YAML::BadFile &ex)
    {
        return make_error(std::format("unable to open config file: {}", ex.what()), {path.string()});
    }
    catch (const YAML::Exception &ex)
    {
        return make_error(std::format("YAML parse error: {}", ex.what()), {path.string()});
    }
}

auto load_config_from_string(std::string_view yaml_text) -> ConfigResult
{
    try
    {
        const auto node = YAML::Load(yaml_text.data());
        return parse_config_node(node);
    }
    catch (const YAML::Exception &ex)
    {
        return make_error(std::format("YAML parse error: {}", ex.what()), {});
    }
}

auto parse_config_node(const YAML::Node &root) -> ConfigResult
{
    if (!root || !root.IsMap())
    {
        return make_error("config root must be a mapping", {});
    }

    Config cfg{};

    // mesh
    const auto mesh_node = root["mesh"];
    if (!mesh_node || !mesh_node.IsMap())
    {
        return make_error("missing 'mesh' section", {"mesh"});
    }
    const auto mesh_path_node = mesh_node["path"];
    if (!mesh_path_node || !mesh_path_node.IsScalar())
    {
        return make_scalar_error("mesh.path must be a scalar string", {"mesh", "path"});
    }
    cfg.mesh_path = std::filesystem::path(mesh_path_node.as<std::string>());

    // materials
    const auto materials_node = root["materials"];
    if (!materials_node || !materials_node.IsSequence() || materials_node.size() == 0U)
    {
        return make_error("materials must be a non-empty sequence", {"materials"});
    }
    cfg.materials.reserve(materials_node.size());
    std::unordered_map<std::string, std::size_t> material_index;
    for (std::size_t i = 0; i < materials_node.size(); ++i)
    {
        const auto node = materials_node[i];
        if (!node.IsMap())
        {
            return make_error("material entry must be a map", {"materials", std::format("[{}]", i)});
        }
        Material mat{};
        try
        {
            mat.name           = node["name"].as<std::string>();
            mat.youngs_modulus = node["E"].as<double>();
            mat.poisson_ratio  = node["nu"].as<double>();
            mat.density        = node["rho"].as<double>();
        }
        catch (const YAML::Exception &ex)
        {
            return make_error(ex.what(), {"materials", std::format("[{}]", i)});
        }

        if (mat.youngs_modulus <= 0.0)
        {
            return make_error("material.E must be > 0", {"materials", std::format("[{}]", i), "E"});
        }
        if (mat.poisson_ratio <= -0.999 || mat.poisson_ratio >= 0.5)
        {
            return make_error("material.nu must be (-0.999, 0.5)",
                              {"materials", std::format("[{}]", i), "nu"});
        }
        if (mat.density <= 0.0)
        {
            return make_error("material.rho must be > 0", {"materials", std::format("[{}]", i), "rho"});
        }
        if (material_index.contains(mat.name))
        {
            return make_error("material names must be unique", {"materials", std::format("[{}]", i), "name"});
        }
        material_index[mat.name] = cfg.materials.size();
        cfg.materials.push_back(std::move(mat));
    }

    // assignments
    const auto assignments_node = root["assignments"];
    if (!assignments_node || !assignments_node.IsSequence() || assignments_node.size() == 0U)
    {
        return make_error("assignments must be a non-empty sequence", {"assignments"});
    }
    cfg.assignments.reserve(assignments_node.size());
    for (std::size_t i = 0; i < assignments_node.size(); ++i)
    {
        const auto node = assignments_node[i];
        if (!node.IsMap())
        {
            return make_error("assignment must be a map", {"assignments", std::format("[{}]", i)});
        }
        Assignment a{};
        try
        {
            a.group    = node["group"].as<std::string>();
            a.material = node["material"].as<std::string>();
        }
        catch (const YAML::Exception &ex)
        {
            return make_error(ex.what(), {"assignments", std::format("[{}]", i)});
        }
        if (!material_index.contains(a.material))
        {
            return make_error("assignment references unknown material",
                              {"assignments", std::format("[{}]", i), "material"});
        }
        cfg.assignments.push_back(std::move(a));
    }

    // damping
    const auto damping_node = root["damping"];
    if (!damping_node || !damping_node.IsMap())
    {
        return make_error("missing damping map", {"damping"});
    }
    try
    {
        cfg.damping.xi = damping_node["xi"].as<double>();
        cfg.damping.w1 = damping_node["w1"].as<double>();
        cfg.damping.w2 = damping_node["w2"].as<double>();
    }
    catch (const YAML::Exception &ex)
    {
        return make_error(ex.what(), {"damping"});
    }
    if (cfg.damping.xi <= 0.0 || cfg.damping.xi >= 1.0)
    {
        return make_error("damping.xi must be (0,1)", {"damping", "xi"});
    }
    if (cfg.damping.w1 <= 0.0)
    {
        return make_error("damping.w1 must be > 0", {"damping", "w1"});
    }
    if (cfg.damping.w2 <= cfg.damping.w1)
    {
        return make_error("damping.w2 must be > damping.w1", {"damping", "w2"});
    }

    // time
    const auto time_node = root["time"];
    if (!time_node || !time_node.IsMap())
    {
        return make_error("missing time map", {"time"});
    }
    try
    {
        cfg.time.initial_dt = time_node["dt"].as<double>();
        cfg.time.adaptive   = time_node["adaptive"].as<bool>();
    }
    catch (const YAML::Exception &ex)
    {
        return make_error(ex.what(), {"time"});
    }
    cfg.time.min_dt = time_node["min_dt"].IsDefined() ? time_node["min_dt"].as<double>() : 0.0;
    cfg.time.max_dt =
        time_node["max_dt"].IsDefined() ? time_node["max_dt"].as<double>() : cfg.time.initial_dt;
    if (cfg.time.initial_dt <= 0.0)
    {
        return make_error("time.dt must be > 0", {"time", "dt"});
    }
    if (cfg.time.min_dt < 0.0)
    {
        return make_error("time.min_dt must be >= 0", {"time", "min_dt"});
    }
    if (cfg.time.max_dt < cfg.time.initial_dt)
    {
        return make_error("time.max_dt must be >= time.dt", {"time", "max_dt"});
    }

    // solver
    const auto solver_node = root["solver"];
    if (!solver_node || !solver_node.IsMap())
    {
        return make_error("missing solver map", {"solver"});
    }
    try
    {
        cfg.solver.type              = solver_node["type"].as<std::string>();
        cfg.solver.preconditioner    = solver_node["preconditioner"].as<std::string>();
        cfg.solver.runtime_tolerance = solver_node["tol_runtime"].as<double>();
        cfg.solver.pause_tolerance   = solver_node["tol_pause"].as<double>();
        cfg.solver.max_iterations    = solver_node["max_iters"].as<std::uint32_t>();
    }
    catch (const YAML::Exception &ex)
    {
        return make_error(ex.what(), {"solver"});
    }
    if (cfg.solver.max_iterations == 0U)
    {
        return make_error("solver.max_iters must be >= 1", {"solver", "max_iters"});
    }
    if (cfg.solver.runtime_tolerance <= 0.0 || cfg.solver.pause_tolerance <= 0.0)
    {
        return make_error("solver tolerances must be > 0", {"solver"});
    }

    // precision
    const auto precision_node = root["precision"];
    if (!precision_node || !precision_node.IsMap())
    {
        return make_error("missing precision map", {"precision"});
    }
    try
    {
        cfg.precision.vector_precision    = precision_node["vectors"].as<std::string>();
        cfg.precision.reduction_precision = precision_node["reductions"].as<std::string>();
    }
    catch (const YAML::Exception &ex)
    {
        return make_error(ex.what(), {"precision"});
    }

    // curves (optional map)
    const auto curves_node = root["curves"];
    if (curves_node && curves_node.IsMap())
    {
        for (const auto &item : curves_node)
        {
            const auto key = item.first.as<std::string>();
            const auto seq = item.second;
            if (!seq.IsSequence() || seq.size() == 0U)
            {
                return make_error("curve must be non-empty sequence", {"curves", key});
            }
            Curve curve{};
            curve.points.reserve(seq.size());
            double previous_time = -std::numeric_limits<double>::infinity();
            for (std::size_t idx = 0; idx < seq.size(); ++idx)
            {
                const auto pair_node = seq[idx];
                if (!pair_node.IsSequence() || pair_node.size() != 2U)
                {
                    return make_error("curve point must be sequence[2]",
                                      {"curves", key, std::format("[{}]", idx)});
                }
                double t{};
                double v{};
                try
                {
                    t = pair_node[0].as<double>();
                    v = pair_node[1].as<double>();
                }
                catch (const YAML::Exception &ex)
                {
                    return make_error(ex.what(), {"curves", key, std::format("[{}]", idx)});
                }
                if (t < previous_time)
                {
                    return make_error("curve times must be non-decreasing",
                                      {"curves", key, std::format("[{}]", idx)});
                }
                previous_time = t;
                curve.points.emplace_back(t, v);
            }
            cfg.curves.emplace(key, std::move(curve));
        }
    }

    // loads
    const auto loads_node = root["loads"];
    if (!loads_node || !loads_node.IsMap())
    {
        return make_error("missing loads map", {"loads"});
    }
    {
        auto gravity_result = node_to_vec3(loads_node["gravity"], {"loads", "gravity"});
        if (!gravity_result)
        {
            return std::unexpected(gravity_result.error());
        }
        cfg.loads.gravity = gravity_result.value();
    }
    const auto tractions_node = loads_node["tractions"];
    if (tractions_node && tractions_node.IsSequence())
    {
        cfg.loads.tractions.reserve(tractions_node.size());
        for (std::size_t i = 0; i < tractions_node.size(); ++i)
        {
            const auto entry = tractions_node[i];
            if (!entry.IsMap())
            {
                return make_error("traction entry must be map",
                                  {"loads", "tractions", std::format("[{}]", i)});
            }
            SurfaceTraction traction{};
            try
            {
                traction.group = entry["group"].as<std::string>();
                traction.scale_curve =
                    entry["scale_curve"].IsDefined() ? entry["scale_curve"].as<std::string>() : std::string{};
            }
            catch (const YAML::Exception &ex)
            {
                return make_error(ex.what(), {"loads", "tractions", std::format("[{}]", i)});
            }
            auto val_result =
                node_to_vec3(entry["value"], {"loads", "tractions", std::format("[{}]", i), "value"});
            if (!val_result)
            {
                return std::unexpected(val_result.error());
            }
            traction.value = val_result.value();
            if (!traction.scale_curve.empty() && !cfg.curves.contains(traction.scale_curve))
            {
                return make_error("traction references unknown curve",
                                  {"loads", "tractions", std::format("[{}]", i), "scale_curve"});
            }
            cfg.loads.tractions.push_back(std::move(traction));
        }
    }

    // dirichlet
    const auto dirichlet_node = root["dirichlet"];
    if (dirichlet_node && dirichlet_node.IsMap())
    {
        const auto fixes_node = dirichlet_node["fixes"];
        if (fixes_node && fixes_node.IsSequence())
        {
            cfg.dirichlet.reserve(fixes_node.size());
            for (std::size_t i = 0; i < fixes_node.size(); ++i)
            {
                const auto entry = fixes_node[i];
                if (!entry.IsMap())
                {
                    return make_error("dirichlet fixed entry must be a map",
                                      {"dirichlet", "fixes", std::format("[{}]", i)});
                }
                DirichletFix fix{};
                try
                {
                    fix.group = entry["group"].as<std::string>();
                }
                catch (const YAML::Exception &ex)
                {
                    return make_error(ex.what(), {"dirichlet", "fixes", std::format("[{}]", i), "group"});
                }
                auto dof_result =
                    node_to_string_vec(entry["dof"], {"dirichlet", "fixes", std::format("[{}]", i), "dof"});
                if (!dof_result)
                {
                    return std::unexpected(dof_result.error());
                }
                if (dof_result->empty())
                {
                    return make_error("dirichlet.dof must not be empty",
                                      {"dirichlet", "fixes", std::format("[{}]", i), "dof"});
                }
                fix.constrain_axis = {false, false, false};
                for (const auto &axis : *dof_result)
                {
                    if (axis == "x")
                    {
                        fix.constrain_axis[0] = true;
                    }
                    else if (axis == "y")
                    {
                        fix.constrain_axis[1] = true;
                    }
                    else if (axis == "z")
                    {
                        fix.constrain_axis[2] = true;
                    }
                    else
                    {
                        return make_error("dirichlet.dof must be subset of {x,y,z}",
                                          {"dirichlet", "fixes", std::format("[{}]", i), "dof"});
                    }
                }
                auto value_result = node_to_optional_vec3(
                    entry["value"], {"dirichlet", "fixes", std::format("[{}]", i), "value"});
                if (!value_result)
                {
                    return std::unexpected(value_result.error());
                }
                fix.value = value_result.value();
                cfg.dirichlet.push_back(std::move(fix));
            }
        }
    }

    // output
    const auto output_node = root["output"];
    if (!output_node || !output_node.IsMap())
    {
        return make_error("missing output map", {"output"});
    }
    try
    {
        cfg.output.vtu_stride = output_node["vtu_stride"].as<std::uint32_t>();
    }
    catch (const YAML::Exception &ex)
    {
        return make_error(ex.what(), {"output", "vtu_stride"});
    }
    if (cfg.output.vtu_stride == 0U)
    {
        return make_error("output.vtu_stride must be >= 1", {"output", "vtu_stride"});
    }
    const auto probes_node = output_node["probes"];
    if (probes_node && probes_node.IsSequence())
    {
        cfg.output.probes.reserve(probes_node.size());
        for (std::size_t i = 0; i < probes_node.size(); ++i)
        {
            try
            {
                cfg.output.probes.emplace_back(probes_node[i].as<std::uint32_t>());
            }
            catch (const YAML::Exception &ex)
            {
                return make_error(ex.what(), {"output", "probes", std::format("[{}]", i)});
            }
        }
    }

    return cfg;
}

} // namespace cwf::config
