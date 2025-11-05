/**
 * @file config.hpp
 * @brief YAML-powered scenario config loader that absolutely slaps uwu
 *
 * this header defines the high-level configuration model for CiviWave-FEM's
 * early pipeline. it parses YAML 1.2 documents into strongly typed C++26 data
 * structures, validates them aggressively, and bubbles up ergonomic errors via
 * std::expected. every struct here is tuned for functional purity so the rest
 * of the codebase can chomp configs without mutating global state.
 *
 * we lean hard into the spec in RefDocs/SPEC.md and RefDocs/TODO.md: materials,
 * damping, solver knobs, YAML curves, and boundary conditions all get explicit
 * representation. this is the contract between user auth YAML files and the
 * GPU-bound FEM core. the loader enforces the schema, catches typos, and
 * preserves vibes (gen-z slang included) because documentation instructions say
 * so ✨
 *
 * @author LukeFrankio
 * @date 2025-11-05
 * @version 1.0
 *
 * @note built assuming Doxygen 1.15 beta (latest) because docs supremacy
 * @note requires GCC 15.2+ with -std=c++2c (aka C++26) for std::expected
 * @note yaml-cpp 0.8.0+ powers parsing but we stay dependency-light elsewhere
 *
 * example (basic usage):
 * @code
 * using cwf::config::load_config_from_file;
 * auto config_result = load_config_from_file("assets/cantilever.yaml");
 * if (!config_result) {
 *     std::cerr << "config error: " << config_result.error().message << '\n';
 *     return EXIT_FAILURE;
 * }
 * const auto& config = *config_result;
 * // config.mesh.path now holds the mesh filepath, ready for the next stage uwu
 * @endcode
 */
#pragma once

#include <array>
#include <expected>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace YAML
{
class Node;
} // namespace YAML

namespace cwf::config
{

/**
 * @brief lit enum for per-dof constraint vibes (functional core ftw)
 *
 * ✨ PURE FUNCTION ✨
 *
 * this enum is pure because:
 * - it just tags degrees of freedom (x/y/z), no state, no I/O
 * - constexpr friendly and works as array indices without UB
 */
enum class DegreeOfFreedom : std::uint8_t
{
    X = 0U,
    Y = 1U,
    Z = 2U
};

/**
 * @brief config error payload with context breadcrumbs for days
 *
 * the loader never throws; instead it returns std::expected with this error
 * struct so callers can bubble issues to UX with nuance. context strings form a
 * breadcrumb trail (e.g., "materials[1].E"), making debugging YAML typos
 * painless.
 */
struct ConfigError
{
    std::string              message; ///< spicy human-readable error message uwu
    std::vector<std::string> context; ///< breadcrumb trail showing where things derailed
};

/**
 * @brief isotropic material definition exactly like the spec demands
 *
 * ✨ PURE FUNCTION ✨ (struct with immutable semantics when used correctly)
 *
 * holds engineering-grade constants for homogeneous linear elastic materials.
 * Values stay in SI units. density = rho, E = Young's modulus, nu = Poisson.
 */
struct Material
{
    std::string name;           ///< unique material nickname, referenced in assignments
    double      youngs_modulus; ///< E [Pa], must be > 0
    double      poisson_ratio;  ///< nu [-], typically 0 < nu < 0.5
    double      density;        ///< rho [kg/m^3], must be > 0
};

/**
 * @brief maps mesh physical groups to material names (per spec Section 3)
 */
struct Assignment
{
    std::string group;    ///< physical group name from mesh (e.g., "SOLID")
    std::string material; ///< material name defined in materials list
};

/**
 * @brief Rayleigh damping coefficients derived from (xi, w1, w2)
 *
 * stores the user input values so preprocessing can compute alpha/beta later.
 */
struct Damping
{
    double xi; ///< damping ratio target (0.0 - 1.0 typically)
    double w1; ///< lower angular frequency for Rayleigh fit [rad/s]
    double w2; ///< upper angular frequency [rad/s]
};

/**
 * @brief simulation time step defaults + bounds (adaptive aware)
 */
struct TimeSettings
{
    double initial_dt; ///< starting timestep [s]
    bool   adaptive;   ///< enable adaptive dt policies per spec
    double min_dt;     ///< optional safety clamp (0 if unspecified)
    double max_dt;     ///< optional safety clamp (> initial dt for safety)
};

/**
 * @brief solver knob pack mirroring the spec (PCG etc.)
 */
struct SolverSettings
{
    std::string   type;              ///< e.g., "pcg"
    std::string   preconditioner;    ///< e.g., "block_jacobi"
    double        runtime_tolerance; ///< tol while sim running (looser)
    double        pause_tolerance;   ///< tol when paused (tighter)
    std::uint32_t max_iterations;    ///< iteration cap per step (>= 1)
};

/**
 * @brief precision options for GPU vectors vs reductions
 */
struct PrecisionSettings
{
    std::string vector_precision;    ///< e.g., "fp32"
    std::string reduction_precision; ///< e.g., "fp64"
};

/**
 * @brief piecewise-linear curve, used by loads/time scaling
 */
struct Curve
{
    std::vector<std::pair<double, double>> points; ///< (time, value) pairs sorted by time
};

/**
 * @brief surface traction definition referencing YAML curves (optional)
 */
struct SurfaceTraction
{
    std::string           group;       ///< surface physical group name
    std::array<double, 3> value;       ///< traction direction + magnitude [Pa]
    std::string           scale_curve; ///< optional curve id ("" if constant)
};

/**
 * @brief aggregated load definitions (body + surface)
 */
struct Loads
{
    std::array<double, 3>        gravity;   ///< global gravity vector [m/s^2]
    std::vector<SurfaceTraction> tractions; ///< list of surface loads
};

/**
 * @brief Dirichlet condition specification from YAML groups
 */
struct DirichletFix
{
    std::string                          group;          ///< named group from mesh (usually surfaces)
    std::array<bool, 3>                  constrain_axis; ///< which dofs are locked (xyz)
    std::array<std::optional<double>, 3> value;          ///< optional displacement targets per axis
};

/**
 * @brief output controls (VTU cadence, probe nodes)
 */
struct OutputSettings
{
    std::uint32_t              vtu_stride; ///< write VTU every N frames (>= 1)
    std::vector<std::uint32_t> probes;     ///< node indices to track
};

/**
 * @brief main configuration object bundling all scenario inputs
 */
struct Config
{
    std::filesystem::path                  mesh_path;   ///< path to mesh (relative allowed)
    std::vector<Material>                  materials;   ///< materials registry
    std::vector<Assignment>                assignments; ///< physical group → material mapping
    Damping                                damping;     ///< Rayleigh damping spec
    TimeSettings                           time;        ///< time stepping configuration
    SolverSettings                         solver;      ///< solver knobs (PCG etc.)
    PrecisionSettings                      precision;   ///< precision toggles
    Loads                                  loads;       ///< body + surface loads
    std::unordered_map<std::string, Curve> curves;      ///< time history curves
    std::vector<DirichletFix>              dirichlet;   ///< locked DoFs definitions
    OutputSettings                         output;      ///< post-processing preferences
};

/**
 * @brief convenience alias for the loader result type (std::expected wrapper)
 */
using ConfigResult = std::expected<Config, ConfigError>;

/**
 * @brief parses YAML config from a file path with aggressive validation
 *
 * ⚠️ IMPURE FUNCTION (has side effects)
 *
 * this helper is impure because:
 * - hits the file system to read YAML
 * - may throw yaml-cpp exceptions internally (captured into expected)
 * - depends on external state (file contents)
 *
 * @param[in] path filesystem location of YAML document
 * @return ConfigResult containing parsed Config or detailed ConfigError
 *
 * @pre file must exist and be readable (otherwise error.context points to IO)
 * @post returned Config obeys schema constraints when success
 *
 * @throws (internally) yaml-cpp exceptions which are caught and mapped
 *
 * @note validations include: non-empty materials, positive parameters, sorted
 *       curves, consistent Dirichlet DOF specs, and assignment coverage.
 *
 * example (edge case handling):
 * @code
 * auto cfg = load_config_from_file("missing.yaml");
 * ASSERT_FALSE(cfg.has_value());
 * EXPECT_THAT(cfg.error().message, testing::HasSubstr("unable to open"));
 * @endcode
 */
[[nodiscard]] auto load_config_from_file(const std::filesystem::path &path) -> ConfigResult;

/**
 * @brief parses YAML config directly from a string buffer (test-friendly)
 *
 * ⚠️ IMPURE FUNCTION (depends on yaml-cpp's global state when parsing)
 *
 * primarily intended for unit tests and tooling where writing temp files is
 * noise. we still validate identically to the file loader.
 *
 * @param[in] yaml_text YAML document contents (UTF-8)
 * @return ConfigResult identical semantics to file loader
 */
[[nodiscard]] auto load_config_from_string(std::string_view yaml_text) -> ConfigResult;

/**
 * @brief low-level parser for already-loaded YAML nodes (advanced usage)
 *
 * ⚠️ IMPURE FUNCTION (depends on yaml-cpp node state)
 *
 * exposed so mesh preprocessing can share validation logic without hitting disk
 * again. still returns std::expected for ergonomic error handling.
 *
 * @param[in] root YAML root node produced by yaml-cpp
 * @return ConfigResult success or ConfigError with breadcrumbs
 */
[[nodiscard]] auto parse_config_node(const YAML::Node &root) -> ConfigResult;

} // namespace cwf::config
