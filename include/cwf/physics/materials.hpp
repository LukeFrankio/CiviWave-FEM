/**
 * @file materials.hpp
 * @brief isotropic elasticity math that keeps FEM stiffness crisp af uwu
 *
 * this header bundles pure helper functions for translating YAML material
 * properties into hardcore constitutive tensors. think lamé parameters,
 * bulk/shear vibes, and the full 6x6 D matrix for 3D linear elasticity.
 * everything is constexpr-friendly, fully documented, and ready for the
 * GPU pipeline to vibe check against the CPU reference solver.
 *
 * the motivation: avoid re-deriving stress-strain relationships every other
 * PR. we expose tiny pure utilities that feed both preprocessing and the
 * upcoming Vulkan kernels while keeping the math transparent.
 *
 * @author LukeFrankio
 * @date 2025-11-06
 * @version 1.0
 *
 * @note uses C++26 features (std::array constexpr goodness) compiled with
 *       GCC 15.2+ in -std=c++2c mode because bleeding edge is praxis.
 * @note documented with Doxygen 1.15 beta (latest) because documentation
 *       supremacy is mandatory.
 */
#pragma once

#include <array>
#include <utility>

#include "cwf/config/config.hpp"

namespace cwf::physics::materials
{

/**
 * @brief lamé parameter duo (alpha + mu) used by isotropic elasticity
 */
struct LamePair
{
    double lambda; ///< first Lamé parameter [Pa]
    double mu;     ///< second Lamé parameter aka shear modulus [Pa]
};

/**
 * @brief packaged elastic goodies: lamé pair, bulk/shear, and stiffness matrix
 */
struct ElasticProperties
{
    double                 youngs_modulus; ///< original E from YAML [Pa]
    double                 poisson_ratio;  ///< original nu (dimensionless)
    double                 bulk_modulus;   ///< K [Pa]
    double                 shear_modulus;  ///< G [Pa]
    LamePair               lame;           ///< lamé parameters (λ, μ)
    std::array<double, 36> stiffness;      ///< 6x6 constitutive matrix (Voigt ordering)
};

/**
 * @brief rayleigh damping coefficients (alpha, beta) in SI units uwu
 */
struct RayleighCoefficients
{
    double alpha; ///< mass-proportional term
    double beta;  ///< stiffness-proportional term
};

/**
 * @brief compute lamé parameters from (E, nu) without mutating anything
 *
 * ✨ PURE FUNCTION ✨
 *
 * @param youngs_modulus Young's modulus [Pa]
 * @param poisson_ratio Poisson ratio [-]
 * @return pair of (lambda, mu)
 */
[[nodiscard]] constexpr auto compute_lame(double youngs_modulus, double poisson_ratio) noexcept -> LamePair;

/**
 * @brief build isotropic 6x6 elasticity matrix for 3D solids (Voigt ordering)
 *
 * ✨ PURE FUNCTION ✨
 *
 * @param youngs_modulus Young's modulus [Pa]
 * @param poisson_ratio Poisson ratio [-]
 * @return stiffness matrix in row-major Voigt form
 */
[[nodiscard]] constexpr auto make_stiffness_matrix(double youngs_modulus, double poisson_ratio) noexcept
    -> std::array<double, 36>;

/**
 * @brief derive bulk/shear/lamé + stiffness from config material definition
 *
 * ✨ PURE FUNCTION ✨
 *
 * @param material validated YAML material struct
 * @return packaged elastic properties ready for assembly
 */
[[nodiscard]] constexpr auto make_properties(const config::Material &material) noexcept -> ElasticProperties;

/**
 * @brief compute rayleigh damping coefficients from YAML damping triple
 *
 * ✨ PURE FUNCTION ✨
 *
 * @param damping Rayleigh damping spec (xi, w1, w2)
 * @return (alpha, beta) pair
 */
[[nodiscard]] constexpr auto compute_rayleigh(const config::Damping &damping) noexcept
    -> RayleighCoefficients;

} // namespace cwf::physics::materials

// --- inline implementations -------------------------------------------------

namespace cwf::physics::materials
{

[[nodiscard]] constexpr auto compute_lame(double youngs_modulus, double poisson_ratio) noexcept -> LamePair
{
    const double denom  = (1.0 + poisson_ratio) * (1.0 - 2.0 * poisson_ratio);
    const double lambda = (poisson_ratio * youngs_modulus) / denom;
    const double mu     = youngs_modulus / (2.0 * (1.0 + poisson_ratio));
    return LamePair{lambda, mu};
}

[[nodiscard]] constexpr auto make_stiffness_matrix(double youngs_modulus, double poisson_ratio) noexcept
    -> std::array<double, 36>
{
    const auto   lame = compute_lame(youngs_modulus, poisson_ratio);
    const double c    = lame.lambda + 2.0 * lame.mu;

    return {c,   lame.lambda, lame.lambda, 0.0,         0.0,         0.0, lame.lambda, c,   lame.lambda,
            0.0, 0.0,         0.0,         lame.lambda, lame.lambda, c,   0.0,         0.0, 0.0,
            0.0, 0.0,         0.0,         lame.mu,     0.0,         0.0, 0.0,         0.0, 0.0,
            0.0, lame.mu,     0.0,         0.0,         0.0,         0.0, 0.0,         0.0, lame.mu};
}

[[nodiscard]] constexpr auto make_properties(const config::Material &material) noexcept -> ElasticProperties
{
    const auto   lame  = compute_lame(material.youngs_modulus, material.poisson_ratio);
    const double bulk  = lame.lambda + (2.0 / 3.0) * lame.mu;
    const double shear = lame.mu;
    return ElasticProperties{material.youngs_modulus,
                             material.poisson_ratio,
                             bulk,
                             shear,
                             lame,
                             make_stiffness_matrix(material.youngs_modulus, material.poisson_ratio)};
}

[[nodiscard]] constexpr auto compute_rayleigh(const config::Damping &damping) noexcept -> RayleighCoefficients
{
    const double denom = damping.w1 + damping.w2;
    const double alpha = 2.0 * damping.xi * damping.w1 * damping.w2 / denom;
    const double beta  = 2.0 * damping.xi / denom;
    return RayleighCoefficients{alpha, beta};
}

} // namespace cwf::physics::materials
