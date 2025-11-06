/**
 * @file newmark.hpp
 * @brief average-acceleration Newmark coefficients + state updates uwu
 */
#pragma once

#include <vector>

#include "cwf/physics/materials.hpp"

namespace cwf::physics::newmark
{

/**
 * @brief discrete dynamical state (displacement, velocity, acceleration)
 */
struct State
{
    std::vector<double> displacement; ///< u
    std::vector<double> velocity;     ///< v
    std::vector<double> acceleration; ///< a
};

/**
 * @brief precomputed Newmark coefficients for the average-acceleration scheme
 */
struct Coefficients
{
    double beta;  ///< default 0.25
    double gamma; ///< default 0.5
    double dt;    ///< timestep [s]
    double a0;
    double a1;
    double a2;
    double a3;
    double a4;
    double a5;
};

/**
 * @brief compute coefficients for given timestep/beta/gamma trio
 *
 * ✨ PURE FUNCTION ✨
 */
[[nodiscard]] auto make_coefficients(double dt, double beta = 0.25, double gamma = 0.5) -> Coefficients;

/**
 * @brief assemble effective stiffness matrix (dense row-major) for Newmark step
 */
[[nodiscard]] auto build_effective_stiffness(const std::vector<double>             &stiffness,
                                             const std::vector<double>             &mass_diag,
                                             const materials::RayleighCoefficients &rayleigh,
                                             const Coefficients &coeffs) -> std::vector<double>;

/**
 * @brief assemble effective right-hand-side vector
 */
[[nodiscard]] auto build_effective_rhs(const std::vector<double>             &external_load,
                                       const std::vector<double>             &stiffness,
                                       const std::vector<double>             &mass_diag,
                                       const materials::RayleighCoefficients &rayleigh,
                                       const Coefficients &coeffs, const State &state) -> std::vector<double>;

/**
 * @brief update dynamic state after solving for delta displacement
 */
[[nodiscard]] auto update_state(const Coefficients &coeffs, const State &previous,
                                const std::vector<double> &delta_displacement) -> State;

} // namespace cwf::physics::newmark
