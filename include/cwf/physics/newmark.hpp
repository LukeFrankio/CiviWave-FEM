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
 * @brief predicted kinematic state produced by the explicit Newmark predictor
 *
 * ✨ PURE FUNCTION ✨ snapshot of what the Newmark scheme thinks the next
 * displacement/velocity will be before the implicit corrector steps in. this
 * mirrors the GPU predictor shader 1:1 so regression tests can keep both sides
 * honest.
 */
struct PredictedState
{
    std::vector<double> displacement; ///< u_{n+1}^{pred}
    std::vector<double> velocity;     ///< v_{n+1}^{pred}
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
 * @brief run the explicit Newmark predictor for displacement and velocity
 *
 * ✨ PURE FUNCTION ✨ — applies the textbook predictor equations using the
 * supplied coefficients. identical math powers the GPU predictor kernel so we
 * can cross-check outputs in unit tests.
 */
[[nodiscard]] auto predict_state(const Coefficients &coeffs, const State &previous) -> PredictedState;

/**
 * @brief scalar factors used by the GPU predictor/update kernels (Phase 9)
 *
 * ✨ PURE FUNCTION ✨ packaging of the coefficients that the Slang update
 * kernel expects (γ/(βΔt) and 1/(βΔt²)). keeps the C++ + GPU code paths aligned
 * and avoids recomputing reciprocals every thread.
 */
struct UpdateScalars
{
    double inv_beta_dt2;       ///< 1 / (β Δt²) for acceleration reconstruction
    double gamma_over_beta_dt; ///< γ / (β Δt) for velocity correction
};

/**
 * @brief compute scalar multipliers required by GPU state update kernels
 *
 * ✨ PURE FUNCTION ✨ helper used by command buffer setup to fill the update
 * cbuffers. documented separately so tests can pin its behavior.
 */
[[nodiscard]] auto compute_update_scalars(const Coefficients &coeffs) -> UpdateScalars;

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
