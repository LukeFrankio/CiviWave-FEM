/**
 * @file newmark.cpp
 * @brief average-acceleration helper implementations
 */
#include "cwf/physics/newmark.hpp"

#include <cmath>

namespace cwf::physics::newmark
{
namespace
{

[[nodiscard]] auto apply_matrix(const std::vector<double> &matrix, const std::vector<double> &vector)
    -> std::vector<double>
{
    const std::size_t   n = vector.size();
    std::vector<double> result(n, 0.0);
    for (std::size_t row = 0; row < n; ++row)
    {
        double        sum     = 0.0;
        const double *row_ptr = matrix.data() + (row * n);
        for (std::size_t col = 0; col < n; ++col)
        {
            sum += row_ptr[col] * vector[col];
        }
        result[row] = sum;
    }
    return result;
}

} // namespace

auto make_coefficients(double dt, double beta, double gamma) -> Coefficients
{
    Coefficients coeffs{};
    coeffs.beta  = beta;
    coeffs.gamma = gamma;
    coeffs.dt    = dt;
    coeffs.a0    = 1.0 / (beta * dt * dt);
    coeffs.a1    = gamma / (beta * dt);
    coeffs.a2    = 1.0 / (beta * dt);
    coeffs.a3    = (1.0 / (2.0 * beta)) - 1.0;
    coeffs.a4    = (gamma / beta) - 1.0;
    coeffs.a5    = dt * ((gamma / (2.0 * beta)) - 1.0);
    return coeffs;
}

auto build_effective_stiffness(const std::vector<double> &stiffness, const std::vector<double> &mass_diag,
                               const materials::RayleighCoefficients &rayleigh, const Coefficients &coeffs)
    -> std::vector<double>
{
    const std::size_t   n               = mass_diag.size();
    std::vector<double> keff            = stiffness;
    const double        stiffness_scale = 1.0 + coeffs.a1 * rayleigh.beta;
    for (auto &value : keff)
    {
        value *= stiffness_scale;
    }
    const double mass_factor = coeffs.a0 + coeffs.a1 * rayleigh.alpha;
    for (std::size_t dof = 0; dof < n; ++dof)
    {
        keff[dof * n + dof] += mass_diag[dof] * mass_factor;
    }
    return keff;
}

auto build_effective_rhs(const std::vector<double> &external_load, const std::vector<double> &stiffness,
                         const std::vector<double>             &mass_diag,
                         const materials::RayleighCoefficients &rayleigh, const Coefficients &coeffs,
                         const State &state) -> std::vector<double>
{
    const std::size_t   n   = state.displacement.size();
    std::vector<double> rhs = external_load;
    std::vector<double> damping_rhs(n, 0.0);

    for (std::size_t i = 0; i < n; ++i)
    {
        const double u            = state.displacement[i];
        const double v            = state.velocity[i];
        const double a            = state.acceleration[i];
        const double mass_term    = mass_diag[i] * (coeffs.a0 * u + coeffs.a2 * v + coeffs.a3 * a);
        const double damping_term = coeffs.a1 * u + coeffs.a4 * v + coeffs.a5 * a;
        rhs[i] += mass_term;
        rhs[i] += rayleigh.alpha * mass_diag[i] * damping_term;
        damping_rhs[i] = damping_term;
    }

    if (rayleigh.beta != 0.0)
    {
        const auto stiffness_part = apply_matrix(stiffness, damping_rhs);
        for (std::size_t i = 0; i < n; ++i)
        {
            rhs[i] += rayleigh.beta * stiffness_part[i];
        }
    }

    return rhs;
}

auto update_state(const Coefficients &coeffs, const State &previous,
                  const std::vector<double> &delta_displacement) -> State
{
    const std::size_t n = delta_displacement.size();
    State             next{};
    next.displacement.resize(n);
    next.velocity.resize(n);
    next.acceleration.resize(n);

    for (std::size_t i = 0; i < n; ++i)
    {
        const double du      = delta_displacement[i];
        const double u0      = previous.displacement[i];
        const double v0      = previous.velocity[i];
        const double a0      = previous.acceleration[i];
        next.displacement[i] = u0 + du;
        next.acceleration[i] = coeffs.a0 * du - coeffs.a2 * v0 - coeffs.a3 * a0;
        next.velocity[i] = v0 + coeffs.dt * ((1.0 - coeffs.gamma) * a0 + coeffs.gamma * next.acceleration[i]);
    }

    return next;
}

} // namespace cwf::physics::newmark
