/**
 * @file solver.hpp
 * @brief CPU reference assembly + conjugate-gradient Newmark stepper
 */
#pragma once

#include <cstddef>
#include <vector>

#include "cwf/config/config.hpp"
#include "cwf/mesh/mesh.hpp"
#include "cwf/mesh/preprocess.hpp"
#include "cwf/physics/materials.hpp"
#include "cwf/physics/newmark.hpp"

namespace cwf::physics::solver
{

/**
 * @brief assembled linear system pieces (dense stiffness + lumped mass)
 */
struct Assembly
{
    std::vector<double> stiffness; ///< dense row-major matrix (dofs x dofs)
    std::vector<double> mass_diag; ///< diagonal mass entries sized dofs
};

/**
 * @brief dirichlet mask + targets per degree of freedom
 */
struct DirichletConditions
{
    std::vector<bool>   mask;    ///< true when DOF constrained
    std::vector<double> targets; ///< absolute displacement target for constrained DOFs
};

/**
 * @brief stats exposed by the CG solve
 */
struct SolveStats
{
    std::size_t iterations{};
    double      residual_norm{};
    bool        converged{};
};

/**
 * @brief result of one Newmark step (state + solver stats)
 */
struct StepResult
{
    newmark::State state;
    SolveStats     stats;
};

/**
 * @brief assemble global stiffness/mass using tetrahedral gradients
 */
[[nodiscard]] auto assemble_linear_system(const mesh::Mesh &mesh, const mesh::pre::Outputs &preprocess,
                                          const std::vector<materials::ElasticProperties> &materials)
    -> Assembly;

/**
 * @brief build dirichlet mask/targets from config + surface groups
 */
[[nodiscard]] auto build_dirichlet_conditions(const mesh::Mesh &mesh, const config::Config &cfg)
    -> DirichletConditions;

/**
 * @brief execute one implicit Newmark step with diagonal preconditioned CG
 */
[[nodiscard]] auto solve_newmark_step(const Assembly                        &assembly,
                                      const materials::RayleighCoefficients &rayleigh,
                                      const DirichletConditions &dirichlet, const mesh::Mesh &mesh,
                                      const config::Config &cfg, const mesh::pre::Outputs &preprocess,
                                      const newmark::Coefficients &coeffs,
                                      const newmark::State &previous_state, double time, double tolerance,
                                      std::size_t max_iterations) -> StepResult;

} // namespace cwf::physics::solver
