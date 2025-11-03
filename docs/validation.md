# Validation Strategy

This document sketches the validation matrix used to keep the simulator honest.
The detailed procedures, acceptance bands, and reference data will grow alongside
implementation milestones.

## Static Benchmarks

- **Cantilever beam:** compare tip deflection and stress profiles against
  analytical beam theory.
- **Thick plate bending:** validate displacement and stress contours using CPU
  reference solutions.
- **Uniaxial compression block:** ensure reaction forces align with applied
  loads and material properties.

## Dynamic Benchmarks

- **Modal analysis:** extract the lowest modes via the GPU solver and cross-check
  them against CPU eigenvalue solutions.
- **Harmonic forcing:** drive sustained oscillations and verify amplitude and
  phase responses match frequency-domain expectations.
- **Transient pulse:** confirm energy balance and damping behaviour across the
  implicit Newmark integrator.

## Regression Policy

- Maintain CPU reference runs for small meshes to detect precision drift when
  shaders or solver pipelines change.
- Capture PCG iteration counts, residual norms, and runtime budgets in CI to
  guard against performance regressions.
- Store validation artifacts (VTU files, plots, logs) under `assets/validation`
  with metadata that cites solver versions and configuration hashes.
