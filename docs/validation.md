# Validation Strategy

This document sketches the validation matrix used to keep the simulator honest.
The detailed procedures, acceptance bands, and reference data will grow alongside
implementation milestones.

## Current coverage (2025-12-11)

- CPU reference paths: matrix-free PCG + Newmark predictor/update verified via
  GoogleTest (see `pcg_test`, `newmark_stepper_test`).
- Mesh and preprocess: Gmsh v4 import, gradN/volume, adjacency, and packers
  validated in `mesh_loader_test`, `preprocess_test`, `pack_shard_upload_test`.
- Physics and outputs: loads, damping, derived fields, VTU writer, and probe
  logging covered in `physics_test`, `derived_fields_test`, `export_writer_test`,
  and `validation_test`.
- Instrumentation: FrameLog YAML serialization validated in
  `instrumentation_test`.

Sample fixtures live under `tests/data/`; helper utilities are under
`tests/support/`.

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
- Store validation artifacts (VTU files, plots, logs) under a tracked
  `validation_artifacts/` (or `tests/data/validation/`) directory with metadata
  that cites solver versions and configuration hashes.

## How to run tests

- Configure and build (e.g., `cmake --preset Debug`, `cmake --build build/Debug`).
- Run all tests for a preset: `ctest --preset Debug` or
  `ctest --test-dir build/Debug --output-on-failure`.
- Filter tests: `ctest --test-dir build/Debug -R pcg`.
- GPU-dependent cases are minimal; ensure a Vulkan 1.3-capable device is
  present if enabling them. Most suites run headless on CPU.
