# CiviWave-FEM Technical Specification

This document provides a contributor-facing summary of the core specification.
The authoritative, fully detailed version lives in
[RefDocs/SPEC.md](../RefDocs/SPEC.md). Keep both documents in sync when the spec
changes.

## Implementation snapshot (2025-12-11)

- YAML loader, mesh importer (Gmsh v4 ASCII), preprocessing (gradN, volumes,
  adjacency), and CPU matrix-free PCG + Newmark reference are implemented and
  exercised in the test suite.
- Vulkan device + buffer + descriptor infrastructure is present; Slang shader
  compilation runs when `slangc` is available. Without `slangc`, stub SPIR-V is
  emitted so the build still succeeds (GPU execution disabled).
- Instrumentation emits FrameLog YAML and RGP markers; optional Tracy
  integration is wired behind `ENABLE_TRACY`.
- No CLI entry point is bundled yet; the library/tests (and optional viewer
  when `BUILD_UI=ON`) are the runnable targets.

## Scope Highlights

- 3D linear elasticity with optional corotational support for moderate
  rotations.
- GPU-first architecture optimized for the AMD Radeon™ integrated GPU with
  wave64 subgroups and a 2 GB buffer cap.
- Mixed-precision matrix-free implicit solver pipeline built on Vulkan 1.4 and
  Slang 2025.18 shaders.
- YAML-driven scenarios covering materials, damping, solver tuning, and output
  control.

## Key Requirements

1. **Toolchain:** GCC 15.2 or newer with `-std=c++2c`, CMake 4.1.2, Vulkan SDK
  1.4.328.1 (runtime target Vulkan 1.3), Slang 2025.18, and Doxygen 1.15+ for
  documentation builds. Prefer latest beta releases when available.
2. **Runtime Features:** shaderFloat64, descriptor indexing with
  `VK_EXT_descriptor_buffer`, buffer device address, timeline semaphores, and
  subgroup size control pinned to 64 lanes. Vulkan 1.3 is required at runtime;
  1.4 is preferred when available.
3. **Performance Targets:** 10–30 Hz for workloads between 50k and 150k DOFs,
   with solver residual tolerances of $2\times10^{-4}$ during runtime and
   tighter thresholds when paused.
4. **Data Layout:** Struct-of-arrays buffers, buffer sharding for allocations
   larger than 2 GB, and gather-based assembly (no floating-point atomics).

## Compliance Checklist

Use this quick list before submitting changes that touch the core spec:

- [ ] Verify new requirements are achievable on the reference AMD iGPU.
- [ ] Update regression and validation assets when expanding physics coverage.
- [ ] Amend [`docs/decisions.yaml`](decisions.yaml) for any spec-breaking change.
- [ ] Align plan milestones and TODO entries with updated scope.
