# CiviWave-FEM: End-to-End Execution Plan and Task List (Slang, YAML, C++26/GCC, Vulkan 1.4, AGPL)
A comprehensive step-by-step guide from empty repository to finished, optimized release. This document emphasizes: Slang shaders, YAML configuration, GCC 15.2 with C++26, extensive compiler extensions (esp. restrict), CMake 4.1.2, Vulkan 1.4 target, and a strict source-first dependency policy using CMake FetchContent.

Note: Use this as a living document. Each checkbox corresponds to specific PRs/issues in the tracker. The pipeline ensures that if any dependency is not detectably ABI-compatible with the current compiler/flags, we build it from source on first configure and never use prebuilt binaries.

## Phase 0 — Repository Bootstrapping and Ground Rules

- [ ] Create GitHub repo “CiviWave-FEM” (public).
- [ ] Add AGPL-3.0-or-later LICENSE file (full text).
- [ ] Add .gitignore for C++, CMake, GCC/MinGW, build outputs, IDE files.
- [ ] Add README.md describing purpose, hardware target (AMD iGPU), and core features.
- [ ] Enable Discussions and create categories (Q&A, Announcements, Tuning, Validation).
- [ ] Create Project board with columns: Backlog, In Progress, Review, Done; milestones M1..M6.
- [ ] Configure branch protections:
  - [ ] main: require PR, 1+ reviews, passing CI, signed commits optional.
  - [ ] dev: require PR, squash merges enabled.
- [ ] Add issue/PR templates (bug, feature, tech debt).
- [ ] Define coding standards and contribution guidelines:
  - [ ] clang-format (.clang-format)
  - [ ] clang-tidy config (advisory)
  - [ ] CONTRIBUTING.md and CODE_OF_CONDUCT.md
- [ ] Add docs skeleton: docs/spec.md, docs/plan.md, docs/validation.md, docs/tuning-amd-igpu.md, docs/decisions.yaml.

## Phase 1 — Developer Environment, Toolchain, and Policies

- [ ] Establish canonical toolchain versions (versions.yaml):
  - [ ] GCC 15.2 (C++26 via -std=c++2c)
  - [ ] CMake 4.1.2
  - [ ] Vulkan SDK 1.4.328.1
  - [ ] Slang 2025.18
  - [ ] VMA 3.3.0
  - [ ] yaml-cpp 0.8.0
- [ ] Document Windows 11 setup (docs/dev-setup-windows.md):
  - [ ] MSYS2/MinGW-w64 with GCC 15.x
  - [ ] CMake 4.1.2
  - [ ] Vulkan SDK 1.4.328.1
  - [ ] Python 3.11+
  - [ ] RGP/RGA
  - [ ] PATH and environment variables
- [ ] Document Linux setup (docs/dev-setup-linux.md):
  - [ ] GCC 15.x, CMake 4.1.2
  - [ ] Vulkan SDK installation
  - [ ] udev rules (if needed) and validation layers
- [ ] Define compiler extensions policy (docs/extensions.md):
  - [ ] __restrict mapping macro CW_RESTRICT
  - [ ] Attribute usage guidelines (always_inline, hot/cold, assume_aligned)
  - [ ] Builtins (assume/expect), alignment guarantees
  - [ ] Loop/vectorization hints; when to use and how to validate impacts
- [ ] Define dependency policy (docs/dependencies.md):
  - [ ] System package acceptance criteria (version, compiler, flags)
  - [ ] FetchContent usage and cache strategy
  - [ ] First-build source compilation
  - [ ] No prebuilt opaque binaries

## Phase 2 — CMake Scaffolding and FetchContent Framework

- [ ] Create top-level CMakeLists.txt:
  - [ ] Require CMake 4.1.2
  - [ ] Project name, version, languages (CXX)
  - [ ] Set C++ standard to 26 (or add -std=c++2c), enforce on all targets
  - [ ] Detect GCC and minimum version >= 15.2
  - [ ] Options: BUILD_TESTS, ENABLE_VALIDATION, ENABLE_TRACY, BUILD_UI, ENABLE_RGP_MARKERS, FORCE_FETCH_DEPS
  - [ ] Global warnings and sanitizers in Debug/Profile
  - [ ] LTO toggles off by default (profile later)
- [ ] Add cmake/Modules:
  - [ ] DetectCompilerABI.cmake (fingerprint compiler/flags → trigger rebuilds)
  - [ ] FetchOrSystem.cmake (wrap find_package with ABI/version validation)
  - [ ] SlangCompile.cmake (custom command to compile .slang → .spv)
- [ ] Add CMakePresets.json (or YAML if using cmake-presets v5 with YAML):
  - [ ] Presets for Debug, Release, RelWithDebInfo, Profile
  - [ ] Cache variables: FORCE_FETCH_DEPS, ENABLE_VALIDATION
- [ ] Implement FetchContent for:
  - [ ] Slang (tag v2025.18) — build slangc and library
  - [ ] VMA v3.3.0 (header-only but keep via FetchContent for pinning)
  - [ ] yaml-cpp v0.8.0 (static, no exceptions if desired)
  - [ ] Optional: Tracy, ImGui (pinned tags)
- [ ] CI workflow:
  - [ ] Windows/Ubuntu jobs installing GCC 15.x and CMake 4.1.2
  - [ ] Configure with FORCE_FETCH_DEPS=ON
  - [ ] Build Debug and Release
  - [ ] Run unit tests
  - [ ] Format check

## Phase 3 — YAML Config, Mesh I/O, Preprocessing

- [ ] Implement YAML config reader:
  - [ ] Schema: materials, damping, time, solver settings, precision, outputs, mesh path, BC/loads groups
  - [ ] Validation and error messages
  - [ ] Defaults handling
- [ ] Mesh I/O:
  - [ ] Gmsh v4 (.msh) reader (nodes, tets/hexes, physical groups)
  - [ ] Optional Abaqus .inp reader
- [ ] Preprocessing:
  - [ ] Build node↔element adjacency and local DOF mapping
  - [ ] Compute gradN and volumes for tets; support hexes later
  - [ ] Lumped mass per node (from ρ and shape functions)
  - [ ] Material assignment by group
- [ ] Sanity checks:
  - [ ] Detect inverted elements
  - [ ] Duplicate nodes/elements detection
  - [ ] Consistency in BC groups

## Phase 4 — CPU Reference Physics and Solver

- [ ] Materials:
  - [ ] Build isotropic elasticity D(E, ν)
- [ ] Loads:
  - [ ] Gravity, nodal point loads
  - [ ] Surface tractions (element face integration on CPU)
  - [ ] Time curves (piecewise-linear from YAML)
- [ ] Integrator (CPU):
  - [ ] Newmark predictor and state updates
- [ ] Solver (CPU):
  - [ ] Assemble CSR K and M (for small test meshes)
  - [ ] CG solver for static/dynamic testing
  - [ ] Verification against analytical solutions:
    - [ ] Cantilever beam deflection
    - [ ] Uniaxial block
- [ ] Unit tests:
  - [ ] Mesh import and properties
  - [ ] gradN/volume correctness
  - [ ] Damping parameter derivation from ξ, ω1, ω2

## Phase 5 — Vulkan Device, Memory, Descriptors, Sync

- [ ] Instance and device selection:
  - [ ] Enumerate devices; pick AMD iGPU by default or device index
  - [ ] Check and enable required features/extensions
  - [ ] Log capability summary
- [ ] Queues:
  - [ ] Compute-capable queue with timestamps
- [ ] Memory:
  - [ ] Integrate VMA; allocators for device-local and host-visible buffers
  - [ ] Staging buffer for uploads; persistent mapped ring
- [ ] Descriptor strategy:
  - [ ] EXT_descriptor_buffer for binding flexibility and low CPU overhead
  - [ ] Descriptor indexing for arrayed bindings
- [ ] Synchronization:
  - [ ] Timeline semaphores for per-iteration sync
  - [ ] Barriers util to chain compute passes
- [ ] Debug utilities:
  - [ ] Names for buffers/pipelines
  - [ ] RGP markers around passes

## Phase 6 — Slang Shader Pipeline

- [ ] Fetch and build Slang (slangc + runtime) via FetchContent.
- [ ] Author base Slang modules:
  - [ ] common.slang: vector/matrix types, math, constants
  - [ ] reductions.slang: subgroup reduce wrappers
  - [ ] bc.slang: mask application and utilities
- [ ] Author compute kernels:
  - [ ] ke_apply_element.slang
  - [ ] ke_gather_node.slang
  - [ ] pcg_precondition.slang
  - [ ] pcg_dot_partials.slang (FP64 partials)
  - [ ] pcg_reduce_final.slang (FP64 final)
  - [ ] pcg_axpy.slang (vector updates)
  - [ ] newmark_predictor.slang
  - [ ] newmark_update.slang
  - [ ] derive_fields.slang
- [ ] CMake integration:
  - [ ] Custom commands compiling each .slang → .spv with slangc
  - [ ] Dependencies from source header modules
  - [ ] Rebuild on Slang or kernel changes
- [ ] Subgroup size control:
  - [ ] RequiredSubgroupSize = 64 in pipelines using subgroup ops

## Phase 7 — Data Packing and Buffer Sharding

- [ ] SoA packing on CPU:
  - [ ] Nodes: pos0, u, v, a, fext, bc_mask, bc_values
  - [ ] Elements: conn, gradN, vol, mat
  - [ ] Solver: p, r, Ap, z, x, partials
- [ ] Sharding:
  - [ ] Implement logical array view that maps onto one or more VkBuffers
  - [ ] Descriptor buffer regions per shard
- [ ] Upload manager:
  - [ ] Single bulk upload for static data
  - [ ] Ring buffer for dynamic uniforms/constants

## Phase 8 — Matrix-Free K_eff Apply and PCG Core

- [ ] Implement ApplyKeff:
  - [ ] Element kernel: ε, σ, fe_local (FP32)
  - [ ] Node gather: sum contributions, add M/C, mask BCs
- [ ] Implement preconditioner:
  - [ ] Block-Jacobi 3×3 per node (FP32 store), clamp near-singular
- [ ] Implement reductions:
  - [ ] Per-workgroup FP64 partial sums via subgroup reduce
  - [ ] Final FP64 reduction kernel to scalar
- [ ] Implement PCG loop orchestration:
  - [ ] p, r, Ap, z vectors in FP32
  - [ ] alpha/beta from FP64 reductions
  - [ ] Warm-start from previous Δu
  - [ ] Timeline semaphore increments per iteration
- [ ] Early stopping:
  - [ ] Relative residual tol 1e-4–3e-4
  - [ ] Cap iterations; telemetry for convergence

## Phase 9 — Implicit Newmark Stepper

- [ ] Predictor stage:
  - [ ] Compute effective RHS
- [ ] K_eff coefficients:
  - [ ] a0, a1 push constants; updated when dt changes
- [ ] Solve Δu with PCG
- [ ] State updates:
  - [ ] u, v, a update kernels
- [ ] Adaptive controls:
  - [ ] Increase dt when iterations low; decrease on stagnation
  - [ ] Tolerance scheduling (looser during runtime; tighter when paused)

## Phase 10 — Derived Fields, Export, Optional Rendering

- [ ] Derived fields:
  - [ ] ε, σ per element
  - [ ] von Mises; principal invariants optional
  - [ ] Node aggregation via gather
- [ ] Export:
  - [ ] VTU writer (binary) every N frames
  - [ ] Probe CSV
- [ ] Optional rendering:
  - [ ] Deformed mesh draw (x0 + u) with stress colors
  - [ ] Camera controls/UI (ImGui)

## Phase 11 — Instrumentation and Profiling

- [ ] Vulkan timestamps per pass; resolve to CPU time units
- [ ] Per-frame log in YAML:
  - [ ] timings: pass_name → ms
  - [ ] pcg: iters, residuals
  - [ ] dt and tol
- [ ] RGP capture scripts; annotated markers
- [ ] Tracy integration (zones, GPU context)

## Phase 12 — Validation and Regression

- [ ] Build assets for validation:
  - [ ] Beam, plate, block meshes
- [ ] Static validations:
  - [ ] Tip deflection, stress distributions
- [ ] Dynamic validations:
  - [ ] Modal frequencies, harmonic steady-state
- [ ] Regression suite:
  - [ ] CPU vs GPU small meshes
  - [ ] Energy checks
- [ ] CI gate:
  - [ ] Run small static validation test with thresholds

## Phase 13 — AMD iGPU Tuning Sprints

- [ ] Subgroup tuning:
  - [ ] Confirm wave64 occupancy; adjust local sizes 128–256
- [ ] Memory traffic:
  - [ ] Profile recompute B vs stored B (gradN vs B-matrix)
- [ ] Kernel fusion opportunities:
  - [ ] Combine AXPY/dots where possible without complicating barriers
- [ ] Preconditioner evaluation:
  - [ ] Measure iterations across problem classes
  - [ ] Decide priority for AMG
- [ ] Buffer sharding:
  - [ ] Validate sharding overhead vs large single buffers (2 GB limit)

## Phase 14 — Advanced Preconditioning (AMG/GMultigrid)

- [ ] Aggregation AMG:
  - [ ] Graph coarsening on GPU
  - [ ] Prolongation/restriction operators
  - [ ] Smoothers (Jacobi/Gauss-Seidel variants on GPU)
  - [ ] Coarse solve on CPU/GPU
- [ ] Integrate as PCG preconditioner:
  - [ ] Configurable levels and smoothers
- [ ] Geometric multigrid (hex meshes) pathway:
  - [ ] Restriction/prolongation on structured grids
  - [ ] V-cycle parameterization

## Phase 15 — Compiler Extensions Hardened

- [ ] Portability header (util/port.hpp):
  - [ ] CW_RESTRICT macro mapping to __restrict__
  - [ ] Alignment utilities and assume_aligned wrappers
  - [ ] Likely/unlikely macros with __builtin_expect
  - [ ] Force inline macros with [[gnu::always_inline]] inline
- [ ] Audit critical loops:
  - [ ] Apply restrict and alignment assumptions where safe
  - [ ] Verify via sanitizers and asserts in Debug
- [ ] Vectorization hints:
  - [ ] Use vector_size where appropriate and measurable
- [ ] Document safety and constraints:
  - [ ] When aliasing guarantees are valid
  - [ ] How to avoid UB from mis-specified restrict

## Phase 16 — Error Handling, Diagnostics, Stability

- [ ] Input validation with precise diagnostics:
  - [ ] YAML schema errors
  - [ ] Mesh integrity issues
- [ ] Runtime checks:
  - [ ] Solver divergence detection and fallback
  - [ ] NaN/Inf guards in shaders (drop to safe values, flag frame)
- [ ] Crash diagnostics:
  - [ ] Windows minidumps; Linux core dump handling
  - [ ] Log rotation and verbosity controls

## Phase 17 — UI/UX (Optional but Beneficial)

- [ ] ImGui panel:
  - [ ] Live residuals, FPS, dt, tol
  - [ ] Toggle preconditioners, precision
  - [ ] Load YAML scenario; reload shaders
- [ ] Camera/scene controls:
  - [ ] Orbit, pan, zoom
  - [ ] Field selection for coloring

## Phase 18 — Packaging, Licensing, Compliance

- [ ] Package script (scripts/package.*):
  - [ ] Bundle exe, shaders, assets, LICENSE, README
- [ ] Versioning scheme:
  - [ ] SemVer, git tags
- [ ] Licensing:
  - [ ] AGPL-3.0-or-later text included
  - [ ] Third-party notices with source URLs
  - [ ] SPDX identifiers in source headers
- [ ] SBOM (optional): CycloneDX generation
- [ ] CodeQL (optional)

## Phase 19 — Documentation and Release

- [ ] User guide:
  - [ ] YAML schema explained
  - [ ] Running scenarios; interpreting outputs
- [ ] Developer guide:
  - [ ] Architecture; adding kernels; FetchContent rules
  - [ ] Adding elements (hex) and new materials
- [ ] Validation report:
  - [ ] Plots vs analytical/CPU reference
  - [ ] Performance tables and RGP screenshots
- [ ] Final checks:
  - [ ] Re-run all validations and profiling
  - [ ] Update screenshots/gifs
- [ ] v1.0.0 release:
  - [ ] Release notes; artifacts published

## Appendix A — YAML Schema Sketch

```yaml
mesh:
  path: assets/meshes/cantilever.msh
materials:
  - name: concrete
    E: 30_000_000_000
    nu: 0.2
    rho: 2500
assignments:
  - group: SOLID
    material: concrete
damping:
  xi: 0.02
  w1: 10.0
  w2: 100.0
time:
  dt: 0.01111
  adaptive: true
solver:
  type: pcg
  preconditioner: block_jacobi
  tol_runtime: 2.0e-4
  tol_pause: 1.0e-5
  max_iters: 120
precision:
  vectors: fp32
  reductions: fp64
loads:
  gravity: [0.0, 0.0, -9.81]
  tractions:
    - group: TOP_FACE
      value: [0, 0, -1e5]
      scale_curve: load_curve1
curves:
  load_curve1:
    - [0.0, 0.0]
    - [1.0, 1.0]
    - [2.0, 0.5]
dirichlet:
  fixes:
    - group: FIXED_BASE
      dof: [x, y, z]
output:
  vtu_stride: 10
  probes:
    - node: 123
    - node: 456
```

## Appendix B — Compiler Flags and Presets

- GCC common:
  - -std=c++2c -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wformat=2
  - -fno-exceptions (optional), -fno-rtti (optional), -ffast-math (careful), -fno-math-errno
  - -march=native (developer builds), configurable target for releases
  - -fno-omit-frame-pointer (Profile)
  - -O3 (Release), -O2 (RelWithDebInfo), -O0 -g (Debug)
- Define portability macros:
  - CW_RESTRICT → __restrict__
  - CW_LIKELY/UNLIKELY → __builtin_expect
  - CW_ASSUME_ALIGNED(p, N) → __builtin_assume_aligned

## Appendix C — Risk Register and Mitigations

- FP64 throughput low:
  - Keep hot loops in FP32; FP64 only for reductions.
- No FP atomicAdd:
  - Gather and two-pass reductions only.
- Memory bandwidth ceiling:
  - Matrix-free, SoA, minimize passes; recompute vs load trade-offs.
- 2 GB buffer cap:
  - Shard buffers; descriptor indexing/descriptor buffer for binding.
- Convergence issues:
  - Warm-start, better preconditioning roadmap, adaptive dt/tol.

This TODO is intended to be exhaustive and prescriptive. Track each item as an issue with a clear definition of done, tests, and profiling snapshots.