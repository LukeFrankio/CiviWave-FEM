# Project: CiviWave-FEM (Vulkan GPU Structural Analysis for Civil Engineering)
Subtitled: Slang shaders, YAML configs, C++26 (GCC), Vulkan 1.4, AGPL

Mission: Build a C++26 Vulkan 1.4 application that computes near real-time structural response (displacement, strain, stress) for homogeneous linear elastic models using FEM, optimized for your AMD integrated GPU (wave64, FP64 supported but low throughput, 2 GB buffer cap). Shaders authored in Slang (SPIR-V), configuration in YAML, GCC as the compiler, CMake as the build system, licensed under AGPL-3.0-or-later, and all dependencies fetched and built from source when needed.

Primary versions (as of 2025-11-03):
- GCC: 15.2 (C++26 via -std=c++2c)
- CMake: 4.1.2
- Vulkan API: 1.4 (use 1.3 device where 1.4 not exposed)
- Vulkan SDK: 1.4.328.1
- Slang: 2025.18 (slangc, slang runtime)
- VMA: 3.3.0
- yaml-cpp: 0.8.0

Dependency policy: Source builds via CMake FetchContent unless a compatible system package is detected. No opaque prebuilt binaries. If compiler family/version/flags mismatch are detected, force FetchContent builds.

Compiler extensions: Use extensively for performance (notably __restrict__), attributes for inlining/hot paths, alignment and vectorization hints. Centralize in a portability header to preserve cross-compiler behavior.

---

## 0) Prerequisites and Decisions

- OS: Windows 11 primary; Linux optional later (GCC toolchain).
- Compiler: GCC 15.2+ required; prefer -std=c++2c (or -std=gnu++2c for guarded GNU extensions).
- Build: CMake 4.1.2+ with presets (Debug, Release, RelWithDebInfo, Profile).
- Shaders: Slang 2025.18 → SPIR-V; integrate slangc as a build step via FetchContent.
- API: Vulkan 1.4 target; runtime feature gating for 1.3-capable devices.
- Config format: YAML (yaml-cpp 0.8.0).
- Memory: VMA 3.3.0.
- Profiling: Tracy (optional), RGP/RGA for GPU, perf counters via Vulkan queries.
- License: AGPL-3.0-or-later (full text in LICENSE).
- Style: clang-format, clang-tidy (advisory), -Wall -Wextra -Werror in CI, -Wno-unknown-attributes when cross-compiling.

Deliverables:
- docs/decisions.yaml tracking key decisions and version pins.

---

## 1) GitHub Repository Creation

- Name: CiviWave-FEM
- Initialize:
  - LICENSE (AGPL-3.0-or-later)
  - .gitignore (C++, CMake, GCC/MinGW, build/, out/, .vscode/, .idea/)
  - README with overview, hardware target, licensing
- Branches:
  - main (protected)
  - dev (default for development PRs)
- Protections:
  - main requires PR + status checks (CI)
  - signed commits optional
- Templates:
  - Issues (bug, feature)
  - PR template
- Project board:
  - Milestones M1..M6 (see Roadmap)
- Discussions: enabled

---

## 2) Local Dev Environment Setup (GCC toolchain)

Windows:
- Install MSYS2 or WinLibs MinGW-w64 (GCC 15.x).
- Install CMake 4.1.2.
- Install Vulkan SDK 1.4.328.1 (for headers/tools; runtime feature gating).
- Install Python 3.11+ (build scripts).
- Install Git, Git LFS.
- Install RGP, RGA.
- Ensure glslang (from SDK) is available if Slang needs it for SPIR-V validation.

Linux:
- GCC 15.x from distro or toolchain PPA.
- CMake 4.1.2 (from Kitware APT or tarball).
- Vulkan SDK as appropriate for distro.

Verify:
- vulkaninfo shows subgroup size = 64; double precision = yes.
- slangc --version prints 2025.18 (or will be built via FetchContent).

---

## 3) Repository Structure

- /cmake/ (toolchain, FetchContent modules, presets)
- /external/ (optional mirrors; generally use FetchContent)
- /src/
  - core/ (app, main, config YAML, logging)
  - gpu/ (device, queues, memory, descriptors, pipelines, sync)
  - physics/ (materials, loads, damping, integrators)
  - mesh/ (io, preprocess, adjacency, packing)
  - solver/ (pcg, preconditioners, reductions)
  - shaders/ (Slang modules and kernels)
  - viz/ (optional graphics)
  - io/ (VTK/VTU export)
  - util/ (portability, extensions, alignment, profiling)
- /assets/ (meshes, scenarios YAML)
- /tests/ (unit, integration)
- /docs/ (spec, plan, validation, tuning)
- /tools/ (converters, profiling scripts)
- /scripts/ (build, run, pack, environment detection)

---

## 4) Build System and Tooling

- CMake:
  - Set CMAKE_CXX_STANDARD 26 (or use -std=c++2c with target_compile_options).
  - Detect GCC≥15.2; error otherwise.
  - Options: BUILD_TESTS, BUILD_UI, ENABLE_TRACY, ENABLE_VALIDATION, ENABLE_RGP_MARKERS, FORCE_FETCH_DEPS.
  - Policies: new behavior for CMP0141+ etc.
  - Presets: Debug, Release, RelWithDebInfo, Profile (LTO off by default).
- FetchContent (source builds if needed):
  - Slang (tag v2025.18)
  - VMA (v3.3.0)
  - yaml-cpp (v0.8.0, BUILD_TESTING=OFF, shared OFF)
  - Optional: Tracy, ImGui
- Shader build:
  - Custom CMake rule to compile .slang to .spv with slangc.
  - Cache artifacts under build/shaders/.
- Formatting and static analysis:
  - .clang-format and .clang-tidy (advisory on GCC build).
  - pre-commit hooks.

---

## 5) CI (GitHub Actions)

- Windows (MSYS2/MinGW GCC) and Ubuntu (GCC):
  - Install GCC 15.x, CMake 4.1.2, Vulkan SDK components (headers only ok), Python.
  - Configure with FORCE_FETCH_DEPS=ON to build dependencies from source.
  - Build Debug/Release, run tests, clang-format check.
- Cache _deps source and build trees keyed by compiler and flags.
- Artifacts: packaged binaries with shaders and sample assets.

---

## 6) Documentation

- README.md: overview, setup, status badges (CI), license note.
- docs/spec.md: this spec (YAML, Slang, C++26, GCC).
- docs/plan.md: this plan.
- docs/validation.md: target cases and acceptance bands.
- docs/tuning-amd-igpu.md: subgroup, bandwidth, buffer sharding tips.
- docs/perf-methodology.md: timestamps, RGP capture, interpretation.
- docs/decisions.yaml: ADRs and version pins.

---

## 7) Core App Scaffolding (CPU first)

- YAML config reader (yaml-cpp 0.8.0).
- Mesh I/O (Gmsh v4):
  - Nodes, elements, groups parsing.
  - Validation: inverted elements, duplicates.
- Preprocess:
  - Build node↔element adjacency + local indices.
  - Compute gradN and volumes for tets; lumped mass.
- Physics:
  - Materials (E, ν, ρ), D matrix.
  - Loads: gravity, nodal, tractions, time curves from YAML.
- Integrator:
  - Newmark predictor formulas (CPU reference).
- CPU reference solver:
  - Assembled CSR + CG (small meshes).
- Unit tests and baseline validations.

---

## 8) Vulkan Device Layer + Slang Integration

- Instance/device creation with required feature negotiation:
  - shaderFloat64, timelineSemaphore, synchronization2, descriptorIndexing, EXT_descriptor_buffer, bufferDeviceAddress, subgroupSizeControl, shaderSubgroupExtendedTypes.
- Queue selection (compute queue with timestamps).
- Memory with VMA.
- Descriptor strategy:
  - Prefer EXT_descriptor_buffer to reduce set churn.
- Sync:
  - Timeline semaphores and pipeline barrier helpers.
- Debug:
  - Validation layers in Debug.
  - Debug utils labels.
- Slang:
  - Fetch slang source; build slangc/slang runtime as part of configure.
  - CMake rule: compile shaders/*.slang to SPIR-V.
  - Slang module structure:
    - common.slang: math, types
    - reductions.slang: subgroup reduction wrappers
    - bc.slang: DOF masks
    - ke_apply_element.slang, ke_gather_node.slang, pcg_precondition.slang, pcg_reduce.slang, newmark_predictor.slang, newmark_update.slang, derive_fields.slang

---

## 9) Data Layout and GPU Buffers

- SoA buffers for nodes and elements as in spec.
- Solver vectors and reduction partials.
- Buffer sharding infra:
  - Logical array view abstraction → multiple VkBuffers bound via descriptor indexing.
- Upload/staging:
  - Single bulk upload; persistent ring buffer for dynamic updates (loads/params).

---

## 10) ApplyKeff (Matrix-Free) and Kernels

- Element pass: compute ε, σ, fe_local (FP32).
- Node gather: sum to y (FP32), add M/C terms, mask BCs.
- FP64 never used in hot loops; only in reductions.
- Workgroup sizing and subgroup size control set to 64.

---

## 11) PCG Solver (GPU) with Two-Pass Reductions

- Vectors in FP32; block-Jacobi preconditioner (3×3 per node).
- Two-pass reductions:
  - Per-workgroup partials (double) via subgroup reduce.
  - Final reduce kernel to scalar (double).
- Orchestrate loop with timeline semaphores; optional device-side loop in future.

---

## 12) Implicit Newmark Integration (GPU-assisted)

- Predictor, assemble RHS.
- K_eff coefficients (a0, a1) via push constants/uniforms.
- PCG solve for Δu.
- State updates.
- Adaptive dt, tolerance scheduling.

---

## 13) BCs and Loads (YAML)

- Dirichlet:
  - bc_mask per DOF via YAML groups; enforce in operator and vectors.
- Neumann:
  - Surface groups and load magnitudes from YAML.
- Time histories:
  - Piecewise-linear curves in YAML; evaluate per-frame on GPU or upload scalar parameters.

---

## 14) Derived Fields and Visualization

- Compute ε, σ per element; von Mises.
- Aggregate to nodes (gather).
- Optional graphics:
  - Simple Vulkan pipeline to draw deformed mesh colored by stress.
- Export VTU for ParaView.

---

## 15) Performance Instrumentation

- Vulkan timestamps around each pass.
- Per-frame JSON/YAML log of pass timings, PCG iterations, residuals.
- RGP markers and capture scripts.
- Tracy zones and GPU context optional.

---

## 16) Validation Suite

- Static: cantilever beam, plate, uniaxial block.
- Dynamic: modal frequencies, harmonic response, pulse.
- Regression:
  - CPU vs GPU mixed precision on small meshes.
  - Energy checks.

---

## 17) AMD iGPU Tuning

- Subgroup size fixed 64 → requireSubgroupSize = 64.
- Memory bandwidth:
  - Coalesced SoA, minimize passes, consider recompute vs load of B.
- Reductions:
  - Two-pass in FP64.
- Preconditioner:
  - Block-Jacobi initially; AMG planned.
- Buffer caps:
  - Shard arrays; use descriptor buffer.

---

## 18) Advanced Preconditioning (Roadmap)

- AMG (aggregation/smoothed aggregation).
- Geometric multigrid for hex meshes.
- Config toggles; performance comparisons.

---

## 19) Error Handling and Stability

- Mesh validity checks.
- BC conflicts detection.
- Solver safeguards for stagnation/NaN.
- Graceful fallback for missing features.
- Logging via spdlog; diagnostics on crash.

---

## 20) UI/UX (Optional)

- ImGui overlay controls (dt, tol, preconditioner, fields).
- Camera controls, probe readouts.

---

## 21) Packaging and Distribution

- Layout:
  - bin/CiviWaveFEM
  - shaders/*.spv
  - assets/ (meshes, YAML scenarios)
  - LICENSE
- No opaque prebuilt deps; everything built from source or verified system packages.
- Releases with versioned artifacts and documentation.

---

## 22) Finishing Passes

- Code cleanup, docs, doxygen (optional).
- Numerical polish (scaling utilities, adaptive tol).
- Kernel fusion and memory traffic reductions.
- Device feature detection and fallbacks hardened.
- Validation benchmark report with plots.
- v1.0.0 release.

---

## 23) Milestones

- M1: CPU reference, YAML I/O, static CSR CG, docs skeleton.
- M2: Vulkan scaffolding, Slang build, matrix-free Keff (FP32), two-pass reductions.
- M3: PCG loop (mixed precision), BCs/loads (YAML), Newmark implicit.
- M4: Block-Jacobi solid, derived fields, optional rendering, perf pass (50k–150k DOFs @ 10–30 Hz).
- M5: AMG preconditioner, corotational option, sharding finalized, UI controls.
- M6: Finishing passes, validation suite complete, packaging, v1.0.0.

---

## 24) Acceptance Criteria

- Correctness: 1–5% bands vs analytical/CPU references.
- Performance: 10–30 FPS for 50k–150k DOFs on AMD iGPU.
- Stability: No validation errors in Debug; graceful feature fallbacks.
- Usability: YAML configs, sample meshes, VTU export, optional UI.

This plan reflects Slang, YAML, C++26 on GCC, Vulkan 1.4, AGPL licensing, and a source-first dependency policy.