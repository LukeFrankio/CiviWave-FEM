# CiviWave-FEM: Vulkan FEM Simulation Spec (Civil Engineering, Near Real-Time, AMD iGPU, Slang, YAML, C++26, GCC)

Goal: Build a C++26 (C++2c) Vulkan 1.4 application that simulates internal/external forces, strain, and stress of homogeneous, linear elastic structures in near real-time on the GPU, tuned for the provided AMD integrated GPU. Use Slang for shaders, YAML for configuration, GCC as primary compiler, CMake as build system, AGPL-3.0-or-later license, and FetchContent to build all dependencies from source when needed.

Applies to: Buildings, bridges, frames, shells, and solids in small-strain regime, optionally corotational for moderate rotations.

License: GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later)

Primary tooling (latest as of 2025-11-03):
- GCC: 15.2 (C++26 via -std=c++2c; gnu++2c optional)
- CMake: 4.1.2
- Vulkan API: 1.4 (drivers/device features gate actual usage)
- Vulkan SDK (LunarG): 1.4.328.1
- Slang (shading language and compiler): 2025.18
- Vulkan Memory Allocator (VMA): 3.3.0
- yaml-cpp: 0.8.0

Compiler extensions usage: Extensive use of compiler extensions for performance (notably restrict via __restrict__/__restrict), attributes, builtin vector ops when beneficial, and GNU extensions if guarded and portable fallbacks exist.

Configuration format: YAML (no JSON). All app scenarios, materials, damping, loads, and runtime options are specified in YAML.

Dependency policy: Prefer system packages if they meet version and compiler ABI constraints; otherwise auto-fetch and build from source via CMake FetchContent on first build. Never rely on prebuilt binaries when compilers/flags/ABIs mismatch.

---

## 0) Target Hardware Profile (your device)

- Device: AMD Radeon(TM) Graphics (integrated GPU)
- Vulkan: 1.3 device reported; target programming model uses Vulkan 1.4 features only when available. Feature/extension gates applied at runtime.
- Key features (from device caps):
  - shaderFloat64: yes (use sparingly; low throughput relative to FP32)
  - Subgroups: wave64 fixed; computeFullSubgroups = 1; subgroup ops in all stages
  - Timeline semaphores: yes
  - Descriptor indexing and EXT_descriptor_buffer: yes
  - Buffer device address: yes
  - Synchronization2: yes
  - EXT_shader_atomic_float present but no atomicAdd for FP32/FP64
- Limits to respect:
  - Max buffer size: 2 GB
  - Workgroup shared memory: 32 KB
  - Subgroup size: 64 (fixed)

Implications:
- Matrix-free, gather-based assembly (no float atomics).
- Mixed precision: FP32 heavy, FP64 for reductions and sensitive accumulators.
- Workgroup sizes 128–256 tuned for wave64.
- Buffer sharding for arrays >2 GB via descriptor indexing/descriptor buffer.

---

## 1) Scope and Success Criteria

- Physics: 3D linear elasticity (optionally corotational), homogeneous isotropic materials.
- Outputs per frame:
  - Displacements u, velocity v, acceleration a
  - Strain ε, stress σ (Cauchy; 2nd PK if corotational)
  - Reaction forces at constrained DOFs
- Dynamics: Transient response with damping; static as special case.
- Near real-time on this iGPU:
  - 10–30 Hz for ~50k–150k DOFs with mixed precision + preconditioning.
- Accuracy: Engineering-grade; validated against analytical/CPU references.
- Stability: Implicit integration favored; explicit optional for small soft problems.

---

## 2) Governing Equations and Model

- Semi-discrete FEM:
  M ü + C u̇ + K u = f_ext(t)
- Linear isotropic:
  σ = D ε, with D from E, ν
- Small strain ε = sym(∇u). Optional corotational formulation for moderate rotations.
- Damping: Rayleigh C = α M + β K
- Loads: Body forces, surface tractions, nodal point loads, time histories.

---

## 3) Discretization and Elements

- Mesh: Tetrahedra (unstructured) first; Hexahedra (structured) optional.
- Nodes: 3 translational DOFs.
- Precompute per element: reference volume, ∇N, B (compact), material constants, lumped mass.
- Boundary conditions:
  - Dirichlet via elimination/masking (preferred to penalty).
  - Neumann via surface integration to nodal loads.

---

## 4) Time Integration (Implicit, iGPU-tuned)

- Newmark-beta (average acceleration): γ = 0.5, β = 0.25.
- Solve per step: (K_eff) Δu = r with K_eff = K + a0 M + a1 C.
- Update:
  - u_{n+1} = u_n + Δu
  - v_{n+1}, a_{n+1} via standard Newmark updates.
- Static: K u = f (set inertia/damping zero or very large Δt).

Timestep guidance:
- Start Δt = 1/120–1/60 s; adapt based on residuals and frame budget.

---

## 5) Solver (Matrix-Free PCG, No Float Atomics)

- PCG for SPD systems; operator is matrix-free.
- y = K_eff x computed via element kernels + diagonal M, C.
- Preconditioners:
  - Start with block-Jacobi 3×3 per node (FP32 store; FP64 optional).
  - Roadmap: aggregation AMG for tets; geometric multigrid for hexes.
- Reductions:
  - Subgroup (wave64) reductions to one value per workgroup.
  - Two-pass global reductions writing per-workgroup partials, then final reduce.
- Convergence:
  - Relative residual tol: 1e-4–3e-4 (configurable).
  - Iteration cap to meet frame budget; warm-start from previous Δu.

---

## 6) Precision Plan (Mixed Precision)

- FP32 for hot loops:
  - u, v, a, f_ext, p, r, Ap, z, element math (B, ε, σ).
- FP64 for sensitive scalars:
  - Dot products, norms, global residuals (two-pass reductions).
  - Optional: preconditioner storage in FP64 if conditioning is poor.
- Visualization fields: FP32.
- Verification mode: FP64-only for small meshes.

---

## 7) GPU Data Layout and Buffering (2 GB cap aware)

- Nodes (N):
  - pos0[N]: float3 (double3 retained CPU-side if needed)
  - u[N], v[N], a[N], fext[N]: float3
  - bc_mask[N]: uint32; bc_values[N]: float3 (optional)
- Elements (Ne):
  - conn[Ne]: uint4 (tets) / uint[8] (hex)
  - gradN[Ne]: compact per-element gradients (float)
  - mat[Ne]: material constants or ids
  - vol[Ne]: float
  - mass_lumped[N]: float
- Adjacency:
  - nodeElemOffsets[N+1], nodeElemIndices[nnz], nodeLocalLIDs[nnz]
- Solver vectors:
  - pcg_x, pcg_r, pcg_p, pcg_Ap, pcg_z (float3)
  - partial_sums[WG_count] (double)
- Buffer sharding:
  - Auto-split arrays >2 GB; bind shards via descriptor indexing/descriptor buffer.

---

## 8) Vulkan and Slang Requirements

- Vulkan target: 1.4 (use 1.3 device if 1.4 not exposed; gate features).
- Required features:
  - shaderFloat64
  - timelineSemaphore
  - synchronization2
  - descriptorIndexing + EXT_descriptor_buffer
  - bufferDeviceAddress
  - subgroupSizeControl (set requiredSubgroupSize = 64)
  - shaderSubgroupExtendedTypes
- Shader language: Slang (emitting SPIR-V).
  - Compile .slang to SPIR-V with slangc (version 2025.18).
  - Use Slang generics/modules for shared math, reductions, BC masking.
- Avoid:
  - FP32/FP64 atomicAdd (not available) — use gather + two-pass reductions.

---

## 9) Per-Frame Compute Passes (Implicit Step)

1) UpdateLoads (FP32)
   - Evaluate time-dependent loads f_ext(t_{n+1}) from YAML-defined curves.

2) Predictor (FP32)
   - Compute predictor terms for RHS assembly.

3) PCG Loop
   - ApplyKeff(p): y = K p + a0 M p + a1 C p
     - Element kernel: ε = B p_e, σ = D ε, fe_local = B^T σ vol
     - Node gather: sum fe_local to nodes; no atomics
     - Add M, C diagonal terms; apply BC masks
   - Precondition: per-node 3×3 block solve
   - Dot products/norms: subgroup reduce (wave64), write partial doubles, final reduce
   - AXPY updates (FP32), iterate to tol or maxIters

4) State Update (FP32)
   - Apply Δu; update u, v, a.

5) Derived Fields (FP32)
   - ε, σ, von Mises; aggregate to nodes for visualization.

6) Render (optional)
   - Deform mesh x = x0 + u; color by scalar fields.

---

## 10) Element Kernels (Matrix-Free, Wave64)

- Prefer recomputing B via gradN on device to reduce bandwidth if profiling indicates benefit.
- Workgroup sizes: 128–256 (2–4 wave64s) per group.
- Subgroup ops for inner reductions.
- Gather pattern via precomputed node→element incidence.

---

## 11) Boundary Conditions

- Dirichlet:
  - Zero constrained DOFs in p and outputs during ApplyKeff; mask residuals.
  - Optional free-DOF index lists to skip masked entries in vector ops.
- Neumann:
  - Surface integration to nodal loads; cache face data where helpful.

---

## 12) Damping Selection

- Rayleigh parameters α, β from target damping ratio ξ at ω1, ω2:
  α = 2 ξ ω1 ω2 / (ω1 + ω2), β = 2 ξ / (ω1 + ω2)
- Configure via YAML.

---

## 13) Precision, Scaling, Conditioning

- Keep units scaled to O(1) to improve conditioning.
- Use mixed precision by default; FP64-only mode for verification.
- Adaptive tolerances/Δt to meet frame budgets while preserving accuracy.

---

## 14) I/O and Preprocessing (YAML-centric)

- Mesh input: Gmsh (.msh v4) or Abaqus .inp.
- Scenario/config: YAML only (materials, damping, Δt, loads, BCs).
- CPU preprocessing:
  - Build adjacency and local indices.
  - Compute gradN, volumes, lumped masses, material mapping.
  - Pack SoA buffers; upload to GPU once.

---

## 15) Performance Targets and Budget (AMD iGPU)

- Example: ~100k nodes (~300k DOFs) mixed precision with block-Jacobi:
  - 15–20 FPS (50–67 ms):
    - Loads/Predictor/Update: 4–8 ms
    - PCG: 35–50 ms for 20–60 iters (mesh/BC/damping dependent)
    - Derived fields: 5–8 ms
    - Render: 2–4 ms

Iteration counts >80 → upgrade preconditioner or adjust Δt/tol.

---

## 16) Validation Plan

- Static: cantilever beam, thick plate, uniaxial block.
- Dynamic: free vibration frequencies, harmonic load, pulse response.
- CPU vs GPU mixed precision comparisons on small meshes.
- Energy consistency checks each step.

---

## 17) Software Architecture

- Core modules:
  - mesh (YAML-driven scenarios, I/O, preprocess)
  - physics (materials, loads, damping)
  - integrator (Newmark)
  - solver (PCG, preconditioners, reductions)
  - gpu (Vulkan device, memory, descriptors, pipelines, sync)
  - shaders (Slang kernels)
  - viz (optional graphics)
  - io (VTU/VTP outputs)
- Build: CMake ≥ 4.1, GCC ≥ 15.2, -std=c++2c
- Dependencies via FetchContent (source builds): Slang, VMA, yaml-cpp, optional Tracy/ImGui
- License: AGPL-3.0-or-later

---

## 18) Compiler and Extensions Policy (GCC-focused)

- Standard: -std=c++2c (C++26), optionally -std=gnu++2c for GNU extensions in non-portable sections guarded by macros.
- Extensions:
  - restrict: use __restrict/__restrict__ for pointer aliasing control; provide macro CW_RESTRICT mapping to compiler-specific tokens.
  - Attributes: [[gnu::always_inline]], [[gnu::flatten]], [[gnu::hot]], [[gnu::cold]], [[gnu::assume_aligned]], [[gnu::malloc]] where beneficial and safe.
  - Builtins: __builtin_assume, __builtin_expect, vector_size for small fixed-width vector math if profiling shows gains.
  - Pragmas: GCC ivdep/loop hints conservatively.
- Safety:
  - Respect strict aliasing rules when asserting restrict; provide defensive fallbacks.
  - Centralize macros in a portability header; compile-time asserts to verify alignments/assumptions.

---

## 19) Dependency Sourcing Policy (FetchContent-first)

- If system packages meet:
  - Minimum version
  - Compiler ABI/flags compatibility (same compiler family/version major)
  - Expected build options (e.g., no exceptions if configured)
  Then allow system package; else FetchContent source build on first configure.
- Never download or use opaque prebuilt binaries for core libs.
- Cache source builds under build/_deps; reconfigure triggers rebuild on compiler/flag change.

---

## 20) Default Configuration (YAML, this iGPU)

- material:
  - E: 30e9
  - nu: 0.2
  - rho: 2500
- damping:
  - xi: 0.02
  - w1: 10.0
  - w2: 100.0
- time:
  - dt: 0.01111 # 1/90 s
  - adaptive: true
- solver:
  - type: pcg
  - tol_runtime: 2.0e-4
  - tol_pause: 1.0e-5
  - max_iters: 120
  - preconditioner: block_jacobi
- precision:
  - vectors: fp32
  - reductions: fp64
- output:
  - vtu_stride: 10
  - probes: [node: 123, node: 456]

This spec reflects Slang shaders, YAML configs, GCC/C++26, AGPL licensing, and a source-first dependency policy for a near real-time, engineering-accurate FEM solver on your AMD iGPU.