# CiviWave-FEM

CiviWave-FEM is a C++26 + Vulkan playground for matrix-free FEM targeting an
AMD Radeon™ integrated GPU (wave64, shaderFloat64). Today the repo ships a
library plus comprehensive tests, Vulkan device/runtime scaffolding, and a
CPU+GPU Newmark/PCG pipeline; an optional ImGui viewer can be enabled when
`BUILD_UI=ON`. No CLI front-end is bundled yet—tests and the viewer are the
entry points for now.

## Hardware vibe check

- Target device: AMD Radeon™ Graphics (wave64 subgroups, shaderFloat64 enabled)
- Buffer cap: ~2 GB device-local allocations (descriptor-buffer sharding keeps
   it chill)
- Mixed precision: FP32 hot loops, FP64 reductions for solver swagger

## Feature snapshot (2025-12-11)

- Matrix-free implicit FEM solver with block-Jacobi PCG (CPU implementation
   mirrors the GPU path), plus Vulkan device/buffer/descriptor orchestration for
   GPU execution.
- Slang shader toolchain (2025.18) compiling to SPIR-V when `slangc` is
   available; stub SPIR-V emitted otherwise so builds still succeed.
- YAML-driven scenarios: materials, damping, loads, curves, solver tolerances,
   precision, Dirichlet constraints, and VTU/probe output controls.
- Source-first dependency policy via CMake FetchContent; system packages used
   only when ABI/version-compatible.
- GoogleTest suite covering config validation, mesh import, preprocessing,
   derived fields, matrix-free PCG, Newmark integration, instrumentation, and
   export.

## Toolchain flex

| Tool            | Minimum | Preferred | Notes |
|-----------------|---------|-----------|-------|
| GCC             | 15.2    | 15.2+     | `-std=c++2c`; Clang/MSVC best-effort |
| CMake           | 4.1.2   | 4.1.2+    | Presets under `CMakePresets.json` |
| Vulkan SDK      | 1.3 (runtime) | 1.4.328.1+ | Build requires 1.3; 1.4 features gated at runtime |
| Slang           | 2025.18 | 2025.18+  | Prefer prebuilt `slangc`; `FETCH_SLANG=ON` is heavy |
| yaml-cpp        | 0.8.0   | 0.8.0+    | Fetched unless ABI-compatible system lib is found |
| Vulkan Memory Allocator | 3.3.0 | 3.3.0+ | Header-only pin |
| Doxygen         | 1.15    | 1.15+     | For API docs (not built by default) |

## Project structure

- `docs/` — contributor-facing docs, ADRs (`decisions.yaml`), setup guides, build/test how-tos.
- `RefDocs/` — long-form spec, plan, and TODO guidance.
- `cmake/` — custom modules for toolchain probing and shader compilation.
- `src/` — core library (config, mesh, preprocessing, physics, GPU orchestration, post-processing, shaders).
- `tests/` — GoogleTest suites plus fixtures under `tests/data/` and helpers under `tests/support/`.
- `scripts/` — profiling helpers (RGP) and utilities.
- `build/` — generated build trees when presets are used (ignored by git).

## Quickstart

1. Install toolchain (see `docs/dev-setup-windows.md` or `docs/dev-setup-linux.md`). Prefer latest stable/beta releases.
2. Configure: `cmake --preset Debug` (or `Release`/`RelWithDebInfo`/`Profile`).
3. Build: `cmake --build build/Debug -j`.
4. Tests: `ctest --preset Debug` (or `--test-dir build/Debug --output-on-failure`).
5. If you have `slangc`, set `SLANGC=/path/to/slangc`; otherwise stub shaders are emitted and GPU execution will be disabled until real SPIR-V is present.

Optional viewer: configure with `-DBUILD_UI=ON` (and set `SLANGC`). CMake copies compiled shaders next to the viewer binary.

## Contribution flow

1. Fork the repo, branch from `dev`, and sync often (no mega-PRs please).
2. Configure with presets and let FetchContent build pinned deps when ABI drift is detected (`FORCE_FETCH_DEPS=ON`).
3. Keep code clang-formatted, clang-tidy clean, and drowning in Doxygen comments with ✨ pure function energy ✨. Run the Doxygen target when comments change.
4. Open a PR into `dev`, expect CI with -Wall -Wextra -Werror, and request review before merging.

The full playbook lives in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Documentation drip

- Quick build/test guide: [`docs/build-and-test.md`](docs/build-and-test.md)
- Hand-holding setup: [`docs/setup-for-dummies.md`](docs/setup-for-dummies.md)
- YAML schema: [`docs/configuration.md`](docs/configuration.md)
- Profiling helpers (RGP/Tracy): [`docs/profiling.md`](docs/profiling.md)
- Big-picture plan: [`docs/plan.md`](docs/plan.md) and [`RefDocs/PLAN.md`](RefDocs/PLAN.md)
- Spec highlights: [`docs/spec.md`](docs/spec.md) and [`RefDocs/SPEC.md`](RefDocs/SPEC.md)
- Validation matrix: [`docs/validation.md`](docs/validation.md)
- AMD tuning cheat sheet: [`docs/tuning-amd-igpu.md`](docs/tuning-amd-igpu.md)
- Architectural decisions: [`docs/decisions.yaml`](docs/decisions.yaml)
- API docs: generate with Doxygen 1.15+ (target forthcoming)

## License

CiviWave-FEM is released under the
[GNU Affero General Public License v3.0 or later](LICENSE). Keep it copyleft,
and keep the vibes immaculate.
