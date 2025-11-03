# CiviWave-FEM

CiviWave-FEM is a C++26 Vulkan 1.4 playground that makes near real-time
structural analysis on an AMD Radeon™ integrated GPU feel effortless. The
mission is simple: embrace Slang shaders, YAML configs, and a source-first
dependency policy to deliver engineering-grade displacements, strains, and
stresses without leaving the comfort of GCC 15.2.

## Hardware vibe check

- Target device: AMD Radeon™ Graphics (wave64 subgroups, shaderFloat64 enabled)
- Buffer cap: 2 GB device-local allocations (descriptor-buffer sharding keeps it
  chill)
- Mixed precision: FP32 hot loops, FP64 reductions for solver swagger

## Feature tour

- Matrix-free implicit FEM solver orchestrated by Vulkan timeline semaphores
- Slang 2025.18 shader toolchain compiling straight to SPIR-V
- YAML-driven scenarios for materials, damping, loads, and solver tuning
- Source-only dependency management with CMake FetchContent (no opaque blobs)
- Validation suite targeting static and dynamic benchmarks with VTU exports

## Toolchain flex

| Tool            | Minimum | Preferred | Notes                                      |
|-----------------|---------|-----------|--------------------------------------------|
| GCC             | 15.2    | 15.2+     | Invoke with `-std=c++2c` (a.k.a. C++26 uwu) |
| CMake           | 4.1.2   | 4.1.2+    | Presets cover Debug/Release/Profile vibes   |
| Vulkan SDK      | 1.4.328 | 1.4.328+  | Feature gating keeps 1.3 hardware happy    |
| Slang           | 2025.18 | 2025.18+  | slangc builds as part of the configure step |
| yaml-cpp        | 0.8.0   | 0.8.0+    | Pulled via FetchContent unless ABI matched  |
| Vulkan Memory Allocator | 3.3.0 | 3.3.0+ | Header-only goodness, still pinned though |
| Doxygen         | 1.15    | 1.15+     | Generates the documentation drip (CI runs it) |

## Project structure (growing soon™)

- `docs/` — contributor-friendly summaries plus `decisions.yaml` for ADRs
- `RefDocs/` — the long-form spec, plan, and TODO guidance
- `cmake/` — custom modules for toolchain probing and shader compilation (TBD)
- `src/` — core application code (matrix-free solver, Vulkan backend)
- `shaders/` — Slang modules compiled to SPIR-V at build time
- `tests/` — Google Test-powered regression and validation harnesses
- `assets/` — canonical meshes, YAML scenarios, and validation snapshots

## Contribution flow

1. Fork the repo, branch from `dev`, and sync often (no mega-PRs please).
2. Run `cmake --preset <preset>` and let FetchContent build pinned deps from
   source if the local toolchain drifts.
3. Keep code clang-formatted, clang-tidy clean, and drowning in Doxygen
   comments with ✨ pure function energy ✨. Run the Doxygen target when docs
   change to keep HTML output vibing.
4. Open a PR into `dev`, expect CI with -Wall -Wextra -Werror, and request
   review before merging.

The full playbook lives in [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Documentation drip

- Big-picture plan: [`docs/plan.md`](docs/plan.md)
- Spec highlights: [`docs/spec.md`](docs/spec.md) (full version in `RefDocs/`)
- Validation matrix: [`docs/validation.md`](docs/validation.md)
- AMD tuning cheat sheet: [`docs/tuning-amd-igpu.md`](docs/tuning-amd-igpu.md)
- Architectural decisions: [`docs/decisions.yaml`](docs/decisions.yaml)
- API docs: generated with Doxygen 1.15+ once the target lands in CMake

## License

CiviWave-FEM is released under the
[GNU Affero General Public License v3.0 or later](LICENSE). Keep it copyleft,
and keep the vibes immaculate.
