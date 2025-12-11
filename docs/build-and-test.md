# Build and test guide

A concise, up-to-date walkthrough for configuring, building, and testing CiviWave-FEM. This reflects the current repository state on 2025-12-11.

## Requirements (latest-preferred)

- GCC 15.2+ (C++26 / `-std=c++2c`). Clang/MSVC are best-effort only.
- CMake 4.1.2+ with presets enabled. Ninja is selected by default via presets.
- Vulkan SDK 1.4.328.1+ (runtime requires Vulkan 1.3; 1.4 features are gated at runtime).
- Slang `slangc` available on `PATH` **or** via `SLANGC` env var. If not provided, the build will emit stub `.spv` outputs so compilation still succeeds.
- Doxygen 1.15+ (for API docs), Graphviz optional for diagrams.
- Python 3.11+ (scripts), Git, Git LFS.

See `docs/versions.yaml` for the canonical pins; prefer the latest stable or beta releases when available.

## Configure with CMake presets

Presets are case-sensitive (`Debug`, `Release`, `RelWithDebInfo`, `Profile`). Binary dirs live under `build/<PresetName>`.

```bash
cmake --preset Debug
cmake --preset Release
```

Key cache options (defaults shown):

- `FORCE_FETCH_DEPS=ON` — build deps from source unless ABI-compatible system packages are found.
- `ENABLE_VALIDATION=ON` — enable Vulkan validation layers in Debug-like builds.
- `ENABLE_SANITIZERS=ON` — ASan/UBSan in Debug/Profile on GCC/Clang.
- `FETCH_SLANG=OFF` — opt-in to building Slang from source (heavy); prefer providing a prebuilt `slangc`.
- `BUILD_UI=OFF` — optional ImGui/GLFW viewer (`cwf_viewer_demo`) when ON.
- `ENABLE_TRACY=OFF` — Tracy profiler integration; pair with `Profile` preset.
- `ENABLE_RGP_MARKERS=ON` — emit RGP markers when profiling on AMD GPUs.
- `REQUIRE_GCC_15=ON` — enforce GCC 15.2+; set OFF only in CI with older toolchains.

Pass overrides on the configure command if needed, e.g.:

```bash
cmake --preset Debug -DFETCH_SLANG=ON -DBUILD_UI=ON
```

## Build

Use matching build presets or build directories:

```bash
cmake --build build/Debug -j
cmake --build build/Release -j
```

Shaders compile to `build/<Preset>/shaders`. If `slangc` is absent, stub SPIR-V files are generated so the build still succeeds (GPU execution will be disabled until real shaders are provided).

## Tests

GoogleTest-based suites live in `tests/` and build into `cwf_core_tests`.

- Run all tests for a preset:

  ```bash
  ctest --preset Debug
  ```

- Or from a build directory:

  ```bash
  ctest --test-dir build/Debug --output-on-failure
  ```

- Filter tests:

  ```bash
  ctest --test-dir build/Debug -R pcg
  ```

GPU-dependent tests are limited; most suites exercise CPU-side packing, preprocessing, solver orchestration, instrumentation, and VTU export. Ensure a Vulkan 1.3-capable device is present for any GPU integration cases.

## Running the optional viewer

If `BUILD_UI=ON`, CMake builds `cwf_viewer_demo` and compiles `viewer_mesh_{vert,frag}.slang` to SPIR-V. Run the viewer from its build output directory so it can discover shaders, or copy the `shaders/` directory alongside the executable (CMake adds a post-build copy step for convenience).

## Environment knobs and paths

- `SLANGC` — absolute path to `slangc` (recommended). If unset, the build falls back to stub shaders.
- `VULKAN_SDK` — set by the SDK installer; must point to a Vulkan 1.4.328.1+ SDK.
- Validation layers — keep `ENABLE_VALIDATION=ON` in Debug to surface Vulkan issues.

## When in doubt

- Start with `cmake --preset Debug`, build, and run `ctest --preset Debug`.
- If shader compilation fails, set `SLANGC` explicitly or enable `FETCH_SLANG` (heavier, slower).
- For profiling workflows, see `docs/profiling.md` and use the provided RGP helper scripts.
