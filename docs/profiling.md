# Profiling and telemetry

A quick guide to collecting performance data with the built-in instrumentation, Radeon GPU Profiler (RGP) helpers, and optional Tracy markers. Status as of 2025-12-11.

## Build-time switches

- `ENABLE_RGP_MARKERS=ON` (default): Emit RGP-friendly debug labels in command streams.
- `ENABLE_TRACY=ON`: Build Tracy client integration (pair with the `Profile` preset). Requires the Tracy server at runtime.
- `ENABLE_VALIDATION=ON`: Keep validation layers enabled in Debug-like builds to catch Vulkan issues early.
- `BUILD_UI=ON`: Needed if you want to profile the optional viewer (`cwf_viewer_demo`).

Ensure `SLANGC` points to a working Slang compiler; otherwise shader stages will be stubbed and GPU passes won’t execute.

## Telemetry structures

The instrumentation layer (see `cwf::gpu::instrumentation`) produces `FrameLog` records:

- Frame index, simulation time, timestep, solver tolerance, pause flag
- Pass timings map (per compute pass durations in ms)
- PCG telemetry (iterations, residual norm, RHS norm, convergence flag)
- Adaptive timestep flags
- Wall-clock budget for the frame

Use `write_frame_log` to serialize a single frame to YAML. Tests in `tests/instrumentation_test.cpp` validate the format.

## RGP capture helpers

Scripts are provided in `scripts/`:

- Windows: `Capture-Rgp.ps1`
- Linux: `capture_rgp.sh`

Prerequisites:

- AMD Radeon GPU Profiler installed and on `PATH`, or `RGP_PATH` env var set.
- Build with `ENABLE_RGP_MARKERS=ON` and, if profiling the viewer, `BUILD_UI=ON`.

Typical usage (after building):

```powershell
# PowerShell
powershell -ExecutionPolicy Bypass -File .\scripts\Capture-Rgp.ps1 -WarmupFrames 120 -CaptureFrames 20
```

```bash
# Linux
./scripts/capture_rgp.sh --warmup 120 --capture 20
```

The scripts:

1. Verify the executable exists (default: `build/bin/cwf_viewer_demo[.exe]`).
2. Locate RGP, prepare environment variables (`AMD_RGP_*`).
3. Launch the app and capture a timestamped `.rgp` trace under `rgp_captures/`.

## Tracy

When `ENABLE_TRACY=ON`, the build links Tracy and enables zones in hot paths. Run the Tracy server locally before launching the instrumented binary. Pair this with the `Profile` preset for `-fno-omit-frame-pointer` and optimizer-friendly flags.

## Tips

- Keep validation layers on while iterating; disable only for final performance sweeps.
- Capture at least two runs: one warm (steady-state) and one cold (to spot upload stalls).
- Note shader revisions and config hashes when comparing captures; log them alongside FrameLog YAML.
- GPU integration depends on having real SPIR-V compiled by `slangc`. If you see stub shaders, set `SLANGC` explicitly or enable `FETCH_SLANG`.
