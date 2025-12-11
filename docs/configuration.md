# YAML configuration schema

This document describes the current YAML schema consumed by the CiviWave-FEM library (2025-12-11). All fields map directly to `cwf::config::Config` in `include/cwf/config/config.hpp`. Units are SI unless noted otherwise.

## Top-level layout

```yaml
mesh:
  path: assets/meshes/cantilever.msh
materials:
  - name: steel
    E: 30.0e9
    nu: 0.3
    rho: 7850
assignments:
  - group: SOLID
    material: steel
curves:
  ramp:
    - [0.0, 0.0]
    - [1.0, 1.0]
loads:
  gravity: [0.0, 0.0, -9.81]
  tractions:
    - group: TOP
      value: [0.0, 0.0, -1.0e5]
      scale_curve: ramp
  points:
    - group: TIP
      value: [0.0, 0.0, -500.0]
      scale_curve: ""
dirichlet:
  - group: FIXED
    constrain_axis: [true, true, true]
    value: [0.0, 0.0, 0.0]
damping:
  xi: 0.02
  w1: 10.0
  w2: 100.0
time:
  initial_dt: 0.01111
  adaptive: true
  min_dt: 0.0
  max_dt: 0.0
solver:
  type: pcg
  preconditioner: block_jacobi
  runtime_tolerance: 2.0e-4
  pause_tolerance: 1.0e-5
  max_iterations: 120
precision:
  vector_precision: fp32
  reduction_precision: fp64
output:
  vtu_stride: 10
  probes: [1, 42, 256]
```

## Sections and constraints

### `mesh`

- `path` (string, required): Mesh file path (Gmsh v4 ASCII currently supported). Relative paths are allowed.

### `materials` (required, non-empty sequence)

Each entry:

- `name` (string): Unique material identifier.
- `E` (double): Young’s modulus (Pa) > 0.
- `nu` (double): Poisson ratio 0 < ν < 0.5.
- `rho` (double): Density (kg/m³) > 0.

### `assignments` (required, non-empty sequence)

Map mesh physical groups to material names:

- `group` (string): Physical group name from the mesh.
- `material` (string): Material name defined above.

### `curves` (optional map)

Piecewise-linear curves keyed by ID. Each curve is an ordered list of `[time, value]` pairs (time must be sorted ascending). Used to scale loads.

### `loads`

- `gravity` (vec3): Gravity vector (m/s²). Set to `[0,0,0]` to disable.
- `tractions` (sequence, optional):
  - `group`: Surface physical group.
  - `value`: vec3 traction (Pa, direction+magnitude per face normal convention).
  - `scale_curve`: curve ID or empty string for constant.
- `points` (sequence, optional):
  - `group`: Node physical group.
  - `value`: vec3 concentrated load per node (N).
  - `scale_curve`: curve ID or empty string.

### `dirichlet`

Sequence of fixed-DOF specifications:

- `group`: Physical group (typically surfaces) to constrain.
- `constrain_axis`: `[bool, bool, bool]` for x/y/z locks.
- `value`: `[double|null, double|null, double|null]` optional displacement targets. Use `null` to leave an axis unconstrained or zero to clamp to zero displacement.

### `damping`

- `xi` (double): Target damping ratio (0–1).
- `w1`, `w2` (double): Angular frequencies (rad/s) used to derive Rayleigh α/β.

### `time`

- `initial_dt` (double > 0): Starting timestep in seconds.
- `adaptive` (bool): Enable adaptive timestep heuristics.
- `min_dt` / `max_dt` (double ≥ 0): Optional clamps (0 means disabled).

### `solver`

- `type`: Currently `pcg`.
- `preconditioner`: Currently `block_jacobi`.
- `runtime_tolerance`: Relative residual tolerance while running.
- `pause_tolerance`: Tighter tolerance when paused.
- `max_iterations`: Positive integer iteration cap per step.

### `precision`

- `vector_precision`: `fp32` (current implementation).
- `reduction_precision`: `fp64` (current implementation for dot products/reductions).

### `output`

- `vtu_stride` (uint ≥ 1): Write VTU every N frames.
- `probes` (uint sequence): Node indices to record for probe logging.

## Validation rules enforced by the loader

- Root must be a mapping; required sections: `mesh`, `materials`, `assignments`, `damping`, `time`, `solver`, `precision`, `loads`, `dirichlet`, `output`.
- Materials must be unique by `name`; assignments must reference existing materials and mesh groups.
- Curves must be time-sorted and non-empty if provided; references must resolve.
- Dirichlet specs must supply a 3-length `constrain_axis`; `value` can be omitted (`null`) per component.
- Tolerances and time steps must be positive; iteration caps ≥ 1.

## Testing configs

See `tests/data/` for sample meshes and YAML fixtures used by the test suite. The config loader is heavily validated in `tests/config_validation_test.cpp`; use those cases as additional references.
