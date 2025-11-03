# CiviWave-FEM Master Plan

This living plan summarizes the execution roadmap for the project and points to
its full reference document. For deep dives, read the
[comprehensive plan](../RefDocs/PLAN.md) that describes every milestone in
exhaustive detail.

## Phase Overview

- **Phase 0 – Repository bootstrapping:** lay down licensing, contribution
  guidelines, project templates, and documentation scaffolding.
- **Phase 1 – Toolchain policy:** document canonical compiler, SDK, Doxygen,
  and helper tooling versions across Windows and Linux.
- **Phase 2 – CMake skeleton:** implement the build system, dependency policy,
  and CI baselines.
- **Phase 3 – CPU groundwork:** YAML config parsing, mesh ingestion, and the CPU
  reference solver.
- **Phases 4–14:** iterative GPU enablement, Vulkan integration, matrix-free
  solvers, and performance tuning aimed at wave64 AMD hardware.
- **Phases 15–19:** ergonomics, diagnostics, packaging, and release polish.

## Tracking Progress

Progress is tracked through GitHub issues that correspond one-to-one with the
checklist in [TODO.md](../RefDocs/TODO.md). Each issue references the relevant
section of the plan so contributors can align tasks with the broader roadmap.

## Change Management

Major deviations from the spec or plan require a new entry in
[`docs/decisions.yaml`](decisions.yaml) that explains the rationale, the impact
on other phases, and any required follow-up actions.
