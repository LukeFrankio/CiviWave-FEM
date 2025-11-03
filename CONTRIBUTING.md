# Contributing to CiviWave-FEM

Thanks for helping make this Vulkan FEM playground extra spicy. This guide keeps
contributions consistent, functional, and dripping with ✨ pure function energy
✨.

## Code of conduct reference

Expectations for kindness, respect, and zero tolerance for harassment live in
[`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md). Read it, live it, enforce it.

## Quickstart

1. Fork the repo and branch from `dev`. Name branches like
   `feature/predictor-kernel` or `bugfix/vulkan-sync` so everyone knows the vibe.
2. Install the toolchain: GCC 15.2+, CMake 4.1.2+, Vulkan SDK 1.4.328.1+,
   Doxygen 1.15+, and Python 3.11+ for scripts. Prefer latest betas when
   available.
3. Configure with CMake presets:
   - `cmake --preset debug` for assertions and validation layers
   - `cmake --preset release` for performance sweeps
   - `cmake --preset profile` when you are ready for Tracy + RGP deep dives
4. Let FetchContent build dependencies from source unless your system packages
   match compiler family, version, and flags exactly. No opaque binaries allowed
   (no cap).

## Coding standards

- Use C++26 with GCC-specific extensions housed in `util/port.hpp` (coming soon).
- Keep everything clang-formatted (`.clang-format`) and clang-tidy clean
  (`.clang-tidy`). Treat warnings as blockers.
- Document every function with maximalist Doxygen comments that call out purity,
  complexity, and Gen-Z commentary. If in doubt, over-document.
- Prefer pure functions, immutable data, and SoA layouts. Side effects live only
  at module boundaries.
- Structure Vulkan code with functional patterns: build immutable descriptors,
  return new states instead of mutating existing ones, and gate features at
  runtime.

## Commit checklist

- [ ] Tests: add or update Google Test coverage for every behavior change.
- [ ] Docs: update Markdown docs, regenerate Doxygen output when comments
   change, and refresh `docs/decisions.yaml` when design choices shift.
- [ ] Format: run `cmake --build build-debug --target clang-format` (or the
      equivalent helper script) before pushing.
- [ ] Static analysis: run clang-tidy via the provided CMake target.
- [ ] Validation: execute the relevant validation scenario or unit test suite.
- [ ] Changelog: note the change in your PR description with context and perf
      signals.

## Pull request flow

1. Push your branch and open a PR targeting `dev`. Title it like
   `[M2] Implement PCG warm-start telemetry` to map back to milestones.
2. Fill out the PR template completely. Link issues from the project board and
   include before/after metrics when touching performance-sensitive code.
3. Expect automated checks:
   - Configure + build (Debug/Release)
   - clang-format and clang-tidy
   - Unit tests (including validation smoke tests)
4. Request review from at least one maintainer. Address feedback quickly and
   keep commits tidy via history rewriting if needed.
5. Once approved, a maintainer will squash-merge into `dev`. Promotion to `main`
   happens via release coordination.

## Docs and ADRs

- Small clarifications go straight into Markdown docs.
- Larger architectural shifts need an ADR entry in `docs/decisions.yaml`. Include
  context, decision, consequences, and follow-up tasks.
- Keep README and docs in sync with actual build or runtime behavior.

## Communication

- Use GitHub Discussions (Q&A, Announcements, Tuning, Validation) for async
  chatter.
- File bugs and feature ideas via the templates. Include platform info, logs,
  and reproduction steps.
- Share performance captures (RGP, Tracy) when discussing GPU tuning. Screenshots
  or it did not happen, fr fr.

Let us keep the solver silky smooth, the shaders unreasonably optimized, and the
community welcoming. uwu 💜
