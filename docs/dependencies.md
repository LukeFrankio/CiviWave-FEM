# Dependency sourcing and build policy (source-first)

We strongly prefer building dependencies from source with versions pinned in `docs/versions.yaml`. System packages may be used if and only if they meet strict criteria; otherwise we fall back to CMake `FetchContent` to fetch and build from source on first configure/build.

## Acceptance criteria for system packages

A system package can be used when all of the following are true:

- Version is greater than or equal to the minimum in `docs/versions.yaml`.
- Compiler family and major version match the project toolchain (e.g., GCC 15.x), or ABI compatibility is guaranteed by the provider.
- Build options match project expectations (examples below):
  - `yaml-cpp`: static library, `BUILD_TESTING=OFF`.
  - VMA: header-only; version must match pin.
  - Doxygen: 1.15+ with Graphviz available for diagrams.
- No vendor-specific patches that change ABI without version bump.

If any check fails, we build from source via `FetchContent`.

## FetchContent usage

- All third-party libraries are declared with exact tags or commit hashes.
- Source is cached under `build/_deps` and reused across builds.
- Reconfiguration triggers rebuilds when the compiler or flags change (ABI fingerprinting in CMake modules).

## Never use opaque prebuilt binaries

- We do not download or ship opaque prebuilt binaries for core libraries.
- Platform packages are acceptable if they pass the acceptance criteria.

## CI integration

- CI jobs set `FORCE_FETCH_DEPS=ON` to ensure consistency across environments.
- Cache `_deps` source and build trees keyed by compiler version and critical flags.
- CI validates versions against `docs/versions.yaml` and fails fast when too old.

## Libraries and expectations

- Slang: build `slangc` and runtime from source at the pinned tag.
- VMA: header-only; include at the pinned version.
- `yaml-cpp`: static library; tests off; exceptions per project policy.

## Upgrade workflow

1. Check upstream releases for Slang, VMA, `yaml-cpp`, and Doxygen.
2. Update `docs/versions.yaml` preferred versions (and minimum if justified).
3. Update CMake pins (FetchContent tags) to match.
4. Build locally with `FORCE_FETCH_DEPS=ON` on Windows and Linux.
5. Run tests and validation; inspect performance snapshots.
6. Submit PR with rationale, changelog notes, and any required code changes.
