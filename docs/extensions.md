# Compiler extensions policy (GCC-focused, C++26)

This document defines how and when we use compiler extensions to squeeze performance while keeping behavior well-defined and portable across compilers as much as practical.

- Standard: C++26 (`-std=c++2c`), optionally `-std=gnu++2c` for guarded GNU extensions.
- Primary compiler: GCC 15.x. Clang/MSVC supported on a best-effort basis.
- Philosophy: prefer zero-cost abstractions; prove wins with profiling; isolate extensions behind macros.

## Restrict semantics (aliasing)

We centralize restrict semantics behind a project macro to assert non-aliasing only when provably safe.

```cpp
// util/port.hpp (sketch)
#ifndef CW_PORT_HPP
#define CW_PORT_HPP

#if defined(__GNUC__) || defined(__clang__)
  #define CW_RESTRICT __restrict__
#else
  #define CW_RESTRICT
#endif

#endif // CW_PORT_HPP
```

Usage guidelines:

- Only mark pointers `CW_RESTRICT` when you can prove the pointed-to ranges do not alias for the lifetime of the call.
- In Debug builds, add runtime asserts to validate non-overlap when feasible.
- Prefer passing spans and indices that naturally avoid aliasing.

## Attributes (hot paths and inlining)

We selectively use attributes that guide inlining and code layout where profiling shows benefit.

```cpp
// Always-inline for leaf, tiny functions on hot paths
#define CW_ALWAYS_INLINE [[gnu::always_inline]] inline

// Branch prediction hints for cold/error paths
#define CW_COLD [[gnu::cold]]
#define CW_HOT  [[gnu::hot]]

// Assume alignment for vectorized loads/stores (pair with static/runtime checks)
#define CW_ASSUME_ALIGNED(ptr, N) (__builtin_assume_aligned((ptr), (N)))
```

Guidelines:

- Do not blanket-mark large functions as always_inline.
- Use `[[gnu::hot]]`/`[[gnu::cold]]` only when profiles show persistent skew.
- Pair `CW_ASSUME_ALIGNED` with actual alignment guarantees (allocators, layout). Add `static_assert` on types and runtime checks in Debug.

## Builtins and branch prediction

We use builtins sparingly to clarify intent to the optimizer.

```cpp
#define CW_LIKELY(x)   (__builtin_expect(!!(x), 1))
#define CW_UNLIKELY(x) (__builtin_expect(!!(x), 0))

// State that a condition holds (no code emitted, UB if false):
#define CW_ASSUME(x)   (__builtin_assume((x)))
```

Guidelines:

- Prefer writing code with obvious control flow; add hints only on proven hot paths.
- `CW_ASSUME` may enable vectorization but introduces UB if the assumption is false—guard with Debug asserts.

## Vectorization hints

For tiny fixed-width math, consider vector extensions after measuring.

```cpp
// Example: 4-wide float vector for small fixed-size kernels
using f32x4 [[gnu::vector_size(16)]] = float;

CW_ALWAYS_INLINE f32x4 fmadd(f32x4 a, f32x4 b, f32x4 c) {
    return a * b + c;
}
```

Guidelines:

- Only apply where data is naturally aligned and contiguous.
- Validate codegen on GCC 15+ and check for fallback on other compilers.

## Loop hints and pragmas

Use GCC loop hints cautiously and only when proven safe.

```cpp
void saxpy(float * CW_RESTRICT y, const float * CW_RESTRICT x, float a, int n) {
    #pragma GCC ivdep
    for (int i = 0; i < n; ++i) {
        y[i] = a * x[i] + y[i];
    }
}
```

Guidelines:

- `ivdep` is appropriate when you guarantee no loop-carried dependencies.
- Prefer data layout changes (SoA) over pragmas to enable vectorization.

## Safety and validation

- Add `static_assert` for type sizes and alignments where assumptions are made.
- In Debug/Profile, add runtime checks for aliasing and alignment assumptions.
- Use sanitizers where applicable; ensure no UB from incorrect `restrict` or `assume` usage.

## Cross-compiler behavior

- Wrap all non-standard features behind macros in a single portability header.
- Provide conservative fallbacks for compilers without the GNU attributes/builtins.
- Gate risky extensions behind `#ifdef __GNUC__` and feature checks.

## Change control

- All extension introductions must include: a profile before/after, effected kernels/functions, safety reasoning, and tests passing on Debug/Release.
- Document each non-trivial addition in the PR description and link to relevant benchmarks.
