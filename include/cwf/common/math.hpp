/**
 * @file math.hpp
 * @brief vibey math helpers for funky FEM vectors uwu!!!
 *
 * this header centralizes the tiny slice of vector math we need for the Phase 3
 * CPU-side preprocessing pipeline. the focus is dead-simple SoA-friendly
 * algebra: dot products, cross products, magnitudes, and friendly normalization
 * that never explodes into NaNs. keeping these helpers in a header ensures
 * constexpr goodness, zero-cost abstraction vibes, and makes the optimizer go
 * absolutely feral in the best way possible.
 *
 * the big idea: expose tiny pure functions that can be inlined everywhere so
 * the FEM preprocessing code stays readable without paying abstraction tax.
 * every helper here targets C++26 and GCC 15.2+ because living on the bleeding
 * edge is praxis. no third-party deps, no Eigen, no GLM – just crisp STL +
 * constexpr loops. results go brrr ✨
 *
 * @author LukeFrankio
 * @date 2025-11-05
 * @version 1.0
 *
 * @note built with Doxygen 1.15 beta because documentation supremacy matters
 * @note compiled with GCC 15.2+ in -std=c++2c mode (C++26 baby!)
 * @note targets math for FEM preprocessing, zero runtime allocation
 * @note designed for AMD iGPU tuning (wave64) even on CPU prep path
 *
 * example (basic usage):
 * @code
 * using namespace cwf::common;
 * constexpr Vec3 a{1.0, 0.0, 0.0};
 * constexpr Vec3 b{0.0, 1.0, 0.0};
 * constexpr Vec3 c = cross(a, b);
 * // c == {0.0, 0.0, 1.0} and the compiler inlines everything uwu
 * @endcode
 */
#pragma once

#include <array>
#include <cmath>
#include <concepts>
#include <limits>
#include <numbers>

namespace cwf::common {

/**
 * @brief spicy 3D vector alias that keeps STL friendly vibes
 *
 * ✨ PURE FUNCTION ✨
 *
 * this alias is pure because:
 * - it's just a type alias (no runtime) and can be used at compile time
 * - it keeps the FEM preprocessing code ergonomic without introducing
 *   third-party baggage
 * - zero allocations, zero side effects, zero drama fr fr
 */
using Vec3 = std::array<double, 3>;

/**
 * @brief blasts a 3D dot product in pure functional style
 *
 * dots two vectors, returns the scalar, and never mutates anything. tuned for
 * FEM assembly math where the dot product keeps showing up in gradients,
 * volumes, and Rayleigh damping derivations.
 *
 * ✨ PURE FUNCTION ✨
 *
 * this function is pure because:
 * - referential transparency applies (same inputs = same output)
 * - no side effects, no throws, just math uwu
 * - constexpr friendly so compile-time evaluation is free
 *
 * @param[in] lhs left vector operand (SoA-friendly)
 * @param[in] rhs right vector operand
 * @return double scalar dot product result (finite unless inputs are wild)
 *
 * @complexity O(1) time (three multiplies + two adds)
 * @complexity O(1) space (no allocations)
 *
 * example (edge case handling):
 * @code
 * constexpr Vec3 zero{0.0, 0.0, 0.0};
 * constexpr Vec3 weird{std::numeric_limits<double>::infinity(), 1.0, 2.0};
 * auto result = dot(zero, weird);
 * // result == 0.0 even with infinities because zero annihilates everything uwu
 * @endcode
 */
[[nodiscard]] constexpr auto dot(const Vec3& lhs, const Vec3& rhs) noexcept -> double
{
    return (lhs[0] * rhs[0]) + (lhs[1] * rhs[1]) + (lhs[2] * rhs[2]);
}

/**
 * @brief computes the right-handed cross product for wave64-ready math
 *
 * cross products are the heart of tetrahedron gradient math. this helper keeps
 * the orientation consistent with Vulkan/SPIR-V conventions so GPU + CPU agree
 * on geometry.
 *
 * ✨ PURE FUNCTION ✨
 *
 * this function is pure because:
 * - deterministic algebra only, zero observable side effects
 * - noexcept and constexpr so the compiler can fold it away
 *
 * @param[in] lhs first vector operand (defines the orientation)
 * @param[in] rhs second vector operand
 * @return Vec3 orthogonal vector following right-hand rule
 *
 * @complexity O(1) time with a fixed handful of multiplies/subtractions
 * @complexity O(1) space because we return by value without allocations
 *
 * example (composition with other functions):
 * @code
 * using namespace cwf::common;
 * constexpr Vec3 u{1.0, 0.0, 0.0};
 * constexpr Vec3 v{0.0, 1.0, 0.0};
 * constexpr Vec3 w{0.0, 0.0, 1.0};
 * auto triple = dot(u, cross(v, w));
 * // triple == 1.0 because standard basis is orthonormal uwu
 * @endcode
 */
[[nodiscard]] constexpr auto cross(const Vec3& lhs, const Vec3& rhs) noexcept -> Vec3
{
    return Vec3{
        (lhs[1] * rhs[2]) - (lhs[2] * rhs[1]),
        (lhs[2] * rhs[0]) - (lhs[0] * rhs[2]),
        (lhs[0] * rhs[1]) - (lhs[1] * rhs[0])
    };
}

/**
 * @brief grabs the Euclidean magnitude with optional safety clamps
 *
 * magnitude pops up for normalization checks and sanity guards. we clamp
 * subnormal magnitudes to zero so downstream code avoids catastrophic
 * cancellation when dividing by vibes near machine epsilon.
 *
 * ✨ PURE FUNCTION ✨
 *
 * this function is pure because:
 * - pure math w/ no state and no observable side effects
 * - constexpr and noexcept so compilers optimize the heck out of it
 * - stable even for denorm minima thanks to explicit clamp logic
 *
 * @param[in] value vector under inspection
 * @return non-negative magnitude (may be zero or +inf depending on inputs)
 *
 * @post return value >= 0.0 (never negative)
 *
 * @note leverages std::hypot for precision and overflow resilience
 * @warning NaN inputs propagate per IEEE 754 (callers should guard if needed)
 */
[[nodiscard]] inline auto magnitude(const Vec3& value) noexcept -> double
{
    const auto hypot = std::hypot(value[0], value[1], value[2]);
    if (hypot < std::numeric_limits<double>::denorm_min()) {
        return 0.0;
    }
    return hypot;
}

/**
 * @brief normalizes a vector but never bulldozes tiny magnitudes
 *
 * this helper normalizes when safe and returns the zero vector when the input
 * magnitude is too tiny to trust. it keeps NaNs from exploding across the
 * preprocessing pipeline.
 *
 * ✨ PURE FUNCTION ✨
 *
 * this function is pure because:
 * - input fully determines output without touching global state
 * - no exceptions, no logging, no side effects whatsoever
 *
 * @param[in] value vector to normalize
 * @return normalized vector (unit length) or zero vector when magnitude tiny
 *
 * @note threshold tuned to 1e-12 to balance precision vs stability
 * @note safe for zero vectors (returns zero vector gracefully)
 */
[[nodiscard]] inline auto safe_normalize(const Vec3& value) noexcept -> Vec3
{
    constexpr double kThreshold = 1.0e-12;
    const auto mag = magnitude(value);
    if (mag < kThreshold || !std::isfinite(mag)) {
        return Vec3{0.0, 0.0, 0.0};
    }
    const auto inv = 1.0 / mag;
    return Vec3{value[0] * inv, value[1] * inv, value[2] * inv};
}

}  // namespace cwf::common