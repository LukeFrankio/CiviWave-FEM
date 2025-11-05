#include <array>
#include <cmath>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <limits>
#include <vector>

#include "cwf/common/math.hpp"

using testing::DoubleNear;
using testing::ElementsAreArray;

namespace
{

constexpr double kEps = 1.0e-12;

[[nodiscard]] auto make_basis_vectors() -> std::vector<cwf::common::Vec3>
{
    return {cwf::common::Vec3{0.0, 0.0, 0.0},  cwf::common::Vec3{1.0, 0.0, 0.0},
            cwf::common::Vec3{0.0, 1.0, 0.0},  cwf::common::Vec3{0.0, 0.0, 1.0},
            cwf::common::Vec3{1.0, 1.0, 1.0},  cwf::common::Vec3{-1.0, -1.0, -1.0},
            cwf::common::Vec3{2.5, -3.0, 4.0}, cwf::common::Vec3{-4.0, 2.0, -1.5}};
}

} // namespace

/**
 * @test verifies dot product symmetry for a chunky dataset of vectors
 */
TEST(CommonMathDot, SymmetryForAllPairs)
{
    const auto vectors = make_basis_vectors();
    for (const auto &lhs : vectors)
    {
        for (const auto &rhs : vectors)
        {
            const auto lhs_rhs = cwf::common::dot(lhs, rhs);
            const auto rhs_lhs = cwf::common::dot(rhs, lhs);
            EXPECT_DOUBLE_EQ(lhs_rhs, rhs_lhs) << "dot product symmetry broke for pair";
        }
    }
}

/**
 * @test checks dot product annihilation when either vector is zero
 */
TEST(CommonMathDot, ZeroVectorAnnihilatesEverything)
{
    const auto zero    = cwf::common::Vec3{0.0, 0.0, 0.0};
    const auto vectors = make_basis_vectors();
    for (const auto &candidate : vectors)
    {
        EXPECT_DOUBLE_EQ(0.0, cwf::common::dot(candidate, zero));
        EXPECT_DOUBLE_EQ(0.0, cwf::common::dot(zero, candidate));
    }
}

/**
 * @test ensures cross of orthogonal basis vectors yields canonical normals
 */
TEST(CommonMathCross, CanonicalRightHandedBasis)
{
    const auto i = cwf::common::Vec3{1.0, 0.0, 0.0};
    const auto j = cwf::common::Vec3{0.0, 1.0, 0.0};
    const auto k = cwf::common::Vec3{0.0, 0.0, 1.0};

    EXPECT_THAT(cwf::common::cross(i, j), ElementsAreArray(k));
    EXPECT_THAT(cwf::common::cross(j, k), ElementsAreArray(i));
    EXPECT_THAT(cwf::common::cross(k, i), ElementsAreArray(j));
}

/**
 * @test cross product orthogonality vs operands for random-ish dataset
 */
TEST(CommonMathCross, OrthogonalityAgainstOperands)
{
    const auto vectors = make_basis_vectors();
    for (std::size_t lhs = 1; lhs < vectors.size(); ++lhs)
    {
        for (std::size_t rhs = lhs + 1; rhs < vectors.size(); ++rhs)
        {
            const auto result  = cwf::common::cross(vectors[lhs], vectors[rhs]);
            const auto lhs_dot = cwf::common::dot(result, vectors[lhs]);
            const auto rhs_dot = cwf::common::dot(result, vectors[rhs]);
            EXPECT_NEAR(lhs_dot, 0.0, 10 * kEps);
            EXPECT_NEAR(rhs_dot, 0.0, 10 * kEps);
        }
    }
}

/**
 * @test magnitude clamps denormals back to zero per helper contract
 */
TEST(CommonMathMagnitude, ClampsDenormToZero)
{
    const cwf::common::Vec3 tiny{std::numeric_limits<double>::denorm_min(),
                                 std::numeric_limits<double>::denorm_min(),
                                 std::numeric_limits<double>::denorm_min()};
    EXPECT_DOUBLE_EQ(0.0, cwf::common::magnitude(tiny));
}

/**
 * @test magnitude handles gigantic values without overflow thanks to hypot
 */
TEST(CommonMathMagnitude, HugeValuesGracefully)
{
    const cwf::common::Vec3 huge{1.0e150, -1.5e150, 2.0e150};
    const auto              mag = cwf::common::magnitude(huge);
    EXPECT_TRUE(std::isfinite(mag));
    EXPECT_NEAR(mag, std::sqrt(1.0e300 + 2.25e300 + 4.0e300), std::abs(mag) * 1e-12);
}

/**
 * @test safe_normalize returns zero vector for degenerate inputs
 */
TEST(CommonMathNormalize, DegenerateInputYieldsZeroVector)
{
    const cwf::common::Vec3 near_zero{1.0e-16, -1.0e-16, 0.0};
    const auto              normalized = cwf::common::safe_normalize(near_zero);
    EXPECT_THAT(normalized, ElementsAreArray(cwf::common::Vec3{0.0, 0.0, 0.0}));
}

/**
 * @test safe_normalize composes to unit vectors for regular inputs
 */
TEST(CommonMathNormalize, ProducesUnitLengthVectors)
{
    const auto vectors = make_basis_vectors();
    for (const auto &vec : vectors)
    {
        if (cwf::common::magnitude(vec) < kEps)
        {
            continue;
        }
        const auto normalized = cwf::common::safe_normalize(vec);
        const auto mag        = cwf::common::magnitude(normalized);
        EXPECT_NEAR(mag, 1.0, 5 * kEps);
    }
}
