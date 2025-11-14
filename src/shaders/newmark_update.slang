#include "common.slang"

//=============================================================================
// @file newmark_update.slang
// @brief corrector stage that folds Δu back into u/v/a with pure Newmark vibes
//
// consumes the predictor outputs + PCG correction vector, then reconstructs the
// final displacement, velocity, and acceleration fields. coefficients are fed
// via cbuffers so the math stays synced with the CPU reference integrator.
//=============================================================================

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> PredictedDisplacement : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> PredictedVelocity : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(2, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> Correction : register(t2, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float3> UpdatedDisplacement : register(u0, space2);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float3> UpdatedVelocity : register(u1, space2);
#if defined(__SLANG__)
[[vk::binding(2, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float3> UpdatedAcceleration : register(u2, space2);

#if defined(__SLANG__)
[[vk::binding(3, CWF_DESCRIPTOR_SET_GLOBALS)]]
#endif
cbuffer UpdateParams : register(b3, space0)
{
    float gammaOverBetaDt; ///< γ / (β Δt)
    float invBetaDt2;      ///< 1 / (β Δt^2)
    float paddingUpdate0;
    float paddingUpdate1;
}

/**
 * @brief corrector kernel a.k.a u/v/a reunion tour
 */
[shader("compute")]
[numthreads(kWaveSize, 1, 1)]
CWF_REQUIRE_SUBGROUP_SIZE
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    CWF_GUARD_SUBGROUP_SIZE();

    const uint idx = dispatchThreadID.x;
    if(!cwf_index_in_range(idx, nodeCount))
    {
        return;
    }

    const float3 delta = Correction[idx];
    const float3 uPred = PredictedDisplacement[idx];
    const float3 vPred = PredictedVelocity[idx];

    const float3 uNew = uPred + delta;
    const float3 aNew = invBetaDt2 * delta;
    const float3 vNew = vPred + gammaOverBetaDt * delta;

    UpdatedDisplacement[idx] = uNew;
    UpdatedVelocity[idx] = vNew;
    UpdatedAcceleration[idx] = aNew;
}
