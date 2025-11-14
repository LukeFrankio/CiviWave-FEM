#include "common.slang"

//=============================================================================
// @file newmark_predictor.slang
// @brief GPU implementation of the explicit Newmark predictor (Phase 9 unlock)
//
// feeds on the current displacement/velocity/acceleration fields and spits out
// the predicted kinematics required by the implicit corrector. the formulas
// come straight from the average-acceleration Newmark variant so CPU + GPU stay
// numerically indistinguishable. tuned for wave64 compute kernels and the
// descriptor buffer layout baked in Phase 8 uwu ✨
//=============================================================================

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> CurrentDisplacement : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> CurrentVelocity : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(2, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> CurrentAcceleration : register(t2, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float3> PredictedDisplacement : register(u0, space2);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float3> PredictedVelocity : register(u1, space2);

#if defined(__SLANG__)
[[vk::binding(2, CWF_DESCRIPTOR_SET_GLOBALS)]]
#endif
cbuffer PredictorParams : register(b2, space0)
{
    float predictorGamma; ///< γ in the Newmark family tree
    float predictorBeta;  ///< β controlling acceleration averaging
    float paddingPredictor0;
    float paddingPredictor1;
}

/**
 * @brief run the explicit predictor across all nodes (pure function energy)
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

    const float3 u = CurrentDisplacement[idx];
    const float3 v = CurrentVelocity[idx];
    const float3 a = CurrentAcceleration[idx];

    const float dt = timeStep;
    const float dt_sq = dt * dt;
    const float disp_factor = 0.5f - predictorBeta;
    const float vel_factor = 1.0f - predictorGamma;

    const float3 predictedU = u + dt * v + disp_factor * dt_sq * a;
    const float3 predictedV = v + vel_factor * dt * a;

    PredictedDisplacement[idx] = predictedU;
    PredictedVelocity[idx] = predictedV;
}
