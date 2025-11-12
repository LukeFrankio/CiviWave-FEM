#include "common.slang"

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
    float predictorGamma;
    float predictorBeta;
    float paddingPredictor0;
    float paddingPredictor1;
}

[shader("compute")]
[numthreads(kWaveSize, 1, 1)]
CWF_REQUIRE_SUBGROUP_SIZE
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    CWF_GUARD_SUBGROUP_SIZE();

    uint idx = dispatchThreadID.x;
    if(!cwf_index_in_range(idx, nodeCount))
    {
        return;
    }

    float3 u = CurrentDisplacement[idx];
    float3 v = CurrentVelocity[idx];
    float3 a = CurrentAcceleration[idx];

    float dt = timeStep;
    float3 predictedU = u + dt * v + 0.5f * dt * dt * a;
    float3 predictedV = v + (1.0f - predictorGamma) * dt * a;

    PredictedDisplacement[idx] = predictedU;
    PredictedVelocity[idx] = predictedV;
}
