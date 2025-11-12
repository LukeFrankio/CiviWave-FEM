#include "common.slang"

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
    float updateA0;
    float updateA1;
    float updateA2;
    float paddingUpdate;
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

    float3 delta = Correction[idx];
    float3 uPred = PredictedDisplacement[idx];
    float3 vPred = PredictedVelocity[idx];

    float3 uNew = uPred + delta;
    float3 vNew = vPred + updateA0 * delta;
    float3 aNew = updateA1 * delta + updateA2 * vNew;

    UpdatedDisplacement[idx] = uNew;
    UpdatedVelocity[idx] = vNew;
    UpdatedAcceleration[idx] = aNew;
}
