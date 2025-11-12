#include "common.slang"
#include "reductions.slang"

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> DisplacementField : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> VelocityField : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float4> DerivedField : register(u0, space2);

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

    float3 u = DisplacementField[idx];
    float3 v = VelocityField[idx];
    float magnitude = length(u);
    float energy = 0.5f * dot(v, v);
    float summary = cwf_reduce_sum(magnitude + energy);
    DerivedField[idx] = float4(u, summary);
}
