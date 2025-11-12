#include "common.slang"

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> VectorX : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> VectorY : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float3> VectorOut : register(u0, space2);

#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_GLOBALS)]]
#endif
cbuffer PcgAxpyParams : register(b1, space0)
{
    float alpha;
    float beta;
    float padding0;
    float padding1;
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

    float3 xValue = VectorX[idx];
    float3 yValue = VectorY[idx];
    VectorOut[idx] = alpha * xValue + beta * yValue;
}
