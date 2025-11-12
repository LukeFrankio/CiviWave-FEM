#include "common.slang"
#include "reductions.slang"

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> VectorA : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> VectorB : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<double> PartialDots : register(u0, space2);

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

    float3 aValue = VectorA[idx];
    float3 bValue = VectorB[idx];
    double contribution = (double)dot(aValue, bValue);
    double reduced = cwf_reduce_sum(contribution);
    PartialDots[idx] = reduced;
}
