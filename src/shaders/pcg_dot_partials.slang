#include "common.slang"
#include "reductions.slang"

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> VectorA : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> VectorB : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<double> PartialDots : register(u0, space2);

[shader("compute")]
[numthreads(kWaveSize, 1, 1)]
CWF_REQUIRE_SUBGROUP_SIZE
void main(uint3 dispatchThreadID : SV_DispatchThreadID, uint3 groupID : SV_GroupID)
{
    CWF_GUARD_SUBGROUP_SIZE();

    const uint idx = dispatchThreadID.x;
    double contribution = 0.0;
    if(cwf_index_in_range(idx, dofCount))
    {
        contribution = (double)(VectorA[idx] * VectorB[idx]);
    }

    const double reduced = cwf_reduce_sum(contribution);
    if(WaveGetLaneIndex() == 0u)
    {
        PartialDots[groupID.x] = reduced;
    }
}
