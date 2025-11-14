#include "common.slang"
#include "reductions.slang"

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<double> PartialDots : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<double> OutputDot : register(u0, space2);

#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_GLOBALS)]]
#endif
cbuffer ReduceParams : register(b1, space0)
{
    uint partialCount;
    uint paddingReduce0;
    uint paddingReduce1;
    uint paddingReduce2;
}

[shader("compute")]
[numthreads(kWaveSize, 1, 1)]
CWF_REQUIRE_SUBGROUP_SIZE
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    CWF_GUARD_SUBGROUP_SIZE();

    double accumulator = 0.0;
    for(uint i = dispatchThreadID.x; i < partialCount; i += kWaveSize)
    {
        accumulator += PartialDots[i];
    }

    const double reduced = cwf_reduce_sum(accumulator);
    if(WaveGetLaneIndex() == 0u)
    {
        OutputDot[0] = reduced;
    }
}
