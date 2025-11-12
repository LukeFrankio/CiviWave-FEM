#include "common.slang"
#include "reductions.slang"
#include "bc.slang"

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<uint> NodeElementOffsets : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<uint> NodeElementIndices : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(2, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> ElementContributions : register(t2, space1);
#if defined(__SLANG__)
[[vk::binding(3, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<DirichletInfo> NodeDirichlet : register(t3, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float3> NodeForces : register(u0, space2);

[shader("compute")]
[numthreads(kWaveSize, 1, 1)]
CWF_REQUIRE_SUBGROUP_SIZE
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    CWF_GUARD_SUBGROUP_SIZE();

    uint nodeIndex = dispatchThreadID.x;
    if(!cwf_index_in_range(nodeIndex, nodeCount))
    {
        return;
    }

    uint rangeBegin = NodeElementOffsets[nodeIndex];
    uint rangeEnd = NodeElementOffsets[nodeIndex + 1];

    float3 accumulated = cwf_zero3();
    for(uint idx = rangeBegin; idx < rangeEnd; ++idx)
    {
        uint elementIndex = NodeElementIndices[idx];
        if(elementIndex < elementCount)
        {
            accumulated += ElementContributions[elementIndex];
        }
    }

    DirichletInfo bc = NodeDirichlet[nodeIndex];
    float3 clamped = cwf_apply_dirichlet(accumulated, bc);
    NodeForces[nodeIndex] = clamped;
}
