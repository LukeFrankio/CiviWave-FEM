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
StructuredBuffer<uint> NodeLocalIndices : register(t2, space1);
#if defined(__SLANG__)
[[vk::binding(3, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> InputVector : register(t3, space1);
#if defined(__SLANG__)
[[vk::binding(4, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<uint> NodeDirichletMask : register(t4, space1);
#if defined(__SLANG__)
[[vk::binding(5, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> LumpedMass : register(t5, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float> ElementInternalForces : register(u0, space2);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float> NodeForces : register(u1, space2);

[shader("compute")]
[numthreads(kWaveSize, 1, 1)]
CWF_REQUIRE_SUBGROUP_SIZE
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    CWF_GUARD_SUBGROUP_SIZE();

    const uint nodeIndex = dispatchThreadID.x;
    if(!cwf_index_in_range(nodeIndex, nodeCount))
    {
        return;
    }

    const uint rangeBegin = NodeElementOffsets[nodeIndex];
    const uint rangeEnd = NodeElementOffsets[nodeIndex + 1u];

    float3 accumulated = cwf_zero3();
    for(uint idx = rangeBegin; idx < rangeEnd; ++idx)
    {
        const uint elementIndex = NodeElementIndices[idx];
        if(elementIndex >= elementCount)
        {
            continue;
        }
        const uint localIndex = NodeLocalIndices[idx];
        const uint forceBase = elementIndex * 8u * 3u + localIndex * 3u;
        accumulated.x += ElementInternalForces[forceBase + 0u];
        accumulated.y += ElementInternalForces[forceBase + 1u];
        accumulated.z += ElementInternalForces[forceBase + 2u];
    }

    const uint mask = NodeDirichletMask[nodeIndex];
    float3 sanitized = cwf_load_dofs(InputVector, nodeIndex);
    sanitized = cwf_zero_dirichlet(sanitized, mask);
    accumulated += sanitized * (massFactor * LumpedMass[nodeIndex]);

    float3 original = cwf_load_dofs(InputVector, nodeIndex);
    if((mask & 0x1u) != 0u)
    {
        accumulated.x = original.x;
    }
    if((mask & 0x2u) != 0u)
    {
        accumulated.y = original.y;
    }
    if((mask & 0x4u) != 0u)
    {
        accumulated.z = original.z;
    }

    cwf_store_dofs(NodeForces, nodeIndex, accumulated);
}
