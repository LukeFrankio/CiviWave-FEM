#include "common.slang"
#include "bc.slang"

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> BlockInverse : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> Residual : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(2, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<uint> NodeDirichletMask : register(t2, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float> Preconditioned : register(u0, space2);

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

    const uint base = nodeIndex * 9u;
    const float3 r = cwf_load_dofs(Residual, nodeIndex);
    float3 result;
    result.x = BlockInverse[base + 0u] * r.x + BlockInverse[base + 1u] * r.y + BlockInverse[base + 2u] * r.z;
    result.y = BlockInverse[base + 3u] * r.x + BlockInverse[base + 4u] * r.y + BlockInverse[base + 5u] * r.z;
    result.z = BlockInverse[base + 6u] * r.x + BlockInverse[base + 7u] * r.y + BlockInverse[base + 8u] * r.z;

    const uint mask = NodeDirichletMask[nodeIndex];
    result = cwf_zero_dirichlet(result, mask);
    cwf_store_dofs(Preconditioned, nodeIndex, result);
}
