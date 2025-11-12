#include "common.slang"
#include "reductions.slang"
#include "bc.slang"

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> ElementLoads : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<DirichletInfo> ElementDirichlet : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float3> ElementInternalForces : register(u0, space2);

[shader("compute")]
[numthreads(kWaveSize, 1, 1)]
CWF_REQUIRE_SUBGROUP_SIZE
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    CWF_GUARD_SUBGROUP_SIZE();

    uint elementIndex = dispatchThreadID.x;
    if(!cwf_index_in_range(elementIndex, elementCount))
    {
        return;
    }

    float3 load = ElementLoads[elementIndex];
    DirichletInfo bc = ElementDirichlet[elementIndex];
    float3 clamped = cwf_apply_dirichlet(load, bc);
    float3 reduced = cwf_reduce_sum(clamped);
    ElementInternalForces[elementIndex] = reduced;
}
