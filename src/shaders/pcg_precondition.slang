#include "common.slang"
#include "bc.slang"

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> InvDiagonal : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float3> Residual : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(2, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<DirichletInfo> NodeDirichlet : register(t2, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float3> Preconditioned : register(u0, space2);

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

    float3 r = Residual[nodeIndex];
    float3 mInv = InvDiagonal[nodeIndex];
    float3 preconditioned = float3(r.x * mInv.x, r.y * mInv.y, r.z * mInv.z);
    DirichletInfo bc = NodeDirichlet[nodeIndex];
    Preconditioned[nodeIndex] = cwf_apply_dirichlet(preconditioned, bc);
}
