#include "common.slang"
#include "bc.slang"

//=============================================================================
// @file pcg_enforce_dirichlet.slang
// @brief clamps solution/residual vectors to respect Dirichlet boundary masks
//
// mirrors the CPU-side enforcement step so GPU PCG solves keep constrained
// DOFs glued to their prescribed displacement values. processing happens per
// node which keeps memory accesses nicely coalesced under our SoA-style DOF
// layout uwu ✨
//=============================================================================

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> RhsVector : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<uint> NodeDirichletMask : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float> SolutionVector : register(u0, space2);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float> ResidualVector : register(u1, space2);

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

    const uint mask = NodeDirichletMask[nodeIndex];
    if(mask == 0u)
    {
        return;
    }

    const uint base = nodeIndex * 3u;
    const float3 rhs = cwf_load_dofs(RhsVector, nodeIndex);

    const float3 solutionInput = float3(SolutionVector[base + 0u], SolutionVector[base + 1u], SolutionVector[base + 2u]);
    const float3 residualInput = float3(ResidualVector[base + 0u], ResidualVector[base + 1u], ResidualVector[base + 2u]);

    float3 solution = solutionInput;
    float3 residual = residualInput;

    if((mask & 0x1u) != 0u)
    {
        solution.x = rhs.x;
        residual.x = 0.0;
    }
    if((mask & 0x2u) != 0u)
    {
        solution.y = rhs.y;
        residual.y = 0.0;
    }
    if((mask & 0x4u) != 0u)
    {
        solution.z = rhs.z;
        residual.z = 0.0;
    }

    cwf_store_dofs(SolutionVector, nodeIndex, solution);
    cwf_store_dofs(ResidualVector, nodeIndex, residual);
}
