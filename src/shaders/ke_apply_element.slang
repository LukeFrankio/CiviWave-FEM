#include "common.slang"
#include "reductions.slang"
#include "bc.slang"

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<uint> ElementConnectivity : register(t0, space1);
#if defined(__SLANG__)
[[vk::binding(1, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> ElementGradients : register(t1, space1);
#if defined(__SLANG__)
[[vk::binding(2, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> ElementVolume : register(t2, space1);
#if defined(__SLANG__)
[[vk::binding(3, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<uint> ElementMaterialIndex : register(t3, space1);
#if defined(__SLANG__)
[[vk::binding(4, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> MaterialStiffness : register(t4, space1);
#if defined(__SLANG__)
[[vk::binding(5, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<float> InputVector : register(t5, space1);
#if defined(__SLANG__)
[[vk::binding(6, CWF_DESCRIPTOR_SET_READ_ONLY)]]
#endif
StructuredBuffer<uint> NodeDirichletMask : register(t6, space1);
#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_READ_WRITE)]]
#endif
RWStructuredBuffer<float> ElementInternalForces : register(u0, space2);

[shader("compute")]
[numthreads(kWaveSize, 1, 1)]
CWF_REQUIRE_SUBGROUP_SIZE
void main(uint3 dispatchThreadID : SV_DispatchThreadID)
{
    CWF_GUARD_SUBGROUP_SIZE();

    const uint elementIndex = dispatchThreadID.x;
    if(!cwf_index_in_range(elementIndex, elementCount))
    {
        return;
    }

    static const uint kLocalNodes = 8u;
    static const uint kStrainCount = 6u;

    float3 localDisplacement[kLocalNodes];
    float3 localGradients[kLocalNodes];
    bool   nodeActive[kLocalNodes];

    const uint connBase = elementIndex * kLocalNodes;
    const uint gradBase = elementIndex * kLocalNodes * 3u;

    for(uint local = 0u; local < kLocalNodes; ++local)
    {
        const uint nodeIndex = ElementConnectivity[connBase + local];
        nodeActive[local] = nodeIndex < nodeCount;
        float3 grad = float3(
            ElementGradients[gradBase + local * 3u + 0u],
            ElementGradients[gradBase + local * 3u + 1u],
            ElementGradients[gradBase + local * 3u + 2u]);
        localGradients[local] = grad;

        if(nodeActive[local])
        {
            float3 displacement = cwf_load_dofs(InputVector, nodeIndex);
            const uint mask = NodeDirichletMask[nodeIndex];
            displacement = cwf_zero_dirichlet(displacement, mask);
            localDisplacement[local] = displacement;
        }
        else
        {
            localDisplacement[local] = cwf_zero3();
        }
    }

    float strain[kStrainCount];
    [unroll]
    for(uint i = 0u; i < kStrainCount; ++i)
    {
        strain[i] = 0.0;
    }

    for(uint local = 0u; local < kLocalNodes; ++local)
    {
        const float3 grad = localGradients[local];
        const float3 u = localDisplacement[local];
        strain[0] += grad.x * u.x;
        strain[1] += grad.y * u.y;
        strain[2] += grad.z * u.z;
        strain[3] += grad.x * u.y + grad.y * u.x;
        strain[4] += grad.y * u.z + grad.z * u.y;
        strain[5] += grad.x * u.z + grad.z * u.x;
    }

    const uint materialIndex = ElementMaterialIndex[elementIndex];
    const uint materialBase = materialIndex * (kStrainCount * kStrainCount);
    float stress[kStrainCount];
    for(uint row = 0u; row < kStrainCount; ++row)
    {
        float sum = 0.0;
        for(uint col = 0u; col < kStrainCount; ++col)
        {
            sum += MaterialStiffness[materialBase + row * kStrainCount + col] * strain[col];
        }
        stress[row] = sum;
    }

    const float volume = ElementVolume[elementIndex] * stiffnessScale;
    const uint forceBase = elementIndex * kLocalNodes * 3u;
    for(uint local = 0u; local < kLocalNodes; ++local)
    {
        const uint writeBase = forceBase + local * 3u;
        if(!nodeActive[local])
        {
            ElementInternalForces[writeBase + 0u] = 0.0;
            ElementInternalForces[writeBase + 1u] = 0.0;
            ElementInternalForces[writeBase + 2u] = 0.0;
            continue;
        }

        const float3 grad = localGradients[local];
        float3 force;
        force.x = grad.x * stress[0] + grad.y * stress[3] + grad.z * stress[5];
        force.y = grad.y * stress[1] + grad.x * stress[3] + grad.z * stress[4];
        force.z = grad.z * stress[2] + grad.y * stress[4] + grad.x * stress[5];
        force *= volume;

        ElementInternalForces[writeBase + 0u] = force.x;
        ElementInternalForces[writeBase + 1u] = force.y;
        ElementInternalForces[writeBase + 2u] = force.z;
    }
}
