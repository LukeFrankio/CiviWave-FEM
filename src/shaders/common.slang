//=============================================================================
// @file common.slang
// @brief cross-kernel constants + helpers that keep every shader perfectly in sync
//
// this module slaps a single source of truth for descriptor sets, cbuffer
// layouts, and micro helpers (like DOF addressing) into place. every compute
// kernel in the project imports this bad boy so we never desync bindings or
// forget to guard subgroup sizes. compile it with slangc 2025.18+ targeting
// SPIR-V 1.6 for Vulkan 1.3, DXIL 6.8 for DirectX 12, or metal2.4 for macOS
// just to flex. functional vibes only, gen-z commentary mandatory uwu ✨
//
// build receipts (pick your poison):
//   slangc -target spirv -profile glsl_460 common.slang -o common.spv
//   slangc -target dxil  -profile sm_6_8  common.slang -o common.dxil
//   slangc -target metal -profile metal2.4 common.slang -o common.metallib
//
// author: LukeFrankio
// date: 2025-11-14
// slang version: 2025.18 (latest nightly preferred, no cap)
// target apis: Vulkan 1.3 / DirectX 12 / Metal 3+
//=============================================================================

#ifndef CWF_COMMON_SLANG
#define CWF_COMMON_SLANG

static const uint kWaveSize = 64;

#if defined(__SLANG__)
#define CWF_DESCRIPTOR_SET_GLOBALS 0
#define CWF_DESCRIPTOR_SET_READ_ONLY 1
#define CWF_DESCRIPTOR_SET_READ_WRITE 2
#endif

#if defined(__SLANG__)
[[vk::binding(0, CWF_DESCRIPTOR_SET_GLOBALS)]]
#endif
cbuffer GlobalSimulationParams : register(b0, space0)
{
    uint  elementCount;
    uint  nodeCount;
    uint  dofCount;
    uint  reductionBlock;
    float stiffnessScale;
    float massFactor;
    float timeStep;
    uint  iterationIndex;
    uint  materialCount;
    uint  maxLocalNodes;
    uint  paddingGlobals0;
    uint  paddingGlobals1;
};

struct DirichletInfo
{
    uint mask;
    float3 value;
};

inline bool cwf_index_in_range(uint idx, uint limit)
{
    return idx < limit;
}

inline float3 cwf_zero3()
{
    return float3(0.0, 0.0, 0.0);
}

inline float cwf_zero()
{
    return 0.0;
}

inline uint cwf_node_dof_base(uint node)
{
    return node * 3u;
}

inline float3 cwf_load_dofs(StructuredBuffer<float> buffer, uint node)
{
    const uint base = cwf_node_dof_base(node);
    return float3(buffer[base + 0u], buffer[base + 1u], buffer[base + 2u]);
}

inline void cwf_store_dofs(RWStructuredBuffer<float> buffer, uint node, float3 value)
{
    const uint base = cwf_node_dof_base(node);
    buffer[base + 0u] = value.x;
    buffer[base + 1u] = value.y;
    buffer[base + 2u] = value.z;
}

#if defined(__SLANG__)
#define CWF_REQUIRE_SUBGROUP_SIZE [WaveSize(64)]
#define CWF_GUARD_SUBGROUP_SIZE()        \
    do                                  \
    {                                   \
        if (WaveGetLaneCount() != kWaveSize) \
        {                               \
            return;                     \
        }                               \
    } while (false)
#else
#define CWF_REQUIRE_SUBGROUP_SIZE
#define CWF_GUARD_SUBGROUP_SIZE() do { } while (0)
#endif

#endif // CWF_COMMON_SLANG
