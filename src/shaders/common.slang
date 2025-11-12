// ============================================================================
// CiviWave-FEM Slang shared helpers (Phase 6)
// ----------------------------------------------------------------------------
// Lightweight scaffolding that provides shared constants, parameter blocks,
// and helper utilities consumed by all compute kernels. These kernels are
// intentionally simple placeholders: they establish the compilation pipeline
// and encode our conventions around bindings, wave size, and scalar helpers.
// ============================================================================

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
    uint elementCount;
    uint nodeCount;
    float timeStep;
    uint iterationIndex;
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
