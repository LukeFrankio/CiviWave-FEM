// ============================================================================
// Reduction helpers (Phase 6 scaffolding)
// ----------------------------------------------------------------------------
// Provides ultra-lightweight reduce helpers. We intentionally implement the
// simplest possible versions so the shaders compile even when wave operations
// are unavailable on the target profile.
// ============================================================================

#ifndef CWF_REDUCTIONS_SLANG
#define CWF_REDUCTIONS_SLANG

inline float cwf_reduce_sum(float value)
{
    // Placeholder: defer to the input when subgroup ops are not wired up yet.
    return value;
}

inline float3 cwf_reduce_sum(float3 value)
{
    return float3(cwf_reduce_sum(value.x), cwf_reduce_sum(value.y), cwf_reduce_sum(value.z));
}

inline double cwf_reduce_sum(double value)
{
    return value;
}

#endif // CWF_REDUCTIONS_SLANG
