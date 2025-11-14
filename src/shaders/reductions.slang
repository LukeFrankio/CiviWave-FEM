// ============================================================================
// Reduction helpers (Phase 6 scaffolding)
// ----------------------------------------------------------------------------
// Provides ultra-lightweight reduce helpers. We intentionally implement the
// simplest possible versions so the shaders compile even when wave operations
// are unavailable on the target profile.
// ============================================================================

#ifndef CWF_REDUCTIONS_SLANG
#define CWF_REDUCTIONS_SLANG

#if defined(__SLANG__)
inline float cwf_reduce_sum(float value)
{
    return WaveActiveSum(value);
}

inline double cwf_reduce_sum(double value)
{
    return WaveActiveSum(value);
}
#else
inline float cwf_reduce_sum(float value)
{
    return value;
}

inline double cwf_reduce_sum(double value)
{
    return value;
}
#endif

inline float3 cwf_reduce_sum(float3 value)
{
    return float3(cwf_reduce_sum(value.x), cwf_reduce_sum(value.y), cwf_reduce_sum(value.z));
}

#endif // CWF_REDUCTIONS_SLANG
