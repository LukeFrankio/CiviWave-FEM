// ============================================================================
// Boundary condition utilities
// ----------------------------------------------------------------------------
// Applies Dirichlet masks to vector quantities. The mask encodes which
// components are clamped (bit 0 -> X, bit 1 -> Y, bit 2 -> Z).
// ============================================================================

#ifndef CWF_BC_SLANG
#define CWF_BC_SLANG

#include "common.slang"

inline float3 cwf_apply_dirichlet(float3 value, DirichletInfo bc)
{
    if((bc.mask & 0x1u) != 0u)
    {
        value.x = bc.value.x;
    }
    if((bc.mask & 0x2u) != 0u)
    {
        value.y = bc.value.y;
    }
    if((bc.mask & 0x4u) != 0u)
    {
        value.z = bc.value.z;
    }
    return value;
}

inline float3 cwf_zero_dirichlet(float3 value, uint mask)
{
    if((mask & 0x1u) != 0u)
    {
        value.x = 0.0;
    }
    if((mask & 0x2u) != 0u)
    {
        value.y = 0.0;
    }
    if((mask & 0x4u) != 0u)
    {
        value.z = 0.0;
    }
    return value;
}

#endif // CWF_BC_SLANG
