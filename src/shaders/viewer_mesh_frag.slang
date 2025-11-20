//============================================================================
// file: viewer_mesh_frag.slang
// brief: gamma-correct fragment shader for the FEM viewer uwu
//
// compiled via slangc to SPIR-V 1.6 (profile ps_6_5) alongside the vertex
// shader. converts interpolated von-mises colors into sRGB so the gradient
// actually slaps on consumer monitors.
//============================================================================

/**
 * @brief interpolated payload from the vertex stage (clip + color)
 */
struct FragmentInput
{
    float4 position : SV_Position;
    float4 color : COLOR0;
    float3 modelPos : TEXCOORD0;
    float  stress : TEXCOORD1;
};

/**
 * @brief converts normalized stress scalar into the familiar blue->green->red gradient
 */
float3 stress_to_color(float stress)
{
    float t = saturate(stress);
    if (t < 0.5)
    {
        const float local = t * 2.0;
        // Blue (0,0,1) to Green (0,1,0)
        return float3(0.0, local, 1.0 - local);
    }
    const float local = (t - 0.5) * 2.0;
    // Green (0,1,0) to Red (1,0,0)
    return float3(local, 1.0 - local, 0.0);
}

/**
 * @brief fragment entry that applies a subtle gamma curve for vibes
 */
[shader("fragment")]
float4 main(FragmentInput input) : SV_Target0
{
    // compute flat normal from model position derivatives
    float3 dx = ddx(input.modelPos);
    float3 dy = ddy(input.modelPos);
    float3 normal = normalize(cross(dx, dy));

    // simple directional lighting (fixed light source)
    // Use a light direction that illuminates the (-,-,-) octant faces differently
    // to reveal the 3D structure.
    // Light coming from roughly (-1, -0.5, -0.2)
    float3 lightDir = normalize(float3(-1.0, -0.5, -0.2));
    float diff = max(dot(normal, lightDir), 0.0);
    
    // add ambient and apply to stress-derived color ramp
    // boost ambient to 0.2 to ensure visibility
    float3 stressColor = stress_to_color(input.stress);
    float3 litColor    = stressColor * (diff * 0.8 + 0.2);

    const float gamma = 1.0 / 2.2;
    float3 srgb = pow(saturate(litColor), float3(gamma, gamma, gamma));
    return float4(srgb, 1.0);
}