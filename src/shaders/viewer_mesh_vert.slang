//============================================================================
// file: viewer_mesh_vert.slang
// brief: transforms packed FEM vertices with a column-major camera matrix uwu
//
// targets the Vulkan viewer demo (Slang 1.3+ compiled to SPIR-V 1.6) so we can
// keep one shader source for every platform. compile via the helper CMake
// target or directly with:
//   slangc viewer_mesh_vert.slang -target spirv -profile vs_6_5 -entry main \
//          -o viewer_mesh_vert.spv
// this shader only handles geometry setup; fragment shading handles the color
// ramp. functional vibes only ✨
//============================================================================

#if defined(__SLANG__)
[[vk::binding(0, 0)]]
#endif
/**
 * @brief camera constant buffer shared with the C++ viewer uwu
 *
 * Using a column-major matrix keeps the data layout aligned with typical
 * Vulkan matrix math. On the CPU we compute view * proj and then transpose
 * the result before copying into the UBO so the shader can consume it as
 * column-major without additional transposition.
 */
cbuffer CameraBuffer : register(b0)
{
    // column-major matrix for Vulkan (no row_major qualifier)
    float4x4 uViewProjection;
};

/**
 * @brief vertex payload straight from the packed mesh buffers uwu
 */
struct VertexInput
{
    float4 position : POSITION; ///< model-space position pulled from CPU buffers (w should be 1.0)
    float4 color : COLOR0; ///< von-mises-driven rainbow gradient ✨
    float stress : TEXCOORD1; ///< normalized von-mises scalar for per-pixel shading
};

/**
 * @brief data forwarded to the fragment shader after clip transform
 */
struct VertexOutput
{
    float4 position : SV_Position; ///< clip-space coordinates consumed by Vulkan
    float4 color : COLOR0; ///< passthrough color for interpolation magic
    float3 modelPos : TEXCOORD0; ///< model-space position for flat shading
    float  stress   : TEXCOORD1; ///< normalized scalar forwarded to pixel shader
};

/**
 * @brief main vertex entry point (pure transform vibes only)
 */
[shader("vertex")]
VertexOutput main(VertexInput input)
{
    VertexOutput output;
    // column-major math: matrix on the left, column-vector on the right
    output.position = mul(uViewProjection, input.position);
    output.color = input.color;
    output.modelPos = input.position.xyz;
    output.stress = input.stress;
    return output;
}