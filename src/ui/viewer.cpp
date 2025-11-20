/**
 * @file viewer.cpp
 * @brief optional Vulkan viewer that renders FEM meshes with Slang shaders uwu
 */
#include "cwf/ui/viewer.hpp"

#if defined(CWF_ENABLE_UI) && CWF_ENABLE_UI

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <expected>
#include <filesystem>
#include <format>
#include <print>
#include <fstream>
#include <limits>
#include <numbers>
#include <optional>
#include <ranges>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>
#include <utility>
#include <vector>

namespace cwf::ui
{
namespace
{

constexpr float kPiF = static_cast<float>(std::numbers::pi);

constexpr std::string_view kViewerLogPrefix = "[cwf::ui::viewer]";

/**
 * @brief tiny logging helper that annotates viewer progress uwu
 *
 * ⚠️ IMPURE FUNCTION (has side effects)
 *
 * this helper prints high-level breadcrumbs to stdout so we can diagnose
 * initialization issues and rendering stalls without attaching a debugger.
 * every call prefixes messages with the viewer tag for easier grepping.
 *
 * @tparam Args formatting argument pack forwarded to std::format
 *
 * @param[in] fmt fmtlib/`std::format`-style string literal with vibe
 * @param[in] args arguments that satisfy the format string requirements
 */
template <typename... Args>
void log_viewer(std::format_string<Args...> fmt, Args &&...args)
{
    std::println("{} {}", kViewerLogPrefix, std::format(fmt, std::forward<Args>(args)...));
}

/**
 * @brief helper that fabricates a ViewerError while keeping the breadcrumb vibes uwu
 */
[[nodiscard]] auto make_error(std::string message, std::initializer_list<std::string> ctx = {}) -> ViewerError
{
    ViewerError err{};
    err.message = std::move(message);
    err.context.assign(ctx.begin(), ctx.end());
    return err;
}

struct Vec3
{
    float x{0.0F};
    float y{0.0F};
    float z{0.0F};
};

struct Vec4
{
    float x{0.0F}; // Vec4 defined above for vertex color and clip transforms
    float y{0.0F};
    float z{0.0F};
    float w{1.0F};
};

struct Vertex
{
    Vec4 position{}; // use Vec4 (position.w=1.0) to ensure 16-byte alignment
    Vec4 color{};    // 16-byte alignment (still used for fallback tinting)
    float stress{0.0F};
    float pad0{0.0F};
    float pad1{0.0F};
    float pad2{0.0F};
};

struct MeshBuffers
{
    std::vector<Vertex>          vertices;
    std::vector<std::uint32_t>   indices;
    std::vector<float>           stress_values;
    std::vector<std::array<std::uint32_t, 2>> edges;
    Vec3                         bounds_min{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                                            std::numeric_limits<float>::infinity()};
    Vec3                         bounds_max{-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                            -std::numeric_limits<float>::infinity()};
    Vec3                         rest_bounds_min{std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(),
                                                 std::numeric_limits<float>::infinity()};
    Vec3                         rest_bounds_max{-std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                                 -std::numeric_limits<float>::infinity()};
    std::vector<Vec4>            rest_positions;
    std::vector<Vec4>            deformed_positions;
    bool                         has_deformation{false};
};

[[nodiscard]] constexpr auto lerp(float a, float b, float t) noexcept -> float
{
    return a + (b - a) * t;
}

[[nodiscard]] auto lerp_color(float t) noexcept -> Vec3
{
    t = std::clamp(t, 0.0F, 1.0F);
    // Stress heatmap: Blue (low) -> Green (mid) -> Red (high)
    if (t < 0.5F)
    {
        const float local = t * 2.0F;
        // Blue (0,0,1) -> Green (0,1,0)
        return Vec3{0.0F, local, 1.0F - local};
    }
    const float local = (t - 0.5F) * 2.0F;
    // Green (0,1,0) -> Red (1,0,0)
    return Vec3{local, 1.0F - local, 0.0F};
}

[[nodiscard]] auto add(Vec3 a, Vec3 b) noexcept -> Vec3 { return Vec3{a.x + b.x, a.y + b.y, a.z + b.z}; }
[[nodiscard]] auto subtract(Vec3 a, Vec3 b) noexcept -> Vec3 { return Vec3{a.x - b.x, a.y - b.y, a.z - b.z}; }
[[nodiscard]] auto scale(Vec3 v, float s) noexcept -> Vec3 { return Vec3{v.x * s, v.y * s, v.z * s}; }
[[nodiscard]] auto dot(Vec3 a, Vec3 b) noexcept -> float { return (a.x * b.x) + (a.y * b.y) + (a.z * b.z); }
[[nodiscard]] auto cross(Vec3 a, Vec3 b) noexcept -> Vec3
{
    return Vec3{(a.y * b.z) - (a.z * b.y), (a.z * b.x) - (a.x * b.z), (a.x * b.y) - (a.y * b.x)};
}
[[nodiscard]] auto length(Vec3 v) noexcept -> float { return std::sqrt(dot(v, v)); }
[[nodiscard]] auto normalize(Vec3 v) noexcept -> Vec3
{
    const float len = length(v);
    if (len < 1.0e-6F)
    {
        return Vec3{};
    }
    return scale(v, 1.0F / len);
}

struct Mat4
{
    std::array<float, 16U> data{1.0F, 0.0F, 0.0F, 0.0F,
                                0.0F, 1.0F, 0.0F, 0.0F,
                                0.0F, 0.0F, 1.0F, 0.0F,
                                0.0F, 0.0F, 0.0F, 1.0F};
};

[[nodiscard]] auto multiply(const Mat4 &a, const Mat4 &b) noexcept -> Mat4
{
    Mat4 result{};
    for (int row = 0; row < 4; ++row)
    {
        for (int col = 0; col < 4; ++col)
        {
            float sum = 0.0F;
            for (int k = 0; k < 4; ++k)
            {
                sum += a.data[row * 4 + k] * b.data[k * 4 + col];
            }
            result.data[row * 4 + col] = sum;
        }
    }
    return result;
}

[[nodiscard]] auto multiply(const Mat4 &m, const Vec4 &v) noexcept -> Vec4
{
    Vec4 result{};
    result.x = (m.data[0] * v.x) + (m.data[1] * v.y) + (m.data[2] * v.z) + (m.data[3] * v.w);
    result.y = (m.data[4] * v.x) + (m.data[5] * v.y) + (m.data[6] * v.z) + (m.data[7] * v.w);
    result.z = (m.data[8] * v.x) + (m.data[9] * v.y) + (m.data[10] * v.z) + (m.data[11] * v.w);
    result.w = (m.data[12] * v.x) + (m.data[13] * v.y) + (m.data[14] * v.z) + (m.data[15] * v.w);
    return result;
}

[[nodiscard]] auto transpose(const Mat4 &m) noexcept -> Mat4
{
    Mat4 res{};
    for (int r = 0; r < 4; ++r)
    {
        for (int c = 0; c < 4; ++c)
        {
            res.data[r * 4 + c] = m.data[c * 4 + r];
        }
    }
    return res;
}

[[nodiscard]] auto make_perspective(float vfov_radians, float aspect, float z_near, float z_far) noexcept -> Mat4
{
    const float f = 1.0F / std::tan(vfov_radians * 0.5F);
    Mat4        mat{};
    // Vulkan style (column-major math used on GPU, [0, 1] Z range):
    // M =
    // [ f/aspect   0        0      0 ]
    // [ 0        -f        0      0 ]
    // [ 0         0    zf/(zn-zf)  -1 ]
    // [ 0         0  (zn*zf)/(zn-zf) 0 ]
    const float C = z_far / (z_near - z_far);
    const float D = (z_near * z_far) / (z_near - z_far);
    mat.data = {f / aspect, 0.0F, 0.0F, 0.0F,
                0.0F, -f, 0.0F, 0.0F,
                0.0F, 0.0F, C, -1.0F,
                0.0F, 0.0F, D, 0.0F};
    return mat;
}

[[nodiscard]] auto make_look_at(Vec3 eye, Vec3 center, Vec3 up) noexcept -> Mat4
{
    // Vulkan-style, column-vector view: forward = eye - target
    const Vec3 f = normalize(subtract(eye, center));
    const Vec3 s = normalize(cross(up, f));
    const Vec3 u = cross(f, s);

    Mat4 mat{};
    mat.data = {
        s.x, u.x, f.x, 0.0F,
        s.y, u.y, f.y, 0.0F,
        s.z, u.z, f.z, 0.0F,
        -dot(s, eye), -dot(u, eye), -dot(f, eye), 1.0F
    };
    return mat;
}



struct CameraState
{
    Vec3 focus{};
    float yaw{3.9F};   // Look from (-,-,-) octant to see 3 faces of the tet
    float pitch{-0.6F}; // Look slightly up
    float distance{5.0F};
    float min_distance{0.1F};
    float max_distance{500.0F};
};

struct CameraInput
{
    bool   rotating{false};
    double last_x{0.0};
    double last_y{0.0};
    float  pending_scroll{0.0F};
};

[[nodiscard]] auto build_mesh_buffers(const mesh::Mesh &mesh,
                                      const mesh::pack::PackingResult &packing,
                                      const post::DerivedFieldSet &derived) -> MeshBuffers
{
    MeshBuffers output{};
    const auto  node_count = static_cast<std::size_t>(packing.metadata.node_count);
    output.vertices.resize(node_count);
    output.rest_positions.resize(node_count);
    output.deformed_positions.resize(node_count);
    output.stress_values.resize(node_count, 0.0F);

    const auto max_nodes = derived.nodes.size();
    float       max_vm    = 0.0F;
    for (const auto &node : derived.nodes)
    {
        max_vm = std::max(max_vm, node.von_mises);
    }
    if (max_vm <= 0.0F)
    {
        max_vm = 1.0F;
    }
    const float inv_max_vm = 1.0F / max_vm;
    bool        deformation_present = false;

    for (std::size_t node = 0; node < node_count; ++node)
    {
        const float rest_x = packing.buffers.nodes.position0.x[node];
        const float rest_y = packing.buffers.nodes.position0.y[node];
        const float rest_z = packing.buffers.nodes.position0.z[node];
        const float disp_x = packing.buffers.nodes.displacement.x[node];
        const float disp_y = packing.buffers.nodes.displacement.y[node];
        const float disp_z = packing.buffers.nodes.displacement.z[node];
        const float px     = rest_x + disp_x;
        const float py     = rest_y + disp_y;
        const float pz     = rest_z + disp_z;
        const float vm = (node < max_nodes) ? derived.nodes[node].von_mises : 0.0F;
        const float normalized_vm = vm * inv_max_vm;
        const Vec3  color = lerp_color(normalized_vm);
        const Vec4  rest_pos{rest_x, rest_y, rest_z, 1.0F};
        const Vec4  def_pos{px, py, pz, 1.0F};
        output.rest_positions[node]    = rest_pos;
        output.deformed_positions[node] = def_pos;
        output.vertices[node]          = Vertex{def_pos, Vec4{color.x, color.y, color.z, 1.0F}, normalized_vm};
        output.stress_values[node]     = vm;
        output.bounds_min.x            = std::min(output.bounds_min.x, px);
        output.bounds_min.y            = std::min(output.bounds_min.y, py);
        output.bounds_min.z            = std::min(output.bounds_min.z, pz);
        output.bounds_max.x            = std::max(output.bounds_max.x, px);
        output.bounds_max.y            = std::max(output.bounds_max.y, py);
        output.bounds_max.z            = std::max(output.bounds_max.z, pz);
        output.rest_bounds_min.x       = std::min(output.rest_bounds_min.x, rest_x);
        output.rest_bounds_min.y       = std::min(output.rest_bounds_min.y, rest_y);
        output.rest_bounds_min.z       = std::min(output.rest_bounds_min.z, rest_z);
        output.rest_bounds_max.x       = std::max(output.rest_bounds_max.x, rest_x);
        output.rest_bounds_max.y       = std::max(output.rest_bounds_max.y, rest_y);
        output.rest_bounds_max.z       = std::max(output.rest_bounds_max.z, rest_z);
        deformation_present = deformation_present || (std::abs(disp_x) > 1.0e-6F || std::abs(disp_y) > 1.0e-6F ||
                                                      std::abs(disp_z) > 1.0e-6F);
    }
    output.has_deformation = deformation_present;

    // Correct winding for tetrahedrons (all faces pointing outwards)
    // Assuming node 3 is the "peak" and 0-1-2 is the base (CCW from outside)
    // Faces: (0,2,1), (0,1,3), (0,3,2), (1,2,3)
    constexpr std::array<std::array<std::uint32_t, 3>, 4> kTetFaces = {{{0U, 2U, 1U}, {0U, 1U, 3U}, {0U, 3U, 2U}, {1U, 2U, 3U}}};
    constexpr std::array<std::array<std::uint32_t, 4>, 6> kHexFaces = {{{0U, 1U, 2U, 3U}, {4U, 5U, 6U, 7U},
                                                                       {0U, 1U, 5U, 4U}, {1U, 2U, 6U, 5U},
                                                                       {2U, 3U, 7U, 6U}, {3U, 0U, 4U, 7U}}};
    constexpr std::array<std::array<std::uint32_t, 2>, 6> kTetEdges = {{{0U, 1U}, {1U, 2U}, {2U, 0U}, {0U, 3U}, {1U, 3U}, {2U, 3U}}};
    constexpr std::array<std::array<std::uint32_t, 2>, 12> kHexEdges = {{{0U, 1U}, {1U, 2U}, {2U, 3U}, {3U, 0U},
                                                                         {4U, 5U}, {5U, 6U}, {6U, 7U}, {7U, 4U},
                                                                         {0U, 4U}, {1U, 5U}, {2U, 6U}, {3U, 7U}}};

    const auto emit = [&](std::uint32_t a, std::uint32_t b, std::uint32_t c) {
        constexpr auto kInvalid = std::numeric_limits<std::uint32_t>::max();
        if (a == kInvalid || b == kInvalid || c == kInvalid)
        {
            return;
        }
        if (a >= node_count || b >= node_count || c >= node_count)
        {
            log_viewer("emit: rejected face ({}, {}, {}) because node_count is {}", a, b, c, node_count);
            return;
        }
        output.indices.push_back(a);
        output.indices.push_back(b);
        output.indices.push_back(c);
    };

    std::unordered_set<std::uint64_t> edge_set;
    edge_set.reserve(mesh.elements.size() * 12U);
    const auto add_edge = [&](std::uint32_t a, std::uint32_t b) {
        constexpr auto kInvalid = std::numeric_limits<std::uint32_t>::max();
        if (a == kInvalid || b == kInvalid)
        {
            return;
        }
        if (a >= node_count || b >= node_count)
        {
            return;
        }
        if (a > b)
        {
            std::swap(a, b);
        }
        const std::uint64_t key = (static_cast<std::uint64_t>(a) << 32U) | static_cast<std::uint64_t>(b);
        if (edge_set.insert(key).second)
        {
            output.edges.push_back({a, b});
        }
    };

    for (const auto &element : mesh.elements)
    {
        if (element.geometry == mesh::ElementGeometry::Tetrahedron4)
        {
            for (const auto &face : kTetFaces)
            {
                emit(element.nodes[face[0]], element.nodes[face[1]], element.nodes[face[2]]);
            }
            for (const auto &edge : kTetEdges)
            {
                add_edge(element.nodes[edge[0]], element.nodes[edge[1]]);
            }
        }
        else
        {
            for (const auto &face : kHexFaces)
            {
                emit(element.nodes[face[0]], element.nodes[face[1]], element.nodes[face[2]]);
                emit(element.nodes[face[0]], element.nodes[face[2]], element.nodes[face[3]]);
            }
            for (const auto &edge : kHexEdges)
            {
                add_edge(element.nodes[edge[0]], element.nodes[edge[1]]);
            }
        }
    }
    log_viewer("build_mesh_buffers: processed {} elements", mesh.elements.size());

    if (!std::isfinite(output.bounds_min.x))
    {
        output.bounds_min = Vec3{-1.0F, -1.0F, -1.0F};
        output.bounds_max = Vec3{1.0F, 1.0F, 1.0F};
    }
    if (!std::isfinite(output.rest_bounds_min.x))
    {
        output.rest_bounds_min = Vec3{-1.0F, -1.0F, -1.0F};
        output.rest_bounds_max = Vec3{1.0F, 1.0F, 1.0F};
    }

    log_viewer("built mesh buffers: {} vertices, {} indices, bounds min ({:.3f}, {:.3f}, {:.3f}) max ({:.3f}, {:.3f}, {:.3f})",
               output.vertices.size(), output.indices.size(), output.bounds_min.x, output.bounds_min.y, output.bounds_min.z,
               output.bounds_max.x, output.bounds_max.y, output.bounds_max.z);
    log_viewer("rest-space bounds min ({:.3f}, {:.3f}, {:.3f}) max ({:.3f}, {:.3f}, {:.3f}) deformation? {}", output.rest_bounds_min.x,
               output.rest_bounds_min.y, output.rest_bounds_min.z, output.rest_bounds_max.x, output.rest_bounds_max.y,
               output.rest_bounds_max.z, output.has_deformation);
    if (!output.vertices.empty())
    {
        const auto &sample = output.vertices.front();
        log_viewer("sample vertex v0 pos ({:.3f}, {:.3f}, {:.3f}) color ({:.3f}, {:.3f}, {:.3f}) stress {:.3f}", sample.position.x,
                   sample.position.y, sample.position.z, sample.color.x, sample.color.y, sample.color.z, sample.stress);
    }

    return output;
}

/**
 * @brief crafts an orbit camera that frames the incoming mesh bounds uwu
 *
 * this helper mirrors the earlier ad-hoc math from run_viewer_once but keeps it
 * reusable so the ImGui "reset" button can snap the view back to a sensible
 * spot. the camera orbits around the mesh centroid, looks slightly down and to
 * the side for depth cues, and scales its radius based on the bounding sphere.
 */
[[nodiscard]] auto make_default_camera(const MeshBuffers &mesh) noexcept -> CameraState
{
    CameraState camera{};
    const Vec3  size = subtract(mesh.bounds_max, mesh.bounds_min);
    const Vec3  focus{lerp(mesh.bounds_min.x, mesh.bounds_max.x, 0.5F),
                     lerp(mesh.bounds_min.y, mesh.bounds_max.y, 0.5F),
                     lerp(mesh.bounds_min.z, mesh.bounds_max.z, 0.5F)};
    const float radius = std::max(length(size) * 0.5F, 0.5F);
    camera.focus        = focus;
    camera.distance     = std::max(radius * 2.5F, 1.0F);
    camera.min_distance = std::max(radius * 0.1F, 0.05F);
    camera.max_distance = camera.distance * 100.0F;
    return camera;
}

struct GlfwContext
{
    GlfwContext()
    {
        if (glfwInit() != GLFW_TRUE)
        {
            throw std::runtime_error("failed to initialize GLFW");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
        window = glfwCreateWindow(1280, 800, "CiviWave FEM", nullptr, nullptr);
        if (!window)
        {
            glfwTerminate();
            throw std::runtime_error("failed to create GLFW window");
        }
    }

    ~GlfwContext()
    {
        if (window)
        {
            glfwDestroyWindow(window);
            window = nullptr;
        }
        glfwTerminate();
    }

    GLFWwindow *window{nullptr};
};

struct SwapchainSupportDetails
{
    VkSurfaceCapabilitiesKHR        capabilities{};
    std::vector<VkSurfaceFormatKHR> formats;
    std::vector<VkPresentModeKHR>   present_modes;
};

struct QueueFamilyIndices
{
    std::optional<std::uint32_t> graphics;
    std::optional<std::uint32_t> present;
    [[nodiscard]] auto complete() const noexcept -> bool { return graphics.has_value() && present.has_value(); }
};

class VulkanViewer
{
public:
    VulkanViewer(GLFWwindow *window, MeshBuffers buffers, CameraState camera, double simulation_time);
    ~VulkanViewer();

    void run();

private:
    void init_vulkan();
    void create_instance();
    void setup_debug_messenger();
    void create_surface();
    void pick_physical_device();
    void create_logical_device();
    void create_swapchain();
    void create_image_views();
    void create_render_pass();
    void create_descriptor_set_layout();
    void create_graphics_pipeline();
    void create_command_pool();
    void create_depth_resources();
    void create_framebuffers();
    void create_vertex_buffer();
    void create_index_buffer();
    void create_uniform_buffers();
    void create_descriptor_pool();
    void create_descriptor_sets();
    void create_command_buffers();
    void create_sync_objects();

    void recreate_swapchain();
    void cleanup_swapchain();

    void draw_frame(ImDrawData *draw_data);
    void record_command_buffer(VkCommandBuffer command_buffer, std::uint32_t image_index, ImDrawData *draw_data);
    void update_uniform_buffer(std::uint32_t frame_index);

    [[nodiscard]] auto query_swapchain_support(VkPhysicalDevice device) const -> SwapchainSupportDetails;
    [[nodiscard]] auto find_queue_families(VkPhysicalDevice device) const -> QueueFamilyIndices;
    [[nodiscard]] auto check_device_suitability(VkPhysicalDevice device) const -> bool;
    [[nodiscard]] auto choose_surface_format(const std::vector<VkSurfaceFormatKHR> &formats) const -> VkSurfaceFormatKHR;
    [[nodiscard]] auto choose_present_mode(const std::vector<VkPresentModeKHR> &modes) const -> VkPresentModeKHR;
    [[nodiscard]] auto choose_swap_extent(const VkSurfaceCapabilitiesKHR &capabilities) const -> VkExtent2D;
    [[nodiscard]] auto find_memory_type(std::uint32_t type_filter, VkMemoryPropertyFlags properties) const -> std::uint32_t;
    [[nodiscard]] auto find_depth_format() const -> VkFormat;
    [[nodiscard]] auto load_shader_module(const std::filesystem::path &path) const -> VkShaderModule;

    void process_camera_input();
    void apply_scroll_delta();
    void update_window_title(double fps);
    void init_imgui();
    void shutdown_imgui();
    void begin_imgui_frame();
    void build_ui();
    void reset_camera();
    void upload_vertex_buffer();
    void set_deformation_enabled(bool enabled);
    void refresh_camera_matrices();
    void update_projected_vertices();
    void update_hover_state();
    void handle_vertex_selection();
    void render_overlays();
    void recompute_display_stress();
    void apply_display_stress_to_vertices();
    void set_stress_anchor(std::size_t vertex_index);
    void normalize_display_stress();
    void apply_deformation_scale(float weight);

    static void framebuffer_resize_callback(GLFWwindow *window, int width, int height);
    static void scroll_callback(GLFWwindow *window, double /*xoffset*/, double yoffset);

private:
    [[nodiscard]] auto stress_direction() const noexcept -> Vec3;
    [[nodiscard]] auto get_vertex_position(std::size_t index) const -> Vec3;
    [[nodiscard]] auto project_position(const Vec4 &position) const -> std::optional<ImVec2>;

    GLFWwindow *window_{};
    MeshBuffers mesh_{};
    CameraState camera_{};
    CameraState initial_camera_{};
    CameraInput camera_input_{};
    double simulation_time_{0.0};

    VkInstance               instance_{VK_NULL_HANDLE};
    VkDebugUtilsMessengerEXT debug_messenger_{VK_NULL_HANDLE};
    VkSurfaceKHR             surface_{VK_NULL_HANDLE};
    VkPhysicalDevice         physical_device_{VK_NULL_HANDLE};
    VkDevice                 device_{VK_NULL_HANDLE};
    VkQueue                  graphics_queue_{VK_NULL_HANDLE};
    VkQueue                  present_queue_{VK_NULL_HANDLE};
    QueueFamilyIndices       queue_family_indices_{};
    VkSwapchainKHR           swapchain_{VK_NULL_HANDLE};
    VkFormat                 swapchain_image_format_{};
    VkExtent2D               swapchain_extent_{};
    std::vector<VkImage>     swapchain_images_{};
    std::vector<VkImageView> swapchain_image_views_{};
    VkRenderPass             render_pass_{VK_NULL_HANDLE};
    VkDescriptorSetLayout    descriptor_set_layout_{VK_NULL_HANDLE};
    VkPipelineLayout         pipeline_layout_{VK_NULL_HANDLE};
    VkPipeline               pipeline_{VK_NULL_HANDLE};
    VkCommandPool            command_pool_{VK_NULL_HANDLE};
    std::vector<VkFramebuffer> framebuffers_{};
    VkImage                    depth_image_{VK_NULL_HANDLE};
    VkDeviceMemory             depth_memory_{VK_NULL_HANDLE};
    VkImageView                depth_image_view_{VK_NULL_HANDLE};

    VkBuffer       vertex_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory vertex_memory_{VK_NULL_HANDLE};
    VkBuffer       index_buffer_{VK_NULL_HANDLE};
    VkDeviceMemory index_memory_{VK_NULL_HANDLE};

    struct FrameResources
    {
        VkBuffer        uniform_buffer{VK_NULL_HANDLE};
        VkDeviceMemory  uniform_memory{VK_NULL_HANDLE};
        void *          mapped{nullptr};
        VkCommandBuffer command_buffer{VK_NULL_HANDLE};
        VkSemaphore     image_available{VK_NULL_HANDLE};
        VkSemaphore     render_finished{VK_NULL_HANDLE};
        VkFence         in_flight{VK_NULL_HANDLE};
    };

    std::vector<FrameResources> frames_{};
    VkDescriptorPool            descriptor_pool_{VK_NULL_HANDLE};
    std::vector<VkDescriptorSet> descriptor_sets_{};
    std::size_t                 current_frame_{0};
    bool                        framebuffer_resized_{false};
    std::filesystem::path       shader_directory_{};
    VkDescriptorPool            imgui_descriptor_pool_{VK_NULL_HANDLE};
    bool                        imgui_initialized_{false};
    std::uint32_t               min_image_count_{0U};
    bool                        drawcall_logged_{false};
    bool                        uniform_log_logged_{false};
    bool                        clip_space_logged_{false};
    bool                        debug_wireframe_{false};
    bool                        debug_disable_depth_{false};
    bool                        deformation_enabled_{true};
    std::vector<Vec3>           deformation_offsets_{};
    float                       deformation_scale_{1.0F};
    float                       current_deformation_weight_{1.0F};
    float                       max_deformation_offset_{0.0F};
    bool                        vertex_data_dirty_{false};
    VkDeviceSize                vertex_buffer_size_{0};
    Mat4                        current_view_matrix_{};
    Mat4                        current_proj_matrix_{};
    Mat4                        current_view_proj_{};
    Mat4                        current_view_proj_cpu_{};
    bool                        camera_matrices_dirty_{true};

    struct StressVectorState
    {
        bool         enabled{false};
        std::size_t  anchor_vertex{0};
        float        magnitude{0.25F};
        float        yaw{0.0F};
        float        pitch{0.0F};
        float        falloff{0.35F};
        float        arrow_length{1.0F};
        bool         visible{true};
    };

    StressVectorState           stress_state_{};
    std::vector<float>          base_stress_{};
    std::vector<float>          display_stress_{};
    std::optional<std::size_t>  hovered_vertex_{};
    std::vector<ImVec2>         projected_vertices_{};
    std::vector<bool>           projected_visible_{};
    float                       edge_outline_thickness_{1.5F};
    float                       vertex_outline_thickness_{1.5F};
    float                       vertex_marker_radius_{5.0F};
    float                       hover_radius_px_{12.0F};
    bool                        overlays_enabled_{true};
    bool                        show_edges_{true};
    bool                        show_vertices_{true};
    bool                        show_hover_labels_{true};
    bool                        require_ctrl_for_selection_{true};
    bool                        selection_in_progress_{false};
};

const std::vector<const char *> kValidationLayers = {"VK_LAYER_KHRONOS_validation"};
const std::vector<const char *> kDeviceExtensions  = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

[[nodiscard]] auto validation_layers_supported() -> bool
{
    std::uint32_t layer_count = 0U;
    vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
    std::vector<VkLayerProperties> layers(layer_count);
    vkEnumerateInstanceLayerProperties(&layer_count, layers.data());
    for (const auto *layer_name : kValidationLayers)
    {
        const bool found = std::ranges::any_of(layers, [layer_name](const VkLayerProperties &props) {
            return std::string_view{props.layerName} == layer_name;
        });
        if (!found)
        {
            return false;
        }
    }
    return true;
}

VulkanViewer::VulkanViewer(GLFWwindow *window, MeshBuffers buffers, CameraState camera, double simulation_time)
    : window_(window), mesh_(std::move(buffers)), camera_(camera), initial_camera_(camera), simulation_time_(simulation_time)
{
    shader_directory_ = std::filesystem::path{CWF_SHADER_BUILD_DIR};
    if (!std::filesystem::exists(shader_directory_))
    {
        throw std::runtime_error("shader directory missing: " + shader_directory_.string());
    }
    if (!window_)
    {
        throw std::runtime_error("GLFW window is null");
    }
    glfwSetWindowUserPointer(window_, this);
    glfwSetFramebufferSizeCallback(window_, framebuffer_resize_callback);
    glfwSetScrollCallback(window_, scroll_callback);
    const Vec3 rest_extent = subtract(mesh_.rest_bounds_max, mesh_.rest_bounds_min);
    const Vec3 def_extent  = subtract(mesh_.bounds_max, mesh_.bounds_min);
    log_viewer("viewer ctor: shader dir '{}', vertices {}, indices {}, rest extent ({:.3f}, {:.3f}, {:.3f}) deformed extent ({:.3f}, {:.3f}, {:.3f})",
               shader_directory_.string(), mesh_.vertices.size(), mesh_.indices.size(), rest_extent.x, rest_extent.y,
               rest_extent.z, def_extent.x, def_extent.y, def_extent.z);
    base_stress_ = mesh_.stress_values;
    if (base_stress_.size() < mesh_.vertices.size())
    {
        base_stress_.resize(mesh_.vertices.size(), 0.0F);
    }
    display_stress_ = base_stress_;
    if (!mesh_.vertices.empty())
    {
        stress_state_.anchor_vertex = std::min(stress_state_.anchor_vertex, mesh_.vertices.size() - 1U);
    }
    projected_vertices_.resize(mesh_.vertices.size(), ImVec2{0.0F, 0.0F});
    projected_visible_.resize(mesh_.vertices.size(), false);
    apply_display_stress_to_vertices();
    if (mesh_.has_deformation)
    {
        const bool rest_match = mesh_.rest_positions.size() == mesh_.vertices.size();
        const bool def_match  = mesh_.deformed_positions.size() == mesh_.vertices.size();
        if (rest_match && def_match)
        {
            deformation_offsets_.resize(mesh_.vertices.size());
            float max_offset = 0.0F;
            for (std::size_t i = 0; i < mesh_.vertices.size(); ++i)
            {
                const Vec4 &rest = mesh_.rest_positions[i];
                const Vec4 &def  = mesh_.deformed_positions[i];
                Vec3        offset{def.x - rest.x, def.y - rest.y, def.z - rest.z};
                deformation_offsets_[i] = offset;
                max_offset              = std::max(max_offset, length(offset));
            }
            current_deformation_weight_ = deformation_enabled_ ? deformation_scale_ : 0.0F;
            log_viewer("viewer ctor: deformation offsets captured (max {:.6f} m)", max_offset);
            max_deformation_offset_ = max_offset;
            if (max_offset < 1.0e-6F)
            {
                mesh_.has_deformation      = false;
                deformation_offsets_.clear();
                deformation_enabled_       = false;
                current_deformation_weight_ = 0.0F;
                max_deformation_offset_    = 0.0F;
            }
            else if (!deformation_enabled_)
            {
                apply_deformation_scale(0.0F);
            }
        }
        else
        {
            log_viewer("viewer ctor: deformation buffers mismatched (rest {}, deformed {}, vertices {}), disabling toggle",
                       mesh_.rest_positions.size(), mesh_.deformed_positions.size(), mesh_.vertices.size());
            mesh_.has_deformation = false;
            deformation_enabled_  = false;
            deformation_offsets_.clear();
            current_deformation_weight_ = 0.0F;
            max_deformation_offset_    = 0.0F;
        }
    }
    else
    {
        deformation_enabled_        = false;
        current_deformation_weight_ = 0.0F;
        max_deformation_offset_     = 0.0F;
    }

    init_vulkan();
    refresh_camera_matrices();
    update_projected_vertices();
    update_hover_state();
}

VulkanViewer::~VulkanViewer()
{
    if (device_ != VK_NULL_HANDLE)
    {
        vkDeviceWaitIdle(device_);
    }

    shutdown_imgui();
    cleanup_swapchain();

    for (auto &frame : frames_)
    {
        if (frame.uniform_buffer)
        {
            vkDestroyBuffer(device_, frame.uniform_buffer, nullptr);
        }
        if (frame.uniform_memory)
        {
            vkFreeMemory(device_, frame.uniform_memory, nullptr);
        }
        if (frame.image_available)
        {
            vkDestroySemaphore(device_, frame.image_available, nullptr);
        }
        if (frame.render_finished)
        {
            vkDestroySemaphore(device_, frame.render_finished, nullptr);
        }
        if (frame.in_flight)
        {
            vkDestroyFence(device_, frame.in_flight, nullptr);
        }
    }

    if (descriptor_pool_)
    {
        vkDestroyDescriptorPool(device_, descriptor_pool_, nullptr);
    }
    if (descriptor_set_layout_)
    {
        vkDestroyDescriptorSetLayout(device_, descriptor_set_layout_, nullptr);
    }
    if (pipeline_)
    {
        vkDestroyPipeline(device_, pipeline_, nullptr);
    }
    if (pipeline_layout_)
    {
        vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
    }
    if (render_pass_)
    {
        vkDestroyRenderPass(device_, render_pass_, nullptr);
    }
    if (command_pool_)
    {
        vkDestroyCommandPool(device_, command_pool_, nullptr);
    }
    if (vertex_buffer_)
    {
        vkDestroyBuffer(device_, vertex_buffer_, nullptr);
    }
    if (vertex_memory_)
    {
        vkFreeMemory(device_, vertex_memory_, nullptr);
    }
    if (index_buffer_)
    {
        vkDestroyBuffer(device_, index_buffer_, nullptr);
    }
    if (index_memory_)
    {
        vkFreeMemory(device_, index_memory_, nullptr);
    }
    if (device_)
    {
        vkDestroyDevice(device_, nullptr);
    }
    if (surface_)
    {
        vkDestroySurfaceKHR(instance_, surface_, nullptr);
    }
    if (debug_messenger_)
    {
        const auto destroy = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (destroy)
        {
            destroy(instance_, debug_messenger_, nullptr);
        }
    }
    if (instance_)
    {
        vkDestroyInstance(instance_, nullptr);
    }
}

void VulkanViewer::run()
{
    auto last_time   = std::chrono::high_resolution_clock::now();
    int  frame_count = 0;
    while (!glfwWindowShouldClose(window_))
    {
        glfwPollEvents();
        if (glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        {
            glfwSetWindowShouldClose(window_, GLFW_TRUE);
        }
        process_camera_input();
        apply_scroll_delta();
        refresh_camera_matrices();
        update_projected_vertices();
        update_hover_state();
        handle_vertex_selection();
        ImDrawData *draw_data = nullptr;
        if (imgui_initialized_)
        {
            begin_imgui_frame();
            build_ui();
            render_overlays();
            ImGui::Render();
            draw_data = ImGui::GetDrawData();
        }
        draw_frame(draw_data);

        ++frame_count;
        const auto now = std::chrono::high_resolution_clock::now();
        if (now - last_time > std::chrono::seconds{1})
        {
            const double seconds = std::chrono::duration<double>(now - last_time).count();
            const double fps     = static_cast<double>(frame_count) / seconds;
            update_window_title(fps);
            frame_count = 0;
            last_time   = now;
        }
    }
}

void VulkanViewer::init_vulkan()
{
    log_viewer("init_vulkan: begin resource boot");
    create_instance();
    setup_debug_messenger();
    create_surface();
    pick_physical_device();
    queue_family_indices_ = find_queue_families(physical_device_);
    create_logical_device();
    create_swapchain();
    create_image_views();
    create_render_pass();
    create_descriptor_set_layout();
    create_graphics_pipeline();
    create_command_pool();
    create_depth_resources();
    create_framebuffers();
    create_vertex_buffer();
    create_index_buffer();
    create_uniform_buffers();
    create_descriptor_pool();
    create_descriptor_sets();
    create_command_buffers();
    create_sync_objects();
    init_imgui();
    log_viewer("init_vulkan: all resources ready ({} swapchain images)", swapchain_images_.size());
}

void VulkanViewer::create_instance()
{
    log_viewer("create_instance: requesting {} validation layers", kValidationLayers.size());
    if (!validation_layers_supported())
    {
        throw std::runtime_error("validation layers requested but unavailable");
    }

    VkApplicationInfo app_info{};
    app_info.sType              = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName   = "CiviWave Viewer";
    app_info.applicationVersion = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.pEngineName        = "cwf";
    app_info.engineVersion      = VK_MAKE_API_VERSION(0, 0, 1, 0);
    app_info.apiVersion         = VK_API_VERSION_1_3;

    std::uint32_t             glfw_count = 0;
    const char **             glfw_ext   = glfwGetRequiredInstanceExtensions(&glfw_count);
    std::vector<const char *> extensions(glfw_ext, glfw_ext + glfw_count);
    extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkInstanceCreateInfo create_info{};
    create_info.sType                   = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo        = &app_info;
    create_info.enabledExtensionCount   = static_cast<std::uint32_t>(extensions.size());
    create_info.ppEnabledExtensionNames = extensions.data();
    create_info.enabledLayerCount       = static_cast<std::uint32_t>(kValidationLayers.size());
    create_info.ppEnabledLayerNames     = kValidationLayers.data();

    VkDebugUtilsMessengerCreateInfoEXT debug_info{};
    debug_info.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    debug_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    debug_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    debug_info.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                    const VkDebugUtilsMessengerCallbackDataEXT *callback, void *) -> VkBool32 {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            std::fprintf(stderr, "[vulkan] %s\n", callback->pMessage);
        }
        return VK_FALSE;
    };
    create_info.pNext = &debug_info;

    if (vkCreateInstance(&create_info, nullptr, &instance_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateInstance failed");
    }
    log_viewer("create_instance: success with {} extensions", extensions.size());
}

void VulkanViewer::setup_debug_messenger()
{
    log_viewer("setup_debug_messenger: installing Vulkan debug callbacks");
    VkDebugUtilsMessengerCreateInfoEXT create_info{};
    create_info.sType           = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    create_info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                  VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    create_info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                              VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    create_info.pfnUserCallback = [](VkDebugUtilsMessageSeverityFlagBitsEXT severity, VkDebugUtilsMessageTypeFlagsEXT,
                                    const VkDebugUtilsMessengerCallbackDataEXT *callback, void *) -> VkBool32 {
        if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        {
            std::fprintf(stderr, "[vulkan] %s\n", callback->pMessage);
        }
        return VK_FALSE;
    };

    const auto create_fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
        vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
    if (!create_fn || create_fn(instance_, &create_info, nullptr, &debug_messenger_) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create debug messenger");
    }
    log_viewer("setup_debug_messenger: ready");
}

void VulkanViewer::create_surface()
{
    log_viewer("create_surface: binding GLFW window to Vulkan");
    if (glfwCreateWindowSurface(instance_, window_, nullptr, &surface_) != VK_SUCCESS)
    {
        throw std::runtime_error("glfwCreateWindowSurface failed");
    }
    log_viewer("create_surface: success");
}

auto VulkanViewer::find_queue_families(VkPhysicalDevice device) const -> QueueFamilyIndices
{
    QueueFamilyIndices indices{};
    std::uint32_t      family_count = 0U;
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, nullptr);
    std::vector<VkQueueFamilyProperties> families(family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(device, &family_count, families.data());
    for (std::uint32_t i = 0; i < family_count; ++i)
    {
        if ((families[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) != 0U)
        {
            indices.graphics = i;
        }
        VkBool32 present_support = VK_FALSE;
        vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface_, &present_support);
        if (present_support == VK_TRUE)
        {
            indices.present = i;
        }
        if (indices.complete())
        {
            break;
        }
    }
    return indices;
}

auto VulkanViewer::query_swapchain_support(VkPhysicalDevice device) const -> SwapchainSupportDetails
{
    SwapchainSupportDetails details{};
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(device, surface_, &details.capabilities);
    std::uint32_t count = 0U;
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &count, nullptr);
    details.formats.resize(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(device, surface_, &count, details.formats.data());

    count = 0U;
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &count, nullptr);
    details.present_modes.resize(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface_, &count, details.present_modes.data());

    return details;
}

auto VulkanViewer::check_device_suitability(VkPhysicalDevice device) const -> bool
{
    const auto indices = find_queue_families(device);
    if (!indices.complete())
    {
        return false;
    }
    std::uint32_t ext_count = 0U;
    vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> available(ext_count);
    vkEnumerateDeviceExtensionProperties(device, nullptr, &ext_count, available.data());
    for (const auto *required : kDeviceExtensions)
    {
        const bool found = std::ranges::any_of(available, [required](const VkExtensionProperties &ext) {
            return std::string_view{ext.extensionName} == required;
        });
        if (!found)
        {
            return false;
        }
    }

    const auto support = query_swapchain_support(device);
    return !support.formats.empty() && !support.present_modes.empty();
}

void VulkanViewer::pick_physical_device()
{
    std::uint32_t device_count = 0U;
    vkEnumeratePhysicalDevices(instance_, &device_count, nullptr);
    if (device_count == 0U)
    {
        throw std::runtime_error("no Vulkan physical devices detected");
    }
    log_viewer("pick_physical_device: evaluating {} adapters", device_count);
    std::vector<VkPhysicalDevice> devices(device_count);
    vkEnumeratePhysicalDevices(instance_, &device_count, devices.data());

    for (const auto device : devices)
    {
        if (check_device_suitability(device))
        {
            physical_device_ = device;
            break;
        }
    }
    if (physical_device_ == VK_NULL_HANDLE)
    {
        throw std::runtime_error("no suitable GPU found for viewer");
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(physical_device_, &props);
    log_viewer("pick_physical_device: selected '{}' (api {}.{}, driver 0x{:08x})", props.deviceName,
               VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion), props.driverVersion);
}

void VulkanViewer::create_logical_device()
{
    log_viewer("create_logical_device: graphics queue {} present queue {}", queue_family_indices_.graphics.value(),
               queue_family_indices_.present.value());
    std::vector<VkDeviceQueueCreateInfo> queue_infos;
    std::vector<std::uint32_t>          unique_indices;
    unique_indices.push_back(queue_family_indices_.graphics.value());
    if (queue_family_indices_.present.value() != queue_family_indices_.graphics.value())
    {
        unique_indices.push_back(queue_family_indices_.present.value());
    }

    const float priority = 1.0F;
    for (const auto family : unique_indices)
    {
        VkDeviceQueueCreateInfo info{};
        info.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        info.queueFamilyIndex = family;
        info.queueCount       = 1U;
        info.pQueuePriorities = &priority;
        queue_infos.push_back(info);
    }

    VkPhysicalDeviceFeatures features{};
    features.fillModeNonSolid = VK_TRUE;

    VkDeviceCreateInfo create_info{};
    create_info.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    create_info.queueCreateInfoCount    = static_cast<std::uint32_t>(queue_infos.size());
    create_info.pQueueCreateInfos       = queue_infos.data();
    create_info.pEnabledFeatures        = &features;
    create_info.enabledExtensionCount   = static_cast<std::uint32_t>(kDeviceExtensions.size());
    create_info.ppEnabledExtensionNames = kDeviceExtensions.data();
    create_info.enabledLayerCount       = static_cast<std::uint32_t>(kValidationLayers.size());
    create_info.ppEnabledLayerNames     = kValidationLayers.data();

    if (vkCreateDevice(physical_device_, &create_info, nullptr, &device_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDevice failed");
    }

    vkGetDeviceQueue(device_, queue_family_indices_.graphics.value(), 0U, &graphics_queue_);
    vkGetDeviceQueue(device_, queue_family_indices_.present.value(), 0U, &present_queue_);
    log_viewer("create_logical_device: device ready");
}

auto VulkanViewer::choose_surface_format(const std::vector<VkSurfaceFormatKHR> &formats) const -> VkSurfaceFormatKHR
{
    for (const auto &fmt : formats)
    {
        if (fmt.format == VK_FORMAT_B8G8R8A8_SRGB && fmt.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR)
        {
            return fmt;
        }
    }
    return formats.front();
}

auto VulkanViewer::choose_present_mode(const std::vector<VkPresentModeKHR> &modes) const -> VkPresentModeKHR
{
    for (const auto mode : modes)
    {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR)
        {
            return mode;
        }
    }
    return VK_PRESENT_MODE_FIFO_KHR;
}

auto VulkanViewer::choose_swap_extent(const VkSurfaceCapabilitiesKHR &capabilities) const -> VkExtent2D
{
    if (capabilities.currentExtent.width != std::numeric_limits<std::uint32_t>::max())
    {
        return capabilities.currentExtent;
    }
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    VkExtent2D extent{};
    extent.width  = std::clamp(static_cast<std::uint32_t>(width), capabilities.minImageExtent.width,
                               capabilities.maxImageExtent.width);
    extent.height = std::clamp(static_cast<std::uint32_t>(height), capabilities.minImageExtent.height,
                               capabilities.maxImageExtent.height);
    return extent;
}

void VulkanViewer::create_swapchain()
{
    log_viewer("create_swapchain: querying surface support");
    const auto support    = query_swapchain_support(physical_device_);
    const auto surface    = choose_surface_format(support.formats);
    const auto present    = choose_present_mode(support.present_modes);
    const auto extent     = choose_swap_extent(support.capabilities);
    std::uint32_t images  = support.capabilities.minImageCount + 1U;
    if (support.capabilities.maxImageCount > 0U && images > support.capabilities.maxImageCount)
    {
        images = support.capabilities.maxImageCount;
    }
    min_image_count_ = images;

    VkSwapchainCreateInfoKHR create_info{};
    create_info.sType            = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    create_info.surface          = surface_;
    create_info.minImageCount    = images;
    create_info.imageFormat      = surface.format;
    create_info.imageColorSpace  = surface.colorSpace;
    create_info.imageExtent      = extent;
    create_info.imageArrayLayers = 1U;
    create_info.imageUsage       = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;

    std::uint32_t queue_indices[] = {queue_family_indices_.graphics.value(), queue_family_indices_.present.value()};
    if (queue_family_indices_.graphics != queue_family_indices_.present)
    {
        create_info.imageSharingMode      = VK_SHARING_MODE_CONCURRENT;
        create_info.queueFamilyIndexCount = 2U;
        create_info.pQueueFamilyIndices   = queue_indices;
    }
    else
    {
        create_info.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    }
    create_info.preTransform   = support.capabilities.currentTransform;
    create_info.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    create_info.presentMode    = present;
    create_info.clipped        = VK_TRUE;
    create_info.oldSwapchain   = VK_NULL_HANDLE;

    if (vkCreateSwapchainKHR(device_, &create_info, nullptr, &swapchain_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateSwapchainKHR failed");
    }

    swapchain_image_format_ = surface.format;
    swapchain_extent_       = extent;

    std::uint32_t image_count = 0U;
    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, nullptr);
    swapchain_images_.resize(image_count);
    vkGetSwapchainImagesKHR(device_, swapchain_, &image_count, swapchain_images_.data());
    log_viewer("create_swapchain: {} images @ {}x{} format {}", image_count, extent.width, extent.height,
               static_cast<int>(surface.format));
    if (imgui_initialized_)
    {
        ImGui_ImplVulkan_SetMinImageCount(min_image_count_);
    }
}

void VulkanViewer::create_image_views()
{
    swapchain_image_views_.resize(swapchain_images_.size());
    for (std::size_t i = 0; i < swapchain_images_.size(); ++i)
    {
        VkImageViewCreateInfo view{};
        view.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        view.image                           = swapchain_images_[i];
        view.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
        view.format                          = swapchain_image_format_;
        view.components.r                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.components.g                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.components.b                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.components.a                    = VK_COMPONENT_SWIZZLE_IDENTITY;
        view.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
        view.subresourceRange.baseMipLevel   = 0U;
        view.subresourceRange.levelCount     = 1U;
        view.subresourceRange.baseArrayLayer = 0U;
        view.subresourceRange.layerCount     = 1U;
        if (vkCreateImageView(device_, &view, nullptr, &swapchain_image_views_[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateImageView failed");
        }
    }
    log_viewer("create_image_views: generated {} views", swapchain_image_views_.size());
}

void VulkanViewer::create_render_pass()
{
    const VkFormat depth_format = find_depth_format();
    log_viewer("create_render_pass: color format {} depth format {}", static_cast<int>(swapchain_image_format_),
               static_cast<int>(depth_format));
    VkAttachmentDescription color{};
    color.format         = swapchain_image_format_;
    color.samples        = VK_SAMPLE_COUNT_1_BIT;
    color.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    color.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    color.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    color.finalLayout    = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;

    VkAttachmentDescription depth{};
    depth.format         = depth_format;
    depth.samples        = VK_SAMPLE_COUNT_1_BIT;
    depth.loadOp         = VK_ATTACHMENT_LOAD_OP_CLEAR;
    depth.storeOp        = VK_ATTACHMENT_STORE_OP_STORE;
    depth.stencilLoadOp  = VK_ATTACHMENT_LOAD_OP_DONT_CARE;
    depth.stencilStoreOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    depth.initialLayout  = VK_IMAGE_LAYOUT_UNDEFINED;
    depth.finalLayout    = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkAttachmentReference color_ref{};
    color_ref.attachment = 0U;
    color_ref.layout     = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;

    VkAttachmentReference depth_ref{};
    depth_ref.attachment = 1U;
    depth_ref.layout     = VK_IMAGE_LAYOUT_DEPTH_STENCIL_ATTACHMENT_OPTIMAL;

    VkSubpassDescription subpass{};
    subpass.pipelineBindPoint       = VK_PIPELINE_BIND_POINT_GRAPHICS;
    subpass.colorAttachmentCount    = 1U;
    subpass.pColorAttachments       = &color_ref;
    subpass.pDepthStencilAttachment = &depth_ref;

    VkSubpassDependency dependency{};
    dependency.srcSubpass    = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass    = 0U;
    dependency.srcStageMask  = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT;
    dependency.dstStageMask  = dependency.srcStageMask;
    dependency.dstAccessMask = VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;

    std::array attachments{color, depth};
    VkRenderPassCreateInfo render_info{};
    render_info.sType           = VK_STRUCTURE_TYPE_RENDER_PASS_CREATE_INFO;
    render_info.attachmentCount = static_cast<std::uint32_t>(attachments.size());
    render_info.pAttachments    = attachments.data();
    render_info.subpassCount    = 1U;
    render_info.pSubpasses      = &subpass;
    render_info.dependencyCount = 1U;
    render_info.pDependencies   = &dependency;

    if (vkCreateRenderPass(device_, &render_info, nullptr, &render_pass_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateRenderPass failed");
    }
    log_viewer("create_render_pass: ok");
}

void VulkanViewer::create_descriptor_set_layout()
{
    log_viewer("create_descriptor_set_layout: building camera UBO binding");
    VkDescriptorSetLayoutBinding camera_binding{};
    camera_binding.binding         = 0U;
    camera_binding.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    camera_binding.descriptorCount = 1U;
    camera_binding.stageFlags      = VK_SHADER_STAGE_VERTEX_BIT;

    VkDescriptorSetLayoutCreateInfo create_info{};
    create_info.sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    create_info.bindingCount = 1U;
    create_info.pBindings    = &camera_binding;
    if (vkCreateDescriptorSetLayout(device_, &create_info, nullptr, &descriptor_set_layout_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorSetLayout failed");
    }
    log_viewer("create_descriptor_set_layout: success");
}

void VulkanViewer::create_graphics_pipeline()
{
    log_viewer("create_graphics_pipeline: loading shaders from '{}'", shader_directory_.string());
    const auto vert_path = shader_directory_ / "viewer_mesh_vert.spv";
    const auto frag_path = shader_directory_ / "viewer_mesh_frag.spv";
    VkShaderModule vert_module = load_shader_module(vert_path);
    VkShaderModule frag_module = load_shader_module(frag_path);

    VkPipelineShaderStageCreateInfo vert_stage{};
    vert_stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    vert_stage.stage  = VK_SHADER_STAGE_VERTEX_BIT;
    vert_stage.module = vert_module;
    vert_stage.pName  = "main";

    VkPipelineShaderStageCreateInfo frag_stage{};
    frag_stage.sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    frag_stage.stage  = VK_SHADER_STAGE_FRAGMENT_BIT;
    frag_stage.module = frag_module;
    frag_stage.pName  = "main";

    VkPipelineShaderStageCreateInfo stages[] = {vert_stage, frag_stage};

    VkVertexInputBindingDescription binding{};
    binding.binding   = 0U;
    binding.stride    = sizeof(Vertex);
    binding.inputRate = VK_VERTEX_INPUT_RATE_VERTEX;

    std::array<VkVertexInputAttributeDescription, 3> attributes{};
    attributes[0] = VkVertexInputAttributeDescription{0U, 0U, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, position)};
    attributes[1] = VkVertexInputAttributeDescription{1U, 0U, VK_FORMAT_R32G32B32A32_SFLOAT, offsetof(Vertex, color)};
    attributes[2] = VkVertexInputAttributeDescription{2U, 0U, VK_FORMAT_R32_SFLOAT, offsetof(Vertex, stress)};

    VkPipelineVertexInputStateCreateInfo vertex_input{};
    vertex_input.sType                           = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO;
    vertex_input.vertexBindingDescriptionCount   = 1U;
    vertex_input.pVertexBindingDescriptions      = &binding;
    vertex_input.vertexAttributeDescriptionCount = static_cast<std::uint32_t>(attributes.size());
    vertex_input.pVertexAttributeDescriptions    = attributes.data();

    VkPipelineInputAssemblyStateCreateInfo input_assembly{};
    input_assembly.sType                  = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO;
    input_assembly.topology               = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    input_assembly.primitiveRestartEnable = VK_FALSE;

    VkPipelineViewportStateCreateInfo viewport{};
    viewport.sType         = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    viewport.viewportCount = 1U;
    viewport.scissorCount  = 1U;

    VkPipelineRasterizationStateCreateInfo raster{};
    raster.sType                   = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO;
    raster.depthClampEnable        = VK_FALSE;
    raster.rasterizerDiscardEnable = VK_FALSE;
    raster.polygonMode             = debug_wireframe_ ? VK_POLYGON_MODE_LINE : VK_POLYGON_MODE_FILL;
    raster.lineWidth               = 1.0F;
    raster.cullMode                = VK_CULL_MODE_BACK_BIT;
    // Vulkan Y-flip in projection matrix inverts winding, so we need CLOCKWISE for front faces
    raster.frontFace               = VK_FRONT_FACE_CLOCKWISE;
    raster.depthBiasEnable         = VK_FALSE;

    VkPipelineMultisampleStateCreateInfo msaa{};
    msaa.sType                = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO;
    msaa.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;

    VkPipelineDepthStencilStateCreateInfo depth{};
    depth.sType            = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    depth.depthTestEnable  = debug_disable_depth_ ? VK_FALSE : VK_TRUE; // configurable via UI
    depth.depthWriteEnable = VK_TRUE;
    depth.depthCompareOp   = VK_COMPARE_OP_LESS;

    VkPipelineColorBlendAttachmentState color_blend{};
    color_blend.colorWriteMask = VK_COLOR_COMPONENT_R_BIT | VK_COLOR_COMPONENT_G_BIT | VK_COLOR_COMPONENT_B_BIT |
                                  VK_COLOR_COMPONENT_A_BIT;
    color_blend.blendEnable = VK_FALSE;

    VkPipelineColorBlendStateCreateInfo blend{};
    blend.sType           = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO;
    blend.attachmentCount = 1U;
    blend.pAttachments    = &color_blend;

    std::array<VkDynamicState, 2> dynamics{VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{};
    dynamic.sType             = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO;
    dynamic.dynamicStateCount = static_cast<std::uint32_t>(dynamics.size());
    dynamic.pDynamicStates    = dynamics.data();

    VkPipelineLayoutCreateInfo layout{};
    layout.sType          = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layout.setLayoutCount = 1U;
    layout.pSetLayouts    = &descriptor_set_layout_;
    if (vkCreatePipelineLayout(device_, &layout, nullptr, &pipeline_layout_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert_module, nullptr);
        vkDestroyShaderModule(device_, frag_module, nullptr);
        throw std::runtime_error("vkCreatePipelineLayout failed");
    }

    VkGraphicsPipelineCreateInfo pipeline_info{};
    pipeline_info.sType               = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO;
    pipeline_info.stageCount          = 2U;
    pipeline_info.pStages             = stages;
    pipeline_info.pVertexInputState   = &vertex_input;
    pipeline_info.pInputAssemblyState = &input_assembly;
    pipeline_info.pViewportState      = &viewport;
    pipeline_info.pRasterizationState = &raster;
    pipeline_info.pMultisampleState   = &msaa;
    pipeline_info.pDepthStencilState  = &depth;
    pipeline_info.pColorBlendState    = &blend;
    pipeline_info.pDynamicState       = &dynamic;
    pipeline_info.layout              = pipeline_layout_;
    pipeline_info.renderPass          = render_pass_;
    pipeline_info.subpass             = 0U;

    if (vkCreateGraphicsPipelines(device_, VK_NULL_HANDLE, 1U, &pipeline_info, nullptr, &pipeline_) != VK_SUCCESS)
    {
        vkDestroyShaderModule(device_, vert_module, nullptr);
        vkDestroyShaderModule(device_, frag_module, nullptr);
        throw std::runtime_error("vkCreateGraphicsPipelines failed");
    }

    vkDestroyShaderModule(device_, vert_module, nullptr);
    vkDestroyShaderModule(device_, frag_module, nullptr);
    log_viewer("create_graphics_pipeline: pipeline ready (vertex stride {} bytes)", sizeof(Vertex));
}

auto VulkanViewer::find_memory_type(std::uint32_t type_filter, VkMemoryPropertyFlags properties) const -> std::uint32_t
{
    VkPhysicalDeviceMemoryProperties mem{};
    vkGetPhysicalDeviceMemoryProperties(physical_device_, &mem);
    for (std::uint32_t i = 0; i < mem.memoryTypeCount; ++i)
    {
        if ((type_filter & (1U << i)) && (mem.memoryTypes[i].propertyFlags & properties) == properties)
        {
            return i;
        }
    }
    throw std::runtime_error("no matching memory type");
}

auto VulkanViewer::find_depth_format() const -> VkFormat
{
    const std::array candidates{VK_FORMAT_D32_SFLOAT, VK_FORMAT_D32_SFLOAT_S8_UINT, VK_FORMAT_D24_UNORM_S8_UINT};
    for (const auto format : candidates)
    {
        VkFormatProperties props{};
        vkGetPhysicalDeviceFormatProperties(physical_device_, format, &props);
        if ((props.optimalTilingFeatures & VK_FORMAT_FEATURE_DEPTH_STENCIL_ATTACHMENT_BIT) != 0U)
        {
            return format;
        }
    }
    return VK_FORMAT_D32_SFLOAT;
}

void VulkanViewer::create_depth_resources()
{
    const VkFormat format = find_depth_format();
    log_viewer("create_depth_resources: {}x{} format {}", swapchain_extent_.width, swapchain_extent_.height, static_cast<int>(format));
    VkImageCreateInfo image{};
    image.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    image.imageType     = VK_IMAGE_TYPE_2D;
    image.extent.width  = swapchain_extent_.width;
    image.extent.height = swapchain_extent_.height;
    image.extent.depth  = 1U;
    image.mipLevels     = 1U;
    image.arrayLayers   = 1U;
    image.format        = format;
    image.tiling        = VK_IMAGE_TILING_OPTIMAL;
    image.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    image.usage         = VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT;
    image.samples       = VK_SAMPLE_COUNT_1_BIT;

    if (vkCreateImage(device_, &image, nullptr, &depth_image_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateImage (depth) failed");
    }
    VkMemoryRequirements requirements{};
    vkGetImageMemoryRequirements(device_, depth_image_, &requirements);
    VkMemoryAllocateInfo alloc{};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = requirements.size;
    alloc.memoryTypeIndex = find_memory_type(requirements.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (vkAllocateMemory(device_, &alloc, nullptr, &depth_memory_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateMemory (depth) failed");
    }
    vkBindImageMemory(device_, depth_image_, depth_memory_, 0U);

    VkImageViewCreateInfo view{};
    view.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    view.image                           = depth_image_;
    view.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
    view.format                          = format;
    view.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_DEPTH_BIT;
    view.subresourceRange.baseMipLevel   = 0U;
    view.subresourceRange.levelCount     = 1U;
    view.subresourceRange.baseArrayLayer = 0U;
    view.subresourceRange.layerCount     = 1U;
    if (vkCreateImageView(device_, &view, nullptr, &depth_image_view_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateImageView (depth) failed");
    }
    log_viewer("create_depth_resources: depth image + view ready");
}

void VulkanViewer::create_framebuffers()
{
    framebuffers_.resize(swapchain_image_views_.size());
    for (std::size_t i = 0; i < swapchain_image_views_.size(); ++i)
    {
        std::array attachments{swapchain_image_views_[i], depth_image_view_};
        VkFramebufferCreateInfo info{};
        info.sType           = VK_STRUCTURE_TYPE_FRAMEBUFFER_CREATE_INFO;
        info.renderPass      = render_pass_;
        info.attachmentCount = static_cast<std::uint32_t>(attachments.size());
        info.pAttachments    = attachments.data();
        info.width           = swapchain_extent_.width;
        info.height          = swapchain_extent_.height;
        info.layers          = 1U;
        if (vkCreateFramebuffer(device_, &info, nullptr, &framebuffers_[i]) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateFramebuffer failed");
        }
    }
    log_viewer("create_framebuffers: {} swapchain framebuffers", framebuffers_.size());
}

void VulkanViewer::create_command_pool()
{
    const auto indices = find_queue_families(physical_device_);
    log_viewer("create_command_pool: graphics queue family {}", indices.graphics.value());
    VkCommandPoolCreateInfo info{};
    info.sType            = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    info.flags            = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    info.queueFamilyIndex = indices.graphics.value();
    if (vkCreateCommandPool(device_, &info, nullptr, &command_pool_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateCommandPool failed");
    }
    log_viewer("create_command_pool: success");
}

void VulkanViewer::create_vertex_buffer()
{
    const VkDeviceSize size = sizeof(Vertex) * mesh_.vertices.size();
    if (size == 0)
    {
        log_viewer("create_vertex_buffer: skipped (no vertices)");
        return;
    }
    log_viewer("create_vertex_buffer: uploading {} vertices ({} bytes)", mesh_.vertices.size(), static_cast<std::size_t>(size));
    VkBufferCreateInfo info{};
    info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size        = size;
    info.usage       = VK_BUFFER_USAGE_VERTEX_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &info, nullptr, &vertex_buffer_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateBuffer (vertex) failed");
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, vertex_buffer_, &req);
    VkMemoryAllocateInfo alloc{};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = req.size;
    alloc.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(device_, &alloc, nullptr, &vertex_memory_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateMemory (vertex) failed");
    }
    vkBindBufferMemory(device_, vertex_buffer_, vertex_memory_, 0U);
    vertex_buffer_size_ = size;
    upload_vertex_buffer();
    vertex_data_dirty_ = false;
    log_viewer("create_vertex_buffer: upload complete");
}

void VulkanViewer::create_index_buffer()
{
    const VkDeviceSize size = sizeof(std::uint32_t) * mesh_.indices.size();
    if (size == 0)
    {
        log_viewer("create_index_buffer: skipped (no indices)");
        return;
    }
    log_viewer("create_index_buffer: uploading {} indices ({} bytes)", mesh_.indices.size(), static_cast<std::size_t>(size));
    VkBufferCreateInfo info{};
    info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    info.size        = size;
    info.usage       = VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &info, nullptr, &index_buffer_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateBuffer (index) failed");
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, index_buffer_, &req);
    VkMemoryAllocateInfo alloc{};
    alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc.allocationSize  = req.size;
    alloc.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    if (vkAllocateMemory(device_, &alloc, nullptr, &index_memory_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateMemory (index) failed");
    }
    vkBindBufferMemory(device_, index_buffer_, index_memory_, 0U);
    void *data = nullptr;
    vkMapMemory(device_, index_memory_, 0U, size, 0U, &data);
    std::memcpy(data, mesh_.indices.data(), static_cast<std::size_t>(size));
    vkUnmapMemory(device_, index_memory_);
    log_viewer("create_index_buffer: upload complete");
}

void VulkanViewer::create_uniform_buffers()
{
    frames_.resize(2U);
    const VkDeviceSize buffer_size = sizeof(Mat4);
    log_viewer("create_uniform_buffers: {} frames, {} bytes each", frames_.size(), static_cast<std::size_t>(buffer_size));
    for (auto &frame : frames_)
    {
        VkBufferCreateInfo info{};
        info.sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        info.size        = buffer_size;
        info.usage       = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
        info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        if (vkCreateBuffer(device_, &info, nullptr, &frame.uniform_buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("vkCreateBuffer (uniform) failed");
        }
        VkMemoryRequirements req{};
        vkGetBufferMemoryRequirements(device_, frame.uniform_buffer, &req);
        VkMemoryAllocateInfo alloc{};
        alloc.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        alloc.allocationSize  = req.size;
        alloc.memoryTypeIndex = find_memory_type(req.memoryTypeBits,
                                                 VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        if (vkAllocateMemory(device_, &alloc, nullptr, &frame.uniform_memory) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateMemory (uniform) failed");
        }
        vkBindBufferMemory(device_, frame.uniform_buffer, frame.uniform_memory, 0U);
        vkMapMemory(device_, frame.uniform_memory, 0U, buffer_size, 0U, &frame.mapped);
    }
    log_viewer("create_uniform_buffers: mapped uniform buffers");
}

void VulkanViewer::create_descriptor_pool()
{
    VkDescriptorPoolSize pool{};
    pool.type            = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    pool.descriptorCount = static_cast<std::uint32_t>(frames_.size());

    VkDescriptorPoolCreateInfo info{};
    info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    info.poolSizeCount = 1U;
    info.pPoolSizes    = &pool;
    info.maxSets       = static_cast<std::uint32_t>(frames_.size());
    if (vkCreateDescriptorPool(device_, &info, nullptr, &descriptor_pool_) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateDescriptorPool failed");
    }
    log_viewer("create_descriptor_pool: {} uniform sets", frames_.size());
}

void VulkanViewer::create_descriptor_sets()
{
    log_viewer("create_descriptor_sets: allocating {} sets", frames_.size());
    std::vector<VkDescriptorSetLayout> layouts(frames_.size(), descriptor_set_layout_);
    VkDescriptorSetAllocateInfo        alloc{};
    alloc.sType              = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    alloc.descriptorPool     = descriptor_pool_;
    alloc.descriptorSetCount = static_cast<std::uint32_t>(frames_.size());
    alloc.pSetLayouts        = layouts.data();
    descriptor_sets_.resize(frames_.size());
    if (vkAllocateDescriptorSets(device_, &alloc, descriptor_sets_.data()) != VK_SUCCESS)
    {
        throw std::runtime_error("vkAllocateDescriptorSets failed");
    }

    for (std::size_t i = 0; i < frames_.size(); ++i)
    {
        VkDescriptorBufferInfo buffer{};
        buffer.buffer = frames_[i].uniform_buffer;
        buffer.offset = 0U;
        buffer.range  = sizeof(Mat4);

        VkWriteDescriptorSet write{};
        write.sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet          = descriptor_sets_[i];
        write.dstBinding      = 0U;
        write.descriptorType  = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        write.descriptorCount = 1U;
        write.pBufferInfo     = &buffer;
        vkUpdateDescriptorSets(device_, 1U, &write, 0U, nullptr);
    }
    log_viewer("create_descriptor_sets: descriptors updated");
}

void VulkanViewer::create_command_buffers()
{
    log_viewer("create_command_buffers: allocating {} primary buffers", frames_.size());
    for (auto &frame : frames_)
    {
        VkCommandBufferAllocateInfo alloc{};
        alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        alloc.commandPool        = command_pool_;
        alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        alloc.commandBufferCount = 1U;
        if (vkAllocateCommandBuffers(device_, &alloc, &frame.command_buffer) != VK_SUCCESS)
        {
            throw std::runtime_error("vkAllocateCommandBuffers failed");
        }
    }
    log_viewer("create_command_buffers: ready");
}

void VulkanViewer::create_sync_objects()
{
    VkSemaphoreCreateInfo sem{};
    sem.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkFenceCreateInfo fence{};
    fence.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fence.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (auto &frame : frames_)
    {
        if (vkCreateSemaphore(device_, &sem, nullptr, &frame.image_available) != VK_SUCCESS ||
            vkCreateSemaphore(device_, &sem, nullptr, &frame.render_finished) != VK_SUCCESS ||
            vkCreateFence(device_, &fence, nullptr, &frame.in_flight) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create synchronization objects");
        }
    }
    log_viewer("create_sync_objects: {} frames worth of semaphores/fences", frames_.size());
}

void VulkanViewer::cleanup_swapchain()
{
    log_viewer("cleanup_swapchain: tearing down {} framebuffers and {} image views", framebuffers_.size(),
               swapchain_image_views_.size());
    for (auto framebuffer : framebuffers_)
    {
        vkDestroyFramebuffer(device_, framebuffer, nullptr);
    }
    framebuffers_.clear();
    if (depth_image_view_)
    {
        vkDestroyImageView(device_, depth_image_view_, nullptr);
        depth_image_view_ = VK_NULL_HANDLE;
    }
    if (depth_image_)
    {
        vkDestroyImage(device_, depth_image_, nullptr);
        depth_image_ = VK_NULL_HANDLE;
    }
    if (depth_memory_)
    {
        vkFreeMemory(device_, depth_memory_, nullptr);
        depth_memory_ = VK_NULL_HANDLE;
    }
    for (auto view : swapchain_image_views_)
    {
        vkDestroyImageView(device_, view, nullptr);
    }
    swapchain_image_views_.clear();
    if (swapchain_)
    {
        vkDestroySwapchainKHR(device_, swapchain_, nullptr);
        swapchain_ = VK_NULL_HANDLE;
    }
    log_viewer("cleanup_swapchain: done");
}

void VulkanViewer::recreate_swapchain()
{
    log_viewer("recreate_swapchain: waiting for non-zero framebuffer size");
    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    while (width == 0 || height == 0)
    {
        glfwWaitEvents();
        glfwGetFramebufferSize(window_, &width, &height);
    }

    vkDeviceWaitIdle(device_);
    cleanup_swapchain();
    create_swapchain();
    create_image_views();
    create_depth_resources();
    create_framebuffers();
    camera_matrices_dirty_ = true;
    log_viewer("recreate_swapchain: rebuilt for {}x{}", swapchain_extent_.width, swapchain_extent_.height);
}

void VulkanViewer::draw_frame(ImDrawData *draw_data)
{
    auto &frame = frames_[current_frame_];
    vkWaitForFences(device_, 1U, &frame.in_flight, VK_TRUE, UINT64_MAX);

    if (vertex_data_dirty_ && vertex_buffer_ != VK_NULL_HANDLE)
    {
        upload_vertex_buffer();
        vertex_data_dirty_ = false;
    }

    std::uint32_t image_index = 0U;
    const VkResult acquire = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, frame.image_available, VK_NULL_HANDLE,
                                                  &image_index);
    if (acquire == VK_ERROR_OUT_OF_DATE_KHR)
    {
        log_viewer("draw_frame: swapchain out of date, recreating");
        recreate_swapchain();
        return;
    }
    else if (acquire != VK_SUCCESS && acquire != VK_SUBOPTIMAL_KHR)
    {
        log_viewer("draw_frame: vkAcquireNextImageKHR returned {}", static_cast<int>(acquire));
    }

    vkResetFences(device_, 1U, &frame.in_flight);
    vkResetCommandBuffer(frame.command_buffer, 0U);
    update_uniform_buffer(static_cast<std::uint32_t>(current_frame_));
    record_command_buffer(frame.command_buffer, image_index, draw_data);

    VkSemaphore wait_semaphores[]      = {frame.image_available};
    VkPipelineStageFlags wait_stages[] = {VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT};
    VkSemaphore signal_semaphores[]    = {frame.render_finished};

    VkSubmitInfo submit{};
    submit.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit.waitSemaphoreCount   = 1U;
    submit.pWaitSemaphores      = wait_semaphores;
    submit.pWaitDstStageMask    = wait_stages;
    submit.commandBufferCount   = 1U;
    submit.pCommandBuffers      = &frame.command_buffer;
    submit.signalSemaphoreCount = 1U;
    submit.pSignalSemaphores    = signal_semaphores;

    if (vkQueueSubmit(graphics_queue_, 1U, &submit, frame.in_flight) != VK_SUCCESS)
    {
        throw std::runtime_error("vkQueueSubmit failed");
    }

    VkPresentInfoKHR present{};
    present.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    present.waitSemaphoreCount = 1U;
    present.pWaitSemaphores    = signal_semaphores;
    present.swapchainCount     = 1U;
    present.pSwapchains        = &swapchain_;
    present.pImageIndices      = &image_index;

    const VkResult present_result = vkQueuePresentKHR(present_queue_, &present);
    if (present_result == VK_ERROR_OUT_OF_DATE_KHR || present_result == VK_SUBOPTIMAL_KHR || framebuffer_resized_)
    {
        log_viewer("draw_frame: present result {} resized flag {}", static_cast<int>(present_result), framebuffer_resized_);
        framebuffer_resized_ = false;
        recreate_swapchain();
    }
    else if (present_result != VK_SUCCESS)
    {
        throw std::runtime_error("vkQueuePresentKHR failed");
    }

    current_frame_ = (current_frame_ + 1U) % frames_.size();
}

void VulkanViewer::record_command_buffer(VkCommandBuffer command_buffer, std::uint32_t image_index, ImDrawData *draw_data)
{
    VkCommandBufferBeginInfo begin{};
    begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    if (vkBeginCommandBuffer(command_buffer, &begin) != VK_SUCCESS)
    {
        throw std::runtime_error("vkBeginCommandBuffer failed");
    }

    VkRenderPassBeginInfo render{};
    render.sType             = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    render.renderPass        = render_pass_;
    render.framebuffer       = framebuffers_[image_index];
    render.renderArea.offset = VkOffset2D{0, 0};
    render.renderArea.extent = swapchain_extent_;

    std::array<VkClearValue, 2> clears{};
    clears[0].color = {{0.02F, 0.02F, 0.03F, 1.0F}};
    clears[1].depthStencil = VkClearDepthStencilValue{1.0F, 0U};
    render.clearValueCount = static_cast<std::uint32_t>(clears.size());
    render.pClearValues    = clears.data();

    vkCmdBeginRenderPass(command_buffer, &render, VK_SUBPASS_CONTENTS_INLINE);

    VkViewport viewport{};
    viewport.x        = 0.0F;
    viewport.y        = 0.0F;
    viewport.width    = static_cast<float>(swapchain_extent_.width);
    viewport.height   = static_cast<float>(swapchain_extent_.height);
    viewport.minDepth = 0.0F;
    viewport.maxDepth = 1.0F;

    VkRect2D scissor{VkOffset2D{0, 0}, swapchain_extent_};
    vkCmdSetViewport(command_buffer, 0U, 1U, &viewport);
    vkCmdSetScissor(command_buffer, 0U, 1U, &scissor);

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_);

    VkBuffer     vertex_buffers[] = {vertex_buffer_};
    VkDeviceSize offsets[]        = {0U};
    if (vertex_buffer_)
    {
        vkCmdBindVertexBuffers(command_buffer, 0U, 1U, vertex_buffers, offsets);
    }
    if (index_buffer_)
    {
        vkCmdBindIndexBuffer(command_buffer, index_buffer_, 0U, VK_INDEX_TYPE_UINT32);
    }

    VkDescriptorSet descriptor = descriptor_sets_[current_frame_];
    vkCmdBindDescriptorSets(command_buffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline_layout_, 0U, 1U, &descriptor, 0U, nullptr);

    if (!drawcall_logged_)
    {
        log_viewer(
            "record_command_buffer: frame {} image {} vertices {} indices {} descriptor bound {}", current_frame_, image_index,
            mesh_.vertices.size(), mesh_.indices.size(), descriptor != VK_NULL_HANDLE);
        drawcall_logged_ = true;
    }

    if (!mesh_.indices.empty())
    {
        vkCmdDrawIndexed(command_buffer, static_cast<std::uint32_t>(mesh_.indices.size()), 1U, 0U, 0, 0U);
    }
    else
    {
        vkCmdDraw(command_buffer, static_cast<std::uint32_t>(mesh_.vertices.size()), 1U, 0U, 0U);
    }

    if (imgui_initialized_ && draw_data)
    {
        ImGui_ImplVulkan_RenderDrawData(draw_data, command_buffer);
    }

    vkCmdEndRenderPass(command_buffer);
    vkEndCommandBuffer(command_buffer);
}

void VulkanViewer::update_uniform_buffer(std::uint32_t frame_index)
{
    if (camera_matrices_dirty_)
    {
        refresh_camera_matrices();
    }
    std::memcpy(frames_[frame_index].mapped, current_view_proj_.data.data(), sizeof(Mat4));
    if (!uniform_log_logged_)
    {
        const Vec3 focus   = camera_.focus;
        log_viewer("update_uniform_buffer: focus ({:.3f}, {:.3f}, {:.3f}) yaw {:.3f} pitch {:.3f} dist {:.3f} extent {}x{}",
                   focus.x, focus.y, focus.z, camera_.yaw, camera_.pitch, camera_.distance, swapchain_extent_.width,
                   swapchain_extent_.height);
        uniform_log_logged_ = true;
    }

}

void VulkanViewer::process_camera_input()
{
    if (imgui_initialized_ && ImGui::GetIO().WantCaptureMouse)
    {
        camera_input_.rotating = false;
        return;
    }
    double xpos = 0.0;
    double ypos = 0.0;
    glfwGetCursorPos(window_, &xpos, &ypos);
    const int pressed = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT);
    if (pressed == GLFW_PRESS)
    {
        if (!camera_input_.rotating)
        {
            camera_input_.rotating = true;
            camera_input_.last_x   = xpos;
            camera_input_.last_y   = ypos;
        }
        const double dx = xpos - camera_input_.last_x;
        const double dy = ypos - camera_input_.last_y;
        camera_input_.last_x = xpos;
        camera_input_.last_y = ypos;
        const float prev_yaw   = camera_.yaw;
        const float prev_pitch = camera_.pitch;
        camera_.yaw -= static_cast<float>(dx) * 0.0035F;
        camera_.pitch -= static_cast<float>(dy) * 0.0035F;
        camera_.pitch = std::clamp(camera_.pitch, -1.4F, 1.4F);
        if (std::abs(camera_.yaw - prev_yaw) > 1.0e-6F || std::abs(camera_.pitch - prev_pitch) > 1.0e-6F)
        {
            camera_matrices_dirty_ = true;
        }
    }
    else
    {
        camera_input_.rotating = false;
    }
}

void VulkanViewer::apply_scroll_delta()
{
    if (camera_input_.pending_scroll != 0.0F)
    {
        const float factor = std::exp(-camera_input_.pending_scroll * 0.15F);
        camera_.distance *= factor;
        camera_matrices_dirty_ = true;
        camera_input_.pending_scroll = 0.0F;
    }
    camera_.distance = std::clamp(camera_.distance, camera_.min_distance, camera_.max_distance);
}

void VulkanViewer::update_window_title(double fps)
{
    std::string title = std::format("CiviWave FEM — {:.2f} FPS — t = {:.4f}s", fps, simulation_time_);
    glfwSetWindowTitle(window_, title.c_str());
}

void VulkanViewer::init_imgui()
{
    log_viewer("init_imgui: bootstrapping Dear ImGui ({} swapchain images, min count {})", swapchain_images_.size(),
               min_image_count_ == 0U ? static_cast<std::uint32_t>(swapchain_images_.size()) : min_image_count_);
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();

    if (!ImGui_ImplGlfw_InitForVulkan(window_, true))
    {
        throw std::runtime_error("ImGui_ImplGlfw_InitForVulkan failed");
    }

    const std::array<VkDescriptorPoolSize, 11> pool_sizes = {
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC, 1000},
        VkDescriptorPoolSize{VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT, 1000}};

    VkDescriptorPoolCreateInfo pool_info{};
    pool_info.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    pool_info.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    pool_info.poolSizeCount = static_cast<std::uint32_t>(pool_sizes.size());
    pool_info.pPoolSizes    = pool_sizes.data();
    pool_info.maxSets       = 1000U * static_cast<std::uint32_t>(pool_sizes.size());
    if (vkCreateDescriptorPool(device_, &pool_info, nullptr, &imgui_descriptor_pool_) != VK_SUCCESS)
    {
        throw std::runtime_error("failed to create ImGui descriptor pool");
    }
    log_viewer("init_imgui: descriptor pool ready ({} pool sizes)", pool_sizes.size());

    ImGui_ImplVulkan_InitInfo init{};
    init.ApiVersion                = VK_API_VERSION_1_3;
    init.Instance                  = instance_;
    init.PhysicalDevice            = physical_device_;
    init.Device                    = device_;
    init.QueueFamily               = queue_family_indices_.graphics.value();
    init.Queue                     = graphics_queue_;
    init.DescriptorPool            = imgui_descriptor_pool_;
    init.MinImageCount             = (min_image_count_ == 0U) ? static_cast<std::uint32_t>(swapchain_images_.size()) : min_image_count_;
    init.ImageCount                = static_cast<std::uint32_t>(swapchain_images_.size());
    init.PipelineInfoMain.RenderPass   = render_pass_;
    init.PipelineInfoMain.Subpass      = 0U;
    init.PipelineInfoMain.MSAASamples  = VK_SAMPLE_COUNT_1_BIT;
    init.PipelineInfoForViewports.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    init.CheckVkResultFn           = nullptr;

    if (!ImGui_ImplVulkan_Init(&init))
    {
        throw std::runtime_error("ImGui_ImplVulkan_Init failed");
    }
    imgui_initialized_ = true;
    log_viewer("init_imgui: Dear ImGui initialized");
}

void VulkanViewer::shutdown_imgui()
{
    if (device_ == VK_NULL_HANDLE)
    {
        log_viewer("shutdown_imgui: skipping (device destroyed)");
        imgui_descriptor_pool_ = VK_NULL_HANDLE;
        imgui_initialized_     = false;
        return;
    }

    if (!imgui_initialized_)
    {
        if (imgui_descriptor_pool_)
        {
            vkDestroyDescriptorPool(device_, imgui_descriptor_pool_, nullptr);
            imgui_descriptor_pool_ = VK_NULL_HANDLE;
        }
        log_viewer("shutdown_imgui: already inactive");
        return;
    }

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    imgui_initialized_ = false;
    log_viewer("shutdown_imgui: destroyed ImGui context");

    if (imgui_descriptor_pool_)
    {
        vkDestroyDescriptorPool(device_, imgui_descriptor_pool_, nullptr);
        imgui_descriptor_pool_ = VK_NULL_HANDLE;
    }
}

void VulkanViewer::begin_imgui_frame()
{
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void VulkanViewer::build_ui()
{
    ImGui::SetNextWindowBgAlpha(0.9F);
    if (ImGui::Begin("Viewer Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
    {
        ImGui::Text("Vertices: %zu", mesh_.vertices.size());
        const std::size_t triangle_count = mesh_.indices.empty() ? (mesh_.vertices.size() / 3U)
                                                                : (mesh_.indices.size() / 3U);
        ImGui::Text("Triangles: %zu", triangle_count);
        const Vec3 rest_extent = subtract(mesh_.rest_bounds_max, mesh_.rest_bounds_min);
        const Vec3 def_extent  = subtract(mesh_.bounds_max, mesh_.bounds_min);
        ImGui::Text("Rest extent: (%.3f, %.3f, %.3f)", rest_extent.x, rest_extent.y, rest_extent.z);
        ImGui::Text("Deformed extent: (%.3f, %.3f, %.3f)", def_extent.x, def_extent.y, def_extent.z);
        ImGui::Separator();
        if (ImGui::Button("Reset Camera"))
        {
            reset_camera();
        }

        ImGui::Separator();
        if (mesh_.has_deformation)
        {
            if (ImGui::CollapsingHeader("Deformation Controls", ImGuiTreeNodeFlags_DefaultOpen))
            {
                bool deform = deformation_enabled_;
                if (ImGui::Checkbox("Show deformation", &deform))
                {
                    set_deformation_enabled(deform);
                }
                ImGui::BeginDisabled(!deformation_enabled_);
                float magnitude = deformation_scale_;
                if (ImGui::SliderFloat("Deformation magnitude", &magnitude, 0.0F, 5.0F, "%.2fx"))
                {
                    deformation_scale_ = magnitude;
                    apply_deformation_scale(deformation_enabled_ ? deformation_scale_ : 0.0F);
                }
                ImGui::Text("Max solver displacement: %.4f m", max_deformation_offset_);
                ImGui::Text("Current weight: %.2fx", deformation_enabled_ ? deformation_scale_ : 0.0F);
                ImGui::EndDisabled();
            }
        }
        else
        {
            ImGui::TextDisabled("No deformation data detected in this mesh (solver reported zero offsets).");
        }

        if (ImGui::CollapsingHeader("Debug"))
        {
            bool wire  = debug_wireframe_;
            bool depth = debug_disable_depth_;
            if (ImGui::Checkbox("Wireframe", &wire))
            {
                debug_wireframe_ = wire;
                vkDeviceWaitIdle(device_);
                vkDestroyPipeline(device_, pipeline_, nullptr);
                vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
                create_graphics_pipeline();
            }
            if (ImGui::Checkbox("Disable Depth Test", &depth))
            {
                debug_disable_depth_ = depth;
                vkDeviceWaitIdle(device_);
                vkDestroyPipeline(device_, pipeline_, nullptr);
                vkDestroyPipelineLayout(device_, pipeline_layout_, nullptr);
                create_graphics_pipeline();
            }
        }

        ImGui::Separator();
        if (ImGui::CollapsingHeader("Stress Vector Controls", ImGuiTreeNodeFlags_DefaultOpen))
        {
            bool enabled = stress_state_.enabled;
            if (ImGui::Checkbox("Enable custom stress", &enabled))
            {
                stress_state_.enabled = enabled;
                recompute_display_stress();
            }
            ImGui::Text("Anchor vertex: %zu", stress_state_.anchor_vertex);
            if (hovered_vertex_.has_value())
            {
                ImGui::Text("Hovered vertex: %zu", *hovered_vertex_);
            }
            if (ImGui::Checkbox("Show stress arrow", &stress_state_.visible))
            {
                // no-op, draw flag toggled via checkbox
            }
            float magnitude = stress_state_.magnitude;
            if (ImGui::SliderFloat("Stress magnitude", &magnitude, 0.0F, 2.0F, "%.2f"))
            {
                stress_state_.magnitude = magnitude;
                recompute_display_stress();
            }
            float yaw_degrees = stress_state_.yaw * (180.0F / kPiF);
            if (ImGui::SliderFloat("Yaw (deg)", &yaw_degrees, -180.0F, 180.0F, "%.1f"))
            {
                stress_state_.yaw = yaw_degrees * (kPiF / 180.0F);
                recompute_display_stress();
            }
            float pitch_degrees = stress_state_.pitch * (180.0F / kPiF);
            if (ImGui::SliderFloat("Pitch (deg)", &pitch_degrees, -85.0F, 85.0F, "%.1f"))
            {
                stress_state_.pitch = pitch_degrees * (kPiF / 180.0F);
                recompute_display_stress();
            }
            if (ImGui::SliderFloat("Falloff", &stress_state_.falloff, 0.05F, 2.0F, "%.2f"))
            {
                recompute_display_stress();
            }
            ImGui::SliderFloat("Arrow length", &stress_state_.arrow_length, 0.1F, 10.0F, "%.2f");
            if (ImGui::Button("Reset Stress Field"))
            {
                stress_state_.enabled = false;
                display_stress_       = base_stress_;
                apply_display_stress_to_vertices();
            }
            ImGui::TextWrapped("Hold Ctrl + Left Click on a highlighted vertex to move the stress vector anchor.");
        }

        if (ImGui::CollapsingHeader("Overlay Highlights", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::Checkbox("Enable overlays", &overlays_enabled_);
            ImGui::BeginDisabled(!overlays_enabled_);
            ImGui::Checkbox("Edge outlines", &show_edges_);
            ImGui::SliderFloat("Edge thickness", &edge_outline_thickness_, 0.5F, 8.0F, "%.2f");
            ImGui::Checkbox("Vertex rims", &show_vertices_);
            ImGui::SliderFloat("Vertex radius", &vertex_marker_radius_, 2.0F, 15.0F, "%.1f");
            ImGui::SliderFloat("Vertex rim thickness", &vertex_outline_thickness_, 0.5F, 6.0F, "%.2f");
            ImGui::Checkbox("Show hover labels", &show_hover_labels_);
            ImGui::SliderFloat("Hover radius (px)", &hover_radius_px_, 4.0F, 40.0F, "%.1f");
            ImGui::Checkbox("Require Ctrl for selection", &require_ctrl_for_selection_);
            ImGui::EndDisabled();
        }
    }
    ImGui::End();
}

void VulkanViewer::reset_camera()
{
    camera_               = initial_camera_;
    camera_input_         = {};
    camera_input_.pending_scroll = 0.0F;
    log_viewer("reset_camera: yaw {:.3f} pitch {:.3f} distance {:.3f}", camera_.yaw, camera_.pitch, camera_.distance);
    camera_matrices_dirty_ = true;
}

/**
 * @brief rewrite the GPU vertex buffer with the current CPU mesh snapshot uwu
 */
void VulkanViewer::upload_vertex_buffer()
{
    if (!device_ || !vertex_buffer_ || !vertex_memory_)
    {
        return;
    }
    if (mesh_.vertices.empty() || vertex_buffer_size_ == 0U)
    {
        return;
    }
    const VkDeviceSize expected_size = sizeof(Vertex) * mesh_.vertices.size();
    if (expected_size != vertex_buffer_size_)
    {
        log_viewer("upload_vertex_buffer: size mismatch (expected {} bytes, actual {} bytes)", vertex_buffer_size_,
                   expected_size);
        return;
    }
    void *data = nullptr;
    vkMapMemory(device_, vertex_memory_, 0U, vertex_buffer_size_, 0U, &data);
    std::memcpy(data, mesh_.vertices.data(), static_cast<std::size_t>(vertex_buffer_size_));
    vkUnmapMemory(device_, vertex_memory_);
}

/**
 * @brief toggle whether the viewer shows deformed or rest positions uwu
 */
void VulkanViewer::set_deformation_enabled(bool enabled)
{
    if (!mesh_.has_deformation || deformation_offsets_.empty())
    {
        deformation_enabled_ = false;
        return;
    }
    if (enabled == deformation_enabled_)
    {
        return;
    }
    deformation_enabled_ = enabled;
    const float target_scale = enabled ? deformation_scale_ : 0.0F;
    apply_deformation_scale(target_scale);
    log_viewer("set_deformation_enabled: now showing {} geometry (scale {:.3f})", enabled ? "deformed" : "rest", target_scale);
}

/**
 * @brief recomputes view/projection matrices used by both CPU overlays and GPU UBO uploads
 */
void VulkanViewer::refresh_camera_matrices()
{
    const Vec3 focus = camera_.focus;
    const float cos_p = std::cos(camera_.pitch);
    const float sin_p = std::sin(camera_.pitch);
    const float cos_y = std::cos(camera_.yaw);
    const float sin_y = std::sin(camera_.yaw);
    Vec3       offset{camera_.distance * cos_p * cos_y, camera_.distance * sin_p, camera_.distance * cos_p * sin_y};
    const Vec3 eye = add(focus, offset);
    const Vec3 up{0.0F, 1.0F, 0.0F};

    current_view_matrix_ = make_look_at(eye, focus, up);
    const float width  = static_cast<float>(std::max(1U, swapchain_extent_.width));
    const float height = static_cast<float>(std::max(1U, swapchain_extent_.height));
    current_proj_matrix_ = make_perspective(60.0F * (kPiF / 180.0F), width / height, 0.01F, 5000.0F);
    current_view_proj_     = multiply(current_view_matrix_, current_proj_matrix_);
    current_view_proj_cpu_ = transpose(current_view_proj_);
    camera_matrices_dirty_ = false;
}

/**
 * @brief recomputes projected 2D coordinates for every vertex so ImGui can outline them
 */
void VulkanViewer::update_projected_vertices()
{
    if (!imgui_initialized_)
    {
        hovered_vertex_.reset();
        return;
    }
    if (!overlays_enabled_ || mesh_.vertices.empty())
    {
        hovered_vertex_.reset();
        if (projected_visible_.size() != mesh_.vertices.size())
        {
            projected_visible_.assign(mesh_.vertices.size(), false);
        }
        else
        {
            std::fill(projected_visible_.begin(), projected_visible_.end(), false);
        }
        return;
    }
    if (projected_vertices_.size() != mesh_.vertices.size())
    {
        projected_vertices_.assign(mesh_.vertices.size(), ImVec2{0.0F, 0.0F});
    }
    if (projected_visible_.size() != mesh_.vertices.size())
    {
        projected_visible_.assign(mesh_.vertices.size(), false);
    }
    for (std::size_t i = 0; i < mesh_.vertices.size(); ++i)
    {
        if (const auto projected = project_position(mesh_.vertices[i].position))
        {
            projected_vertices_[i] = *projected;
            projected_visible_[i]  = true;
        }
        else
        {
            projected_visible_[i] = false;
        }
    }
}

/**
 * @brief updates hovered vertex tracking using the latest projected coordinates
 */
void VulkanViewer::update_hover_state()
{
    if (!imgui_initialized_)
    {
        hovered_vertex_.reset();
        return;
    }
    if (!overlays_enabled_ || mesh_.vertices.empty())
    {
        hovered_vertex_.reset();
        return;
    }
    const ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        hovered_vertex_.reset();
        return;
    }
    double mouse_x = 0.0;
    double mouse_y = 0.0;
    glfwGetCursorPos(window_, &mouse_x, &mouse_y);
    const float radius = hover_radius_px_;
    float       best_dist = radius;
    std::optional<std::size_t> candidate{};
    for (std::size_t i = 0; i < projected_vertices_.size(); ++i)
    {
        if (!projected_visible_[i])
        {
            continue;
        }
        const ImVec2 pos = projected_vertices_[i];
        const float  dx  = pos.x - static_cast<float>(mouse_x);
        const float  dy  = pos.y - static_cast<float>(mouse_y);
        const float  dist = std::sqrt((dx * dx) + (dy * dy));
        if (dist < best_dist)
        {
            best_dist = dist;
            candidate  = i;
        }
    }
    hovered_vertex_ = candidate;
}

/**
 * @brief handles Ctrl+LMB selection to retarget the synthetic stress vector
 */
void VulkanViewer::handle_vertex_selection()
{
    if (!imgui_initialized_)
    {
        selection_in_progress_ = false;
        return;
    }
    if (!overlays_enabled_)
    {
        selection_in_progress_ = false;
        return;
    }
    const ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureMouse)
    {
        selection_in_progress_ = false;
        return;
    }
    if (!hovered_vertex_.has_value())
    {
        selection_in_progress_ = false;
        return;
    }
    const bool control_active = !require_ctrl_for_selection_ ||
                                glfwGetKey(window_, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS ||
                                glfwGetKey(window_, GLFW_KEY_RIGHT_CONTROL) == GLFW_PRESS;
    const bool left_pressed = glfwGetMouseButton(window_, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    if (control_active && left_pressed && !selection_in_progress_)
    {
        set_stress_anchor(*hovered_vertex_);
        selection_in_progress_ = true;
    }
    else if (!left_pressed)
    {
        selection_in_progress_ = false;
    }
}

/**
 * @brief draws vertex/edge outlines plus the stress direction arrow via ImGui foreground draw lists
 */
void VulkanViewer::render_overlays()
{
    if (!imgui_initialized_ || !overlays_enabled_)
    {
        return;
    }
    auto *draw = ImGui::GetForegroundDrawList();
    const ImU32 edge_color = IM_COL32(0, 0, 0, 220);
    const ImU32 vertex_color = IM_COL32(0, 0, 0, 200);
    const ImU32 anchor_color = IM_COL32(72, 207, 173, 255);
    const ImU32 hover_color  = IM_COL32(255, 214, 0, 255);

    if (show_edges_ && edge_outline_thickness_ > 0.0F)
    {
        for (const auto &edge : mesh_.edges)
        {
            if (edge.size() != 2U)
            {
                continue;
            }
            const auto a = edge[0];
            const auto b = edge[1];
            if (a >= projected_vertices_.size() || b >= projected_vertices_.size())
            {
                continue;
            }
            if (!projected_visible_[a] || !projected_visible_[b])
            {
                continue;
            }
            draw->AddLine(projected_vertices_[a], projected_vertices_[b], edge_color, edge_outline_thickness_);
        }
    }

    if (show_vertices_ && vertex_marker_radius_ > 0.0F)
    {
        for (std::size_t i = 0; i < projected_vertices_.size(); ++i)
        {
            if (!projected_visible_[i])
            {
                continue;
            }
            draw->AddCircle(projected_vertices_[i], vertex_marker_radius_, vertex_color, 20, vertex_outline_thickness_);
        }
    }

    if (stress_state_.visible && stress_state_.anchor_vertex < projected_vertices_.size() && projected_visible_[stress_state_.anchor_vertex])
    {
        const ImVec2 anchor_screen = projected_vertices_[stress_state_.anchor_vertex];
        const auto   anchor_world  = get_vertex_position(stress_state_.anchor_vertex);
        const Vec3   dir           = stress_direction();
        const float  arrow_extent  = std::max(0.0F, stress_state_.magnitude) * stress_state_.arrow_length;
        const Vec3   tip_world     = add(anchor_world, scale(dir, arrow_extent));
        const Vec4   tip_vec4{tip_world.x, tip_world.y, tip_world.z, 1.0F};
        const auto   tip_screen    = project_position(tip_vec4);
        if (arrow_extent > 1.0e-4F && tip_screen.has_value())
        {
            const ImU32 arrow_color = IM_COL32(255, 128, 0, 255);
            draw->AddLine(anchor_screen, *tip_screen, arrow_color, vertex_outline_thickness_ * 1.2F);
            const ImVec2 dir_screen = ImVec2{(*tip_screen).x - anchor_screen.x, (*tip_screen).y - anchor_screen.y};
            const float  len        = std::sqrt((dir_screen.x * dir_screen.x) + (dir_screen.y * dir_screen.y));
            if (len > 1.0F)
            {
                const float inv = 1.0F / len;
                const ImVec2 norm_dir{dir_screen.x * inv, dir_screen.y * inv};
                const ImVec2 left_head{-norm_dir.y, norm_dir.x};
                const float  head_size = 8.0F;
                const ImVec2 head_a{(*tip_screen).x - norm_dir.x * head_size + left_head.x * head_size * 0.5F,
                                    (*tip_screen).y - norm_dir.y * head_size + left_head.y * head_size * 0.5F};
                const ImVec2 head_b{(*tip_screen).x - norm_dir.x * head_size - left_head.x * head_size * 0.5F,
                                    (*tip_screen).y - norm_dir.y * head_size - left_head.y * head_size * 0.5F};
                draw->AddTriangleFilled(*tip_screen, head_a, head_b, arrow_color);
            }
        }
        draw->AddCircle(anchor_screen, vertex_marker_radius_ * 1.4F, anchor_color, 24, vertex_outline_thickness_ * 1.5F);
    }

    if (hovered_vertex_.has_value() && projected_visible_[*hovered_vertex_])
    {
        draw->AddCircle(projected_vertices_[*hovered_vertex_], vertex_marker_radius_ * 1.6F, hover_color, 24,
                        vertex_outline_thickness_ * 1.3F);
        if (show_hover_labels_)
        {
            const std::string label = std::format("v{}", *hovered_vertex_);
            const ImVec2 text_pos{projected_vertices_[*hovered_vertex_].x + 6.0F, projected_vertices_[*hovered_vertex_].y - 18.0F};
            draw->AddText(text_pos, IM_COL32(255, 255, 255, 255), label.c_str());
        }
    }
}

/**
 * @brief recomputes the synthetic per-vertex stress field based on the UI sliders
 */
void VulkanViewer::recompute_display_stress()
{
    if (base_stress_.empty())
    {
        return;
    }
    display_stress_ = base_stress_;
    if (!stress_state_.enabled || mesh_.vertices.empty() || stress_state_.anchor_vertex >= display_stress_.size())
    {
        apply_display_stress_to_vertices();
        return;
    }

    const Vec3 anchor_position = get_vertex_position(stress_state_.anchor_vertex);
    const Vec3 direction       = stress_direction();
    for (std::size_t i = 0; i < mesh_.vertices.size(); ++i)
    {
        const auto &pos4 = mesh_.vertices[i].position;
        Vec3 delta       = subtract(Vec3{pos4.x, pos4.y, pos4.z}, anchor_position);
        const float distance = length(delta);
        if (distance < 1.0e-5F)
        {
            display_stress_[i] += stress_state_.magnitude;
            continue;
        }
        delta = scale(delta, 1.0F / distance);
        const float alignment = dot(delta, direction);
        if (alignment <= 0.0F)
        {
            continue;
        }
        const float attenuation = std::exp(-distance * stress_state_.falloff);
        const float influence   = stress_state_.magnitude * alignment * attenuation;
        display_stress_[i] += influence;
    }
    apply_display_stress_to_vertices();
}

/**
 * @brief writes the latest stress scalars + derived colors into the CPU vertex array and marks the GPU buffer dirty
 */
void VulkanViewer::apply_display_stress_to_vertices()
{
    if (mesh_.vertices.size() != display_stress_.size())
    {
        return;
    }
    normalize_display_stress();
    for (std::size_t i = 0; i < mesh_.vertices.size(); ++i)
    {
        const float stress = display_stress_[i];
        const Vec3  color  = lerp_color(stress);
        mesh_.vertices[i].color  = Vec4{color.x, color.y, color.z, 1.0F};
        mesh_.vertices[i].stress = stress;
    }
    vertex_data_dirty_ = true;
}

void VulkanViewer::normalize_display_stress()
{
    if (display_stress_.empty())
    {
        return;
    }
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    for (const float value : display_stress_)
    {
        if (!std::isfinite(value))
        {
            continue;
        }
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }
    if (!std::isfinite(min_value) || !std::isfinite(max_value))
    {
        std::fill(display_stress_.begin(), display_stress_.end(), 0.0F);
        return;
    }
    const float range = max_value - min_value;
    if (range < 1.0e-6F)
    {
        const float fallback = (std::abs(min_value) > 1.0e-6F) ? 1.0F : 0.0F;
        for (auto &value : display_stress_)
        {
            value = std::isfinite(value) ? fallback : 0.0F;
        }
        return;
    }
    const float inv_range = 1.0F / range;
    for (auto &value : display_stress_)
    {
        if (!std::isfinite(value))
        {
            value = 0.0F;
            continue;
        }
        value = (value - min_value) * inv_range;
    }
}

/**
 * @brief switches the stress vector anchor to the requested vertex index
 */
void VulkanViewer::set_stress_anchor(std::size_t vertex_index)
{
    if (mesh_.vertices.empty() || vertex_index >= mesh_.vertices.size())
    {
        return;
    }
    stress_state_.anchor_vertex = vertex_index;
    stress_state_.enabled       = true;
    recompute_display_stress();
}

/**
 * @brief applies the deformation blend weight and defers the vertex buffer upload until the next frame
 */
void VulkanViewer::apply_deformation_scale(float weight)
{
    if (!mesh_.has_deformation || deformation_offsets_.empty())
    {
        return;
    }
    std::size_t count = std::min(mesh_.vertices.size(), mesh_.rest_positions.size());
    count             = std::min(count, deformation_offsets_.size());
    if (count == 0)
    {
        return;
    }
    for (std::size_t i = 0; i < count; ++i)
    {
        const Vec4 &rest   = mesh_.rest_positions[i];
        const Vec3 &offset = deformation_offsets_[i];
        mesh_.vertices[i].position = Vec4{rest.x + offset.x * weight, rest.y + offset.y * weight, rest.z + offset.z * weight, 1.0F};
    }
    current_deformation_weight_ = weight;
    vertex_data_dirty_ = true;
    camera_matrices_dirty_ = true;
}

/**
 * @brief computes the normalized stress direction based on yaw/pitch sliders
 */
auto VulkanViewer::stress_direction() const noexcept -> Vec3
{
    const float cos_p = std::cos(stress_state_.pitch);
    return Vec3{cos_p * std::cos(stress_state_.yaw), std::sin(stress_state_.pitch), cos_p * std::sin(stress_state_.yaw)};
}

/**
 * @brief fetches the world-space vertex currently displayed (rest vs deformed)
 */
auto VulkanViewer::get_vertex_position(std::size_t index) const -> Vec3
{
    if (index >= mesh_.vertices.size())
    {
        return Vec3{};
    }
    const auto &pos = mesh_.vertices[index].position;
    return Vec3{pos.x, pos.y, pos.z};
}

/**
 * @brief projects a homogeneous position into framebuffer pixel coordinates
 */
auto VulkanViewer::project_position(const Vec4 &position) const -> std::optional<ImVec2>
{
    if (swapchain_extent_.width == 0U || swapchain_extent_.height == 0U)
    {
        return std::nullopt;
    }
    const Vec4 clip = multiply(current_view_proj_cpu_, position);
    if (std::abs(clip.w) < 1.0e-5F)
    {
        return std::nullopt;
    }
    const float inv_w = 1.0F / clip.w;
    const float ndc_x = clip.x * inv_w;
    const float ndc_y = clip.y * inv_w;
    const float ndc_z = clip.z * inv_w;
    if (ndc_z < -1.0F || ndc_z > 1.5F)
    {
        return std::nullopt;
    }
    if (clip.w <= 0.0F)
    {
        return std::nullopt;
    }
    const float screen_x = ((ndc_x * 0.5F) + 0.5F) * static_cast<float>(swapchain_extent_.width);
    const float screen_y = ((ndc_y * 0.5F) + 0.5F) * static_cast<float>(swapchain_extent_.height);
    if (!imgui_initialized_)
    {
        return ImVec2{screen_x, screen_y};
    }
    const ImVec2 scale = ImGui::GetIO().DisplayFramebufferScale;
    ImVec2       logical{screen_x, screen_y};
    if (scale.x > 0.0F)
    {
        logical.x /= scale.x;
    }
    if (scale.y > 0.0F)
    {
        logical.y /= scale.y;
    }
    return logical;
}

/**
 * @brief chooses which position buffer should be treated as "world space" for overlays
 */
void VulkanViewer::framebuffer_resize_callback(GLFWwindow *window, int /*width*/, int /*height*/)
{
    if (auto *viewer = static_cast<VulkanViewer *>(glfwGetWindowUserPointer(window)))
    {
        viewer->framebuffer_resized_ = true;
        log_viewer("framebuffer_resize_callback: resize requested");
    }
}

void VulkanViewer::scroll_callback(GLFWwindow *window, double, double yoffset)
{
    if (auto *viewer = static_cast<VulkanViewer *>(glfwGetWindowUserPointer(window)))
    {
        if (viewer->imgui_initialized_ && ImGui::GetIO().WantCaptureMouse)
        {
            return;
        }
        viewer->camera_input_.pending_scroll += static_cast<float>(yoffset);
    }
}

auto VulkanViewer::load_shader_module(const std::filesystem::path &path) const -> VkShaderModule
{
    std::ifstream file{path, std::ios::binary | std::ios::ate};
    if (!file)
    {
        throw std::runtime_error("failed to open shader: " + path.string());
    }
    const auto size = file.tellg();
    if (size <= 0)
    {
        throw std::runtime_error("shader file empty: " + path.string());
    }
    log_viewer("load_shader_module: '{}' ({} bytes)", path.string(), static_cast<std::size_t>(size));
    file.seekg(0, std::ios::beg);
    const std::size_t byte_count = static_cast<std::size_t>(size);
    if ((byte_count % sizeof(std::uint32_t)) != 0U)
    {
        throw std::runtime_error("shader byte size misaligned: " + path.string());
    }
    std::vector<std::uint32_t> buffer(byte_count / sizeof(std::uint32_t));
    file.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(byte_count));

    VkShaderModuleCreateInfo info{};
    info.sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    info.codeSize = byte_count;
    info.pCode    = buffer.data();

    VkShaderModule module = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device_, &info, nullptr, &module) != VK_SUCCESS)
    {
        throw std::runtime_error("vkCreateShaderModule failed for " + path.string());
    }
    return module;
}

} // namespace

[[nodiscard]] auto run_viewer_once(const mesh::Mesh &mesh,
                                   const mesh::pack::PackingResult &packing,
                                   const post::DerivedFieldSet &derived,
                                   double simulation_time) -> std::expected<void, ViewerError>
{
    try
    {
        const auto buffers = build_mesh_buffers(mesh, packing, derived);
        const CameraState camera = make_default_camera(buffers);
        log_viewer("launching viewer: t = {:.4f}s, vertices = {}, indices = {} (camera dist {:.3f})", simulation_time,
                   buffers.vertices.size(), buffers.indices.size(), camera.distance);

        GlfwContext glfw{};
        {
            VulkanViewer viewer(glfw.window, buffers, camera, simulation_time);
            viewer.run();
        }
        return {};
    }
    catch (const std::exception &ex)
    {
        return std::unexpected(make_error(ex.what(), {"vulkan_viewer"}));
    }
}

} // namespace cwf::ui

#endif // CWF_ENABLE_UI