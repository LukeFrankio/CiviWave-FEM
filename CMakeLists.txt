cmake_minimum_required(VERSION 4.1.2 FATAL_ERROR)

# Project definition
project(CiviWaveFEM
    VERSION 0.1.0
    DESCRIPTION "Vulkan GPU Structural Analysis (FEM)"
    LANGUAGES CXX
)

# C++ standard (prefer C++26)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Options
option(BUILD_TESTS "Build unit/integration tests" ON)
option(BUILD_UI "Build optional UI (ImGui)" OFF)
option(ENABLE_TRACY "Enable Tracy profiling zones" OFF)
option(ENABLE_VALIDATION "Enable extra validation and Vulkan layers" ON)
option(ENABLE_RGP_MARKERS "Enable RGP markers in GPU command streams" ON)
option(FORCE_FETCH_DEPS "Force building dependencies from source (ignore system packages)" ON)
option(ENABLE_SANITIZERS "Enable sanitizers in Debug/Profile (GCC/Clang)" ON)
option(ENABLE_LTO "Enable link-time optimization (Release only)" OFF)
option(FETCH_SLANG "Fetch and build Slang (heavy; only enable when needed)" OFF)
option(REQUIRE_GCC_15 "Require GCC 15.2+ (set OFF in CI to allow older compilers)" ON)

# Configure policies to NEW where applicable
if(POLICY CMP0141)
    cmake_policy(SET CMP0141 NEW)
endif()

# Detect compiler and versions
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(REQUIRE_GCC_15)
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.2)
            message(FATAL_ERROR "GCC 15.2+ required. Detected ${CMAKE_CXX_COMPILER_VERSION}.")
        else()
            message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION} detected (C++26 mode)")
        endif()
    else()
        if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS 15.2)
            message(WARNING "GCC ${CMAKE_CXX_COMPILER_VERSION} detected; REQUIRE_GCC_15=OFF so configure will proceed but build may fail on older compilers.")
        else()
            message(STATUS "GCC ${CMAKE_CXX_COMPILER_VERSION} detected (C++26 mode)")
        endif()
    endif()
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    message(WARNING "Clang detected; primary toolchain is GCC 15+. Proceeding best-effort.")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    message(WARNING "MSVC detected; primary toolchain is GCC 15+. Proceeding best-effort.")
endif()

# Global warnings
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wformat=2)
endif()

# Sanitizers and profiling-friendly flags (GCC/Clang)
if((CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang"))
    if(CMAKE_BUILD_TYPE MATCHES "^Debug$|^RelWithDebInfo$|^Profile$")
        add_compile_options(-fno-omit-frame-pointer)
        add_link_options(-fno-omit-frame-pointer)
        if(ENABLE_SANITIZERS)
            add_compile_options(-fsanitize=address,undefined)
            add_link_options(-fsanitize=address,undefined)
        endif()
    endif()
    if(ENABLE_LTO AND CMAKE_BUILD_TYPE MATCHES "^Release$")
        add_compile_options(-flto)
        add_link_options(-flto)
    endif()
endif()

# Generator: prefer Ninja where not set (advisory)
# (Handled by presets; do not override here.)

# Provide project-wide output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Modules
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/Modules")
include(DetectCompilerABI OPTIONAL)
include(FetchOrSystem OPTIONAL)
include(SlangCompile OPTIONAL)
include(FormatAndTidy OPTIONAL)

# FetchContent setup
include(FetchContent)

# Export compile commands to allow clang-tidy to run
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# yaml-cpp (prefer system unless forced)
set(YAML_CPP_MIN_VERSION 0.8.0)
include(FetchOrSystem)
cw_fetch_or_system_yaml_cpp(${YAML_CPP_MIN_VERSION})

# VMA (header-only) — fetch header and create IMPORTED INTERFACE target
set(VMA_VERSION 3.3.0)
FetchContent_Declare(
    vma_src
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG v3.3.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(vma_src)
# Create an interface target only if the subproject hasn't already created one
if(NOT TARGET VulkanMemoryAllocator)
    add_library(VulkanMemoryAllocator INTERFACE)
    if(DEFINED vma_src_SOURCE_DIR)
        target_include_directories(VulkanMemoryAllocator INTERFACE "${vma_src_SOURCE_DIR}/include")
    endif()
else()
    message(STATUS "VulkanMemoryAllocator target provided by subproject; skipping local interface creation")
endif()

# Slang (compiler + runtime)
# Note: We no longer fetch and build Slang from source by default. Instead, we
# prefer a prebuilt `slangc` tool that is either on PATH or specified via the
# SLANGC environment variable. Building Slang from source is platform-heavy
# (MSVC-friendly) and can pull many submodules and external downloads.
message(STATUS "Slang integration: prefer prebuilt 'slangc' tool. Ensure 'slangc' is on PATH or set SLANGC environment variable to its path.")

# Optional: Tracy and ImGui can be added later via FetchContent (not required for Phase 2)
if(ENABLE_TRACY)
    message(STATUS "ENABLE_TRACY=ON: Fetching Tracy via FetchContent")
    FetchContent_Declare(
        tracy_src
        GIT_REPOSITORY https://github.com/wolfpld/tracy.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(tracy_src)
    if(DEFINED tracy_src_SOURCE_DIR)
        add_library(tracy_client INTERFACE)
        target_include_directories(tracy_client INTERFACE "${tracy_src_SOURCE_DIR}/client")
    endif()
endif()

if(BUILD_UI)
    message(STATUS "BUILD_UI=ON: Fetching ImGui via FetchContent")
    FetchContent_Declare(
        imgui_src
        GIT_REPOSITORY https://github.com/ocornut/imgui.git
        GIT_TAG master
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(imgui_src)
    if(DEFINED imgui_src_SOURCE_DIR)
        add_library(imgui_interface INTERFACE)
        target_include_directories(imgui_interface INTERFACE "${imgui_src_SOURCE_DIR}")
    endif()
endif()

# Optional Vulkan find (do not fail configure if missing; we will enforce later when gpu module lands)
find_package(Vulkan 1.3 QUIET)
if(Vulkan_FOUND)
    message(STATUS "Vulkan found: ${Vulkan_VERSION}")
else()
    message(STATUS "Vulkan SDK not found at configure time (this is OK for Phase 2; will be required later)")
endif()

# Placeholder for future subdirectories
if(EXISTS "${CMAKE_SOURCE_DIR}/src/CMakeLists.txt")
    add_subdirectory(src)
endif()

# Tests placeholder
if(BUILD_TESTS AND EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "Configure complete for ${PROJECT_NAME} ${PROJECT_VERSION}")
