# FetchOrSystem.cmake
# Prefer system packages when they meet version and ABI requirements unless FORCE_FETCH_DEPS=ON.
# Provides helper functions to wrap find_package and fall back to FetchContent.

include(FetchContent)

# Example wrapper for yaml-cpp
function(cw_fetch_or_system_yaml_cpp MIN_VERSION)
    if(NOT FORCE_FETCH_DEPS)
        find_package(yaml-cpp ${MIN_VERSION} QUIET CONFIG)
    endif()

    if(NOT TARGET yaml-cpp)
        message(STATUS "Fetching yaml-cpp ${MIN_VERSION} via FetchContent")
        # Some upstream CMakeLists in older projects use legacy cmake_minimum_required
        # semantics that newer CMake versions may reject. Set a compatible policy
        # version or hint for the subproject configuration.
        # The yaml-cpp project may still call cmake_minimum_required with an older
        # version; provide a cache variable to relax policy enforcement during the
        # subproject configure step.
        set(CMAKE_POLICY_VERSION_MINIMUM 3.5 CACHE STRING "Minimum policy version for external projects" FORCE)
        FetchContent_Declare(
            yaml_cpp_src
            GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
            GIT_TAG 0.8.0
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(yaml_cpp_src)
        # Ensure position independent code for linking into shared libs if needed
        if(TARGET yaml-cpp)
            set_target_properties(yaml-cpp PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
    endif()
endfunction()

# Generic helper to populate a header-only repo and expose an INTERFACE target
function(cw_header_only_import NAME GIT_URL GIT_TAG INCLUDE_SUBDIR)
    FetchContent_Declare(
        ${NAME}
        GIT_REPOSITORY ${GIT_URL}
        GIT_TAG ${GIT_TAG}
        GIT_SHALLOW TRUE
    )
    FetchContent_GetProperties(${NAME})
    if(NOT ${NAME}_POPULATED)
        FetchContent_Populate(${NAME})
    endif()
    add_library(${NAME}_interface INTERFACE)
    target_include_directories(${NAME}_interface INTERFACE "${${NAME}_SOURCE_DIR}/${INCLUDE_SUBDIR}")
endfunction()
