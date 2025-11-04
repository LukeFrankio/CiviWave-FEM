## FormatAndTidy.cmake
# Adds `format` and `tidy` custom targets when clang-format/clang-tidy are available.

# Usage: include(FormatAndTidy OPTIONAL)

find_program(CLANG_FORMAT_EXE NAMES clang-format clang-format.exe)
find_program(CLANG_TIDY_EXE NAMES clang-tidy clang-tidy.exe)

if(CLANG_FORMAT_EXE)
    # Collect source files to format
    file(GLOB_RECURSE CW_FORMAT_SOURCES
        "${CMAKE_SOURCE_DIR}/*.c"
        "${CMAKE_SOURCE_DIR}/*.cc"
        "${CMAKE_SOURCE_DIR}/*.cpp"
        "${CMAKE_SOURCE_DIR}/*.cxx"
        "${CMAKE_SOURCE_DIR}/*.h"
        "${CMAKE_SOURCE_DIR}/*.hpp"
        "${CMAKE_SOURCE_DIR}/*.hh"
    )

    # provide a safe 'format' target that formats in-place using the repository .clang-format
    add_custom_target(cw-format
        COMMAND ${CLANG_FORMAT_EXE} -style=file -i ${CW_FORMAT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-format on source files"
    )
endif()

if(CLANG_TIDY_EXE)
    # Tidy requires compile commands; assume build tree exports them
    if(NOT EXISTS "${CMAKE_BINARY_DIR}/compile_commands.json")
        message(STATUS "clang-tidy available but compile_commands.json not found; enable CMAKE_EXPORT_COMPILE_COMMANDS in your build to use tidy target.")
    endif()

    # Collect C/C++ source files to run tidy on (same patterns as formatting)
    file(GLOB_RECURSE CW_TIDY_SOURCES
        "${CMAKE_SOURCE_DIR}/*.c"
        "${CMAKE_SOURCE_DIR}/*.cc"
        "${CMAKE_SOURCE_DIR}/*.cpp"
        "${CMAKE_SOURCE_DIR}/*.cxx"
    )

    # Create a tidy target that runs clang-tidy over the source files with the build tree as -p
    add_custom_target(cw-tidy
        COMMAND ${CMAKE_COMMAND} -E echo "Running clang-tidy..."
        COMMAND ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR} ${CW_TIDY_SOURCES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Running clang-tidy on C/C++ sources"
    )
endif()
