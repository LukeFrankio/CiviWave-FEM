cmake_minimum_required(VERSION 4.1.2)

include(FetchContent)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.15.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

# Generate test configuration header with paths
file(TO_CMAKE_PATH "${CMAKE_SOURCE_DIR}/tests/data" CWF_TEST_DATA_DIR)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/test_config.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/test_config.hpp
    @ONLY
)

add_executable(cwf_core_tests
    test_sample.cpp
    config_validation_test.cpp
    mesh_loader_test.cpp
    preprocess_test.cpp
    physics_test.cpp
)

target_link_libraries(cwf_core_tests PRIVATE
    cwf_core
    GTest::gtest
    GTest::gmock
    GTest::gtest_main
)

target_include_directories(cwf_core_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/support
    ${CMAKE_CURRENT_BINARY_DIR}  # for generated test_config.hpp
)

include(GoogleTest)
gtest_discover_tests(cwf_core_tests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    PROPERTIES
        TIMEOUT 180
)
