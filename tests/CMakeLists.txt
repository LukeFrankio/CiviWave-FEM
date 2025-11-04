cmake_minimum_required(VERSION 3.14)

# Small unit tests using GoogleTest
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG release-1.12.1
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(googletest)

add_executable(test_sample test_sample.cpp)
target_link_libraries(test_sample PRIVATE GTest::gtest_main)
add_test(NAME sample_test COMMAND test_sample)
