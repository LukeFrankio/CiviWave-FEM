/**
 * @file instrumentation_test.cpp
 * @brief Phase 11 tests for GPU profiling and telemetry infrastructure uwu
 *
 * this test suite validates the instrumentation subsystem:
 * - FrameLog serialization to YAML
 * - PassTiming data structures
 * - timestamp resolution math
 * - YAML output format compliance
 *
 * GPU-dependent tests (GpuTimestamps) are marked as integration tests and
 * require a Vulkan device to run. Unit tests for pure functions run headless.
 *
 * @author LukeFrankio
 * @date 2025-11-25
 * @version 1.0
 *
 * @note documented with Doxygen 1.15 beta because testing is praxis ✨
 */

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#include "cwf/gpu/instrumentation.hpp"
#include "cwf/gpu/pcg.hpp"
#include <yaml-cpp/yaml.h>

namespace cwf::gpu::instrumentation
{
namespace
{

/**
 * @brief Test fixture for instrumentation unit tests (no GPU required)
 */
class InstrumentationTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        // Clean up any leftover test files
        test_output_dir = std::filesystem::temp_directory_path() / "cwf_instrumentation_test";
        std::filesystem::create_directories(test_output_dir);
    }

    void TearDown() override
    {
        // Remove test output directory
        std::error_code ec;
        std::filesystem::remove_all(test_output_dir, ec);
    }

    std::filesystem::path test_output_dir;
};

// -----------------------------------------------------------------------------
// FrameLog serialization tests
// -----------------------------------------------------------------------------

TEST_F(InstrumentationTest, FrameLogToYaml_EmptyLog_ValidYaml)
{
    FrameLog log{};
    log.frame_index       = 0U;
    log.simulation_time_s = 0.0;
    log.time_step_s       = 0.01;
    log.solver_tolerance  = 1.0e-4;
    log.paused_mode       = false;
    log.wall_clock_ms     = 16.7;

    const std::string yaml = frame_log_to_yaml(log);

    // Should be valid YAML
    ASSERT_NO_THROW({
        YAML::Node node = YAML::Load(yaml);
        EXPECT_EQ(node["frame"].as<std::uint64_t>(), 0U);
        EXPECT_DOUBLE_EQ(node["time_step_s"].as<double>(), 0.01);
        EXPECT_FALSE(node["paused_mode"].as<bool>());
    });
}

TEST_F(InstrumentationTest, FrameLogToYaml_WithPassTimings_ContainsTimingsMap)
{
    FrameLog log{};
    log.frame_index       = 42U;
    log.simulation_time_s = 0.42;
    log.time_step_s       = 0.01;
    log.solver_tolerance  = 2.0e-4;
    log.paused_mode       = false;
    log.wall_clock_ms     = 33.3;

    log.pass_timings = {
        PassTiming{.name = "predictor", .duration_ms = 1.5, .start_tick = 0U, .end_tick = 1000U},
        PassTiming{.name = "pcg_solve", .duration_ms = 25.0, .start_tick = 1000U, .end_tick = 26000U},
        PassTiming{.name = "update", .duration_ms = 2.0, .start_tick = 26000U, .end_tick = 28000U},
    };

    const std::string yaml = frame_log_to_yaml(log);
    YAML::Node        node = YAML::Load(yaml);

    ASSERT_TRUE(node["timings"].IsDefined());
    ASSERT_TRUE(node["timings"].IsMap());
    EXPECT_NEAR(node["timings"]["predictor"].as<double>(), 1.5, 0.01);
    EXPECT_NEAR(node["timings"]["pcg_solve"].as<double>(), 25.0, 0.01);
    EXPECT_NEAR(node["timings"]["update"].as<double>(), 2.0, 0.01);
}

TEST_F(InstrumentationTest, FrameLogToYaml_WithPcgTelemetry_ContainsPcgSection)
{
    FrameLog log{};
    log.frame_index       = 100U;
    log.simulation_time_s = 1.0;
    log.time_step_s       = 0.01;
    log.solver_tolerance  = 3.0e-4;
    log.paused_mode       = true;
    log.wall_clock_ms     = 50.0;

    log.pcg_iterations    = 45U;
    log.pcg_residual_norm = 2.5e-5;
    log.pcg_rhs_norm      = 1.0e3;
    log.pcg_converged     = true;

    const std::string yaml = frame_log_to_yaml(log);
    YAML::Node        node = YAML::Load(yaml);

    ASSERT_TRUE(node["pcg"].IsDefined());
    ASSERT_TRUE(node["pcg"].IsMap());
    EXPECT_EQ(node["pcg"]["iterations"].as<std::size_t>(), 45U);
    EXPECT_NEAR(node["pcg"]["residual_norm"].as<double>(), 2.5e-5, 1.0e-10);
    EXPECT_TRUE(node["pcg"]["converged"].as<bool>());
}

TEST_F(InstrumentationTest, FrameLogToYaml_WithAdaptiveFlags_ContainsAdaptiveSection)
{
    FrameLog log{};
    log.frame_index       = 200U;
    log.simulation_time_s = 2.0;
    log.time_step_s       = 0.02;
    log.solver_tolerance  = 2.0e-4;
    log.paused_mode       = false;
    log.wall_clock_ms     = 40.0;

    log.dt_increased   = true;
    log.dt_decreased   = false;
    log.dt_clamped_min = false;
    log.dt_clamped_max = false;

    const std::string yaml = frame_log_to_yaml(log);
    YAML::Node        node = YAML::Load(yaml);

    ASSERT_TRUE(node["adaptive"].IsDefined());
    EXPECT_TRUE(node["adaptive"]["dt_increased"].as<bool>());
    EXPECT_FALSE(node["adaptive"]["dt_decreased"].as<bool>());
    EXPECT_FALSE(node["adaptive"]["dt_clamped_min"].as<bool>());
    EXPECT_FALSE(node["adaptive"]["dt_clamped_max"].as<bool>());
}

// -----------------------------------------------------------------------------
// write_frame_log tests
// -----------------------------------------------------------------------------

TEST_F(InstrumentationTest, WriteFrameLog_ValidPath_CreatesFile)
{
    FrameLog log{};
    log.frame_index       = 123U;
    log.simulation_time_s = 1.23;
    log.time_step_s       = 0.01;
    log.solver_tolerance  = 1.0e-4;
    log.wall_clock_ms     = 16.0;

    const auto path   = test_output_dir / "test_frame.yaml";
    auto       result = write_frame_log(log, path);

    ASSERT_TRUE(result.has_value()) << "write_frame_log failed: " << result.error().message;
    EXPECT_TRUE(std::filesystem::exists(path));

    // Verify file contents are valid YAML
    std::ifstream const file{path};
    std::stringstream buffer;
    buffer << file.rdbuf();

    YAML::Node node;
    ASSERT_NO_THROW(node = YAML::Load(buffer.str()));
    EXPECT_EQ(node["frame"].as<std::uint64_t>(), 123U);
}

TEST_F(InstrumentationTest, WriteFrameLog_InvalidPath_ReturnsError)
{
    FrameLog log{};
    log.frame_index = 0U;

    // Use a path that cannot be created (invalid characters or restricted)
    const std::filesystem::path invalid_path = "/nonexistent/deeply/nested/invalid/path/frame.yaml";
    auto                        result       = write_frame_log(log, invalid_path);

    // Should return error for invalid path
    EXPECT_FALSE(result.has_value());
}

// -----------------------------------------------------------------------------
// FrameLogger tests
// -----------------------------------------------------------------------------

TEST_F(InstrumentationTest, FrameLogger_BeginEndFrame_RecordsWallClock)
{
    InstrumentationConfig config{};
    config.enable_yaml_logging = false; // disable file output for this test

    auto logger = FrameLogger::create(config);
    logger.begin_frame(0U, 0.0, 0.01, 1.0e-4, false);

    // Simulate some work
    volatile int dummy = 0;
    for (int i = 0; i < 100000; ++i)
    {
        dummy += i;
    }
    (void) dummy;

    auto log = logger.end_frame();

    // Wall clock should be non-zero (some time elapsed)
    EXPECT_GT(log.wall_clock_ms, 0.0);
}

TEST_F(InstrumentationTest, FrameLogger_RecordPass_AccumulatesTimings)
{
    InstrumentationConfig config{};
    config.enable_yaml_logging = false;

    auto logger = FrameLogger::create(config);
    logger.begin_frame(10U, 0.1, 0.01, 1.0e-4, false);

    logger.record_pass(PassTiming{.name = "pass_a", .duration_ms = 5.0});
    logger.record_pass(PassTiming{.name = "pass_b", .duration_ms = 10.0});

    auto log = logger.end_frame();

    ASSERT_EQ(log.pass_timings.size(), 2U);
    EXPECT_EQ(log.pass_timings[0].name, "pass_a");
    EXPECT_DOUBLE_EQ(log.pass_timings[0].duration_ms, 5.0);
    EXPECT_EQ(log.pass_timings[1].name, "pass_b");
    EXPECT_DOUBLE_EQ(log.pass_timings[1].duration_ms, 10.0);
}

TEST_F(InstrumentationTest, FrameLogger_RecordPcg_StoresTelemetry)
{
    InstrumentationConfig config{};
    config.enable_yaml_logging = false;

    auto logger = FrameLogger::create(config);
    logger.begin_frame(20U, 0.2, 0.01, 2.0e-4, true);

    pcg::PcgTelemetry pcg_telemetry{};
    pcg_telemetry.iterations    = 50U;
    pcg_telemetry.residual_norm = 1.5e-5;
    pcg_telemetry.rhs_norm      = 500.0;
    pcg_telemetry.converged     = true;

    logger.record_pcg(pcg_telemetry);

    auto log = logger.end_frame();

    EXPECT_EQ(log.pcg_iterations, 50U);
    EXPECT_DOUBLE_EQ(log.pcg_residual_norm, 1.5e-5);
    EXPECT_DOUBLE_EQ(log.pcg_rhs_norm, 500.0);
    EXPECT_TRUE(log.pcg_converged);
}

TEST_F(InstrumentationTest, FrameLogger_RecordAdaptive_StoresFlags)
{
    InstrumentationConfig config{};
    config.enable_yaml_logging = false;

    auto logger = FrameLogger::create(config);
    logger.begin_frame(30U, 0.3, 0.01, 1.0e-4, false);

    logger.record_adaptive(false, true, false, true);

    auto log = logger.end_frame();

    EXPECT_FALSE(log.dt_increased);
    EXPECT_TRUE(log.dt_decreased);
    EXPECT_FALSE(log.dt_clamped_min);
    EXPECT_TRUE(log.dt_clamped_max);
}

TEST_F(InstrumentationTest, FrameLogger_WithYamlOutput_WritesFile)
{
    InstrumentationConfig config{};
    config.enable_yaml_logging = true;
    config.log_directory       = test_output_dir;
    config.log_prefix          = "test_";
    config.log_stride          = 1U;

    auto logger = FrameLogger::create(config);
    logger.begin_frame(0U, 0.0, 0.01, 1.0e-4, false);
    logger.record_pass(PassTiming{.name = "test_pass", .duration_ms = 1.0});

    auto log = logger.end_frame();

    const auto expected_path = test_output_dir / "test_000000.yaml";
    EXPECT_TRUE(std::filesystem::exists(expected_path)) << "Expected: " << expected_path;
}

TEST_F(InstrumentationTest, FrameLogger_LogStride_SkipsFrames)
{
    InstrumentationConfig config{};
    config.enable_yaml_logging = true;
    config.log_directory       = test_output_dir;
    config.log_prefix          = "stride_";
    config.log_stride          = 5U; // Only log every 5th frame

    auto logger = FrameLogger::create(config);

    // Frame 0 - should be logged (0 % 5 == 0)
    logger.begin_frame(0U, 0.0, 0.01, 1.0e-4, false);
    static_cast<void>(logger.end_frame());

    // Frame 1 - should NOT be logged
    logger.begin_frame(1U, 0.01, 0.01, 1.0e-4, false);
    static_cast<void>(logger.end_frame());

    // Frame 5 - should be logged (5 % 5 == 0)
    logger.begin_frame(5U, 0.05, 0.01, 1.0e-4, false);
    static_cast<void>(logger.end_frame());

    EXPECT_TRUE(std::filesystem::exists(test_output_dir / "stride_000000.yaml"));
    EXPECT_FALSE(std::filesystem::exists(test_output_dir / "stride_000001.yaml"));
    EXPECT_TRUE(std::filesystem::exists(test_output_dir / "stride_000005.yaml"));
}

// -----------------------------------------------------------------------------
// PassTiming structure tests
// -----------------------------------------------------------------------------

TEST_F(InstrumentationTest, PassTiming_DefaultConstruction_ZeroValues)
{
    PassTiming const timing{};
    EXPECT_TRUE(timing.name.empty());
    EXPECT_DOUBLE_EQ(timing.duration_ms, 0.0);
    EXPECT_EQ(timing.start_tick, 0U);
    EXPECT_EQ(timing.end_tick, 0U);
}

TEST_F(InstrumentationTest, PassTiming_DesignatedInit_SetsAllFields)
{
    PassTiming const timing{
        .name        = "test_pass",
        .duration_ms = 42.5,
        .start_tick  = 1000U,
        .end_tick    = 2000U,
    };

    EXPECT_EQ(timing.name, "test_pass");
    EXPECT_DOUBLE_EQ(timing.duration_ms, 42.5);
    EXPECT_EQ(timing.start_tick, 1000U);
    EXPECT_EQ(timing.end_tick, 2000U);
}

// -----------------------------------------------------------------------------
// InstrumentationConfig tests
// -----------------------------------------------------------------------------

TEST_F(InstrumentationTest, InstrumentationConfig_Defaults_ReasonableValues)
{
    InstrumentationConfig const config{};

    EXPECT_TRUE(config.enable_gpu_timestamps);
    EXPECT_TRUE(config.enable_yaml_logging);
    EXPECT_TRUE(config.enable_rgp_markers);
    EXPECT_FALSE(config.enable_tracy); // Tracy off by default
    EXPECT_EQ(config.max_passes, 32U);
    EXPECT_TRUE(config.log_directory.empty());
    EXPECT_EQ(config.log_stride, 1U);
}

} // namespace
} // namespace cwf::gpu::instrumentation
