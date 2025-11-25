#!/bin/bash
# capture_rgp.sh - RGP capture automation for CiviWave-FEM on Linux uwu
#
# SYNOPSIS
#   ./capture_rgp.sh [OPTIONS]
#
# DESCRIPTION
#   Automates AMD Radeon GPU Profiler capture for CiviWave-FEM workloads.
#   Requires RGP installed and the application built with ENABLE_RGP_MARKERS=ON.
#
# OPTIONS
#   -e, --executable PATH   Path to executable (default: build/bin/cwf_viewer_demo)
#   -o, --output DIR        Output directory (default: ./rgp_captures)
#   -w, --warmup N          Warmup frames (default: 60)
#   -c, --capture N         Capture frames (default: 10)
#   -s, --scenario FILE     YAML scenario file
#   -h, --help              Show this help message
#
# EXAMPLES
#   ./capture_rgp.sh --warmup 120 --capture 20
#   ./capture_rgp.sh -s scenarios/cantilever.yaml
#
# AUTHOR
#   LukeFrankio (2025-11-25)

set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration defaults
# -----------------------------------------------------------------------------
EXECUTABLE="build/bin/cwf_viewer_demo"
OUTPUT_DIR="rgp_captures"
WARMUP_FRAMES=60
CAPTURE_FRAMES=10
SCENARIO_FILE=""

# -----------------------------------------------------------------------------
# Color output helpers
# -----------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
GRAY='\033[0;90m'
NC='\033[0m' # No Color

log_status() {
    echo -e "${GRAY}[cwf::rgp]${NC} ${CYAN}$1${NC}"
}

log_success() {
    echo -e "${GRAY}[cwf::rgp]${NC} ${GREEN}$1${NC}"
}

log_warning() {
    echo -e "${GRAY}[cwf::rgp]${NC} ${YELLOW}$1${NC}"
}

log_error() {
    echo -e "${GRAY}[cwf::rgp]${NC} ${RED}$1${NC}"
}

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
show_help() {
    sed -n '/^# SYNOPSIS/,/^# AUTHOR/p' "$0" | sed 's/^# //' | head -n -1
    exit 0
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--executable)
            EXECUTABLE="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP_FRAMES="$2"
            shift 2
            ;;
        -c|--capture)
            CAPTURE_FRAMES="$2"
            shift 2
            ;;
        -s|--scenario)
            SCENARIO_FILE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            log_error "Unknown option: $1"
            show_help
            ;;
    esac
done

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
find_rgp() {
    # Check RGP_PATH environment variable
    if [[ -n "${RGP_PATH:-}" ]] && [[ -x "$RGP_PATH" ]]; then
        echo "$RGP_PATH"
        return 0
    fi

    # Common Linux installation paths
    local paths=(
        "/opt/rocm/bin/rocprofiler"
        "/opt/amd/rgp/RadeonGPUProfiler"
        "$HOME/RadeonGPUProfiler/RadeonGPUProfiler"
        "/usr/local/bin/RadeonGPUProfiler"
    )

    for path in "${paths[@]}"; do
        if [[ -x "$path" ]]; then
            echo "$path"
            return 0
        fi
    done

    # Check PATH
    if command -v RadeonGPUProfiler &> /dev/null; then
        command -v RadeonGPUProfiler
        return 0
    fi

    return 1
}

get_timestamp() {
    date +"%Y%m%d_%H%M%S"
}

# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
echo -e "${MAGENTA}CiviWave-FEM RGP Capture Script${NC}"
echo -e "${MAGENTA}================================${NC}"

# Validate executable
if [[ ! -x "$EXECUTABLE" ]]; then
    log_error "Executable not found or not executable: $EXECUTABLE"
    log_warning "Build the project first with BUILD_UI=ON"
    exit 1
fi
log_success "Executable: $EXECUTABLE"

# Find RGP installation
RGP_PATH=$(find_rgp) || {
    log_error "AMD Radeon GPU Profiler not found"
    log_warning "Install RGP from: https://gpuopen.com/rgp/"
    log_warning "Or set RGP_PATH environment variable"
    exit 1
}
log_success "RGP found: $RGP_PATH"

# Create output directory
mkdir -p "$OUTPUT_DIR"
log_success "Output directory: $OUTPUT_DIR"

# Generate capture filename
TIMESTAMP=$(get_timestamp)
CAPTURE_FILE="${OUTPUT_DIR}/cwf_capture_${TIMESTAMP}.rgp"
log_success "Capture output: $CAPTURE_FILE"

# Build command-line arguments
APP_ARGS=()
if [[ -n "$SCENARIO_FILE" ]] && [[ -f "$SCENARIO_FILE" ]]; then
    APP_ARGS+=("--scenario" "$SCENARIO_FILE")
    log_success "Scenario: $SCENARIO_FILE"
fi

# Configure RGP environment
export AMD_RGP_CAPTURE_ON_PRESENT=1
export AMD_RGP_WARMUP_FRAMES=$WARMUP_FRAMES
export AMD_RGP_CAPTURE_FRAMES=$CAPTURE_FRAMES
export AMD_RGP_OUTPUT_FILE=$CAPTURE_FILE

log_status "Configuration:"
echo -e "${GRAY}  Warmup frames: $WARMUP_FRAMES${NC}"
echo -e "${GRAY}  Capture frames: $CAPTURE_FRAMES${NC}"
echo ""

# Launch application
log_status "Launching application with RGP injection..."
log_warning "Press Ctrl+Shift+C during execution to trigger manual capture"
echo ""

if [[ ${#APP_ARGS[@]} -gt 0 ]]; then
    "$EXECUTABLE" "${APP_ARGS[@]}"
else
    "$EXECUTABLE"
fi

EXIT_CODE=$?
if [[ $EXIT_CODE -eq 0 ]]; then
    log_success "Application exited cleanly"
else
    log_warning "Application exited with code: $EXIT_CODE"
fi

# Check capture results
if [[ -f "$CAPTURE_FILE" ]]; then
    SIZE_MB=$(du -m "$CAPTURE_FILE" | cut -f1)
    log_success "SUCCESS: Capture saved (${SIZE_MB} MB)"
    echo -e "${GRAY}  Path: $CAPTURE_FILE${NC}"
    echo ""
    log_status "Open in RGP:"
    echo -e "  $RGP_PATH $CAPTURE_FILE"
else
    log_warning "No capture file found at expected location"
    log_warning "Capture may have failed or been saved elsewhere"

    # Check for any .rgp files
    RGP_FILES=$(find "$OUTPUT_DIR" -name "*.rgp" -type f 2>/dev/null || true)
    if [[ -n "$RGP_FILES" ]]; then
        log_status "Found RGP files in output directory:"
        echo "$RGP_FILES" | while read -r f; do echo "  $f"; done
    fi
fi

echo ""
echo -e "${MAGENTA}Done! uwu${NC}"
