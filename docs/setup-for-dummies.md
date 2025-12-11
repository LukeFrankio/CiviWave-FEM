# Setup for Dummies (Windows & Linux)

Zero-to-build, no assumptions beyond a fresh OS install. Target hardware: a GPU with Vulkan 1.3 support (AMD iGPU preferred). If you skip installing `slangc`, the build will still succeed using stub shaders; GPU execution stays disabled until real SPIR-V is compiled.

## What you get right now

- Library + GoogleTest suite (runnable on CPU-only machines).
- Optional ImGui viewer (`cwf_viewer_demo`) when `BUILD_UI=ON` and `slangc` is available.
- No standalone CLI app yet.

## Windows 11 (fresh machine)

1. **Install Git & Git LFS**  
   - Download Git for Windows (latest). Enable Git LFS during setup or run `git lfs install` later.

2. **Install Python 3.11+**  
   - Grab the latest Python installer, check "Add Python to PATH".

3. **Install CMake 4.1.2+**  
   - Either install from cmake.org (x64) **or** run:  
     `python -m pip install --upgrade pip cmake`

4. **Install a GCC 15.x toolchain** (pick one):
   - **MSYS2**: install MSYS2, open the UCRT64 shell, run `pacman -Syu` then `pacman -S mingw-w64-ucrt-x86_64-gcc mingw-w64-ucrt-x86_64-gdb ninja`. Add `C:\msys64\ucrt64\bin` to PATH.
   - **WinLibs**: download the latest WinLibs GCC 15.x archive, extract to `C:\Dev\winlibs-mingw64`, add `C:\Dev\winlibs-mingw64\mingw64\bin` to PATH.

5. **Install Vulkan SDK 1.4.328.1+**  
   - Install from LunarG. Ensure `VULKAN_SDK` is set (installer usually does) and `VULKAN_SDK\Bin` is on PATH. Verify: `vulkaninfo | findstr Version`.

6. **Optional: Slang `slangc`**  
   - Download a prebuilt `slangc` matching 2025.18 and set `SLANGC` to its full path, e.g.:  
     `setx SLANGC "C:\\VulkanSDK\\1.4.328.1\\Bin\\slangc.exe"`

7. **Clone the repo**

   ```powershell
   git clone https://github.com/LukeFrankio/CiviWave-FEM.git
   cd CiviWave-FEM
   git lfs install
   ```

8. **Configure with presets**

   ```powershell
   cmake --preset Debug   # or Release / RelWithDebInfo / Profile
   ```

9. **Build**

   ```powershell
   cmake --build build/Debug -j
   ```

10. **Run tests** (CPU-safe)

    ```powershell
    ctest --preset Debug --output-on-failure
    ```

11. **Run the optional viewer** (requires `SLANGC`, `BUILD_UI=ON`)

    ```powershell
    cmake --preset Debug -DBUILD_UI=ON
    cmake --build build/Debug -j
    ./build/Debug/bin/cwf_viewer_demo.exe
    ```

    CMake copies compiled shaders next to the binary.

## Linux (Ubuntu 24.04+ example)

1. **Install base tools**

   ```bash
   sudo apt update
   sudo apt install -y git git-lfs python3 python3-pip ninja-build
   git lfs install
   ```

2. **Install GCC 15.x** (use distro toolchain or PPA)

   ```bash
   sudo apt install -y gcc-15 g++-15
   export CC=gcc-15
   export CXX=g++-15
   ```

3. **Install CMake 4.1.2+**

   ```bash
   python3 -m pip install --upgrade pip cmake
   ```

4. **Install Vulkan SDK / tools**
   - Install LunarG SDK for your distro **or** install headers/tools:  
     `sudo apt install -y libvulkan-dev vulkan-tools`  
     Verify: `vulkaninfo | head -n 20` (ensure 1.3 support).

5. **Optional: Slang `slangc`**  
   - Place `slangc` on PATH or set `SLANGC=/path/to/slangc` (2025.18 recommended).

6. **Clone the repo**

   ```bash
   git clone https://github.com/LukeFrankio/CiviWave-FEM.git
   cd CiviWave-FEM
   git lfs install
   ```

7. **Configure with presets**

   ```bash
   cmake --preset Debug   # or Release / RelWithDebInfo / Profile
   ```

8. **Build**

   ```bash
   cmake --build build/Debug -j
   ```

9. **Run tests**

   ```bash
   ctest --preset Debug --output-on-failure
   ```

10. **Optional viewer** (requires `SLANGC`, `BUILD_UI=ON`)

    ```bash
    cmake --preset Debug -DBUILD_UI=ON
    cmake --build build/Debug -j
    ./build/Debug/bin/cwf_viewer_demo
    ```

## Troubleshooting quick hits

- `vulkaninfo` missing or failing: re-open the shell after SDK install; ensure PATH and `VULKAN_SDK` are set.
- `slangc` not found: set `SLANGC` to the full executable path; otherwise stub shaders are emitted and GPU execution will be disabled.
- Compiler too old: verify `gcc --version` reports 15.x; set `CC`/`CXX` if multiple GCC versions are installed.
- Tests failing on GPU-specific cases: ensure a Vulkan 1.3-capable device is present, or filter tests with `ctest -R <name>`.

You’re done when `ctest` passes. Add `-DBUILD_UI=ON` plus `SLANGC` for the viewer; otherwise the library + tests are the supported deliverables today.
