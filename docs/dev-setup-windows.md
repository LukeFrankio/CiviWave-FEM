# Windows 11 developer setup (GCC/MinGW, CMake, Vulkan SDK)

A concise, zero-drama setup to build and run CiviWave-FEM on Windows 11 using GCC 15.x, CMake 4.1.2, and the Vulkan SDK. Versions below reflect the canonical minimums/as-of pins; prefer newer if available.

## Toolchain targets (as of 2025-11-03)

- GCC: 15.2 (C++26 via `-std=c++2c`)
- CMake: 4.1.2
- Vulkan SDK: 1.4.328.1
- Doxygen: 1.15+ (beta preferred)
- Python: 3.11+
- Git + Git LFS
- Optional: AMD RGP/RGA (profiling)

See `docs/versions.yaml` for the living source of truth.

## 1. Install GCC (MinGW-w64)

Choose one of the two supported distributions. Both are fine—use whichever you prefer.

- MSYS2 (recommended for package management) — install MSYS2, then install the UCRT MinGW toolchain packages for GCC 15.x.
- WinLibs (standalone zip) — download the latest WinLibs MinGW-w64 GCC 15.x build, extract to a fixed path (e.g., `C:\Dev\winlibs-mingw64`).

Add the MinGW `bin` directory to your PATH (UCRT flavor for MSYS2):

```powershell
# Replace the path with your actual MinGW bin folder
$MinGW = "C:\\Dev\\winlibs-mingw64\\mingw64\\bin"
[Environment]::SetEnvironmentVariable("Path", $Env:Path + ";" + $MinGW, "User")
```

Verify:

```powershell
gcc --version
g++ --version
```

The output should report GCC 15.x.

## 2. Install CMake 4.1.2

Install the official CMake binaries (x64) and add them to PATH. Alternatively, you can install from Python via pip (installs command-line tools):

```powershell
python --version
python -m pip install --upgrade pip
python -m pip install --upgrade cmake
cmake --version
```

Ensure the reported version is 4.1.2 or newer.

## 3. Install Vulkan SDK 1.4.328.1

Install the LunarG Vulkan SDK and enable the Validation Layers during installation. The installer typically sets `VULKAN_SDK` and updates PATH. If not, set `VULKAN_SDK` manually and append its `\Bin` to PATH:

```powershell
# Example path; adjust if your SDK is installed elsewhere
$Vulkan = "C:\\VulkanSDK\\1.4.328.1"
[Environment]::SetEnvironmentVariable("VULKAN_SDK", $Vulkan, "User")
[Environment]::SetEnvironmentVariable("Path", $Env:Path + ";" + (Join-Path $Vulkan "Bin"), "User")
```

Verify:

```powershell
vulkaninfo | Select-String -Pattern "Vulkan Instance Version|deviceName|apiVersion" -SimpleMatch
```

## 4. Install Doxygen 1.15+

Install Doxygen (prefer the latest beta if available) and add it to PATH. Install Graphviz as well for call graphs.

Verify:

```powershell
doxygen --version
```

## 5. Install Python 3.11+

Install Python 3.11 or newer and ensure `python` is on PATH. The project uses Python for auxiliary scripts.

```powershell
python --version
```

## 6. Install Git and Git LFS

Install Git for Windows and Git LFS, then enable LFS for the repository if needed.

```powershell
git --version
git lfs install
```

## 7. Optional: AMD RGP and RGA

Install Radeon GPU Profiler (RGP) and Radeon GPU Analyzer (RGA) for GPU profiling and analysis. These are optional but highly recommended for performance work.

## 8. PATH sanity check

Ensure the following are on your PATH (order matters—prefer your MinGW over conflicting toolchains):

- MinGW-w64 `bin` (GCC 15.x)
- CMake `bin`
- Python `Scripts` and base dir
- Vulkan SDK `Bin`
- Doxygen `bin` (and Graphviz `bin`)

```powershell
$paths = $Env:Path -split ";"
$paths | ForEach-Object { $_ } | Select-String -Pattern "mingw|cmake|Python|Vulkan|doxygen|graphviz" -SimpleMatch
```

## 9. Quick verification

From the repository root (`CiviWave-FEM`), run these checks. They should complete quickly and print versions:

```powershell
Get-Location
cmake --version
"GCC: " + (g++ --version | Select-Object -First 1)
$Env:VULKAN_SDK
vulkaninfo | Select-String -Pattern "apiVersion" -SimpleMatch | Select-Object -First 1
```

## 10. First configure (no build yet)

After cloning the repo and setting up PATHs, do an initial CMake configure. We recommend the Ninja generator; if you do not have Ninja, CMake will fall back to the default generator.

```powershell
# Create build directory and configure
New-Item -ItemType Directory -Force -Path build | Out-Null
cmake -S . -B build -G Ninja -DFORCE_FETCH_DEPS=ON -DENABLE_VALIDATION=ON
```

If this is your first configure on this machine, the project will fetch and build dependencies from source as needed on the first full build. That step may take a while; the command above only configures.

## Troubleshooting tips

- If `vulkaninfo` fails, re-open your terminal to refresh environment variables or verify `VULKAN_SDK`.
- If `gcc` is not found, confirm you added the correct MinGW `bin` to PATH (UCRT for MSYS2).
- When running PowerShell scripts in this repo, prefer ExecutionPolicy Bypass:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\YourScript.ps1
```

Happy compiling—now go make that FEM solver fly 🛫