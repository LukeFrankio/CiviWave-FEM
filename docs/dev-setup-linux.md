# Linux developer setup (GCC, CMake, Vulkan SDK)

Guidance to build and run CiviWave-FEM on Linux using GCC 15.x and CMake 4.1.2 with the Vulkan SDK installed. Prefer the latest stable or beta versions; pins below are minimum known-good.

## Toolchain targets (as of 2025-11-03)

- GCC: 15.x (C++26 via `-std=c++2c`)
- CMake: 4.1.2
- Vulkan SDK: 1.4.x (headers, validation layers, tools)
- Doxygen: 1.15+
- Python: 3.11+
- Git + Git LFS

See `docs/versions.yaml` for current pins.

## 1. Install GCC 15.x and CMake 4.1.2

Use your distribution's packages or official repositories. Examples below for common distros—adapt as needed.

### Ubuntu (24.04+)

```bash
sudo apt update
# GCC 15 toolchain (from official or toolchain PPA if needed)
sudo apt install -y gcc-15 g++-15
# Latest CMake from PyPI (command-line tools)
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade cmake
cmake --version
```

Set compilers (optional, for multi-compiler setups):

```bash
export CC=gcc-15
export CXX=g++-15
```

### Fedora (40+)

```bash
sudo dnf install -y gcc gcc-c++
python3 -m pip install --upgrade pip cmake
cmake --version
```

If GCC 15 is not yet the default on your distro, install from your distro's toolchain repo or build GCC from source (recommended only if required).

## 2. Install Vulkan SDK and validation layers

Install the LunarG Vulkan SDK appropriate for your distro, or use your package manager for headers and validation layers. Ensure `vulkaninfo` works and validation layers are available in Debug.

Verify:

```bash
vulkaninfo | egrep "Vulkan Instance Version|apiVersion|deviceName" | head -n 5
```

## 3. Install Doxygen 1.15+ and Graphviz

Install Doxygen (prefer the latest) and Graphviz for diagrams.

```bash
# Example for Ubuntu
sudo apt install -y doxygen graphviz
# If your distro version is older, consider building Doxygen from source.
doxygen --version
```

## 4. Install Python 3.11+ and Git LFS

```bash
python3 --version
sudo apt install -y git git-lfs  # or your distro equivalent
git lfs install
```

## 5. Quick verification

From the repository root (`CiviWave-FEM`), run:

```bash
pwd
cmake --version
$CXX --version || g++ --version
vulkaninfo | head -n 20
```

## 6. First configure (no build yet)

```bash
mkdir -p build
cmake -S . -B build -G Ninja -DFORCE_FETCH_DEPS=ON -DENABLE_VALIDATION=ON
```

If this is the first configure on the machine, dependencies will be fetched and built on the first full build. That step may take a while.

## Troubleshooting tips

- If `vulkaninfo` is missing, install it from the Vulkan SDK or your distro's `vulkan-tools` package.
- Ensure the validation layers are accessible in Debug; set `VK_LAYER_PATH` if you installed them to a custom location.
- If you have multiple GCC versions, export `CC` and `CXX` to point to GCC 15.x before configuring with CMake.
