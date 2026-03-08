# CUDA Ray Tracer

A high-performance GPU-accelerated path tracer implemented in CUDA, with CPU fallback using OpenMP.

## Features

- **Path Tracing**: Physically-based rendering with global illumination
- **Dual Rendering**: GPU (CUDA) and CPU (OpenMP) rendering modes
- **Materials**: Lambertian (diffuse), Metal (reflective), Dielectric (glass)
- **Primitives**: Spheres, Planes
- **Acceleration**: BVH with iterative GPU/CPU traversal
- **Camera**: Configurable FOV, depth of field
- **Post-processing**: Reinhard tone mapping, gamma correction
- **Output**: PNG, JPG, BMP, TGA, PPM formats
- **Interactive Mode**: Real-time rendering with WASD + mouse controls (optional)
- **Profiling**: Built-in timing with `--profile` flag

## Requirements

### GPU Build (default)
- CUDA Toolkit 11.0+ (tested with 12.6)
- CMake 3.18+
- C++17 compatible compiler (GCC 9+)
- NVIDIA GPU with compute capability 7.0+
- OpenMP (optional, for `--cpu` mode)

### Interactive Mode (optional)
- GLFW 3.3+
- OpenGL 3.3+
- GLEW

### CPU-Only Build
- CMake 3.18+
- C++17 compatible compiler
- OpenMP

## Quick Start

### Build (GPU)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc)
```

### Build (CPU-Only)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_CPU_ONLY=ON
make -j$(nproc)
```

### Run

```bash
# GPU rendering (default)
./raytracer -o output.png

# CPU rendering
./raytracer --cpu -o output.png

# With profiling
./raytracer --profile -o output.png

# Show GPU info
./raytracer --info
```

## Usage

```
./raytracer [options]

Rendering options:
  -w, --width <int>      Image width (default: 800)
  -h, --height <int>     Image height (default: 600)
  -s, --samples <int>    Samples per pixel (default: 100)
  -d, --depth <int>      Max ray bounces (default: 50)
  -o, --output <file>    Output file (default: output.png)
  --cpu                  Use CPU rendering (OpenMP)
  --profile              Show detailed timing
  --info                 Display GPU information
  --quiet                Suppress progress output
  --help                 Show help

Interactive mode (requires ENABLE_INTERACTIVE):
  --interactive          Enable interactive mode (WASD + mouse)
  --ispp <int>           Samples per frame during movement (default: 1)
  --max-spp <int>        Max accumulated samples when still (default: 1000)
  --sensitivity <float>  Mouse sensitivity (default: 0.002)
  --speed <float>        Movement speed (default: 5.0)
```

## Build Options

| Option | Description | Default |
|--------|-------------|---------|
| `BUILD_CPU_ONLY` | Build CPU-only version (no CUDA required) | OFF |
| `BUILD_TESTS` | Build unit tests | OFF |
| `ENABLE_INTERACTIVE` | Enable interactive mode (GLFW/OpenGL) | ON |
| `ENABLE_FAST_MATH` | Enable fast math optimizations | ON |
| `CMAKE_CUDA_ARCHITECTURES` | Target GPU architecture (70, 80, 90) | 90 |

## Interactive Mode Controls

| Key | Action |
|-----|--------|
| WASD | Move camera |
| Mouse | Look around |
| Space / Q | Move up |
| Ctrl / E | Move down |
| Shift | Fast movement |
| R | Reset accumulation |
| P | Take screenshot |
| Esc | Quit |

## Project Structure

```
RayTracing/
├── include/raytracer/     # Headers (.cuh)
│   ├── core/              # Vec3, Ray, AABB, Interval, Timer
│   ├── geometry/          # Sphere, Plane, Hittable
│   ├── materials/         # Lambertian, Metal, Dielectric
│   ├── textures/          # Solid color
│   ├── acceleration/      # BVH, BVH Builder
│   ├── camera/            # Camera
│   ├── environment/       # Sky
│   ├── rendering/         # Renderer, CPU Renderer, Tone mapping
│   └── interactive/       # Window, GL Interop, Input (optional)
├── src/                   # Source files
│   ├── main.cu            # GPU entry point
│   ├── main_cpu.cpp       # CPU-only entry point
│   ├── cli/               # Argument parsing
│   ├── io/                # Image writer
│   └── scene/             # Default scene setup
├── external/              # Dependencies (stb_image)
├── tests/                 # Unit tests
└── output/                # Render output
```

## Performance

Typical performance (RTX 4060, 800x600, 100 SPP):

| Mode | Time | Performance |
|------|------|-------------|
| GPU (CUDA) | ~0.3s | ~160 MRays/s |
| CPU (OpenMP, 8 threads) | ~17s | ~2.8 MRays/s |

GPU is approximately 50x faster than CPU for path tracing.

## License

MIT License

## References

- [Ray Tracing in One Weekend](https://raytracing.github.io/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Physically Based Rendering](https://www.pbr-book.org/)
