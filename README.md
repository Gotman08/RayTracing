# CUDA Ray Tracer for HPC

A high-performance GPU-accelerated path tracer implemented in CUDA, optimized for HPC clusters with NVIDIA GPUs (Romeo2025 GH200/H100).

## Features

- **Path Tracing**: Physically-based rendering with global illumination
- **Materials**: Lambertian, Metal, Dielectric (glass), Emissive, Isotropic
- **Textures**: Solid color, Checker patterns, Perlin noise, Image textures
- **Primitives**: Spheres, Planes, Quads, Triangles, Boxes
- **Acceleration**: BVH with iterative GPU traversal
- **Camera**: Configurable FOV, depth of field, motion blur
- **Post-processing**: ACES/Reinhard tone mapping, gamma correction
- **Output**: PNG, JPG, PPM formats

## Requirements

- CUDA Toolkit 11.0+ (tested with 12.6)
- CMake 3.18+
- C++17 compatible compiler (GCC 9+)
- NVIDIA GPU with compute capability 7.0+

## Quick Start

### Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc)
```

### Run

```bash
# Default scene
./raytracer -o output.png

# Custom scene
./raytracer -w 1920 -h 1080 -s 100 --scene scenes/cornell_box.json -o render.png

# Show GPU info
./raytracer --info
```

## Usage

```
./raytracer [options]

Options:
  -w, --width <int>      Image width (default: 800)
  -h, --height <int>     Image height (default: 600)
  -s, --samples <int>    Samples per pixel (default: 100)
  -d, --depth <int>      Max ray bounces (default: 50)
  -i, --scene <file>     Scene file (JSON format)
  -o, --output <file>    Output file (default: output.png)
  --hdri <file>          HDR environment map
  --exposure <float>     Exposure adjustment (default: 0.0)
  --info                 Display GPU information
  --quiet                Suppress progress output
  --help                 Show help
```

## Running on Romeo2025

### Submit a job

```bash
# Edit account in script first
nano scripts/slurm/run_raytracer.slurm

# Submit
sbatch scripts/slurm/run_raytracer.slurm
```

### Interactive session

```bash
salloc -p gpu --account=your_account --gres=gpu:1 --time=01:00:00
```

### Benchmark

```bash
sbatch scripts/slurm/run_benchmark.slurm
```

## Scene Format (JSON)

```json
{
    "camera": {
        "lookfrom": [13, 2, 3],
        "lookat": [0, 0, 0],
        "vup": [0, 1, 0],
        "fov": 20,
        "aperture": 0.1,
        "focus_dist": 10
    },
    "background": "sky",
    "objects": [
        {
            "type": "sphere",
            "center": [0, 1, 0],
            "radius": 1.0,
            "material": {
                "type": "dielectric",
                "ior": 1.5
            }
        }
    ]
}
```

### Object Types

- `sphere`: center, radius
- `quad`: Q (corner), u, v (edge vectors)
- `box`: min, max (corners)

### Material Types

- `lambertian`: color
- `metal`: color, fuzz
- `dielectric`/`glass`: ior
- `emissive`/`light`: color, strength
- `checker`: color1, color2, scale

## Project Structure

```
RayTracing/
├── include/raytracer/     # Headers (.cuh)
│   ├── core/              # Vec3, Ray, AABB, utilities
│   ├── geometry/          # Primitives
│   ├── materials/         # Shading models
│   ├── textures/          # Texture types
│   ├── acceleration/      # BVH
│   ├── camera/            # Camera
│   ├── lighting/          # PDF, importance sampling
│   ├── environment/       # Sky, HDRI
│   └── rendering/         # Renderer, integrator
├── src/                   # Source files
├── external/              # Dependencies (stb, json)
├── scenes/                # Example scenes
├── scripts/slurm/         # HPC job scripts
└── output/                # Render output
```

## Performance

Typical performance on Romeo2025 (H100):

| Scene       | Resolution | SPP | Time   | MRays/s |
|-------------|------------|-----|--------|---------|
| Spheres     | 1920x1080  | 100 | ~2s    | ~100    |
| Cornell Box | 1920x1080  | 100 | ~3s    | ~70     |
| Showcase    | 3840x2160  | 500 | ~30s   | ~140    |

## License

MIT License

## References

- [Ray Tracing in One Weekend](https://raytracing.github.io/)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/)
- [Physically Based Rendering](https://www.pbr-book.org/)
