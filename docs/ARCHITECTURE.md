# Architecture du Ray Tracer CUDA

## Vue d'ensemble

Ce ray tracer utilise une architecture optimisée pour les GPUs NVIDIA, évitant les virtual functions et la récursion pour maximiser les performances.

## Composants Principaux

### 1. Core (`include/raytracer/core/`)

- **vec3.cuh**: Vecteur 3D avec toutes les opérations marquées `__host__ __device__`
- **ray.cuh**: Structure rayon (origine + direction + temps)
- **aabb.cuh**: Axis-Aligned Bounding Box avec intersection rapide
- **interval.cuh**: Intervalle [min, max] pour les tests d'intersection
- **cuda_utils.cuh**: Macros CUDA_CHECK, helpers mémoire

### 2. Géométrie (`include/raytracer/geometry/`)

**Dispatch par enum (évite virtual functions)**:
```cuda
enum class HittableType {
    SPHERE, MOVING_SPHERE, PLANE, QUAD, TRIANGLE, BOX
};
```

Chaque primitive implémente:
- `hit()`: Test d'intersection rayon-objet
- `bounding_box()`: Calcul AABB pour BVH

### 3. Matériaux (`include/raytracer/materials/`)

**Types supportés**:
- Lambertian (diffus)
- Metal (réflectif avec fuzz)
- Dielectric (verre avec Fresnel-Schlick)
- Emissive (lumières)
- Isotropic (volumétrique)

### 4. Accélération (`include/raytracer/acceleration/`)

**BVH (Bounding Volume Hierarchy)**:
- Construction CPU avec tri par axe le plus long
- Traversée GPU itérative avec stack de 64 éléments
- Évite la récursion (limite de stack CUDA)

### 5. Rendu (`include/raytracer/rendering/`)

**Path Tracing Itératif**:
```cuda
for (depth = 0; depth < max_depth; depth++) {
    if (hit) {
        accumulated += attenuation * emission;
        scatter(); // Rebond

        // Russian Roulette après N bounces
        if (depth > 3 && random() > throughput) break;
    } else {
        accumulated += attenuation * background;
        break;
    }
}
```

## Flux de Données

```
Scene JSON → Host Memory → Device Memory
                              ↓
                        CUDA Kernels
                              ↓
                        Frame Buffer
                              ↓
                     Host Memory → Image
```

## Optimisations Clés

1. **Structure of Arrays vs Array of Structures**
   - Union pour types d'objets différents
   - Accès mémoire coalescent

2. **Évitement de divergence**
   - Pas de virtual functions
   - Dispatch par switch/enum

3. **Russian Roulette**
   - Terminaison probabiliste
   - Réduit les calculs pour rayons peu significatifs

4. **Traversée BVH itérative**
   - Stack local de 64 éléments
   - Pas de récursion GPU

## Configuration Kernel

```cuda
dim3 blocks((width + 7) / 8, (height + 7) / 8);
dim3 threads(8, 8);  // 64 threads par bloc
```

Chaque thread traite un pixel avec tous ses samples.
