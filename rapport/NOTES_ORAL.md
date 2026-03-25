# Notes pour l'oral - Problèmes rencontrés

## 1. Profiling NVIDIA sous WSL2

### Problème
```
==ERROR== ERR_NVGPUCTRPERM - The user does not have permission to access NVIDIA GPU Performance Counters
```

### Cause
- WSL2 ne permet pas l'accès aux compteurs de performance GPU par défaut
- `ncu` (Nsight Compute) nécessite des permissions administrateur ou une configuration spéciale
- `nsys` génère des fichiers `.qdstrm` au lieu de `.nsys-rep` (importer manquant)

### Solution adoptée
- Implémentation de timers internes avec `std::chrono` et `cudaEvent`
- Analyse empirique des performances en variant les paramètres

---

## 2. nvprof obsolète

### Problème
```
Warning: nvprof is not supported on devices with compute capability 8.0 and higher
```

### Cause
- RTX 4070 = Ada Lovelace (compute capability 8.9)
- nvprof abandonné par NVIDIA depuis les architectures Ampere/Ada

### Solution
- Utiliser `nsys`/`ncu` (qui ont leurs propres problèmes sous WSL2)
- Ou timers internes (solution adoptée)

---

## 3. OpenMP ignoré par NVCC

### Problème
```
warning: ignoring '#pragma omp parallel' [-Wunknown-pragmas]
```

### Cause
- NVCC compile le code CUDA mais ne comprend pas les pragmas OpenMP
- Les pragmas sont passés au compilateur hôte mais sans les flags appropriés

### Solution
```cmake
target_compile_options(raytracer PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
```
- `-Xcompiler` passe le flag `-fopenmp` au compilateur hôte (gcc)

---

## 4. Conflit namespace avec <random>

### Problème
```
error: namespace "rt::std" has no member "numeric_limits"
```

### Cause
- `#include <random>` était fait DANS le namespace `rt`
- `std::mt19937` cherchait `rt::std::numeric_limits` au lieu de `::std::numeric_limits`

### Solution
- Inclure `<random>` AVANT d'ouvrir le namespace `rt`
- Ou fermer le namespace, inclure, puis rouvrir

---

## 5. Forward declaration insuffisante

### Problème
```
error: call of an object of a class type without appropriate operator()
```

### Cause
- `class CPURandom;` (forward declaration) ne suffit pas pour appeler `rng()`
- Le compilateur a besoin de la définition complète de la classe

### Solution
- Inclure le header complet `random.cuh` au lieu d'une forward declaration
- Réorganiser les headers pour éviter les dépendances circulaires

---

## 6. perf non disponible

### Problème
```
perf not found
```

### Cause
- `linux-tools-generic` non installé sous WSL2
- Nécessite le package correspondant au kernel WSL

### Solution
- Utiliser les timers internes pour le CPU
- Alternative: `gprof` avec `-pg` ou `valgrind --tool=callgrind`

---

## Résumé des solutions

| Problème | Solution |
|----------|----------|
| Profiling GPU WSL2 | Timers internes (`cudaEvent`) |
| nvprof obsolète | Timers internes |
| OpenMP + NVCC | `-Xcompiler=-fopenmp` |
| namespace <random> | Include hors namespace |
| Forward declaration | Include complet |
| perf manquant | Timers `std::chrono` |
