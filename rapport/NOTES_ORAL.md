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

## 7. Pas de fonctions virtuelles sur GPU

### Problème
- Les `virtual` fonctions nécessitent une vtable (pointeur indirect)
- CUDA ne supporte pas les vtables efficacement sur device
- Performances catastrophiques avec polymorphisme classique

### Solution
- Enum-based dispatch (`MaterialType`, `HittableType`) avec `switch`
- Union pour stocker les données (`HittableData`)
- Pas d'héritage, pas de `new` côté device

---

## 8. Récursion interdite dans les kernels

### Problème
- La traversée BVH classique est récursive
- CUDA a une pile d'appel très limitée par thread (~1KB)
- Récursion profonde = stack overflow silencieux

### Solution
- Traversée itérative avec pile explicite : `int stack[64]`
- Même chose pour `ray_color` : boucle `for(depth)` au lieu de récursion

---

## 9. Gestion mémoire Host↔Device (fixup de pointeurs)

### Problème
- Les objets contiennent des pointeurs vers les matériaux (ex: `sphere.mat`)
- Après `cudaMemcpy` Host→Device, ces pointeurs pointent toujours vers la RAM host
- Accès à un pointeur host depuis un kernel = crash ou corruption

### Solution
- Technique de "pointer fixup" : parcourir tous les objets avant l'upload
- Pour chaque objet, retrouver l'index du matériau dans le tableau host
- Remplacer le pointeur host par `d_materials + index` (pointeur device)
- C'est moche mais nécessaire sans allocateur unifié (managed memory)

---

## 10. curand_init très lent

### Problème
- `curand_init(seed, sequence, offset, state)` est extrêmement coûteux
- Sur une image 1920x1080, l'init RNG prenait plus de temps que le rendu

### Solution
- Wang hash pour pré-mélanger le seed : O(1) au lieu de O(sequence)
- `curand_init(wang_hash(seed + pixel_index), 0, 0, &state)` : sequence=0, offset=0
- Résultat visuellement identique, init ~50x plus rapide

---

# Améliorations à implémenter

## A1. Mémoire `__constant__` pour Camera et RenderConfig

### Pourquoi
- Camera (~120 octets) et RenderConfig (~64 octets) sont passés par valeur à chaque kernel
- Tous les threads lisent exactement les mêmes valeurs → broadcast parfait
- La constant memory a un cache dédié, lecture en ~5 cycles vs ~500 pour la globale

### Comment
- Déclarer `__constant__ Camera d_const_camera;` dans renderer.cuh
- Utiliser `cudaMemcpyToSymbol()` avant chaque lancement de kernel
- Retirer Camera/RenderConfig des paramètres des kernels

### Impact attendu
- Moins de pression sur les registres (pas de copie des params)
- Lecture plus rapide grâce au cache constant

---

## A2. Réduction parallèle pour tone mapping adaptatif

### Pourquoi
- Le tone mapping Reinhard actuel est "local" (par pixel, sans contexte global)
- Un tone mapping adaptatif a besoin de la luminance moyenne de toute l'image
- Calculer une moyenne sur ~2M pixels nécessite une réduction parallèle

### Comment
- Kernel de réduction avec `__shared__ float sdata[256]`
- Arbre binaire intra-bloc avec `__syncthreads()`
- `atomicAdd` pour combiner les résultats des blocs
- Reinhard étendu : `scaled = key * hdr / avg_lum; result = scaled / (1 + scaled)`

### Concepts du cours utilisés
- `__shared__` memory
- `__syncthreads()`
- `atomicAdd()`
- Réduction parallèle (arbre binaire)

---

## A3. CUDA Streams (recouvrement d'opérations)

### Pourquoi
- Actuellement tout est synchrone sur le stream par défaut
- L'upload des matériaux et l'init des RNG sont indépendants → peuvent se chevaucher
- Le D2H du framebuffer et la préparation fichier aussi

### Comment
- Créer 2 streams : `stream_compute` et `stream_transfer`
- `cudaMemcpyAsync` pour les transferts sur stream_transfer
- Kernels sur stream_compute
- Synchroniser avant les dépendances

### Impact attendu
- Faible gain de perf (le kernel de rendu domine), mais démontre le concept

---

## A4. Thrust pour la réduction (comparaison librairie)

### Pourquoi
- Montrer la maîtrise des librairies CUDA au-delà de cuRAND
- Thrust fournit `transform_reduce` qui fait exactement notre réduction

### Comment
- `thrust::transform_reduce(d_ptr, d_ptr + n, LuminanceFunctor(), 0.0f, thrust::plus<float>())`
- Comparer le temps avec notre kernel custom
- Thrust est header-only, inclus dans le CUDA Toolkit

---

## A5. Occupancy et profilage avancé

### Pourquoi
- L'occupancy = ratio warps actifs / warps max par SM
- `cudaOccupancyMaxPotentialBlockSize` donne la taille de bloc optimale
- Permet de justifier le choix de BLOCK_SIZE=16

### Comment
- Appeler l'API d'occupancy pour chaque kernel
- Afficher : blocs actifs/SM, warps actifs, % occupancy
- Estimer la bande passante effective : bytes_rw / temps_kernel

---

## A6. Tests GPU (kernels CUDA)

### Pourquoi
- Les 144 tests actuels sont CPU-only (pas de validation GPU)
- Besoin de vérifier que les kernels CUDA produisent des résultats corrects

### Comment
- Nouvel exécutable `test_cuda_kernels.cu` compilé avec CUDA
- Tests : init RNG, réduction, tone mapping, constant memory, mini-rendu
- CMake conditionnel : seulement si CUDA est disponible

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
| Pas de virtual sur GPU | Enum dispatch + switch |
| Récursion interdite | Pile explicite itérative |
| Pointeurs Host sur Device | Pointer fixup avant upload |
| curand_init lent | Wang hash O(1) |

## Résumé des améliorations prévues

| Amélioration | Concepts du cours | Impact |
|-------------|-------------------|--------|
| `__constant__` memory | Types de mémoire | Cache broadcast, moins de registres |
| Réduction parallèle | `__shared__`, `__syncthreads`, `atomicAdd` | Tone mapping adaptatif |
| CUDA Streams | Recouvrement calcul/transfert | Pipeline async |
| Thrust | Librairies CUDA | Comparaison custom vs librairie |
| Occupancy API | Architecture GPU, profilage | Justification block size |
| Tests GPU | Validation CUDA | Couverture complète |
