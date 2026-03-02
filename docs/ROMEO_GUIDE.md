# Guide Romeo2025 - Ray Tracer CUDA

## Configuration du Cluster

Romeo2025 utilise des **Superchips GH200** qui combinent:
- CPU ARM Grace
- GPU H100 (384 Go HBM3 par nœud)
- Architecture CUDA sm_90 (Hopper)

## Modules à Charger

```bash
module purge
module load cuda/12.6
module load cmake/3.28
module load gcc/13.2
```

## Compilation

```bash
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=90
make -j8
```

**Note**: L'architecture `90` cible spécifiquement les H100/GH200.

## Soumission de Jobs

### Configuration SLURM

```bash
#SBATCH --partition=gpu
#SBATCH --account=VOTRE_COMPTE    # Modifier avec votre allocation
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
```

### Soumettre un job

```bash
# Éditer le compte dans le script
sed -i 's/your_account/VOTRE_COMPTE/' scripts/slurm/run_raytracer.slurm

# Soumettre
sbatch scripts/slurm/run_raytracer.slurm
```

### Session interactive

```bash
salloc -p gpu --account=VOTRE_COMPTE --gres=gpu:1 --time=01:00:00
```

## Vérification GPU

```bash
# Info GPU
nvidia-smi

# Test compilation
./build/raytracer --info
```

Sortie attendue:
```
=== CUDA Device Information ===
Device 0: NVIDIA H100
  Compute capability: 9.0
  Total global memory: ~80 GB
  Multiprocessors: 132
```

## Performance Attendue

Sur H100 avec ce ray tracer:

| Résolution | SPP  | Temps estimé |
|------------|------|--------------|
| 1920x1080  | 100  | ~2-3s        |
| 3840x2160  | 100  | ~8-10s       |
| 1920x1080  | 1000 | ~20-25s      |

## Dépannage

### Erreur "out of memory"
Réduire la résolution ou les samples:
```bash
./build/raytracer -w 1280 -h 720 -s 50
```

### Module non trouvé
Vérifier les modules disponibles:
```bash
module avail cuda
module avail cmake
```

### Compte SLURM incorrect
Vérifier vos allocations:
```bash
sacctmgr show associations user=$USER
```

## Bonnes Pratiques

1. **Tester localement d'abord** avec peu de samples
2. **Utiliser --quiet** dans les jobs batch
3. **Sauvegarder les outputs** dans des dossiers datés
4. **Monitorer** avec `squeue -u $USER`
