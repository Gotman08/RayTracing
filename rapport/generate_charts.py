#!/usr/bin/env python3
"""
Génération des graphiques de benchmark pour le rapport du Ray Tracer CUDA.
Comparaison RTX 4070 Laptop vs NVIDIA GH200 (Romeo HPC).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# S'assurer que les chemins sont relatifs au dossier du script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')

# Style global
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 200,
})

NVIDIA_GREEN = '#76B900'
NVIDIA_DARK = '#1A1A2E'
BLUE = '#2196F3'
ORANGE = '#FF9800'
RED = '#E53935'
GRAY = '#757575'

# =============================================================================
# Données
# =============================================================================

# Romeo GH200 - GPU H100
gh200_gpu = {
    '640x480':   {'spp': 100, 'time': 0.045, 'mrays': 683},
    '1280x720':  {'spp': 100, 'time': 0.073, 'mrays': 1262},
    '1080p':     {'spp': 100, 'time': 0.147, 'mrays': 1411},
    '1080p_500': {'spp': 500, 'time': 0.719, 'mrays': 1442},
    '4K':        {'spp': 100, 'time': 0.569, 'mrays': 1458},
    '4K_500':    {'spp': 500, 'time': 2.804, 'mrays': 1479},
}

# Romeo GH200 - CPU ARM Neoverse V2 (16 threads)
gh200_cpu = {
    '640x480':  {'spp': 100, 'time': 0.933, 'mrays': 33},
    '1280x720': {'spp': 100, 'time': 2.558, 'mrays': 36},
    '1080p':    {'spp': 100, 'time': 5.805, 'mrays': 36},
}

# RTX 4070 Laptop
rtx4070_gpu = {
    '800x600':   {'spp': 100, 'time': 0.219, 'mrays': 219.2},
    '1080p':     {'spp': 100, 'time': 0.757, 'mrays': 273.9},
    '1080p_500': {'spp': 500, 'time': 3.843, 'mrays': 269.7},
}

rtx4070_cpu = {
    '800x600':   {'spp': 100, 'time': 15.98, 'mrays': 3.0},
    '1080p':     {'spp': 100, 'time': 124.88, 'mrays': 1.66},
    '1080p_500': {'spp': 500, 'time': 386.89, 'mrays': 2.68},
}

# =============================================================================
# Graphique 1 : Comparaison GPU - RTX 4070 vs GH200 H100
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

labels = ['1080p\n100 SPP', '1080p\n500 SPP']
rtx_vals = [rtx4070_gpu['1080p']['mrays'], rtx4070_gpu['1080p_500']['mrays']]
gh200_vals = [gh200_gpu['1080p']['mrays'], gh200_gpu['1080p_500']['mrays']]

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, rtx_vals, width, label='RTX 4070 Laptop', color=BLUE, edgecolor='white', zorder=3)
bars2 = ax.bar(x + width/2, gh200_vals, width, label='GH200 H100 (Romeo)', color=NVIDIA_GREEN, edgecolor='white', zorder=3)

# Annotations speedup
for i in range(len(labels)):
    ratio = gh200_vals[i] / rtx_vals[i]
    ax.annotate(f'x{ratio:.1f}',
                xy=(x[i] + width/2, gh200_vals[i]),
                xytext=(0, 8), textcoords='offset points',
                ha='center', fontweight='bold', fontsize=11, color=NVIDIA_GREEN)

# Valeurs sur les barres
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 15,
            f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=9, color=NVIDIA_GREEN)

ax.set_ylabel('Performance (MRays/s)')
ax.set_title('Comparaison GPU : RTX 4070 Laptop vs GH200 H100')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(loc='upper left')
ax.set_ylim(0, max(gh200_vals) * 1.2)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_gpu_comparison.png'), bbox_inches='tight')
plt.close()
print("chart_gpu_comparison.png OK")

# =============================================================================
# Graphique 2 : CPU vs GPU sur GH200 (barres, échelle log)
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

labels = ['640x480\n100 SPP', '1280x720\n100 SPP', '1920x1080\n100 SPP']
cpu_times = [gh200_cpu['640x480']['time'], gh200_cpu['1280x720']['time'], gh200_cpu['1080p']['time']]
gpu_times = [gh200_gpu['640x480']['time'], gh200_gpu['1280x720']['time'], gh200_gpu['1080p']['time']]

x = np.arange(len(labels))
width = 0.35

bars1 = ax.bar(x - width/2, cpu_times, width, label='CPU (ARM Neoverse V2, 16t)', color=ORANGE, edgecolor='white', zorder=3)
bars2 = ax.bar(x + width/2, gpu_times, width, label='GPU (H100)', color=NVIDIA_GREEN, edgecolor='white', zorder=3)

# Annotations speedup
for i in range(len(labels)):
    speedup = cpu_times[i] / gpu_times[i]
    mid_x = x[i]
    ax.annotate(f'{speedup:.0f}x',
                xy=(mid_x, max(cpu_times[i], gpu_times[i])),
                xytext=(0, 12), textcoords='offset points',
                ha='center', fontweight='bold', fontsize=12, color=RED,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor=RED, alpha=0.8))

# Valeurs
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.05,
            f'{bar.get_height():.2f}s', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.05,
            f'{bar.get_height():.3f}s', ha='center', va='bottom', fontsize=8)

ax.set_ylabel('Temps de rendu (secondes, échelle log)')
ax.set_title('CPU vs GPU sur NVIDIA GH200 (Romeo)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yscale('log')
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_cpu_vs_gpu.png'), bbox_inches='tight')
plt.close()
print("chart_cpu_vs_gpu.png OK")

# =============================================================================
# Graphique 3 : Scaling GPU - MRays/s vs résolution (GH200)
# =============================================================================

fig, ax = plt.subplots(figsize=(8, 5))

# Données 100 SPP seulement pour le scaling
resolutions = ['640x480', '1280x720', '1920x1080', '3840x2160']
pixels = [640*480, 1280*720, 1920*1080, 3840*2160]
mrays_100 = [683, 1262, 1411, 1458]

ax.plot(pixels, mrays_100, 'o-', color=NVIDIA_GREEN, linewidth=2.5, markersize=10,
        label='GH200 H100 (100 SPP)', zorder=3)

# Ligne RTX 4070 pour comparaison (1080p seulement, ligne horizontale)
ax.axhline(y=274, color=BLUE, linestyle='--', alpha=0.7, linewidth=1.5, label='RTX 4070 Laptop (~274 MRays/s)')

# Annotations
for i, (res, mray) in enumerate(zip(resolutions, mrays_100)):
    ax.annotate(f'{res}\n{mray} MRays/s',
                xy=(pixels[i], mray),
                xytext=(10, -25 if i % 2 == 0 else 15), textcoords='offset points',
                fontsize=8, ha='left',
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=0.8) if i > 0 else None)

ax.set_xlabel('Nombre de pixels')
ax.set_ylabel('Performance (MRays/s)')
ax.set_title('Scaling GPU : throughput en fonction de la résolution (GH200)')
ax.set_xscale('log')
ax.legend(loc='center right')
ax.set_ylim(0, 1700)

# Format x-axis
ax.set_xticks(pixels)
ax.set_xticklabels(['640x480\n(307K)', '1280x720\n(922K)', '1920x1080\n(2.1M)', '3840x2160\n(8.3M)'])

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'chart_scaling.png'), bbox_inches='tight')
plt.close()
print("chart_scaling.png OK")

print("\nTous les graphiques ont ete generes avec succes.")
