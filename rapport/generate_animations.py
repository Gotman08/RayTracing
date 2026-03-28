#!/usr/bin/env python3
"""
Generation des animations GIF pour le rapport du Ray Tracer CUDA.
4 animations : pipeline GPU, reduction parallele, streams timeline, warp divergence.
Ouvrable dans n'importe quel navigateur (Chrome, Edge, Firefox...).
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'figure.dpi': 150,
})

GREEN  = '#4CAF50'
BLUE   = '#2196F3'
ORANGE = '#FF9800'
PURPLE = '#9C27B0'
CYAN   = '#00BCD4'
RED    = '#E53935'
GRAY   = '#9E9E9E'
LIGHT  = '#E0E0E0'

# =============================================================================
# Animation 1 : Pipeline GPU 3 passes (5 etapes)
# =============================================================================
def make_pipeline_gif():
    stages = [
        ('init_rand\nstates', GREEN),
        ('render\nkernel', GREEN),
        ('compute_avg\nluminance', ORANGE),
        ('tonemap\nkernel', PURPLE),
        ('D→H\ncopy', CYAN),
    ]
    labels_below = [
        'Wang hash RNG',
        'Color* HDR brut',
        '__shared__ sdata[256]',
        'Reinhard + gamma',
        'cudaMemcpyAsync',
    ]
    titles = [
        'Étape 1/5 : Initialisation RNG',
        'Étape 2/5 : Rendu path tracing (SPP rays/pixel)',
        'Étape 3/5 : Réduction parallèle (luminance moyenne)',
        'Étape 4/5 : Tone mapping adaptatif (HDR → SDR)',
        'Étape 5/5 : Téléchargement framebuffer (D→H)',
    ]

    fig, ax = plt.subplots(figsize=(10, 3.5))

    def draw_frame(step):
        ax.clear()
        ax.set_xlim(-0.5, 10.5)
        ax.set_ylim(-2, 2)
        ax.axis('off')
        ax.set_title(titles[step], fontsize=12, fontweight='bold', pad=10)

        for i, (name, color) in enumerate(stages):
            x = i * 2.2
            if i < step:
                fc = '#E3F2FD'
                ec = '#90CAF9'
                tc = '#90CAF9'
            elif i == step:
                fc = color
                ec = 'black'
                tc = 'white'
            else:
                fc = LIGHT
                ec = GRAY
                tc = GRAY
            rect = mpatches.FancyBboxPatch((x - 0.7, -0.45), 1.4, 0.9,
                                            boxstyle="round,pad=0.1",
                                            facecolor=fc, edgecolor=ec, linewidth=2 if i == step else 1)
            ax.add_patch(rect)
            ax.text(x, 0, name, ha='center', va='center', fontsize=8,
                    fontfamily='monospace', color=tc, fontweight='bold' if i == step else 'normal')

            if i == step:
                ax.text(x, -0.9, labels_below[i], ha='center', va='top', fontsize=8,
                        fontstyle='italic', color=color)

            if i < len(stages) - 1:
                ax.annotate('', xy=((i+1)*2.2 - 0.7, 0), xytext=(x + 0.7, 0),
                           arrowprops=dict(arrowstyle='->', color=GRAY if i >= step else '#90CAF9',
                                          lw=1.5))

        # constant memory annotation
        ax.text(4.4, -1.7, '▸ __constant__ Camera + RenderConfig (broadcast cache)',
                ha='center', fontsize=8, color=PURPLE, fontstyle='italic')

    anim = FuncAnimation(fig, draw_frame, frames=5, interval=1500, repeat=True)
    anim.save(os.path.join(OUTPUT_DIR, 'anim_pipeline_gpu.gif'),
              writer=PillowWriter(fps=1))
    plt.close()
    print("anim_pipeline_gpu.gif OK")


# =============================================================================
# Animation 2 : Reduction parallele (8 valeurs, 3 strides + warp shuffle)
# =============================================================================
def make_reduction_gif():
    frames_data = [
        {
            'title': 'État initial : chaque thread charge la luminance',
            'values': [0.21, 0.71, 0.07, 0.50, 0.21, 0.71, 0.07, 0.50],
            'active': list(range(8)),
            'arrows': [],
            'note': 'sdata[tid] = luminance(pixel[gid])',
        },
        {
            'title': 'Stride = 4 : sdata[i] += sdata[i+4]',
            'values': [0.42, 1.42, 0.14, 1.00, 0.21, 0.71, 0.07, 0.50],
            'active': [0, 1, 2, 3],
            'arrows': [(4,0), (5,1), (6,2), (7,3)],
            'note': '__syncthreads() - 4 threads actifs',
        },
        {
            'title': 'Stride = 2 : sdata[i] += sdata[i+2]',
            'values': [0.56, 2.42, 0.14, 1.00, None, None, None, None],
            'active': [0, 1],
            'arrows': [(2,0), (3,1)],
            'note': '__syncthreads() - 2 threads actifs',
        },
        {
            'title': 'Stride = 1 : __shfl_down_sync (warp shuffle)',
            'values': [2.98, 2.42, None, None, None, None, None, None],
            'active': [0],
            'arrows': [(1,0)],
            'note': 'Pas de __syncthreads - échange de registres intra-warp !',
        },
        {
            'title': 'Résultat : atomicAdd(d_total_luminance, 2.98)',
            'values': [2.98, None, None, None, None, None, None, None],
            'active': [0],
            'arrows': [],
            'note': 'Thread 0 accumule le résultat partiel du bloc',
        },
    ]

    fig, ax = plt.subplots(figsize=(9, 4))

    def draw_frame(f):
        ax.clear()
        data = frames_data[f]
        ax.set_xlim(-0.8, 8.5)
        ax.set_ylim(-2.5, 2.5)
        ax.axis('off')
        ax.set_title(data['title'], fontsize=12, fontweight='bold', pad=10)

        for i, v in enumerate(data['values']):
            x = i * 1.05
            if v is None:
                fc = '#F5F5F5'
                ec = '#E0E0E0'
                txt = ''
                tc = GRAY
            elif i in data['active']:
                fc = GREEN if f < 3 else (ORANGE if f == 3 else '#FF6F00')
                ec = 'black'
                txt = f'{v:.2f}'
                tc = 'white'
            else:
                fc = '#E8E8E8'
                ec = GRAY
                txt = f'{v:.2f}'
                tc = GRAY

            rect = mpatches.FancyBboxPatch((x - 0.4, -0.35), 0.8, 0.7,
                                            boxstyle="round,pad=0.05",
                                            facecolor=fc, edgecolor=ec, linewidth=1.5)
            ax.add_patch(rect)
            ax.text(x, 0, txt, ha='center', va='center', fontsize=10,
                    fontweight='bold', color=tc)
            ax.text(x, -0.65, f'[{i}]', ha='center', fontsize=7, color=GRAY)

        for src, dst in data['arrows']:
            color = PURPLE if f == 3 else RED
            style = '--' if f == 3 else '-'
            ax.annotate('', xy=(dst * 1.05, 0.5), xytext=(src * 1.05, 0.5),
                       arrowprops=dict(arrowstyle='->', color=color, lw=2,
                                      linestyle=style, connectionstyle='arc3,rad=0.3'))

        ax.text(4, -1.5, data['note'], ha='center', fontsize=10, fontstyle='italic',
                color=PURPLE if f >= 3 else RED)

        # Progress bar
        for i in range(5):
            c = GREEN if i <= f else LIGHT
            ax.add_patch(mpatches.Circle((2.5 + i * 0.8, -2.2), 0.15, color=c))

    anim = FuncAnimation(fig, draw_frame, frames=5, interval=2000, repeat=True)
    anim.save(os.path.join(OUTPUT_DIR, 'anim_reduction.gif'),
              writer=PillowWriter(fps=1))
    plt.close()
    print("anim_reduction.gif OK")


# =============================================================================
# Animation 3 : CUDA Streams timeline (4 phases)
# =============================================================================
def make_streams_gif():
    fig, ax = plt.subplots(figsize=(10, 3.5))

    phases = [
        {'cursor': 1.0, 'title': 'Phase 1 : Overlap - init_rand + materials upload',
         'highlight_compute': (0, 2), 'highlight_transfer': (0, 2), 'overlap': (0, 2)},
        {'cursor': 5.0, 'title': 'Phase 2 : Rendu GPU (phase dominante)',
         'highlight_compute': (2.5, 8.5), 'highlight_transfer': None, 'overlap': None},
        {'cursor': 9.0, 'title': 'Phase 3 : Réduction + tone mapping',
         'highlight_compute': (8.5, 10.5), 'highlight_transfer': None, 'overlap': None},
        {'cursor': 11.0, 'title': 'Phase 4 : Overlap - FB download + CPU save',
         'highlight_compute': None, 'highlight_transfer': (10.5, 12), 'overlap': (10.5, 12)},
    ]

    all_compute = [(0, 2, 'init_rand'), (2.5, 8.5, 'render_kernel'), (8.5, 9.5, 'reduce'), (9.5, 10.5, 'TM')]
    all_transfer = [(0, 2, 'mat H→D'), (10.5, 12, 'FB D→H')]

    def draw_frame(f):
        ax.clear()
        phase = phases[f]
        ax.set_xlim(-1.5, 13)
        ax.set_ylim(-2, 2.5)
        ax.axis('off')
        ax.set_title(phase['title'], fontsize=12, fontweight='bold', pad=10)

        # Labels
        ax.text(-1.2, 1, 'compute', fontsize=9, fontweight='bold', va='center')
        ax.text(-1.2, 0, 'transfer', fontsize=9, fontweight='bold', va='center')

        # Overlap zone
        if phase['overlap']:
            x0, x1 = phase['overlap']
            rect = mpatches.FancyBboxPatch((x0, -0.35), x1 - x0, 1.7,
                                            boxstyle="round,pad=0.1",
                                            facecolor='#FFF3E0', edgecolor=ORANGE,
                                            linewidth=1.5, linestyle='--')
            ax.add_patch(rect)
            ax.text((x0 + x1) / 2, -0.6, 'OVERLAP', ha='center', fontsize=8,
                    fontweight='bold', color=ORANGE)

        # Compute blocks
        for x0, x1, name in all_compute:
            is_active = phase['highlight_compute'] and x0 >= phase['highlight_compute'][0] - 0.1 and x1 <= phase['highlight_compute'][1] + 0.1
            fc = GREEN if is_active else '#E8F5E9'
            ec = 'black' if is_active else '#C8E6C9'
            tc = 'white' if is_active else '#A5D6A7'
            rect = mpatches.FancyBboxPatch((x0, 0.65), x1 - x0, 0.7,
                                            boxstyle="round,pad=0.05",
                                            facecolor=fc, edgecolor=ec, linewidth=1.5 if is_active else 1)
            ax.add_patch(rect)
            ax.text((x0 + x1) / 2, 1, name, ha='center', va='center', fontsize=7,
                    fontfamily='monospace', color=tc)

        # Transfer blocks
        for x0, x1, name in all_transfer:
            is_active = phase['highlight_transfer'] and x0 >= phase['highlight_transfer'][0] - 0.1 and x1 <= phase['highlight_transfer'][1] + 0.1
            fc = BLUE if is_active else '#E3F2FD'
            ec = 'black' if is_active else '#BBDEFB'
            tc = 'white' if is_active else '#90CAF9'
            rect = mpatches.FancyBboxPatch((x0, -0.35), x1 - x0, 0.7,
                                            boxstyle="round,pad=0.05",
                                            facecolor=fc, edgecolor=ec, linewidth=1.5 if is_active else 1)
            ax.add_patch(rect)
            ax.text((x0 + x1) / 2, 0, name, ha='center', va='center', fontsize=7,
                    fontfamily='monospace', color=tc)

        # Timeline
        ax.plot([0, 12.5], [-1.2, -1.2], color=GRAY, linewidth=1)
        ax.annotate('temps', xy=(12.5, -1.2), fontsize=8, color=GRAY)

        # Cursor
        ax.plot([phase['cursor'], phase['cursor']], [-1.0, 1.6], color=RED,
                linewidth=2, linestyle='--')
        ax.plot(phase['cursor'], -1.2, 'o', color=RED, markersize=8)

    anim = FuncAnimation(fig, draw_frame, frames=4, interval=2000, repeat=True)
    anim.save(os.path.join(OUTPUT_DIR, 'anim_streams.gif'),
              writer=PillowWriter(fps=1))
    plt.close()
    print("anim_streams.gif OK")


# =============================================================================
# Animation 4 : Divergence de warp (3 cas)
# =============================================================================
def make_divergence_gif():
    fig, ax = plt.subplots(figsize=(10, 4))

    cases = [
        {
            'title': 'Cas 1 : Pas de divergence - tous Lambertian',
            'colors': [GREEN] * 16,
            'branches': ['scatter_lambertian()'],
            'branch_colors': [GREEN],
            'efficiency': '100%',
            'eff_color': GREEN,
            'note': '[OK] Tous les threads executent la meme branche',
        },
        {
            'title': 'Cas 2 : Divergence modérée - frontière métal/diffus',
            'colors': [GREEN]*10 + [BLUE]*6,
            'branches': ['scatter_lambertian() - pass 1', 'scatter_metal() - pass 2'],
            'branch_colors': [GREEN, BLUE],
            'efficiency': '~60%',
            'eff_color': ORANGE,
            'note': '△ 2 branches exécutées séquentiellement',
        },
        {
            'title': 'Cas 3 : Divergence maximale - 3 matériaux',
            'colors': [GREEN]*5 + [BLUE]*6 + [PURPLE]*5,
            'branches': ['lambertian - pass 1', 'metal - pass 2', 'dielectric - pass 3'],
            'branch_colors': [GREEN, BLUE, PURPLE],
            'efficiency': '~33%',
            'eff_color': RED,
            'note': '[X] 3 branches sequentielles -> efficacite divisee par 3',
        },
    ]

    def draw_frame(f):
        ax.clear()
        case = cases[f]
        ax.set_xlim(-1.5, 13)
        ax.set_ylim(-3, 2.5)
        ax.axis('off')
        ax.set_title(case['title'], fontsize=12, fontweight='bold', pad=10)

        # Warp label
        ax.text(-1, 0.8, 'Warp 0\n(32 threads)', fontsize=9, fontweight='bold',
                ha='center', va='center')

        # Threads
        for i in range(16):
            x = i * 0.6
            rect = mpatches.FancyBboxPatch((x, 0.4), 0.45, 0.8,
                                            boxstyle="round,pad=0.02",
                                            facecolor=case['colors'][i],
                                            edgecolor='white', linewidth=1)
            ax.add_patch(rect)

        # "..." pour les 16 restants
        ax.text(10, 0.8, '...', fontsize=14, ha='center', va='center', color=GRAY)

        # Branches
        for j, (branch, bc) in enumerate(zip(case['branches'], case['branch_colors'])):
            y = -0.5 - j * 0.7
            rect = mpatches.FancyBboxPatch((0, y - 0.2), 6, 0.4,
                                            boxstyle="round,pad=0.05",
                                            facecolor=bc, alpha=0.2, edgecolor=bc, linewidth=1)
            ax.add_patch(rect)
            ax.text(3, y, branch, ha='center', va='center', fontsize=9,
                    fontfamily='monospace', color=bc)

        # Efficiency badge
        ax.text(8.5, -0.8, f'Efficacité SIMT : {case["efficiency"]}',
                fontsize=14, fontweight='bold', color=case['eff_color'],
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                         edgecolor=case['eff_color'], linewidth=2))

        # Note
        ax.text(5, -2.7, case['note'], ha='center', fontsize=10,
                fontstyle='italic', color=case['eff_color'])

    anim = FuncAnimation(fig, draw_frame, frames=3, interval=2500, repeat=True)
    anim.save(os.path.join(OUTPUT_DIR, 'anim_warp_divergence.gif'),
              writer=PillowWriter(fps=1))
    plt.close()
    print("anim_warp_divergence.gif OK")


# =============================================================================
# Main
# =============================================================================
if __name__ == '__main__':
    make_pipeline_gif()
    make_reduction_gif()
    make_streams_gif()
    make_divergence_gif()
    print("\nToutes les animations GIF ont ete generees avec succes.")
    print(f"Fichiers dans : {os.path.abspath(OUTPUT_DIR)}")
