#!/usr/bin/env python3
"""Generate rapport_raytracer.pdf using reportlab."""

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, HRFlowable, Preformatted
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import os

# -----------------------------------------------
# Styles
# -----------------------------------------------
styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    'CustomTitle',
    parent=styles['Title'],
    fontSize=22,
    spaceAfter=8,
    textColor=colors.HexColor('#1a1a2e'),
    alignment=TA_CENTER,
)
subtitle_style = ParagraphStyle(
    'Subtitle',
    parent=styles['Normal'],
    fontSize=14,
    spaceAfter=4,
    textColor=colors.HexColor('#16213e'),
    alignment=TA_CENTER,
)
author_style = ParagraphStyle(
    'Author',
    parent=styles['Normal'],
    fontSize=11,
    spaceAfter=2,
    textColor=colors.grey,
    alignment=TA_CENTER,
)
h1_style = ParagraphStyle(
    'H1',
    parent=styles['Heading1'],
    fontSize=16,
    spaceBefore=20,
    spaceAfter=8,
    textColor=colors.HexColor('#1a1a2e'),
    borderPad=4,
)
h2_style = ParagraphStyle(
    'H2',
    parent=styles['Heading2'],
    fontSize=13,
    spaceBefore=14,
    spaceAfter=6,
    textColor=colors.HexColor('#16213e'),
)
body_style = ParagraphStyle(
    'Body',
    parent=styles['Normal'],
    fontSize=10.5,
    leading=16,
    spaceAfter=6,
    alignment=TA_JUSTIFY,
)
bullet_style = ParagraphStyle(
    'Bullet',
    parent=styles['Normal'],
    fontSize=10.5,
    leading=15,
    spaceAfter=3,
    leftIndent=20,
    bulletIndent=5,
)
code_style = ParagraphStyle(
    'Code',
    parent=styles['Code'],
    fontSize=8.5,
    leading=12,
    fontName='Courier',
    backColor=colors.HexColor('#f4f4f4'),
    leftIndent=10,
    rightIndent=10,
    spaceAfter=8,
    spaceBefore=4,
)
caption_style = ParagraphStyle(
    'Caption',
    parent=styles['Normal'],
    fontSize=9,
    textColor=colors.grey,
    alignment=TA_CENTER,
    spaceAfter=10,
)
abstract_style = ParagraphStyle(
    'Abstract',
    parent=styles['Normal'],
    fontSize=10.5,
    leading=16,
    leftIndent=40,
    rightIndent=40,
    spaceAfter=8,
    alignment=TA_JUSTIFY,
    borderColor=colors.HexColor('#1a1a2e'),
    borderWidth=1,
    borderPad=10,
    borderRadius=4,
    backColor=colors.HexColor('#f0f4ff'),
)

# -----------------------------------------------
# Table helper
# -----------------------------------------------
def make_table(headers, rows, col_widths=None):
    data = [headers] + rows
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 9.5),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f5f7ff')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cccccc')),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
    ]))
    return t

def b(text):
    return f'<b>{text}</b>'

def code_block(text):
    return Preformatted(text, code_style)

# -----------------------------------------------
# Build document
# -----------------------------------------------
def build():
    output = os.path.join(os.path.dirname(__file__), 'rapport_raytracer.pdf')
    doc = SimpleDocTemplate(
        output,
        pagesize=A4,
        leftMargin=2.5*cm,
        rightMargin=2.5*cm,
        topMargin=2.5*cm,
        bottomMargin=2.5*cm,
        title='Rapport Technique – CUDA Ray Tracer',
        author='Projet RayTracing',
    )

    story = []

    # ---- Page de titre ----
    story.append(Spacer(1, 2*cm))
    story.append(Paragraph('Rapport Technique', title_style))
    story.append(Paragraph('CUDA Ray Tracer', ParagraphStyle('T2', parent=title_style, fontSize=26, textColor=colors.HexColor('#0f3460'))))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph('Moteur de rendu physiquement réaliste', subtitle_style))
    story.append(Paragraph('GPU (CUDA) / CPU (OpenMP)', subtitle_style))
    story.append(Spacer(1, 0.8*cm))
    story.append(HRFlowable(width="80%", thickness=2, color=colors.HexColor('#1a1a2e'), spaceAfter=8))
    story.append(Paragraph('Projet RayTracing - Mars 2026', author_style))
    story.append(Spacer(1, 1.5*cm))

    # Abstract
    story.append(Paragraph(
        'Ce rapport présente l\'implémentation d\'un moteur de ray tracing physiquement réaliste '
        'tirant parti des capacités de calcul parallèle des GPU modernes via CUDA. Le projet '
        'propose deux chemins de rendu : un chemin GPU utilisant CUDA pour des performances '
        'optimales (~160 MRays/s sur RTX 4060), et un chemin CPU multi-thread utilisant OpenMP. '
        '
        'L\'accélération GPU est de l\'ordre de <b>×56</b> par rapport au rendu CPU.',
        abstract_style
    ))

    story.append(PageBreak())

    # ---- 1. Introduction ----
    story.append(Paragraph('1. Introduction', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))
    story.append(Paragraph(
        'Le ray tracing est une technique de rendu qui simule le comportement physique de la lumière '
        'en traçant des rayons depuis la caméra vers la scène. Contrairement à la rastérisation, '
        'cette approche produit naturellement des effets tels que les réflexions, les réfractions, '
        'les ombres douces et la profondeur de champ.',
        body_style
    ))
    story.append(Paragraph(
        'Ce projet implémente un <b>path tracer Monte Carlo</b> : pour chaque pixel, de nombreux '
        'rayons sont lancés avec des perturbations aléatoires, et leurs contributions sont moyennées '
        'pour obtenir une image convergée. La nature massivement parallèle du problème '
        '(chaque pixel est indépendant) le rend idéal pour les GPU.',
        body_style
    ))

    story.append(Paragraph('1.1 Objectifs', h2_style))
    for obj in [
        'Implémenter un path tracer complet avec matériaux physiques (Lambertian, Métal, Diélectrique)',
        'Exploiter CUDA pour une accélération GPU massive',
        'Fournir un chemin de rendu CPU avec OpenMP comme alternative portable',
        'Mode benchmark multi-résolution pour l\'analyse de performance',
        'Construire un pipeline de tests unitaires sans dépendance CUDA',
    ]:
        story.append(Paragraph(f'• {obj}', bullet_style))

    story.append(Paragraph('1.2 Technologies utilisées', h2_style))
    t = make_table(
        ['Technologie', 'Rôle'],
        [
            ['CUDA 12.6',       'Rendu GPU parallèle'],
            ['OpenMP',          'Rendu CPU multi-thread'],
            ['stb_image_write', 'Export PNG/BMP/JPG/TGA/PPM'],
            ['CMake 3.18+',     'Système de build'],
            ['CTest',           'Tests unitaires'],
        ],
        col_widths=[6*cm, 10*cm]
    )
    story.append(t)
    story.append(Paragraph('Tableau 1 - Technologies et bibliothèques du projet', caption_style))

    # ---- 2. Architecture ----
    story.append(Paragraph('2. Architecture générale', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))
    story.append(Paragraph(
        'Le projet est structuré autour de bibliothèques d\'en-têtes (fichiers <b>.cuh</b>) '
        'organisées par domaine fonctionnel, ce qui favorise la réutilisation du code entre '
        'les chemins GPU et CPU.',
        body_style
    ))

    story.append(Paragraph('2.1 Organisation des modules', h2_style))
    t = make_table(
        ['Module', 'Contenu'],
        [
            ['Utilitaires core',  'vec3, ray, aabb, interval, timer, générateur aléatoire'],
            ['Géométrie',         'sphere, plane, interface hittable'],
            ['Matériaux',         'lambertian, metal, dielectric'],
            ['Accélération',      'Arbre BVH (variantes GPU et CPU)'],
            ['Rendu',             'Renderers GPU/CPU, tone mapping, dispatch matériaux'],
            ['Caméra',            'Caméra perspective avec DoF et motion blur'],
            ['Mode interactif',   'Fenêtre, interop GL, contrôleur caméra, entrées'],
        ],
        col_widths=[5*cm, 11*cm]
    )
    story.append(t)
    story.append(Paragraph('Tableau 2 - Organisation modulaire du projet', caption_style))

    story.append(Paragraph('2.2 Trois modes de fonctionnement', h2_style))
    for i, item in enumerate([
        '<b>Mode GPU</b> (défaut) : rendu CUDA, export image',
        '<b>Mode CPU</b> (--cpu) : rendu OpenMP, export image',
        '<b>Mode benchmark</b> (--benchmark) : benchmark multi-résolution avec CUDA events',
    ], 1):
        story.append(Paragraph(f'{i}. {item}', bullet_style))

    # ---- 3. Algorithme ----
    story.append(Paragraph('3. Algorithme de path tracing', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    story.append(Paragraph('3.1 Principe du Monte Carlo path tracing', h2_style))
    story.append(Paragraph(
        'Pour chaque pixel (i, j), on lance N rayons avec des perturbations aléatoires. '
        'La couleur finale est la moyenne des contributions :',
        body_style
    ))
    story.append(Paragraph('<b>C(i,j) = (1/N) × Σ trace(r_k)</b>', ParagraphStyle('formula',
        parent=body_style, fontSize=11, alignment=TA_CENTER, spaceAfter=8,
        backColor=colors.HexColor('#f0f4ff'), leftIndent=60, rightIndent=60, borderPad=6)))
    story.append(Paragraph(
        'Chaque rayon est suivi de rebond en rebond jusqu\'à une profondeur maximale d_max = 50 '
        'ou jusqu\'à ce qu\'il s\'échappe vers l\'arrière-plan.',
        body_style
    ))

    story.append(Paragraph('3.2 Traversée BVH itérative', h2_style))
    story.append(Paragraph(
        'L\'accélération par <b>Bounding Volume Hierarchy (BVH)</b> réduit la complexité de '
        'l\'intersection de O(n) à O(log n). La traversée est implémentée de façon itérative '
        'avec une pile (profondeur max : 64 nœuds), ce qui est critique pour les performances '
        'GPU où la récursion est coûteuse.',
        body_style
    ))
    story.append(Paragraph(
        'La construction du BVH utilise une stratégie de <b>median split</b> sur l\'axe le plus '
        'long, avec partitionnement par centroïde.',
        body_style
    ))

    story.append(Paragraph('3.3 Accumulation et tone mapping', h2_style))
    story.append(Paragraph(
        'Les couleurs sont accumulées en virgule flottante (HDR). La compression vers l\'espace SDR '
        'utilise le <b>tone mapping de Reinhard</b> : c\' = c / (1 + c), '
        'suivi d\'une <b>correction gamma γ = 2.2</b> : c_out = c\'^(1/γ).',
        body_style
    ))

    # ---- 4. Matériaux ----
    story.append(Paragraph('4. Matériaux', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    story.append(Paragraph('4.1 Lambertian (diffus)', h2_style))
    story.append(Paragraph(
        'Le matériau Lambertien modélise les surfaces mattes avec une distribution '
        'cosinus-pondérée. La direction de diffusion est : '
        '<b>d_scatter = n̂ + r_unit</b>, où r_unit est un vecteur unitaire aléatoire. '
        'L\'atténuation est la couleur albedo du matériau.',
        body_style
    ))

    story.append(Paragraph('4.2 Métal (réflexion spéculaire)', h2_style))
    story.append(Paragraph(
        'La réflexion spéculaire avec flou contrôlé : '
        '<b>d_reflect = v - 2(v·n̂)n̂ + f·r_unit</b>, '
        'où f ∈ [0,1] est le paramètre de <i>fuzz</i> contrôlant le flou '
        '(0 = miroir parfait).',
        body_style
    ))

    story.append(Paragraph('4.3 Diélectrique (verre/réfraction)', h2_style))
    story.append(Paragraph(
        'Le diélectrique combine réflexion et réfraction via <b>l\'approximation de Schlick</b> '
        'pour les équations de Fresnel : R(θ) ≈ R₀ + (1 - R₀)(1 - cosθ)⁵, '
        'avec R₀ = ((n₁ - n₂)/(n₁ + n₂))². '
        'La réfraction suit la loi de Snell-Descartes : n₁ sinθ₁ = n₂ sinθ₂. '
        'La réflexion interne totale est gérée quand n₁/n₂ · sinθ₁ > 1. '
        'IOR par défaut : 1.5 (verre).',
        body_style
    ))

    t = make_table(
        ['Matériau', 'Modèle', 'Paramètres', 'Usage'],
        [
            ['Lambertian', 'Diffus cosinus',    'albedo (RGB)',            'Surfaces mattes'],
            ['Metal',      'Réflexion + flou',  'albedo (RGB), fuzz [0,1]','Métaux, miroirs'],
            ['Dielectric', 'Snell + Schlick',   'IOR (défaut 1.5)',        'Verre, eau'],
        ],
        col_widths=[3.2*cm, 4*cm, 5.2*cm, 3.6*cm]
    )
    story.append(t)
    story.append(Paragraph('Tableau 3 - Résumé des matériaux implémentés', caption_style))

    # ---- 5. Géométrie ----
    story.append(Paragraph('5. Géométrie', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    story.append(Paragraph('5.1 Sphère', h2_style))
    story.append(Paragraph(
        'L\'intersection rayon-sphère est résolue analytiquement via l\'équation quadratique '
        '<b>‖o + td - c‖² = r²</b>, soit at² + bt + c = 0 avec a = ‖d‖², '
        'b = 2d·(o - c), c = ‖o - c‖² - r².',
        body_style
    ))
    story.append(Paragraph(
        'La normale de surface est calculée correctement pour les rayons entrant et sortant '
        '(front/back face). Les coordonnées UV sphériques sont également disponibles pour '
        'le texturage. La AABB est le cube [c - r, c + r] sur chaque axe.',
        body_style
    ))

    story.append(Paragraph('5.2 Plan', h2_style))
    story.append(Paragraph(
        'Le plan infini est supporté comme primitive géométrique supplémentaire, '
        'permettant la construction de sols et de surfaces plates arbitraires.',
        body_style
    ))

    # ---- 6. Caméra ----
    story.append(Paragraph('6. Caméra', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    story.append(Paragraph('6.1 Projection perspective', h2_style))
    for p in [
        '<b>FOV vertical</b> : angle de champ de vision (défaut : 20°)',
        '<b>Position</b> : vecteur look-from',
        '<b>Cible</b> : vecteur look-at',
        '<b>Up vector</b> : orientation verticale',
    ]:
        story.append(Paragraph(f'• {p}', bullet_style))

    story.append(Paragraph('6.2 Profondeur de champ (Depth of Field)', h2_style))
    story.append(Paragraph(
        'La simulation de la profondeur de champ utilise un modèle de lentille mince : '
        'les rayons sont échantillonnés depuis un disque de rayon proportionnel à l\'ouverture '
        '(<i>defocus angle</i>), tous pointant vers le plan de mise au point. '
        'Par défaut : ouverture ≈ f/22, distance de mise au point = 10 unités.',
        body_style
    ))

    story.append(Paragraph('6.3 Motion blur', h2_style))
    story.append(Paragraph(
        'Les rayons sont générés à des instants aléatoires entre les temps d\'ouverture '
        'et de fermeture de l\'obturateur, permettant de simuler le flou de mouvement.',
        body_style
    ))

    # ---- 7. GPU ----
    story.append(PageBreak())
    story.append(Paragraph('7. Implémentation GPU (CUDA)', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    story.append(Paragraph('7.1 Organisation des threads', h2_style))
    story.append(Paragraph(
        'Le rendu est organisé en grille 2D de blocs 16×16 (256 threads/bloc). '
        'Chaque thread traite un pixel de l\'image de manière indépendante :',
        body_style
    ))
    story.append(code_block(
        '// Organisation de la grille\ndim3 block(16, 16);\n'
        'dim3 grid(ceil(W/16), ceil(H/16));\nrender_kernel<<<grid, block>>>(...);'
    ))

    story.append(Paragraph('7.2 Kernels CUDA', h2_style))
    t = make_table(
        ['Kernel', 'Rôle'],
        [
            ['init_rand_states_fast',   'Initialisation des états RNG (Wang hash)'],
            ['render_kernel',           'Rendu principal (SPP samples par pixel)'],
            ['render_kernel_accumulate','Accumulation progressive (mode interactif)'],
            ['convert_to_rgba8',        'Conversion HDR→RGBA8 avec tone mapping'],
        ],
        col_widths=[6*cm, 10*cm]
    )
    story.append(t)
    story.append(Paragraph('Tableau 4 - Kernels CUDA du projet', caption_style))

    story.append(Paragraph('7.3 Gestion mémoire', h2_style))
    for item in [
        '<b>Host→Device</b> : matériaux, nœuds BVH, primitives (via cudaMemcpy)',
        '<b>Device→Host</b> : framebuffer final après rendu',
        '<b>Fixup de pointeurs</b> : redirection host→device des pointeurs de matériaux en O(n)',
        '<b>États RNG</b> : un curandState par pixel (mémoire device)',
    ]:
        story.append(Paragraph(f'• {item}', bullet_style))

    story.append(Paragraph('7.4 Initialisation RNG rapide (Wang hash)', h2_style))
    story.append(Paragraph(
        'Au lieu de curand_init (coûteux), le projet utilise un <b>Wang hash</b> '
        'pour une initialisation O(1) par pixel :',
        body_style
    ))
    story.append(code_block(
        '__device__ uint32_t wang_hash(uint32_t seed) {\n'
        '    seed = (seed ^ 61) ^ (seed >> 16);\n'
        '    seed *= 9;\n'
        '    seed = seed ^ (seed >> 4);\n'
        '    seed *= 0x27d4eb2d;\n'
        '    seed = seed ^ (seed >> 15);\n'
        '    return seed;\n'
        '}'
    ))

    # ---- 8. CPU ----
    story.append(Paragraph('8. Implémentation CPU (OpenMP)', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    story.append(Paragraph('8.1 Parallélisation', h2_style))
    story.append(code_block(
        '#pragma omp parallel for schedule(dynamic, 1)\n'
        'for (int j = 0; j < height; j++) {\n'
        '    CPURandom rng(seed + j);\n'
        '    for (int i = 0; i < width; i++) {\n'
        '        // rendu du pixel (i, j)\n'
        '    }\n'
        '}'
    ))
    story.append(Paragraph(
        'L\'<b>ordonnancement dynamique</b> (schedule(dynamic, 1)) assure un équilibrage de '
        'charge optimal, certaines lignes étant plus coûteuses selon la complexité de la scène.',
        body_style
    ))

    story.append(Paragraph('8.2 Comparaison GPU vs CPU', h2_style))
    t = make_table(
        ['Paramètre', 'GPU (RTX 4060)', 'CPU (8 threads)'],
        [
            ['Résolution',         '800×600',      '800×600'],
            ['Samples per pixel',  '100',           '100'],
            ['Temps de rendu',     '~0.3 s',        '~17 s'],
            ['Débit',              '~160 MRays/s',  '~2.8 MRays/s'],
            ['Speedup GPU/CPU',    '×56',           '-'],
        ],
        col_widths=[5.5*cm, 5.5*cm, 5*cm]
    )
    story.append(t)
    story.append(Paragraph('Tableau 5 - Comparaison des performances GPU vs CPU', caption_style))

    # ---- 9. Build ----
    story.append(Paragraph('10. Système de build (CMake)', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    t = make_table(
        ['Option CMake', 'Description'],
        [
            ['(défaut)',              'Build GPU complet (CUDA + CPU)'],
            ['BUILD_CPU_ONLY=ON',     'Build CPU seul, sans CUDA'],
            ['BUILD_TESTS_ONLY=ON',   'Tests unitaires seuls, sans CUDA'],
        ],
        col_widths=[5.5*cm, 10.5*cm]
    )
    story.append(t)
    story.append(Paragraph('Tableau 7 - Modes de build CMake', caption_style))

    story.append(Paragraph('10.1 Flags CUDA notables', h2_style))
    for flag in [
        '<b>-lineinfo</b> : informations de débogage ligne par ligne',
        '<b>--expt-relaxed-constexpr</b> : expressions constantes relaxées',
        '<b>--extended-lambda</b> : support des lambdas étendus',
        '<b>-Xcompiler=-Wall</b> : propagation des warnings vers le compilateur host',
        '<b>--use_fast_math</b> : fonctions mathématiques optimisées (si activé)',
    ]:
        story.append(Paragraph(f'• {flag}', bullet_style))

    story.append(Paragraph('10.2 Exemples de build', h2_style))
    story.append(code_block(
        '# Build GPU Release\ncmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=90\n'
        'make -j$(nproc)\n\n'
        '# Rendu avec profiling\n./raytracer -w 1920 -h 1440 -s 500 -d 100 --profile -o output.png\n\n'
        '# Mode benchmark\n./raytracer --benchmark\n\n'
        '# Build CPU uniquement\ncmake .. -DBUILD_CPU_ONLY=ON && make\n./raytracer --cpu -o output_cpu.png\n\n'
        '# Tests\ncmake .. -DBUILD_TESTS_ONLY=ON && make\n./raytracer_tests'
    ))

    # ---- 11. Tests ----
    story.append(Paragraph('11. Tests unitaires', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))
    story.append(Paragraph(
        'La suite de tests est compilable <b>sans CUDA</b> (BUILD_TESTS_ONLY=ON), '
        'ce qui permet de valider les composants mathématiques et géométriques '
        'de manière indépendante du GPU.',
        body_style
    ))

    t = make_table(
        ['Fichier', 'Composant testé'],
        [
            ['test_vec3.cpp',     'Opérations vectorielles (construction, arithmétique, normalisation, produits)'],
            ['test_ray.cpp',      'Paramétrisation des rayons'],
            ['test_interval.cpp', 'Intersection et containment d\'intervalles'],
            ['test_aabb.cpp',     'Intersection et fusion de boîtes englobantes (AABB)'],
            ['test_sphere.cpp',   'Intersection rayon-sphère'],
            ['test_plane.cpp',    'Géométrie des plans'],
            ['test_main.cpp',     'Framework de tests et statistiques'],
        ],
        col_widths=[5*cm, 11*cm]
    )
    story.append(t)
    story.append(Paragraph('Tableau 8 - Suite de tests unitaires', caption_style))

    # ---- 12. Optimisations ----
    story.append(Paragraph('12. Optimisations de performance', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    opts = [
        ('Fast math', 'Flags --use_fast_math et -ffast-math pour des opérations FPU plus rapides'),
        ('AABB sans branchement', 'Code déroulé pour minimiser les divergences de warp sur GPU'),
        ('Wang hash RNG', 'Initialisation O(1) des états aléatoires, évite le coûteux curand_init'),
        ('BVH itératif', 'Pile explicite (64 nœuds max) évitant la récursion, favorable au cache GPU'),
        ('Échantillonnage stratifié', 'Jittering des positions de pixel pour réduire l\'aliasing'),
        ('BVH median split', 'Partitionnement par centroïde sur l\'axe le plus long'),
        ('Terminaison anticipée', 'Un rayon s\'échappe immédiatement si sa direction est invalide'),
        ('Row-major layout', 'Accès mémoire coalescés pour le framebuffer GPU'),
        ('Fixup de pointeurs', 'Redirection host→device des matériaux en O(n) avant transfert'),
        ('Ordonnancement dynamique', 'schedule(dynamic, 1) pour l\'équilibrage de charge OpenMP'),
    ]
    for i, (name, desc) in enumerate(opts, 1):
        story.append(Paragraph(f'{i}. <b>{name}</b> : {desc}', bullet_style))

    # ---- 13. Scène par défaut ----
    story.append(Paragraph('13. Scène par défaut', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    story.append(Paragraph(
        'La scène par défaut (create_default_scene) comprend les éléments suivants :',
        body_style
    ))
    for item in [
        '<b>Caméra</b> : position (13, 2, 3), cible (0, 0, 0), FOV 20°, ouverture f/22, distance de mise au point 10',
        '<b>Sol</b> : grande sphère (rayon 1000) Lambertienne grise',
        '<b>Sphère centrale</b> : verre diélectrique (IOR 1.5), rayon 1.0',
        '<b>Sphère gauche</b> : Lambertienne brune, rayon 1.0',
        '<b>Sphère droite</b> : métal poli, rayon 1.0',
        '<b>Champ de marbres</b> : 100 petites sphères (grille 10×10) - 60% diffuses, 25% métalliques, 15% verre',
    ]:
        story.append(Paragraph(f'• {item}', bullet_style))

    # ---- 14. Historique ----
    story.append(Paragraph('14. Historique du développement', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    t = make_table(
        ['Commit', 'Description'],
        [
            ['Fondation',    'Structure de base du ray tracer CUDA, primitives et matériaux'],
            ['fb5df74',      'Refactorisation de l\'architecture CUDA et optimisations de performance'],
            ['d1b2f20',      'Suite de tests unitaires complète compilable sans CUDA'],
            ['10164b5',      'Support du rendu CPU avec OpenMP et optimisations supplémentaires'],
            ['d6e1518',      'Mise à jour de la documentation README'],
            ['475bfe1 ✓',    'Pipeline complet GPU avec réduction parallèle et tone mapping adaptatif (état actuel)'],
        ],
        col_widths=[3.5*cm, 12.5*cm]
    )
    story.append(t)
    story.append(Paragraph('Tableau 9 - Historique des commits principaux', caption_style))

    # ---- 15. Conclusion ----
    story.append(PageBreak())
    story.append(Paragraph('15. Conclusion', h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#1a1a2e'), spaceAfter=10))

    story.append(Paragraph(
        'Ce projet démontre une implémentation complète et optimisée d\'un path tracer '
        'physiquement réaliste exploitant le parallélisme massif des GPU modernes. '
        'L\'architecture modulaire, la double implémentation GPU/CPU et le mode interactif '
        'en font un moteur de rendu à la fois performant et extensible.',
        body_style
    ))

    story.append(Paragraph('Points forts du projet :', h2_style))
    for item in [
        '<b>Double implémentation GPU/CPU</b> offrant flexibilité et portabilité',
        '<b>Accélération ×56</b> grâce à CUDA par rapport au rendu CPU',
        '<b>Mode benchmark multi-résolution</b> avec CUDA events pour l\'analyse de performance',
        '<b>Matériaux physiques</b> couvrant les principaux types de surfaces réelles',
        '<b>Pipeline de tests</b> indépendant de CUDA pour une validation fiable',
        '<b>Build système flexible</b> avec CMake supportant plusieurs configurations',
        '<b>Architecture modulaire</b> facilitant l\'extension avec nouveaux matériaux et primitives',
    ]:
        story.append(Paragraph(f'• {item}', bullet_style))

    story.append(Paragraph('Perspectives d\'amélioration :', h2_style))
    for item in [
        'Sources de lumière explicites (area lights, HDR environment maps)',
        'Support des textures (texture mapping UV)',
        'Sampling par importance multiple (MIS) pour une convergence plus rapide',
        'Volumes participatifs (brouillard, fumée, subsurface scattering)',
        'Débruitage par apprentissage (OIDN, OptiX denoiser)',
    ]:
        story.append(Paragraph(f'• {item}', bullet_style))

    story.append(Spacer(1, 1*cm))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#cccccc'), spaceAfter=8))
    story.append(Paragraph('Rapport généré le 19 mars 2026 - CUDA Ray Tracer v1.0', caption_style))

    # ---- Build ----
    doc.build(story)
    print(f'PDF généré : {output}')
    return output

if __name__ == '__main__':
    build()
