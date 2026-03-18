/**
 * @file args.cuh
 * @brief Gestion des arguments en ligne de commande du ray tracer
 * @details Ce fichier definit la structure contenant tous les parametres
 *          configurables du ray tracer ainsi que les fonctions de parsing
 *          des arguments passes par l'utilisateur via argc/argv.
 */

#ifndef RAYTRACER_CLI_ARGS_CUH
#define RAYTRACER_CLI_ARGS_CUH

#include <iostream>
#include <string>
#include <cstdlib>

namespace rt {

/**
 * @brief Structure regroupant tous les arguments en ligne de commande
 * @details Contient les parametres de rendu (resolution, samples, profondeur),
 *          les options de sortie, le choix du mode CPU/GPU, ainsi que les
 *          parametres du mode interactif (controles camera, accumulation).
 *          Chaque champ possede une valeur par defaut raisonnable.
 */
struct Args {
    int width = 800;                        ///< Largeur de l'image en pixels (defaut : 800)
    int height = 600;                       ///< Hauteur de l'image en pixels (defaut : 600)
    int samples = 100;                      ///< Nombre de samples (rayons) par pixel (defaut : 100)
    int depth = 50;                         ///< Profondeur maximale de rebonds des rayons (defaut : 50)
    std::string output_file = "output.png"; ///< Chemin du fichier image de sortie (defaut : "output.png")
    bool show_info = false;                 ///< Afficher les informations du GPU et quitter
    bool quiet = false;                     ///< Mode silencieux : supprime les messages de progression

    bool use_cpu = false;                   ///< Forcer le rendu CPU avec OpenMP au lieu du GPU CUDA
    bool profile = false;                   ///< Afficher le rapport de timing detaille apres le rendu

    bool interactive = false;               ///< Activer le mode interactif avec fenetre OpenGL
    int interactive_spp = 1;                ///< Nombre de samples par frame en mode interactif (defaut : 1)
    int max_accumulated_spp = 1000;         ///< Nombre maximal de samples accumules en mode interactif (defaut : 1000)
    float mouse_sensitivity = 0.002f;       ///< Sensibilite de la souris pour le controle de la camera (defaut : 0.002)
    float move_speed = 5.0f;                ///< Vitesse de deplacement de la camera avec WASD (defaut : 5.0)
};

/**
 * @brief Affiche le message d'aide avec toutes les options disponibles
 * @details Liste toutes les options en ligne de commande avec leur description,
 *          leur format attendu et leur valeur par defaut. Inclut les options
 *          de rendu standard ainsi que les options du mode interactif.
 * @param program Nom du programme (argv[0]), utilise pour afficher la syntaxe d'utilisation
 */
inline void print_usage(const char* program) {
    std::cout << "CUDA Ray Tracer\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -w, --width <int>      Largeur image (defaut: 800)\n";
    std::cout << "  -h, --height <int>     Hauteur image (defaut: 600)\n";
    std::cout << "  -s, --samples <int>    Samples par pixel (defaut: 100)\n";
    std::cout << "  -d, --depth <int>      Rebonds max (defaut: 50)\n";
    std::cout << "  -o, --output <file>    Fichier sortie (defaut: output.png)\n";
    std::cout << "  --info                 Afficher info GPU\n";
    std::cout << "  --quiet                Mode silencieux\n";
    std::cout << "  --cpu                  Rendu CPU (OpenMP)\n";
    std::cout << "  --profile              Afficher timing detaille\n";
    std::cout << "  --help                 Afficher cette aide\n";
    std::cout << "\nInteractive mode:\n";
    std::cout << "  --interactive          Mode interactif (WASD + souris)\n";
    std::cout << "  --ispp <int>           Samples par frame interactif (defaut: 1)\n";
    std::cout << "  --max-spp <int>        Max samples accumules (defaut: 1000)\n";
    std::cout << "  --sensitivity <float>  Sensibilite souris (defaut: 0.002)\n";
    std::cout << "  --speed <float>        Vitesse deplacement (defaut: 5.0)\n";
}

/**
 * @brief Parse les arguments de la ligne de commande et retourne une structure Args
 * @details Parcourt les arguments argv un par un et remplit la structure Args
 *          en consequence. Les options non reconnues sont silencieusement ignorees.
 *          Si --help est detecte, l'aide est affichee et le programme se termine.
 * @param argc Nombre d'arguments passes au programme
 * @param argv Tableau de chaines de caracteres contenant les arguments
 * @return Structure Args remplie avec les valeurs parsees (ou les valeurs par defaut)
 */
inline Args parse_args(int argc, char** argv) {
    Args args;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        } else if (arg == "--info") {
            args.show_info = true;
        } else if (arg == "--quiet") {
            args.quiet = true;
        } else if (arg == "--cpu") {
            args.use_cpu = true;
        } else if (arg == "--profile") {
            args.profile = true;
        } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            args.width = std::stoi(argv[++i]);
        } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
            args.height = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--samples") && i + 1 < argc) {
            args.samples = std::stoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--depth") && i + 1 < argc) {
            args.depth = std::stoi(argv[++i]);
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            args.output_file = argv[++i];
        } else if (arg == "--interactive") {
            args.interactive = true;
        } else if (arg == "--ispp" && i + 1 < argc) {
            args.interactive_spp = std::stoi(argv[++i]);
        } else if (arg == "--max-spp" && i + 1 < argc) {
            args.max_accumulated_spp = std::stoi(argv[++i]);
        } else if (arg == "--sensitivity" && i + 1 < argc) {
            args.mouse_sensitivity = std::stof(argv[++i]);
        } else if (arg == "--speed" && i + 1 < argc) {
            args.move_speed = std::stof(argv[++i]);
        }
    }

    return args;
}

}

#endif
