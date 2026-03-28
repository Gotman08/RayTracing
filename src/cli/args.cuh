/** @file args.cuh
 * @brief Parsing des args CLI du ray tracer */

#ifndef RAYTRACER_CLI_ARGS_CUH
#define RAYTRACER_CLI_ARGS_CUH

#include <iostream>
#include <string>
#include <cstdlib>

namespace rt {

/** @brief Args CLI : resolution, samples, profondeur, mode CPU/GPU, etc. */
struct Args {
    int width = 800;
    int height = 600;
    int samples = 100;
    int depth = 50;
    std::string output_file = "output/output.png";
    bool show_info = false;
    bool quiet = false;

    bool use_cpu = false;
    bool profile = false;
    bool benchmark = false;

};

/** @brief Affiche l'aide CLI */
inline void print_usage(const char* program) {
    std::cout << "CUDA Ray Tracer\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -w, --width <int>      Largeur image (defaut: 800)\n";
    std::cout << "  -h, --height <int>     Hauteur image (defaut: 600)\n";
    std::cout << "  -s, --samples <int>    Samples par pixel (defaut: 100)\n";
    std::cout << "  -d, --depth <int>      Rebonds max (defaut: 50)\n";
    std::cout << "  -o, --output <file>    Fichier sortie (defaut: output/output.png)\n";
    std::cout << "  --info                 Afficher info GPU\n";
    std::cout << "  --quiet                Mode silencieux\n";
    std::cout << "  --cpu                  Rendu CPU (OpenMP)\n";
    std::cout << "  --profile              Afficher timing detaille\n";
    std::cout << "  --benchmark            Benchmark multi-resolution (4 res., 50 SPP)\n";
    std::cout << "  --help                 Afficher cette aide\n";
}

/** @brief Parse argc/argv et retourne un Args rempli */
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
        } else if (arg == "--benchmark") {
            args.benchmark = true;
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
        }
    }

    return args;
}

}

#endif
