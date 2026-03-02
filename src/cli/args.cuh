/**
 * Command Line Argument Parsing
 */

#ifndef RAYTRACER_CLI_ARGS_CUH
#define RAYTRACER_CLI_ARGS_CUH

#include <iostream>
#include <string>
#include <cstdlib>

namespace rt {

struct Args {
    int width = 800;
    int height = 600;
    int samples = 100;
    int depth = 50;
    std::string scene_file = "";
    std::string output_file = "output.png";
    std::string hdri_file = "";
    bool show_info = false;
    bool quiet = false;
    float exposure = 0.0f;
};

inline void print_usage(const char* program) {
    std::cout << "CUDA Ray Tracer v1.0 - Optimized for Romeo2025 (GH200/H100)\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -w, --width <int>      Image width (default: 800)\n";
    std::cout << "  -h, --height <int>     Image height (default: 600)\n";
    std::cout << "  -s, --samples <int>    Samples per pixel (default: 100)\n";
    std::cout << "  -d, --depth <int>      Max ray bounces (default: 50)\n";
    std::cout << "  -i, --scene <file>     Scene file (JSON format)\n";
    std::cout << "  -o, --output <file>    Output file (default: output.png)\n";
    std::cout << "  --hdri <file>          HDR environment map\n";
    std::cout << "  --exposure <float>     Exposure adjustment (default: 0.0)\n";
    std::cout << "  --info                 Display GPU information\n";
    std::cout << "  --quiet                Suppress progress output\n";
    std::cout << "  --help                 Show this help\n";
}

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
        } else if ((arg == "-w" || arg == "--width") && i + 1 < argc) {
            args.width = std::stoi(argv[++i]);
        } else if ((arg == "-h" || arg == "--height") && i + 1 < argc) {
            args.height = std::stoi(argv[++i]);
        } else if ((arg == "-s" || arg == "--samples") && i + 1 < argc) {
            args.samples = std::stoi(argv[++i]);
        } else if ((arg == "-d" || arg == "--depth") && i + 1 < argc) {
            args.depth = std::stoi(argv[++i]);
        } else if ((arg == "-i" || arg == "--scene") && i + 1 < argc) {
            args.scene_file = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            args.output_file = argv[++i];
        } else if (arg == "--hdri" && i + 1 < argc) {
            args.hdri_file = argv[++i];
        } else if (arg == "--exposure" && i + 1 < argc) {
            args.exposure = std::stof(argv[++i]);
        }
    }

    return args;
}

} // namespace rt

#endif // RAYTRACER_CLI_ARGS_CUH
