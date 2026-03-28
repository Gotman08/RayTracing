#ifndef RAYTRACER_CORE_TIMER_CUH
#define RAYTRACER_CORE_TIMER_CUH

/** @file timer.cuh
 *  @brief Chronos CPU (std::chrono) et GPU (CUDA events) */

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

namespace rt {

/** @brief Chrono CPU multi-etapes avec rapport */
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double, std::milli>;

    /** @brief Entree chrono (nom + duree ms) */
    struct TimerEntry {
        std::string name;
        double duration_ms;
    };

    /** @brief Demarre une etape nommee */
    void start(const std::string& name) {
        current_name = name;
        start_time = Clock::now();
    }

    /** @brief Stop + enregistre la duree */
    void stop() {
        auto end_time = Clock::now();
        Duration duration = end_time - start_time;
        entries.push_back({current_name, duration.count()});
    }

    /** @brief Temps ecoule (ms) sans stopper */
    double elapsed_ms() const {
        auto now = Clock::now();
        Duration duration = now - start_time;
        return duration.count();
    }

    /** @brief Reset toutes les entrees */
    void reset() {
        entries.clear();
    }

    /** @brief Affiche le rapport (durees + pourcentages) */
    void print_report(const std::string& title = "Timing Report") const {
        if (entries.empty()) return;

        double total = 0;
        for (const auto& e : entries) {
            total += e.duration_ms;
        }

        std::cout << "\n=== " << title << " ===\n";
        std::cout << std::fixed << std::setprecision(3);

        for (const auto& e : entries) {
            double pct = (e.duration_ms / total) * 100.0;
            std::cout << "  " << std::setw(25) << std::left << e.name
                      << std::setw(10) << std::right << e.duration_ms << " ms"
                      << "  (" << std::setw(5) << std::right << pct << "%)\n";
        }

        std::cout << "  " << std::string(45, '-') << "\n";
        std::cout << "  " << std::setw(25) << std::left << "TOTAL"
                  << std::setw(10) << std::right << total << " ms\n";
        std::cout << "\n";
    }

    /** @brief Acces aux entrees */
    const std::vector<TimerEntry>& get_entries() const { return entries; }

private:
    std::string current_name;
    TimePoint start_time;
    std::vector<TimerEntry> entries;
};

#ifdef __CUDACC__
/** @brief Chrono GPU via CUDA events (RAII) */
class CudaTimer {
public:
    /** @brief Cree les events start/stop */
    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    /** @brief Detruit les events */
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    /** @brief Record event de debut */
    void start() {
        cudaEventRecord(start_event);
    }

    /** @brief Record event de fin + sync */
    void stop() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
    }

    /** @brief Temps ecoule en ms (precision GPU) */
    float elapsed_ms() const {
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }

private:
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
};
#endif

}

#endif
