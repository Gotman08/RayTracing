#ifndef RAYTRACER_CORE_TIMER_CUH
#define RAYTRACER_CORE_TIMER_CUH

/**
 * @file timer.cuh
 * @brief Classes de chronometrage pour mesurer les performances CPU et GPU
 * @details Ce fichier fournit deux classes de mesure du temps :
 *          - Timer : utilise std::chrono pour le chronometrage cote CPU
 *          - CudaTimer : utilise les CUDA events pour le chronometrage precis cote GPU
 *          Ces outils permettent de profiler les differentes etapes du rendu
 *          et de generer des rapports de performance detailles.
 */

#include <chrono>
#include <string>
#include <vector>
#include <iostream>
#include <iomanip>

namespace rt {

/**
 * @class Timer
 * @brief Chronometre CPU pour mesurer le temps d'execution des differentes etapes
 * @details Cette classe permet de chronometrer plusieurs etapes successives
 *          et de les enregistrer avec un nom. A la fin, on peut afficher un rapport
 *          complet avec la duree de chaque etape et son pourcentage du temps total.
 *          Elle utilise std::chrono::high_resolution_clock pour une precision maximale.
 */
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;   ///< Horloge haute resolution
    using TimePoint = Clock::time_point;                ///< Point dans le temps
    using Duration = std::chrono::duration<double, std::milli>; ///< Duree en millisecondes

    /**
     * @brief Structure representant une entree de chronometrage
     * @details Stocke le nom de l'etape et sa duree en millisecondes.
     */
    struct TimerEntry {
        std::string name;   ///< Nom de l'etape chronometree
        double duration_ms; ///< Duree de l'etape en millisecondes
    };

    /**
     * @brief Demarre le chronometrage d'une nouvelle etape
     * @param name Le nom de l'etape a chronometrer (ex : "Construction BVH")
     */
    void start(const std::string& name) {
        current_name = name;
        start_time = Clock::now();
    }

    /**
     * @brief Arrete le chronometrage et enregistre l'etape
     * @details Calcule la duree ecoulee depuis le dernier appel a start()
     *          et ajoute une entree dans la liste avec le nom et la duree.
     */
    void stop() {
        auto end_time = Clock::now();
        Duration duration = end_time - start_time;
        entries.push_back({current_name, duration.count()});
    }

    /**
     * @brief Retourne le temps ecoule depuis le dernier start() sans arreter le chrono
     * @details Utile pour obtenir un temps intermediaire pendant qu'une etape
     *          est encore en cours d'execution.
     * @return Le temps ecoule en millisecondes
     */
    double elapsed_ms() const {
        auto now = Clock::now();
        Duration duration = now - start_time;
        return duration.count();
    }

    /**
     * @brief Reinitialise le chronometre en supprimant toutes les entrees
     */
    void reset() {
        entries.clear();
    }

    /**
     * @brief Affiche un rapport detaille des temps mesures
     * @details Affiche un tableau formate avec le nom de chaque etape,
     *          sa duree en millisecondes et son pourcentage du temps total.
     *          Une ligne de total est ajoutee a la fin.
     * @param title Titre du rapport (defaut : "Timing Report")
     */
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

    /**
     * @brief Retourne la liste de toutes les entrees enregistrees
     * @return Reference constante vers le vecteur d'entrees TimerEntry
     */
    const std::vector<TimerEntry>& get_entries() const { return entries; }

private:
    std::string current_name;        ///< Nom de l'etape en cours de chronometrage
    TimePoint start_time;            ///< Instant de debut de l'etape courante
    std::vector<TimerEntry> entries;  ///< Historique de toutes les etapes chronometrees
};

#ifdef __CUDACC__
/**
 * @class CudaTimer
 * @brief Chronometre GPU utilisant les CUDA events pour une mesure precise
 * @details Les CUDA events permettent de mesurer le temps d'execution
 *          des operations GPU avec une grande precision, car la mesure
 *          est faite directement par le GPU (et non par le CPU). C'est
 *          la methode recommandee pour profiler les kernels CUDA.
 *          Le constructeur cree les events et le destructeur les libere
 *          automatiquement (RAII).
 */
class CudaTimer {
public:
    /**
     * @brief Constructeur qui cree les deux CUDA events (debut et fin)
     */
    CudaTimer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    /**
     * @brief Destructeur qui libere les CUDA events
     */
    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    /**
     * @brief Demarre la mesure du temps GPU
     * @details Enregistre un event dans le stream CUDA courant.
     *          Toutes les operations GPU lancees apres cet appel
     *          seront chronometrees.
     */
    void start() {
        cudaEventRecord(start_event);
    }

    /**
     * @brief Arrete la mesure du temps GPU
     * @details Enregistre l'event de fin et synchronise le CPU
     *          pour attendre que le GPU ait termine toutes les operations.
     */
    void stop() {
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
    }

    /**
     * @brief Calcule le temps ecoule entre start() et stop()
     * @details Utilise cudaEventElapsedTime pour calculer la duree
     *          entre les deux events avec la precision du GPU.
     * @return Le temps ecoule en millisecondes
     */
    float elapsed_ms() const {
        float ms = 0;
        cudaEventElapsedTime(&ms, start_event, stop_event);
        return ms;
    }

private:
    cudaEvent_t start_event; ///< Event CUDA marquant le debut de la mesure
    cudaEvent_t stop_event;  ///< Event CUDA marquant la fin de la mesure
};
#endif

}

#endif
