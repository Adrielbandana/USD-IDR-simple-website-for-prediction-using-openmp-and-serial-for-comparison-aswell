#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using Window = std::array<double, 3>;
using Coefficients = std::array<double, 3>;
constexpr double differenceStep = 1.e-10;
constexpr double baseLearningRate = 0.0001;
constexpr double referenceWindowCount = 241.0;

struct TrainingStep {
    std::array<int, 6> threads{};
    double errorBefore = 0;
};

double squaredError(const std::vector<Window>& trainData, const Coefficients& coefficients)
{
    double sum = 0;
    for (const auto& [x1, x2, target] : trainData) {
        const double residual = target - x1 * coefficients[0] - x2 * coefficients[1] - coefficients[2];
        sum += residual * residual;
    }
    return sum;
}

template <bool ShowStep = false>
void train(const std::vector<Window>& trainData, Coefficients& coefficients,
           int iterations, int requestedThreads, int& workers, TrainingStep& step)
{
    std::array<double, 6> errors{};
    // Keep the original learning rate for the supplied dataset and scale it for larger inputs.
    const double windowCount = static_cast<double>(trainData.size());
    const double learningRate = baseLearningRate * referenceWindowCount / std::max(referenceWindowCount, windowCount);

#ifdef _OPENMP
    omp_set_dynamic(0);
    #pragma omp parallel num_threads(requestedThreads)
#else
    (void)requestedThreads;
    workers = 1;
#endif
    {
#ifdef _OPENMP
        #pragma omp single
        workers = omp_get_num_threads();
#endif
        for (int iteration = 0; iteration < iterations; ++iteration) {
            // The six central-difference loss evaluations share a read-only coefficient snapshot.
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int evaluation = 0; evaluation < 6; ++evaluation) {
                auto trialCoefficients = coefficients;
                const double offset = evaluation % 2 == 0 ? differenceStep : -differenceStep;
                trialCoefficients[evaluation / 2] += offset;
                errors[evaluation] = squaredError(trainData, trialCoefficients);
#ifdef _OPENMP
                if constexpr (ShowStep) {
                    if (iteration == iterations - 1) step.threads[evaluation] = omp_get_thread_num();
                }
#endif
            }
#ifdef _OPENMP
            #pragma omp single
#endif
            {
                if constexpr (ShowStep) {
                    if (iteration == iterations - 1) step.errorBefore = squaredError(trainData, coefficients);
                }
                for (int index = 0; index < 3; ++index) {
                    const double forwardError = errors[2 * index];
                    const double backwardError = errors[2 * index + 1];
                    const double derivative = (forwardError - backwardError) / (2 * differenceStep);
                    coefficients[index] -= learningRate * derivative;
                }
            }
        }
    }
}

int main(int argc, char* argv[])
{
    try {
        int count, iterations, steps, requestedThreads;
        if (!(std::cin >> count >> iterations >> steps >> requestedThreads)
            || count < 3 || count > 10000 || iterations < 1 || iterations > 20000
            || steps < 1 || steps > 60 || requestedThreads < 1 || requestedThreads > 6) {
            throw std::runtime_error("Invalid model settings.");
        }

        std::vector<double> normalized(count);
        for (double& value : normalized) {
            if (!(std::cin >> value) || !std::isfinite(value) || value <= 0) {
                throw std::runtime_error("Expected positive, finite exchange rates.");
            }
        }

        const double mean = std::accumulate(normalized.begin(), normalized.end(), 0.0) / count;
        double squaredDeviation = 0;
        for (double value : normalized) {
            squaredDeviation += (value - mean) * (value - mean);
        }
        const double deviation = std::sqrt(squaredDeviation / count);
        if (!std::isfinite(deviation) || deviation == 0) {
            throw std::runtime_error("The exchange rates must have finite, nonzero variation.");
        }
        for (double& value : normalized) {
            value = (value - mean) / deviation;
        }

        std::vector<Window> trainData;
        trainData.reserve(count - 2);
        for (int index = 0; index < count - 2; ++index) {
            trainData.push_back({normalized[index], normalized[index + 1], normalized[index + 2]});
        }

        Coefficients coefficients{0.1, 0.1, 0.1};
        int workers = 1;
        TrainingStep step;
        const bool showStep = argc == 2 && std::string_view(argv[1]) == "--demo";
        const auto start = std::chrono::steady_clock::now();
        if (showStep) {
            train<true>(trainData, coefficients, iterations, requestedThreads, workers, step);
        } else {
            train(trainData, coefficients, iterations, requestedThreads, workers, step);
        }
        const std::chrono::duration<double, std::milli> elapsed = std::chrono::steady_clock::now() - start;

        const double errorAfter = squaredError(trainData, coefficients);
        const double rmse = std::sqrt(errorAfter / trainData.size()) * deviation;
        if (!std::isfinite(rmse)) {
            throw std::runtime_error("Training did not produce a finite result.");
        }

        if (showStep) {
            std::cout << std::setprecision(17) << "{\"workers\":" << workers
                      << ",\"error_before\":" << step.errorBefore << ",\"error_after\":" << errorAfter
                      << ",\"threads\":[";
            for (std::size_t index = 0; index < step.threads.size(); ++index) {
                if (index > 0) std::cout << ',';
                std::cout << step.threads[index];
            }
            std::cout << "]}\n";
            return 0;
        }

        std::vector<double> forecast;
        double x1 = normalized[count - 2];
        double x2 = normalized[count - 1];
        for (int step = 0; step < steps; ++step) {
            const double next = coefficients[0] * x1 + coefficients[1] * x2 + coefficients[2];
            const double rate = next * deviation + mean;
            if (!std::isfinite(rate)) {
                throw std::runtime_error("The forecast became unstable.");
            }
            forecast.push_back(rate);
            x1 = x2;
            x2 = next;
        }

        std::cout << std::setprecision(17)
                  << "{\"training_ms\":" << elapsed.count() << ",\"workers\":" << workers
                  << ",\"rmse\":" << rmse << ",\"forecast\":[";
        for (int index = 0; index < steps; ++index) {
            if (index > 0) std::cout << ',';
            std::cout << forecast[index];
        }
        std::cout << "]}\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
