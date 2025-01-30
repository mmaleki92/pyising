// #include "indicators.hpp"

// class ParallelProgress {
//     std::vector<indicators::ProgressBar> bars;
//     std::mutex mtx;

// public:
//     ParallelProgress(size_t count) {
//         for (size_t i = 0; i < count; ++i) {
//             bars.emplace_back(indicators::option::BarWidth{50},
//                 indicators::option::Start{"["},
//                 indicators::option::Fill{"█"},
//                 indicators::option::Lead{"█"},
//                 indicators::option::Remainder{"-"},
//                 indicators::option::End{"]"},
//                 indicators::option::ShowPercentage{true},
//                 indicators::option::ShowElapsedTime{true},
//                 indicators::option::ShowRemainingTime{true},
//                 indicators::option::PrefixText{"T" + std::to_string(i+1)});
//         }
//     }

//     void update(size_t index, float progress) {
//         std::lock_guard<std::mutex> lock(mtx);
//         if (index < bars.size()) {
//             bars[index].set_progress(progress * 100);
//         }
//     }

//     void complete() {
//         for (auto& bar : bars) {
//             bar.set_progress(100);
//             bar.mark_as_completed();
//         }
//     }
// };
