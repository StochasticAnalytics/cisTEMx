/*
 * Modern C++ implementation of Powell's conjugate direction minimizer (VA04A).
 *
 * This is a readable rewrite of the f2c-translated va04.cpp. The mathematical
 * behavior is intended to be identical: same algorithm, same convergence
 * criteria, same direction replacement heuristic. The difference is purely
 * structural: structured control flow, named variables, RAII, and std::vector.
 *
 * Algorithm: Powell's method (1964) minimizes a function of n variables without
 * derivatives. It maintains n search directions (initially coordinate axes),
 * performs line minimization along each using parabolic interpolation, and
 * replaces the direction of greatest improvement with the overall displacement.
 *
 * Note: Despite the legacy class name "ConjugateGradient", this is NOT a
 * conjugate gradient method (which requires derivatives). It is a conjugate
 * DIRECTION method — derivative-free.
 */

#ifndef _SRC_CORE_CONJUGATE_GRADIENT_REFACTOR2026_H_
#define _SRC_CORE_CONJUGATE_GRADIENT_REFACTOR2026_H_

// cisTEM_ENABLE_CG_REFACTOR_2026 — Feature gate for the CG refactor experiment.
// Comment out the following line to disable the refactored PowellConjugateGradient
// class project-wide. The .cpp compiles to empty and all dependent tests are skipped.
// To find all gated code: grep -rn cisTEM_ENABLE_CG_REFACTOR_2026
#define cisTEM_ENABLE_CG_REFACTOR_2026

#ifdef cisTEM_ENABLE_CG_REFACTOR_2026

// cisTEM_USE_CG_REFACTOR_2026 — Routes production callers (refine3d, ctffind,
// find_dqe) through the refactored optimizer instead of the original va04a.
// Requires cisTEM_ENABLE_CG_REFACTOR_2026 above. Commented out by default.
// To find all call-site ifdefs: grep -rn cisTEM_USE_CG_REFACTOR_2026
#define cisTEM_USE_CG_REFACTOR_2026

#include <functional>
#include <vector>

class PowellConjugateGradient {

  public:
    using ObjectiveFunction = std::function<float(float*)>;

    PowellConjugateGradient( );
    ~PowellConjugateGradient( ) = default;

    // Modern interface: pass any callable (lambda, functor, etc.)
    float Init(ObjectiveFunction objective,
               int               num_dimensions,
               const float       starting_values[],
               const float       accuracy[],
               float             escale = 100.0f);

    // Legacy interface: function pointer + void* context (backward compatible)
    float Init(float (*function_to_minimize)(void* parameters, float[]),
               void*       parameters,
               int         num_dimensions,
               const float starting_values[],
               const float accuracy[],
               float       escale = 100.0f);

    // Run the minimization. Returns the best score found.
    float Run(int max_iterations = 50);

    // Accessors (matching ConjugateGradient API)
    inline float GetBestValue(int index) const { return best_values_[index]; }

    inline float GetBestScore( ) const { return static_cast<float>(best_score_); }

    inline float* GetPointerToBestValues( ) { return best_values_.data( ); }

    inline int GetFunctionCallCount( ) const { return function_call_count_; }

  private:
    // --- Algorithm phases ---

    // Set direction matrix to scaled identity, initialize scales
    void InitializeDirections( );

    // Evaluate objective at current_position_, with call-count safety limit.
    // Inlined: hot path, called on every function evaluation.
    double EvaluateObjective( ) {
        if ( safety_limit_reached_ ) {
            return best_score_;
        }

        function_call_count_++;
        if ( function_call_count_ > max_function_calls_ ) {
            safety_limit_reached_ = true;
            return best_score_;
        }

        // Optimization 2: bypass std::function dispatch for legacy callers
        if ( use_legacy_path_ )
            return static_cast<double>(legacy_function_(legacy_params_, float_buffer_.data( )));
        return static_cast<double>(objective_(float_buffer_.data( )));
    }

    // Move current_position_ along direction[dir_index] by step.
    // Inlined: hot path, called on every line search probe.
    // Optimization 1: incrementally syncs float_buffer_ alongside position update.
    void MoveAlongDirection(int dir_index, double step) {
        int offset = dir_index * num_dimensions_;
        for ( int i = 0; i < num_dimensions_; i++ ) {
            current_position_[i] += step * direction_matrix_[offset + i];
            float_buffer_[i] = static_cast<float>(current_position_[i]);
        }
    }

    // Line search result
    struct LineSearchResult {
        double best_step; // Step at which the minimum was found (di)
        double best_value; // Function value at the minimum (fi)
        bool   terminated; // True if the entire optimization should stop
    };

    // Perform parabolic-interpolation line search along direction dir_index.
    // mode: 1 = normal sweep, 3 = direction replacement
    // For mode 3, func_at_cycle_start and func_after_sweep provide the
    // bracket endpoints (the optimization reuses prior evaluations).
    LineSearchResult PerformLineSearch(int    dir_index,
                                       int    mode,
                                       double func_at_cycle_start = 0.0,
                                       double func_after_sweep    = 0.0);

    // Remove direction at the given index, shifting subsequent directions back
    void ShiftDirectionsRemove(int direction_to_remove);

    // Convert current_position_ (double) to float_buffer_ for objective calls.
    // Inlined: retained for rare bulk-sync cases (Init, safety_exit).
    void UpdateFloatBuffer( ) {
        for ( int i = 0; i < num_dimensions_; i++ ) {
            float_buffer_[i] = static_cast<float>(current_position_[i]);
        }
    }

    // Copy current_position_ to best_values_ (double -> float).
    // Inlined: called at convergence/exit points.
    void UpdateBestValues( ) {
        for ( int i = 0; i < num_dimensions_; i++ ) {
            best_values_[i] = static_cast<float>(current_position_[i]);
        }
    }

    // --- Data ---

    ObjectiveFunction objective_;
    int               num_dimensions_;
    int               function_call_count_;
    int               max_function_calls_;
    bool              is_initialized_;
    bool              safety_limit_reached_;

    // Optimization 2: direct legacy function pointer to bypass std::function dispatch
    float (*legacy_function_)(void*, float[]);
    void* legacy_params_;
    bool  use_legacy_path_;

    // Current function value (evolves during optimization; final value on return)
    double best_score_;
    double escale_;

    // Algorithm parameters (evolve during optimization)
    double ddmag_; // Step magnitude control (adapts each iteration)
    double scer_; // Convergence scaling: 0.05 / escale
    int    gradient_mode_; // 1 = gradient parabolic, 2 = standard (replaces isgrad)

    // Working storage
    std::vector<double> current_position_; // n elements: current parameter values
    std::vector<double> initial_position_; // n elements: starting values (safety restore)
    std::vector<double> accuracy_; // n elements: per-dimension accuracy
    std::vector<double> direction_scales_; // n elements: max step per direction line
    std::vector<double> direction_matrix_; // n*n elements: direction vectors (row-major)
    std::vector<double> saved_position_; // n elements: position at start of sweep
    std::vector<double> gradient_workspace_; // n elements: auxiliary for gradient phase
    std::vector<float>  best_values_; // n elements: float copy for external API
    std::vector<float>  float_buffer_; // n elements: reusable buffer for objective
};

#endif // cisTEM_ENABLE_CG_REFACTOR_2026

#endif // _SRC_CORE_CONJUGATE_GRADIENT_REFACTOR2026_H_
