/*
 * Modern C++ implementation of Powell's VA04A conjugate direction minimizer (2026 refactor).
 * See conjugate_gradient_refactor2026.h for algorithm description.
 *
 * This file is a structured translation of va04.cpp. Comments reference the
 * original label numbers (e.g., "// [L7]") for traceability during comparison
 * testing. Once the refactored code is validated, these references can be
 * removed.
 */

#include "core_headers.h"

#ifdef cisTEM_ENABLE_CG_REFACTOR_2026

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

// ============================================================================
// Construction and initialization
// ============================================================================

PowellConjugateGradient::PowellConjugateGradient( )
    : num_dimensions_(0),
      function_call_count_(0),
      max_function_calls_(0),
      is_initialized_(false),
      safety_limit_reached_(false),
      legacy_function_(nullptr),
      legacy_params_(nullptr),
      use_legacy_path_(false),
      best_score_(std::numeric_limits<float>::max( )),
      escale_(0.0),
      ddmag_(0.0),
      scer_(0.0),
      gradient_mode_(2) {
}

float PowellConjugateGradient::Init(ObjectiveFunction objective,
                                    int               num_dimensions,
                                    const float       starting_values[],
                                    const float       accuracy[],
                                    float             escale) {
    MyDebugAssertTrue(num_dimensions > 0,
                      "Initializing PowellConjugateGradient with zero dimensions");

    objective_       = std::move(objective);
    num_dimensions_  = num_dimensions;
    escale_          = static_cast<double>(escale);
    use_legacy_path_ = false;

    // Allocate and initialize storage
    current_position_.resize(num_dimensions_);
    initial_position_.resize(num_dimensions_);
    accuracy_.resize(num_dimensions_);
    direction_scales_.resize(num_dimensions_);
    direction_matrix_.resize(num_dimensions_ * num_dimensions_);
    saved_position_.resize(num_dimensions_);
    gradient_workspace_.resize(num_dimensions_, 0.0);
    best_values_.resize(num_dimensions_);
    float_buffer_.resize(num_dimensions_);

    for ( int i = 0; i < num_dimensions_; i++ ) {
        current_position_[i] = static_cast<double>(starting_values[i]);
        initial_position_[i] = current_position_[i];
        float_buffer_[i]     = starting_values[i];
        accuracy_[i]         = static_cast<double>(accuracy[i]);
    }

    // Derived constants [va04.cpp lines 108-109]
    ddmag_ = escale_ * 0.1;
    scer_  = 0.05 / escale_;

    // Initialize direction set to scaled identity [va04.cpp lines 120-138]
    InitializeDirections( );

    // Algorithm state
    gradient_mode_        = 2; // standard (no gradient) initially
    function_call_count_  = 0;
    max_function_calls_   = std::numeric_limits<int>::max( ); // Until Run() sets proper limit
    safety_limit_reached_ = false;
    is_initialized_       = true;

    // Evaluate at starting point [va04.cpp lines 143-147]
    best_score_ = EvaluateObjective( );
    UpdateBestValues( );

    return static_cast<float>(best_score_);
}

float PowellConjugateGradient::Init(
        float (*function_to_minimize)(void* parameters, float[]),
        void*       parameters,
        int         num_dimensions,
        const float starting_values[],
        const float accuracy[],
        float       escale) {
    // Store raw pointer for direct dispatch (Optimization 2)
    legacy_function_ = function_to_minimize;
    legacy_params_   = parameters;

    // Wrap the C-style function pointer + void* into a std::function via lambda
    // (retained for API compatibility; the hot path bypasses this via use_legacy_path_)
    void* params_copy = parameters;
    auto  wrapper     = [function_to_minimize, params_copy](float* x) -> float {
        return function_to_minimize(params_copy, x);
    };
    // Note: Init() below sets use_legacy_path_ = false; override it after the call.
    float result     = Init(ObjectiveFunction(wrapper), num_dimensions, starting_values,
                            accuracy, escale);
    use_legacy_path_ = true;
    return result;
}

// ============================================================================
// Internal helpers
// ============================================================================

void PowellConjugateGradient::InitializeDirections( ) {
    // Direction matrix = identity scaled by |accuracy[i]| on diagonal
    // Direction scales = escale for all directions
    // [va04.cpp lines 120-138]
    std::fill(direction_matrix_.begin( ), direction_matrix_.end( ), 0.0);
    for ( int i = 0; i < num_dimensions_; i++ ) {
        direction_matrix_[i * num_dimensions_ + i] = std::abs(accuracy_[i]);
        direction_scales_[i]                       = escale_;
    }
}

void PowellConjugateGradient::ShiftDirectionsRemove(int dir_to_remove) {
    // Shift directions [dir_to_remove+1 .. n-1] back by one, removing dir_to_remove.
    // Also shift direction_scales_ correspondingly.
    // [va04.cpp labels L60/L62/L97: lines 506-517]
    int n = num_dimensions_;

    // Shift direction matrix rows
    for ( int d = dir_to_remove; d < n - 1; d++ ) {
        for ( int i = 0; i < n; i++ ) {
            direction_matrix_[d * n + i] = direction_matrix_[(d + 1) * n + i];
        }
    }

    // Shift direction scales
    for ( int d = dir_to_remove; d < n - 1; d++ ) {
        direction_scales_[d] = direction_scales_[d + 1];
    }
}

// ============================================================================
// Line search: parabolic interpolation along a single direction
// ============================================================================

PowellConjugateGradient::LineSearchResult
PowellConjugateGradient::PerformLineSearch(int    dir_index,
                                           int    mode,
                                           double func_at_cycle_start,
                                           double func_after_sweep) {
    // Local abbreviations
    int    n         = num_dimensions_;
    double max_step  = direction_scales_[dir_index]; // dmax [L7: line 162]
    double step_acc  = max_step * scer_; // dacc [L7: line 163]
    double step_mag  = std::min(ddmag_, max_step * 0.1); // dmag [L7: lines 165-166]
    step_mag         = std::max(step_mag, step_acc * 20.0); // [L7: lines 168-169]
    double max_trial = step_mag * 10.0; // ddmax [L7: line 170]

    // Bracket: three points (step, value) for parabolic interpolation
    double step_a, val_a;
    double step_b, val_b;
    double step_c, val_c;

    // Track position along the direction line (incremental moves)
    double previous_step = 0.0;

    // Parabolic fit coefficients (reused at finalization)
    double parabolic_a = 0.0, parabolic_b = 0.0;

    // Best bracket point (set during refinement, used at finalization)
    double final_step = 0.0, final_value = best_score_;

    // =======================================================================
    // Phase 1: Build initial 3-point bracket
    // =======================================================================

    if ( mode != 3 ) {
        // --- Normal initialization [L70: lines 176-182] ---
        previous_step     = 0.0;
        double trial_step = step_mag;
        val_a             = best_score_;
        step_a            = 0.0;

        // First probe [is=5 -> L14]
        double dd     = trial_step - previous_step;
        previous_step = trial_step;
        MoveAlongDirection(dir_index, dd);
        double probe_val = EvaluateObjective( );
        if ( safety_limit_reached_ ) {
            return {0.0, best_score_, true};
        }

        // Classify first probe [L14: lines 211-219]
        if ( probe_val == val_a ) {
            // Function unchanged — double step until something changes [L16-L18]
            while ( std::abs(trial_step) <= max_step ) {
                trial_step *= 2.0; // [L17: line 228]
                dd            = trial_step - previous_step;
                previous_step = trial_step;
                MoveAlongDirection(dir_index, dd);
                probe_val = EvaluateObjective( );
                if ( safety_limit_reached_ )
                    return {0.0, best_score_, true};
                if ( probe_val != val_a )
                    break;
            }
            if ( probe_val == val_a ) {
                // [L18] "Maximum change does not alter function" -> terminate
                best_score_ = probe_val;
                return {0.0, probe_val, true};
            }
        }

        if ( probe_val < val_a ) {
            // [L15] Better: new point becomes b
            step_b = trial_step;
            val_b  = probe_val;
        }
        else {
            // [L24] Worse or first-time-different: swap so b is the better point
            step_b = step_a;
            val_b  = val_a;
            step_a = trial_step;
            val_a  = probe_val;
        }

        // Second probe — depends on gradient_mode_ [L21: lines 246-249]
        if ( gradient_mode_ == 2 ) {
            // Standard mode [L23]: reflected point
            double d2     = 2.0 * step_b - step_a; // [L23: line 251]
            dd            = d2 - previous_step;
            previous_step = d2;
            MoveAlongDirection(dir_index, dd);
            double probe2 = EvaluateObjective( );
            if ( safety_limit_reached_ )
                return {0.0, best_score_, true};
            step_c = d2;
            val_c  = probe2;
            // -> fall through to refinement loop (3 points ready)
        }
        else {
            // Gradient mode [L83]: parabolic interpolant from 2 points
            double denom    = step_a - step_b;
            double d_interp = (step_a + step_b -
                               (val_a - val_b) / denom) *
                              0.5; // [L83: line 255]

            if ( (step_a - d_interp) * (d_interp - step_b) >= 0.0 ) {
                // Interpolant is between step_a and step_b — probe there [is=4]
                dd            = d_interp - previous_step;
                previous_step = d_interp;
                MoveAlongDirection(dir_index, dd);
                double probe_interp = EvaluateObjective( );
                if ( safety_limit_reached_ )
                    return {0.0, best_score_, true};

                if ( probe_interp >= val_a ) {
                    // [L13->L23] Worse than best: fall back to standard reflection
                    double d2     = 2.0 * step_b - step_a;
                    dd            = d2 - previous_step;
                    previous_step = d2;
                    MoveAlongDirection(dir_index, dd);
                    double probe2 = EvaluateObjective( );
                    if ( safety_limit_reached_ )
                        return {0.0, best_score_, true};
                    step_c = d2;
                    val_c  = probe2;
                }
                else {
                    // [L13->L28->L29] Better: update bracket
                    step_c = step_b;
                    val_c  = val_b;
                    step_b = d_interp;
                    val_b  = probe_interp;
                }
            }
            else {
                // [L83->L25] Interpolant outside bracket
                if ( std::abs(d_interp - step_b) <= max_trial ) {
                    // [L25->L8, is=1] Within range: probe at interpolant
                    dd            = d_interp - previous_step;
                    previous_step = d_interp;
                    MoveAlongDirection(dir_index, dd);
                    double probe2 = EvaluateObjective( );
                    if ( safety_limit_reached_ )
                        return {0.0, best_score_, true};
                    step_c = d_interp;
                    val_c  = probe2;
                }
                else {
                    // [L25->L26] Expand search
                    double expand_step = step_b +
                                         std::copysign(max_trial, step_b - step_a);
                    max_trial *= 2.0; // [L26: line 275]
                    ddmag_ *= 2.0; // [L26: line 276]
                    if ( max_trial > max_step )
                        max_trial = max_step; // [L27]
                    dd            = expand_step - previous_step;
                    previous_step = expand_step;
                    MoveAlongDirection(dir_index, dd);
                    double probe2 = EvaluateObjective( );
                    if ( safety_limit_reached_ )
                        return {0.0, best_score_, true};
                    step_c = expand_step;
                    val_c  = probe2;
                }
            }
        }
    }
    else {
        // --- Direction replacement initialization [L71: lines 322-329] ---
        // Three bracket points from prior evaluations (no new function calls):
        //   step = -1: position at start of cycle (value = func_at_cycle_start)
        //   step =  0: position at end of sweep    (value = func_after_sweep)
        //   step = +1: extrapolated position        (value = current best_score_)
        previous_step = 1.0;
        max_trial     = 5.0;
        step_a        = -1.0;
        val_a         = func_at_cycle_start;
        step_b        = 0.0;
        val_b         = func_after_sweep;
        step_c        = 1.0;
        val_c         = best_score_;
    }

    // =======================================================================
    // Phase 2: Parabolic refinement loop [L30-L49]
    // =======================================================================

    bool skip_convergence = (mode == 3); // First pass in replacement mode skips check

    while ( true ) {
        // Three-point parabolic fit [L30: lines 334-335]
        parabolic_a = (step_b - step_c) * (val_a - val_c);
        parabolic_b = (step_c - step_a) * (val_b - val_c);

        if ( (parabolic_a + parabolic_b) * (step_a - step_c) <= 0.0 ) {
            // [L33] Invalid fit: rearrange bracket and expand
            step_a = step_b;
            val_a  = val_b;
            step_b = step_c;
            val_b  = val_c;

            // [L26] Expand: probe at expanded point
            double expand_step = step_b +
                                 std::copysign(max_trial, step_b - step_a);
            max_trial *= 2.0;
            ddmag_ *= 2.0;
            if ( max_trial > max_step )
                max_trial = max_step;

            double dd     = expand_step - previous_step;
            previous_step = expand_step;
            MoveAlongDirection(dir_index, dd);
            val_c  = EvaluateObjective( );
            step_c = expand_step;
            if ( safety_limit_reached_ )
                return {0.0, best_score_, true};
            continue;
        }

        // Valid fit: compute parabolic minimum [L34: line 349]
        double d_min = (parabolic_a * (step_b + step_c) +
                        parabolic_b * (step_a + step_c)) *
                       0.5 / (parabolic_a + parabolic_b);

        // Best of the bracket points. va04 [L44] selects only between {b, c}
        // for the convergence target, relying on the invariant that endpoint
        // a never holds the lowest value (val_a >= val_b in a well-formed
        // bracket). The "invalid fit" branch above [L33] violates that
        // invariant: its a<-b, b<-c rearrangement can move the lowest point
        // into a. When the parabolic minimum d_min then converges onto a, a
        // {b, c}-only check cannot see it, the probe regenerates a coincident
        // point, and the search cycles without terminating. va04 escapes this
        // only by float rounding keeping bracket values unequal; the
        // double-precision position accumulation here can reach exact equality
        // and expose the latent cycle. Including a in the selection closes it:
        // for a well-formed bracket b is still chosen (no behavior change),
        // and the degenerate case converges at the true minimum carried in a.
        double best_bracket_step = step_b, best_bracket_val = val_b;
        if ( val_c < best_bracket_val ) {
            best_bracket_step = step_c;
            best_bracket_val  = val_c;
        }
        if ( val_a < best_bracket_val ) {
            best_bracket_step = step_a;
            best_bracket_val  = val_a;
        }

        // Mode handling [L85/L86: lines 362-383]
        if ( skip_convergence ) {
            // [L85] Direction replacement first pass: skip convergence check
            skip_convergence = false; // itone = 2
            // Fall through to probing
        }
        else {
            // [L86] Normal convergence check
            double gap = std::abs(d_min - best_bracket_step);
            if ( gap <= step_acc ||
                 gap <= std::abs(d_min) * 0.03 ) {
                // Converged [L41]
                final_step  = best_bracket_step;
                final_value = best_bracket_val;
                break;
            }
        }

        // Check bracket arrangement [L45: lines 385-390]
        if ( (step_a - step_c) * (step_c - d_min) >= 0.0 ) {
            // [L46] Bad arrangement: rearrange, try d_min or expand
            step_a = step_b;
            val_a  = val_b;
            step_b = step_c;
            val_b  = val_c;

            // [L25] Check if d_min is within range
            if ( std::abs(d_min - step_b) <= max_trial ) {
                // Probe at d_min [is=1 -> L10]
                double dd     = d_min - previous_step;
                previous_step = d_min;
                MoveAlongDirection(dir_index, dd);
                val_c  = EvaluateObjective( );
                step_c = d_min;
                if ( safety_limit_reached_ )
                    return {0.0, best_score_, true};
            }
            else {
                // [L26] Expand
                double expand_step = step_b +
                                     std::copysign(max_trial, step_b - step_a);
                max_trial *= 2.0;
                ddmag_ *= 2.0;
                if ( max_trial > max_step )
                    max_trial = max_step;

                double dd     = expand_step - previous_step;
                previous_step = expand_step;
                MoveAlongDirection(dir_index, dd);
                val_c  = EvaluateObjective( );
                step_c = expand_step;
                if ( safety_limit_reached_ )
                    return {0.0, best_score_, true};
            }
            continue;
        }

        // Probe at the parabolic minimum [L47/L48: lines 397-407]
        double dd     = d_min - previous_step;
        previous_step = d_min;
        MoveAlongDirection(dir_index, dd);
        double probe_val = EvaluateObjective( );
        if ( safety_limit_reached_ )
            return {0.0, best_score_, true};

        // Update bracket based on where d_min fell relative to {b, c}
        if ( (step_b - d_min) * (d_min - step_c) >= 0.0 ) {
            // [L47, is=2 -> L11] d_min is between step_b and step_c
            if ( probe_val >= val_b ) {
                // [L11->L10] Worse than center: replace c
                step_c = d_min;
                val_c  = probe_val;
            }
            else {
                // [L11->L32->L29] Better: a<-b, b<-probe
                step_a = step_b;
                val_a  = val_b;
                step_b = d_min;
                val_b  = probe_val;
            }
        }
        else {
            // [L48, is=3 -> L12] d_min is on the other side
            if ( probe_val <= val_b ) {
                // [L12->L28->L29] Better than center: c<-b, b<-probe
                step_c = step_b;
                val_c  = val_b;
                step_b = d_min;
                val_b  = probe_val;
            }
            else {
                // [L12->L31] Worse: replace a
                step_a = d_min;
                val_a  = probe_val;
            }
        }
    } // end refinement loop

    // =======================================================================
    // Phase 3: Finalize — move to best point, rescale direction [L41: 408-422]
    // =======================================================================

    best_score_ = final_value;

    // Move from current position to best bracket point
    double displacement = final_step - previous_step;
    MoveAlongDirection(dir_index, displacement);

    // Compute curvature estimate and rescale direction vector [L41: lines 411-422]
    double curvature_arg = (step_c - step_b) * (step_c - step_a) *
                           (step_a - step_b) / (parabolic_a + parabolic_b);
    double curvature = (curvature_arg > 0.0) ? std::sqrt(curvature_arg) : 0.0;

    if ( curvature == 0.0 )
        curvature = 1e-10; // [L41: lines 419-421]

    // Scale direction vector by curvature
    int offset = dir_index * n;
    for ( int i = 0; i < n; i++ ) {
        direction_matrix_[offset + i] *= curvature;
    }

    // Update direction scale inversely by curvature
    direction_scales_[dir_index] /= curvature;

    return {final_step, final_value, false};
}

// ============================================================================
// Main optimization loop
// ============================================================================

float PowellConjugateGradient::Run(int max_iterations) {
    MyDebugAssertTrue(is_initialized_,
                      "PowellConjugateGradient::Run called before Init");

    int n    = num_dimensions_;
    int icon = 1; // Convergence mode: 1=immediate, 2=gradient verify

    max_function_calls_   = 100 * max_iterations;
    safety_limit_reached_ = false;

    // Safety backup [va04.cpp line 147]
    double fkeep = 2.0 * std::abs(best_score_);

    // Outer iteration state
    int  phase               = 1; // 1=normal, 2=gradient verify [ind]
    bool gradient_check_done = false; // [inn: 1=not done, 2=done]

    for ( int iteration = 0; iteration < max_iterations; iteration++ ) {

        double func_at_cycle_start = best_score_; // [fp, L5: line 150]
        double max_improvement     = 0.0; // [sum]
        int    best_direction      = 0; // [jil, 0-based]

        // Save current position [L5/L6: lines 152-158]
        for ( int i = 0; i < n; i++ ) {
            saved_position_[i] = current_position_[i];
        }

        // ===================================================================
        // Sweep all directions [L7-L55]
        // ===================================================================

        for ( int dir = 0; dir < n; dir++ ) {
            double func_before_line = best_score_; // [fprev]

            auto ls = PerformLineSearch(dir, 1); // mode=1, normal

            if ( ls.terminated || safety_limit_reached_ ) {
                goto safety_exit;
            }

            // Track direction with greatest improvement [L55: lines 441-449]
            double improvement = func_before_line - best_score_;
            if ( improvement >= max_improvement ) {
                max_improvement = improvement;
                best_direction  = dir;
            }
        }

        // ===================================================================
        // Direction update phase [L84-L97]
        // ===================================================================

        {
            double max_relative_change = 0.0; // [aaa]

            if ( phase == 1 ) {
                // --- Normal phase [L92] ---

                // Save score at end of sweep
                double func_after_sweep = best_score_; // [fhold]

                // Compute displacement = current - saved [L92: lines 466-471]
                for ( int i = 0; i < n; i++ ) {
                    saved_position_[i] = current_position_[i] - saved_position_[i];
                }

                // Extrapolation probe: move by 1.0 * displacement [L92->L58]
                for ( int i = 0; i < n; i++ ) {
                    current_position_[i] += saved_position_[i];
                    float_buffer_[i] = static_cast<float>(current_position_[i]);
                }
                best_score_ = EvaluateObjective( );
                if ( safety_limit_reached_ )
                    goto safety_exit;

                // Powell's direction replacement test [L112/L91: lines 479-497]
                bool should_replace = false;
                if ( func_at_cycle_start > best_score_ ) {
                    double r      = func_at_cycle_start - best_score_;
                    double d_test = 2.0 * (func_at_cycle_start + best_score_ - 2.0 * func_after_sweep) /
                                    (r * r);
                    double s = func_at_cycle_start - func_after_sweep -
                               max_improvement;
                    should_replace = (d_test * s * s < max_improvement);
                }

                if ( should_replace ) {
                    // [L87] Replace direction: remove best_direction, add displacement

                    // Shift directions [L60/L62/L97]
                    ShiftDirectionsRemove(best_direction);

                    // Place displacement as the last direction [L61: lines 519-545]
                    int last_dir        = n - 1;
                    int offset          = last_dir * n;
                    max_relative_change = 0.0;
                    for ( int i = 0; i < n; i++ ) {
                        direction_matrix_[offset + i] = saved_position_[i];
                        double rel                    = std::abs(saved_position_[i] / accuracy_[i]);
                        if ( rel > max_relative_change ) {
                            max_relative_change = rel;
                        }
                    }
                    ddmag_ = 1.0; // [L540]
                    if ( max_relative_change == 0.0 )
                        max_relative_change = 1e-10;
                    direction_scales_[last_dir] = escale_ / max_relative_change;

                    // Line search along the new displacement direction [L7, itone=3]
                    auto ls = PerformLineSearch(last_dir, 3,
                                                func_at_cycle_start,
                                                func_after_sweep);
                    if ( ls.terminated || safety_limit_reached_ ) {
                        goto safety_exit;
                    }

                    // [L38] Scale convergence measure by line search step
                    max_relative_change *= (ls.best_step + 1.0);
                }
                else {
                    // [L37] Don't replace: undo extrapolation
                    best_score_         = func_after_sweep;
                    max_relative_change = 0.0;
                    for ( int i = 0; i < n; i++ ) {
                        current_position_[i] -= saved_position_[i];
                        float_buffer_[i] = static_cast<float>(current_position_[i]);
                        double rel       = std::abs(saved_position_[i] / accuracy_[i]);
                        if ( rel > max_relative_change ) {
                            max_relative_change = rel;
                        }
                    }
                }

                // =============================================================
                // Convergence check [L53->L109: lines 584-594]
                // =============================================================

                if ( max_relative_change <= 0.1 ) {
                    // [L89] Converged
                    if ( icon == 1 ) {
                        // Immediate termination [L89->L20]
                        UpdateBestValues( );
                        return static_cast<float>(best_score_);
                    }
                    // icon == 2: enter gradient verify phase [L116]
                    phase = 2;
                    if ( ! gradient_check_done ) {
                        // [L100] First gradient step: save, perturb, evaluate
                        gradient_check_done = true;
                        for ( int i = 0; i < n; i++ ) {
                            gradient_workspace_[i] = current_position_[i];
                            current_position_[i] += accuracy_[i] * 10.0;
                            float_buffer_[i] = static_cast<float>(current_position_[i]);
                        }
                        fkeep       = best_score_;
                        best_score_ = EvaluateObjective( );
                        if ( safety_limit_reached_ )
                            goto safety_exit;
                        ddmag_ = 0.0;
                        continue; // [L108: next iteration]
                    }
                    else {
                        // [L101] Return from gradient step
                        int jil_flag;
                        if ( best_score_ < fkeep ) {
                            jil_flag            = 1; // [L105] perturbed is better
                            func_at_cycle_start = fkeep;
                        }
                        else if ( best_score_ == fkeep ) {
                            // [L78] accuracy limited
                            UpdateBestValues( );
                            return static_cast<float>(best_score_);
                        }
                        else {
                            jil_flag            = 2; // [L104] perturbed is worse
                            func_at_cycle_start = best_score_;
                            best_score_         = fkeep;
                        }

                        // [L105-L113] Set up displacement for direction update
                        for ( int i = 0; i < n; i++ ) {
                            if ( jil_flag == 1 ) {
                                saved_position_[i] = gradient_workspace_[i];
                            }
                            else {
                                saved_position_[i]   = current_position_[i];
                                current_position_[i] = gradient_workspace_[i];
                                float_buffer_[i]     = static_cast<float>(current_position_[i]);
                            }
                        }

                        // [goto L92] Redo direction update with gradient displacement
                        // Save displacement
                        double func_after_sweep_2 = best_score_;
                        for ( int i = 0; i < n; i++ ) {
                            saved_position_[i] = current_position_[i] -
                                                 saved_position_[i];
                        }
                        // Extrapolation probe
                        for ( int i = 0; i < n; i++ ) {
                            current_position_[i] += saved_position_[i];
                            float_buffer_[i] = static_cast<float>(current_position_[i]);
                        }
                        best_score_ = EvaluateObjective( );
                        if ( safety_limit_reached_ )
                            goto safety_exit;

                        // Powell's test
                        bool should_replace_2 = false;
                        if ( func_at_cycle_start > best_score_ ) {
                            double r = func_at_cycle_start - best_score_;
                            double d = 2.0 * (func_at_cycle_start + best_score_ - 2.0 * func_after_sweep_2) /
                                       (r * r);
                            double s = func_at_cycle_start - func_after_sweep_2 -
                                       max_improvement;
                            should_replace_2 = (d * s * s < max_improvement);
                        }

                        if ( should_replace_2 ) {
                            // Same replacement as normal phase
                            ShiftDirectionsRemove(best_direction);
                            int offset_2        = (n - 1) * n;
                            max_relative_change = 0.0;
                            for ( int i = 0; i < n; i++ ) {
                                direction_matrix_[offset_2 + i] =
                                        saved_position_[i];
                                double rel = std::abs(saved_position_[i] /
                                                      accuracy_[i]);
                                if ( rel > max_relative_change )
                                    max_relative_change = rel;
                            }
                            ddmag_ = 1.0;
                            if ( max_relative_change == 0.0 )
                                max_relative_change = 1e-10;
                            direction_scales_[n - 1] = escale_ /
                                                       max_relative_change;

                            auto ls2 = PerformLineSearch(n - 1, 3,
                                                         func_at_cycle_start,
                                                         func_after_sweep_2);
                            if ( ls2.terminated || safety_limit_reached_ ) {
                                goto safety_exit;
                            }
                            max_relative_change *= (ls2.best_step + 1.0);
                        }
                        else {
                            best_score_         = func_after_sweep_2;
                            max_relative_change = 0.0;
                            for ( int i = 0; i < n; i++ ) {
                                current_position_[i] -= saved_position_[i];
                                float_buffer_[i] = static_cast<float>(current_position_[i]);
                                double rel       = std::abs(saved_position_[i] /
                                                            accuracy_[i]);
                                if ( rel > max_relative_change )
                                    max_relative_change = rel;
                            }
                        }

                        // [L106] Check convergence after gradient replacement
                        if ( max_relative_change <= 0.1 ) {
                            UpdateBestValues( );
                            return static_cast<float>(best_score_);
                        }

                        // [L107] Not converged: reset and continue
                        gradient_check_done = false;
                        // Fall through to L35
                    }
                }
                else {
                    // [L76] Not converged
                    if ( best_score_ >= func_at_cycle_start ) {
                        // [L78] "accuracy limited by errors in function"
                        UpdateBestValues( );
                        return static_cast<float>(best_score_);
                    }
                    // Fall through to update ddmag_ [L35]
                }
            }
            else {
                // phase == 2: gradient verify [L72->L53->L88]
                phase = 1; // [L88: ind = 1]
                // Fall through to convergence check at L72->L53->L109
                // But in the original, L72 with iprint<2 goes to L53,
                // and L53 with ind=2 goes to L88 (set ind=1, fall through to L35).
                // So effectively: reset phase and fall through to L35.
            }

            // [L35] Update step magnitude for next iteration [lines 637-644]
            double score_improvement = func_at_cycle_start - best_score_;
            if ( score_improvement > 0.0 ) {
                ddmag_ = std::sqrt(score_improvement) * 0.4;
            }
            else {
                ddmag_ = 0.0;
            }
            gradient_mode_ = 1; // Enable gradient parabolic for next iteration

        } // end direction update block

    } // end outer iteration loop

    // [L81] Max iterations reached [lines 654-671]
    if ( best_score_ > fkeep ) {
        // Restore from gradient workspace (only meaningful if gradient phase ran)
        best_score_ = fkeep;
        for ( int i = 0; i < n; i++ ) {
            current_position_[i] = gradient_workspace_[i];
            float_buffer_[i]     = static_cast<float>(current_position_[i]);
        }
    }
    UpdateBestValues( );
    return static_cast<float>(best_score_);

safety_exit:
    // [L999] Safety limit: restore initial position [lines 720-727]
    for ( int i = 0; i < n; i++ ) {
        current_position_[i] = initial_position_[i];
        float_buffer_[i]     = static_cast<float>(current_position_[i]);
    }
    // One final evaluation at the restored position
    if ( use_legacy_path_ )
        best_score_ = static_cast<double>(legacy_function_(legacy_params_, float_buffer_.data( )));
    else
        best_score_ = static_cast<double>(objective_(float_buffer_.data( )));
    UpdateBestValues( );
    return static_cast<float>(best_score_);
}

#endif // cisTEM_ENABLE_CG_REFACTOR_2026
