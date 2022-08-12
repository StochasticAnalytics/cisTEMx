#ifndef _SRC_PROGRAMS_REFINE3D_BATCH_SIZE_OPTIMIZER_H_
#define _SRC_PROGRAMS_REFINE3D_BATCH_SIZE_OPTIMIZER_H_

#include <chrono>
#include <cmath>
#include <omp.h> // FIXME: this should of course depend on weither we are using OpenMP or not

// #define PRINT_OPT_DEBUG
class BatchSizeOptimizer {

  public:
    BatchSizeOptimizer( ){ };
    ~BatchSizeOptimizer( ){ };

    int start( ) {
        start_time = std::chrono::high_resolution_clock::now( );
        if ( t == -2 ) {
            t++;
            current_batch[0] = initial_start - initial_step;
            return current_batch[0];
        }
        if ( t == -1 ) {
            t++;
            current_batch[1] = initial_start + initial_step;
            return current_batch[1];
        }
        if ( omp_get_thread_num( ) == 0 )
            std::cerr << "Current batch " << current_batch[1] << std::endl;
        return current_batch[1];
    };

    void lap( ) {
        ns_count = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now( ) - start_time).count( );
        // Reduce to milliseconds
        time_in_ms = ((ns_count / 100000) % 10 >= 5) ? 1 : 0;
        time_in_ms += (ns_count / 1000000);
#ifdef PRINT_OPT_DEBUG
        if ( omp_get_thread_num( ) == 0 )
            std::cerr << "At time t: ms = " << t << " " << time_in_ms << std::endl;
#endif
    };

    void update_batch_size( ) {
        // Get the measured time.
        lap( );
        if ( t == -1 ) {
            current_time[0] = time_in_ms;
        }
        else {

            current_time[1] = time_in_ms;

            calc_gradient( );
            if ( calculate_alpha ) {
                // Because we don't know how large the steps will be, choose alpha to be ~half the intial step size
                alpha           = fabsf(ceilf(10.f * float(initial_step)) / gradient);
                calculate_alpha = false;
            }

            t++;
            beta1_of_t       = powf(beta1, t);
            beta2_of_t       = powf(beta2, t);
            m[1]             = beta1 * m[0] + (1 - beta1) * gradient;
            v[1]             = beta2 * v[0] + (1 - beta2) * gradient * gradient;
            m_hat            = m[1] / (1 - beta1_of_t);
            v_hat            = v[1] / (1 - beta2_of_t);
            current_step     = int(std::round(alpha * m_hat / (sqrtf(v_hat) + epsilon)));
            current_batch[0] = current_batch[1];
            current_batch[1] -= current_step;
            current_time[0] = current_time[1];

            m[0] = m[1];
            v[0] = v[1];
#ifdef PRINT_OPT_DEBUG
            if ( omp_get_thread_num( ) == 0 ) {
                std::cerr << "Batch size: " << current_batch[1] << " step " << current_step << std::endl;
                std::cerr << "Gradient and alpha : " << gradient << " " << alpha << std::endl;
                std::cerr << "m_hat and v_hat : " << m_hat << " " << v_hat << std::endl;
                std::cerr << "m[1] and v[1] : " << m[1] << " " << v[1] << "\n\n"
                          << std::endl;
            }
#endif
        }
        return;
    };

    inline void calc_gradient( ) {

        // these are unsigned ints so watch out for wrap around if negative - leave as double
        gradient = float((double(current_time[1]) - double(current_time[0])) / double(2 * std::max(1, current_step)));
#ifdef PRINT_OPT_DEBUG
        if ( omp_get_thread_num( ) == 0 ) {
            std::cerr << "Time t, gradient calc: " << t << " " << current_time[0] << " " << current_time[1] << std::endl;
            std::cerr << "Gradient: " << gradient << std::endl;
        }
#endif
    };

  private:
    float       alpha   = 0.0;
    const float beta1   = 0.9;
    const float beta2   = 0.999;
    const float epsilon = 1e-8;
    float       beta1_of_t;
    float       beta2_of_t;
    int         t        = -2;
    float       m[2]     = {0.f, 0.f};
    float       v[2]     = {0.f, 0.f};
    float       m_hat    = 0.0;
    float       v_hat    = 0.0;
    float       gradient = 0.0;
    int         current_batch[2];
    uint64_t    current_time[2];

    std::chrono::time_point<std::chrono::high_resolution_clock> start_time = std::chrono::high_resolution_clock::now( );

    uint64_t ns_count;
    uint64_t time_in_ms;

    const int initial_step  = 5;
    const int initial_start = 15;

    int current_step    = initial_step;
    int best_batch_size = initial_start;

    // We set the initial learning rate to produce a step size that is 1/2 the initial step assuming we have a decent idea where the optimum is
    bool calculate_alpha = true;
};

#endif