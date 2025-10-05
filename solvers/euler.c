#include "solvers.h"

void solver(Float* state, Float* params, Float* derivs,
            unsigned state_size, unsigned steps, Float dt) {
    
    for (unsigned step = 0; step < steps; step++) {
        forward(&state[step * state_size], params, derivs);
        for (unsigned i = 0; i < state_size; i++) {
            state[(step + 1) * state_size + i] = state[step * state_size + i] + dt * derivs[i];
        }
    }
}
