#include <metal_stdlib>
using namespace metal;

kernel void forward_pricing(
    device const float* spot_prices [[buffer(0)]],  
    device const float* interest_rates [[buffer(1)]],  
    device const float* maturities [[buffer(2)]],  
    device float* forward_prices [[buffer(3)]],  
    uint gid [[thread_position_in_grid]]  
) {
    if (gid >= buffer_size) return;

    float S0 = spot_prices[gid];
    float r = interest_rates[gid];
    float T = maturities[gid];

    forward_prices[gid] = S0 * exp(r * T);
}

