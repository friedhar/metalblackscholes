#include <metal_stdlib>
using namespace metal;

constexpr float INV_SQRT_2PI = 0.3989422804014337; // 1 / sqrt(2 * π)

float norm_cdf(float x) {
    float k = 1.0 / (1.0 + 0.2316419 * fabs(x));
    float poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));
    float phi = INV_SQRT_2PI * exp(-0.5 * x * x);
    float cdf = (x >= 0) ? (1.0 - phi * poly) : (phi * poly);
    return cdf;
}

kernel void black_scholes_optimized(
    device const float4* S0,     
    device const float4* X,     
    device const float4* r,   
    device const float4* sigma,   
    device const float4* t,     
    device float4* result,       
    uint id [[thread_position_in_grid]]
) {
    float4 S = S0[id];
    float4 K = X[id];
    float4 R = r[id];
    float4 Sigma = sigma[id];
    float4 T = t[id];

    float4 sqrtT = sqrt(T);
    float4 d1 = (log(S / K) + (R + 0.5f * Sigma * Sigma) * T) / (Sigma * sqrtT);
    float4 d2 = d1 - Sigma * sqrtT;

    float4 C = S * norm_cdf(d1) - K * exp(-R * T) * norm_cdf(d2);

    result[id] = C;
}
