# Optimized Parallel, Batched Kernels For Intrinstic Value Computation Of Common Derivatives.

## European Options
Assumes Log-Normal return distribution, no dividends, completely efficent markets, and constant risk-free rate.
Utilizes a fast approximation of the CDF, and `float4` for faster computations / smaller memory fotoprint. \\
```metal
constexpr float INV_SQRT_2PI = 0.3989422804014337; // 1 / sqrt(2 * π)

float norm_cdf(float x) {
    float k = 1.0 / (1.0 + 0.2316419 * fabs(x));
    float poly = k * (0.319381530 + k * (-0.356563782 + k * (1.781477937 + k * (-1.821255978 + k * 1.330274429))));
    float phi = INV_SQRT_2PI * exp(-0.5 * x * x);
    float cdf = (x >= 0) ? (1.0 - phi * poly) : (phi * poly);
    return cdf;
}
```


$$C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)$$ \
where: \\
$d_1 = \frac{\ln(S_0 / K) + (r + \sigma^2 / 2) T}{\sigma \sqrt{T}}$ \\
$d_2 = d_1 - \sigma \sqrt{T}$ 

## Forward Contracts

Much simpler pricing. \
$s_0 * e^{r * T}$ \
where: \ 
$s_0$ is the current spot price. \
$r$ is the risk free rate \
$T$ is time to exercise. 
