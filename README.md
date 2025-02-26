# Optimized Parallel, Batched Kernels For Intrinstic Value Computation Of Common Derivatives.

## European Options
Assumes Log-Normal return distribution, no dividends, completely efficent markets, and constant risk-free rate.
\
$$C = S_0 \Phi(d_1) - K e^{-rT} \Phi(d_2)$$ \
where: \
$d_1 = \frac{\ln(S_0 / K) + (r + \sigma^2 / 2) T}{\sigma \sqrt{T}}$ \
$d_2 = d_1 - \sigma \sqrt{T}$ \

