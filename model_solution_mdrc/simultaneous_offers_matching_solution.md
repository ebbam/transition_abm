# Expected Matches in a Directed Search Model with Simultaneous Offers (Workers Take Highest)

## Problem Setup

- **U** workers, **V** vacancies  
- Vacancy wages: \(w_v \sim \text{Lognormal}(\mu,\sigma)\) i.i.d.  
- Each worker samples **\(\nu\)** vacancies **with replacement**, considers only those with wage \(\ge r\), and applies to the top **\(a\)** among them  
- **Offer stage (simultaneous):** every vacancy with \(\ge 1\) applicant makes **one** offer to a randomly chosen applicant  
- Workers who receive multiple offers accept the **highest-wage** offer; all other offers are rejected and those vacancies remain open (no re-issuing)

Goal: compute \(\mathbb E[M]\), the expected number of filled vacancies (matches).

---

## Closed-Form Solution

### Key Quantities

**Probability a vacancy exceeds reservation wage:**
\[
p_r = 1-\Phi\!\left(\frac{\ln r-\mu}{\sigma}\right)
\]

**Tail probability that a random vacancy wage exceeds \(w\):**
\[
q(w)=\Pr(W'>w)=1-\Phi\!\left(\frac{\ln w-\mu}{\sigma}\right)
\]

**With-replacement inclusion probability (a given vacancy is sampled at least once):**
\[
q_V = 1-\left(1-\frac{1}{V}\right)^{\nu}
\]

**Top-\(a\) selection probability (conditional on sampling the vacancy):**  
A worker applies to a wage-\(w\) vacancy if fewer than \(a\) of the other \(\nu-1\) sampled draws have wage exceeding \(w\):
\[
H(w)=\Pr\!\big(\text{Binomial}(\nu-1,q(w))\le a-1\big)
=\sum_{k=0}^{a-1}\binom{\nu-1}{k}q(w)^k(1-q(w))^{\nu-1-k}
\]

**Per-worker application probability to a wage-\(w\) vacancy:**
\[
\alpha(w)=q_V\,\mathbf 1\{w\ge r\}\,H(w)
\]

**Mean applications to a wage-\(w\) vacancy (Poisson approximation):**
\[
\boxed{\lambda(w)=U\,\alpha(w)=U\,q_V\,\mathbf 1\{w\ge r\}\,H(w)}
\]

**Lognormal density:**
\[
f(w;\mu,\sigma)=\frac{1}{w\sigma\sqrt{2\pi}}
\exp\!\left(-\frac{(\ln w-\mu)^2}{2\sigma^2}\right)
\]

---

### Step 1: Expected Number of Offers Issued

A vacancy with wage \(w\) has at least one applicant with probability
\[
1-e^{-\lambda(w)}.
\]

Therefore, the expected number of vacancies that issue an offer is
\[
\boxed{
M_0
=
V\int_{r}^{\infty} f(w;\mu,\sigma)\,\big(1-e^{-\lambda(w)}\big)\,dw
}
\]
(Interpretation: \(M_0\) is the expected number of offers sent.)

---

### Step 2: From Offers to Matches (Occupancy / “At Least One Offer”)

If \(m\) offers are sent and each offer targets (approximately) a uniformly random worker, then for any worker:

- probability of receiving no offer is \((1-\tfrac1U)^m\)
- probability of receiving at least one offer is \(1-(1-\tfrac1U)^m\)

So
\[
\mathbb E[M\mid m]=U\Big(1-(1-\tfrac1U)^m\Big).
\]

Using a standard large-market approximation \(m\approx \text{Poisson}(M_0)\), we get
\[
\mathbb E\!\left[(1-\tfrac1U)^m\right]\approx e^{-M_0/U}
\quad\Rightarrow\quad
\boxed{
\mathbb E[M]\;\approx\;U\Big(1-e^{-M_0/U}\Big)
}.
\]

---

## Main Result (Closed Form)

Combining the pieces:

\[
\boxed{
\mathbb E[M]
\;\approx\;
U\left(1-\exp\!\left[-\frac{1}{U}\,V\int_{r}^{\infty} f(w;\mu,\sigma)\,\big(1-e^{-\lambda(w)}\big)\,dw\right]\right)
}
\]

where
\[
\lambda(w)=U\,q_V\,\mathbf 1\{w\ge r\}\,H(w),
\quad
q_V = 1-\left(1-\frac{1}{V}\right)^{\nu},
\quad
H(w)=\Pr(\text{Bin}(\nu-1,q(w))\le a-1).
\]

---

## Alternative “Wage-by-Wage” Expression (Equivalent)

The expected matches can also be written as
\[
\boxed{
\mathbb E[M]\approx
V\int_r^\infty f(w)\,\big(1-e^{-\lambda(w)}\big)\,
\exp\!\left(
-\frac{V}{U}\int_w^\infty f(x)\,\big(1-e^{-\lambda(x)}\big)\,dx
\right)\,dw
}
\]
which explicitly encodes: “an offer at wage \(w\) is accepted iff the worker receives no higher-wage offer.”

---

## Computational Algorithm (Deterministic Numerical Evaluation)

```
Input: U, V, ν, a, μ, σ, r

1. Compute qV = 1 - (1 - 1/V)^ν

2. Create wage grid w_i over [r, w_max], compute:
   - q_i = 1 - Φ((ln(w_i) - μ)/σ)
   - H_i = BinomCDF(a-1, ν-1, q_i)
   - λ_i = U * qV * H_i
   - f_i = LognormalPDF(w_i; μ, σ)

3. Compute M0 = V * Σ_i f_i * (1 - exp(-λ_i)) * Δw

4. Return expected matches:
   M = U * (1 - exp(-M0 / U))
```

---

## Special Cases

### Case 1: No congestion (offers very sparse)
If \(M_0/U\ll 1\), then \(1-e^{-M_0/U}\approx M_0/U\), so
\[
\mathbb E[M]\approx M_0
\]
(i.e., almost every offer is accepted; very few workers receive multiple offers).

### Case 2: Very slack market (many offers per worker)
If \(M_0/U\gg 1\), then \(1-e^{-M_0/U}\to 1\), so
\[
\mathbb E[M]\to U
\]
(each worker gets at least one offer, so essentially all workers are matched).

---

## Summary

The simultaneous-offers version reduces to:

1) compute expected offers \(M_0\) from the application process via \(\lambda(w)\);  
2) map offers to matches using an occupancy step.

\[
\boxed{
\mathbb E[M]\approx U\Big(1-e^{-M_0/U}\Big),
\quad
M_0=V\int_r^\infty f(w)\big(1-e^{-\lambda(w)}\big)\,dw,
\quad
\lambda(w)=U\Big(1-(1-\tfrac1V)^\nu\Big)H(w).
}
\]
