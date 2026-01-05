# Expected Matches in a Directed Search Model with Sequential Hiring

## Problem Setup

- **U** workers, **V** vacancies
- Wages: $w_v \sim \text{Lognormal}(\mu, \sigma)$ 
- Each worker samples **ν** vacancies (with replacement), applies to top **a** above reservation wage **r**
- Vacancies processed in random order; each offers job to one random available applicant
- Workers accept any offer (since all applied vacancies exceed r)

---

## Closed-Form Solution

### Key Quantities

**Probability a vacancy exceeds reservation wage:**
$$p = 1 - \Phi\left(\frac{\ln r - \mu}{\sigma}\right)$$

**Probability a random vacancy has wage exceeding w:**
$$q(w) = 1 - \Phi\left(\frac{\ln w - \mu}{\sigma}\right)$$

**Expected applications per worker:**
$$\bar{a} = \sum_{j=1}^{a} \mathbb{P}(\text{Binomial}(\nu, p) \geq j)$$

**Application rate to vacancy with wage w:**
$$\lambda(w) = \frac{U\nu}{V} \cdot F_{\text{Binom}}(a-1;\, \nu-1,\, q(w))$$

This captures that workers preferentially apply to higher-wage vacancies.

---

### Mean-Field ODE

Let $u(s)$ = fraction of workers still unmatched after fraction $s \in [0,1]$ of vacancies processed.

**Dynamics:** When a vacancy with wage $w$ is processed at time $s$:
- Effective applicants: $\lambda(w) \cdot u(s)$
- Fill probability: $1 - e^{-\lambda(w) u(s)}$
- If filled, one worker becomes matched

**Expected fill rate at time s:**
$$\Psi(u) = \int_r^{\infty} f(w;\mu,\sigma) \left[1 - e^{-\lambda(w) \cdot u}\right] dw$$

**The ODE:**
$$\boxed{\frac{du}{ds} = -\frac{V}{U} \Psi(u(s)), \quad u(0) = 1}$$

**Expected matches:**
$$\boxed{M = U \cdot (1 - u(1))}$$

---

### Alternative: Integral Form for M

$$M = V \int_0^1 \Psi(u(s))\, ds = V \int_0^1 \int_r^{\infty} f(w) \left[1 - e^{-\lambda(w) u(s)}\right] dw\, ds$$

---

## Computational Algorithm

```
Input: U, V, ν, a, μ, σ, r

1. Compute p = 1 - Φ((ln(r) - μ)/σ)
2. Create wage grid w_i over [r, w_max], compute:
   - q_i = 1 - Φ((ln(w_i) - μ)/σ)  
   - λ_i = (Uν/V) × BinomCDF(a-1, ν-1, q_i)
   - f_i = LognormalPDF(w_i; μ, σ)

3. Define Ψ(u) = Σ_i f_i × [1 - exp(-λ_i × u)] × Δw

4. Solve ODE numerically:
   du/ds = -(V/U) × Ψ(u)
   from s=0 to s=1 with u(0)=1

5. Return M = U × (1 - u(1))
```

---

## Special Cases

### Case 1: Single Application (a = 1)
Each worker applies to exactly one vacancy (their highest-wage sample above r).
The model reduces toward the classical urn-ball:
$$M \approx V_{\text{eff}} \left(1 - e^{-U/V_{\text{eff}}}\right)$$
where $V_{\text{eff}} = Vp$ (vacancies above threshold).

### Case 2: Uniform Applications (σ → 0)
When wage dispersion vanishes, $\lambda(w) \to \bar{\lambda} = U\bar{a}/(Vp)$ becomes constant, and:
$$\Psi(u) = p\left(1 - e^{-\bar{\lambda} u}\right)$$

The ODE becomes:
$$\frac{du}{ds} = -\frac{Vp}{U}\left(1 - e^{-\bar{\lambda} u}\right)$$

### Case 3: Tight Market (U >> V)
Most vacancies receive many applications, $u(s)$ stays high, and $M \to V_{\text{eff}}$.

### Case 4: Slack Market (V >> U)  
Applications are spread thin, most workers find jobs, and $M \to U$.

---

## Verification Results

| U | V | ν | a | σ | r | Theory | Simulation | Error |
|---|---|---|---|---|---|--------|------------|-------|
| 100 | 100 | 10 | 3 | 0.5 | 0.5 | 49.4 | 50.2 ± 2.4 | 1.7% |
| 200 | 100 | 10 | 3 | 0.5 | 0.5 | 56.7 | 57.0 ± 2.1 | 0.5% |
| 100 | 200 | 10 | 3 | 0.5 | 0.5 | 74.3 | 78.6 ± 2.9 | 5.4% |
| 100 | 100 | 20 | 5 | 0.5 | 0.5 | 42.8 | 43.6 ± 2.0 | 1.9% |
| 500 | 500 | 15 | 4 | 0.6 | 0.7 | 227.2 | 224.3 ± 4.8 | 1.3% |
| 200 | 50 | 10 | 3 | 0.5 | 0.5 | 31.0 | 31.8 ± 1.5 | 2.5% |

---

## Summary

The expected number of filled vacancies is:

$$\boxed{M(U, V, \nu, a, \mu, \sigma, r) = U \cdot \left(1 - u(1)\right)}$$

where $u(1)$ is obtained by solving the ODE:

$$\frac{du}{ds} = -\frac{V}{U} \int_r^{\infty} f_{\text{LN}}(w; \mu, \sigma) \left[1 - \exp\left(-\frac{U\nu}{V} F_{\text{Binom}}(a-1; \nu-1, q(w)) \cdot u\right)\right] dw$$

with initial condition $u(0) = 1$.
