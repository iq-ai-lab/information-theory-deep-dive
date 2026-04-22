# 2.5 Wasserstein Distance — Optimal Transport의 정보이론

## 🎯 핵심 질문

> **분포의 support 가 겹치지 않을 때도 의미 있는 "거리"를 주는 척도는 무엇인가?**
> **Kantorovich–Rubinstein 쌍대성** 은 왜 1-Lipschitz 함수를 등장시키는가?
> Wasserstein GAN 은 JSD 의 어떤 치명적 문제를 해결하는가?

---

## 🔍 왜 AI에서 중요한가

- **WGAN**: JSD의 gradient vanishing 문제를 해결, **earth mover's distance** 로 학습 안정화.
- **OT(Optimal Transport) 붐**: Sinkhorn divergence, Entropic OT, 도메인 적응, color transfer, 스타일 변환.
- **Diffusion bridges**: forward/backward SDE의 path-space OT 해석.
- **Fairness & Distribution Shift**: Wasserstein DRO(distributionally robust optimization).
- **정책 최적화**: trust region 설계 시 Wasserstein trust region.
- **generative evaluation**: FID, KID 는 Gaussian/Feature 공간의 2-Wasserstein 혹은 MMD 근사.

**f-divergence의 한계** (Theorem 2.3.6: support disjoint → 상수) 를 기하학적으로 돌파하는 것이 OT.

---

## 📐 선행 학습 지식

- [2.1 KL의 정의](./01-kl-definition-nonnegativity.md), [2.3 JSD](./03-js-divergence.md), [2.4 f-divergence](./04-f-divergence.md)
- Metric space의 coupling (joint distribution with fixed marginals)
- Duality (Lagrangian, Fenchel), Lipschitz 함수
- 기본 측도론 (measurable space, marginals)

---

## 📖 직관

### "Earth Mover's Distance"

두 분포 $p, q$ 를 **흙 더미**에 비유: $p$ 는 현재 흙 분포, $q$ 는 목표 분포. 한 점에서 다른 점으로 흙을 옮기는 데 **거리×양** 만큼 비용이 든다. **최소 총비용** 이 Wasserstein 거리.

- 1D 의 예: $p = \delta_0$ (점질량 0), $q = \delta_a$ (점질량 $a$). 모든 질량을 $a$ 만큼 옮겨야 → $W_1(p, q) = a$.
- 반면 JSD 는 이 둘에 대해 항상 $\log 2$ (disjoint support). KL은 $\infty$.

### 왜 support mismatch 에 강한가

거리는 **metric 공간 위에서 정의된 cost function** 을 사용하므로, support 가 겹치지 않아도 점 사이의 **지리적 거리**로 비용을 잰다. 그래서 "점점 가까워진다"는 연속 개념이 살아있음 → gradient가 사라지지 않는다.

---

## ✏️ 공식 정의

**정의 2.5.1 (Coupling / Transport plan)**
두 확률측도 $p, q$ 에 대해 marginals 가 각각 $p, q$ 인 $\mathcal{X}\times\mathcal{X}$ 위 joint 측도 전체를 $\Pi(p, q)$ 라 한다. 원소 $\gamma \in \Pi(p, q)$ 를 coupling (수송계획) 이라 부른다.

**정의 2.5.2 (p-Wasserstein 거리)**
$(\mathcal{X}, d)$ metric 공간에서 $p \ge 1$ 에 대해
$$
\boxed{\ W_p(p, q) \;=\; \left( \inf_{\gamma \in \Pi(p, q)} \int d(x, y)^p\, d\gamma(x, y) \right)^{1/p}\ }
$$

**주요 특수경우**:
- $W_1$: Earth Mover's Distance (L1, linear transport cost).
- $W_2$: $L^2$ OT, displacement convexity의 중심.

**정의 2.5.3 (1D에서 CDF로)** 1차원 실수축에서
$$
W_p(p, q) = \left(\int_0^1 |F_p^{-1}(u) - F_q^{-1}(u)|^p\, du\right)^{1/p}
$$
즉 두 cumulative 분포 함수 역함수 간 $L^p$ 거리.

---

## 🔬 정리와 증명

### Theorem 2.5.1 (Metric 성질)

**진술.** $d$ 가 $\mathcal{X}$ 위 metric 이면 $W_p$ 도 확률측도 공간 $\mathcal{P}_p(\mathcal{X})$ 위 metric (비음성, 대칭, 삼각부등식, 정부호).

**증명 스케치.**
- **비음성·대칭·정부호** 는 정의 + 비용 함수 성질로 자명.
- **삼각부등식**: $\gamma_{12} \in \Pi(p_1, p_2), \gamma_{23} \in \Pi(p_2, p_3)$ 에 대해 glueing lemma 로 $\gamma_{123}$ 구성, marginal $\gamma_{13} \in \Pi(p_1, p_3)$ 가 $\gamma_{123}$ 의 $x_2$ 주변화로 얻어진다. Minkowski 부등식 적용. $\blacksquare$

### Theorem 2.5.2 (Kantorovich–Rubinstein Duality, $p=1$)

**진술.**
$$
W_1(p, q) = \sup_{\|f\|_L \le 1} \left\{ \mathbb{E}_p[f(X)] - \mathbb{E}_q[f(X)] \right\}
$$
여기서 $\|f\|_L = \sup_{x\ne y} |f(x)-f(y)|/d(x,y)$ 는 Lipschitz 상수.

**증명 스케치.** Kantorovich primal: $\inf \int d(x,y) d\gamma(x,y)$. Lagrangian:
$$
\inf_{\gamma \ge 0}\, \int d(x,y) d\gamma - \int f(x)\,(d\gamma_1 - dp) - \int g(y)\,(d\gamma_2 - dq)
$$
primal 최적성 조건으로 $f(x) + g(y) \le d(x,y)$ 가 도출. 1-Lipschitz $f$ 와 $g = -f$ 로 제한하면 sup 에 도달. Strong duality (Kantorovich 1942, Villani 2008). $\blacksquare$

> **함의**: WGAN 의 판별기는 $f$ (critic), 1-Lipschitz 제약 하에서 $\mathbb{E}_p[f] - \mathbb{E}_q[f]$ 를 최대화 = $W_1$ 을 추정.

### Theorem 2.5.3 ($W_p$ 의 연속성)

**진술.** Sequence $p_n \to p$ in $W_p$ iff $p_n \to p$ weakly **and** $\int d(x_0, x)^p dp_n \to \int d(x_0, x)^p dp$.

**증명**: Villani 2008 Theorem 6.9. (생략)

> **함의**: Wasserstein 은 weak convergence 와 거의 같은 topology — 샘플 분포의 수렴에 **연속적**으로 반응. 반면 TV/KL 은 sample 분포 간 discrete 해서 수렴이 불연속.

### Theorem 2.5.4 (Disjoint support case: $W_p$ 가 **상수 아님**)

**진술.** $p = \delta_0, q = \delta_a$ (각각 점질량). $W_p(p, q) = a$.

**증명.** Coupling 은 하나뿐: $\gamma = \delta_{(0, a)}$. 비용 $d(0, a)^p = a^p$. $W_p = a$. $\blacksquare$

> **비교**: JSD 는 $\log 2$ (상수), KL 은 $\infty$. Wasserstein은 **$a$ 에 연속**.

### Theorem 2.5.5 (Gaussian간 $W_2$ 공식)

**진술.** $p = \mathcal{N}(\mu_1, \Sigma_1), q = \mathcal{N}(\mu_2, \Sigma_2)$ 이면
$$
W_2^2(p, q) = \|\mu_1 - \mu_2\|^2 + \mathrm{tr}\!\left(\Sigma_1 + \Sigma_2 - 2(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}\right)
$$

**증명 스케치.** Optimal coupling은 linear map $y = \mu_2 + A(x-\mu_1)$, $A = \Sigma_1^{-1/2}(\Sigma_1^{1/2}\Sigma_2\Sigma_1^{1/2})^{1/2}\Sigma_1^{-1/2}$. 대수적 계산. (Takatsu 2011) $\blacksquare$

> **응용**: FID (Fréchet Inception Distance) = 두 Gaussian 간 $W_2^2$ — 생성 모델의 표준 평가지표.

### Theorem 2.5.6 (Entropic OT / Sinkhorn)

**진술.** 엔트로피 정규화 OT:
$$
W_\epsilon(p, q) = \inf_{\gamma \in \Pi(p, q)} \int d(x,y)^2\, d\gamma + \epsilon\, H(\gamma \| p \otimes q)
$$
는 $\epsilon > 0$ 일 때 **Sinkhorn iteration** 으로 $O(n^2)$ 해결 가능. $\epsilon \to 0$ 에서 $W_2^2$ 로 수렴.

**증명 스케치.** Dual 에 KKT 적용 → $\gamma^*(x,y) \propto \exp((u(x) + v(y) - c(x,y))/\epsilon) p(x)q(y)$. $u, v$ 에 대한 고정점 반복 = Sinkhorn. $\blacksquare$

> **함의**: 대용량 OT 는 Sinkhorn으로 GPU상 실행가능. POT(Python Optimal Transport), geomloss 라이브러리.

---

## 💻 NumPy로 직접 확인

### 1D Wasserstein-1

```python
import numpy as np
from scipy.stats import wasserstein_distance

# 이산 1D 예제
rng = np.random.default_rng(0)
x = rng.normal(0, 1, 10000)
y = rng.normal(1.0, 1, 10000)
print("W1(N(0,1), N(1,1)) =", wasserstein_distance(x, y))
# ≈ 1.0 (평균 차이)
```

### Gaussian $W_2$ (FID 공식)

```python
from scipy.linalg import sqrtm

def w2_gauss(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    covmean = sqrtm(cov1 @ cov2)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return diff @ diff + np.trace(cov1 + cov2 - 2*covmean)

d = 4
mu1, mu2 = np.zeros(d), np.ones(d)
cov1, cov2 = np.eye(d), 2*np.eye(d)
w2_sq = w2_gauss(mu1, cov1, mu2, cov2)
print(f"W2^2(N(0,I), N(1,2I)) = {w2_sq:.4f}")
```

### Support disjoint에서 Wasserstein 이 연속적

```python
from scipy.stats import wasserstein_distance
# delta_0 와 delta_a 샘플 근사
a_vals = np.linspace(0, 5, 6)
for a in a_vals:
    x = np.zeros(1000); y = np.full(1000, a)
    print(f"a={a:.1f}  W1={wasserstein_distance(x, y):.3f}")
```
출력:
```
a=0.0  W1=0.000
a=1.0  W1=1.000
a=2.0  W1=2.000
...
```
($a$에 비례) — KL/JSD 와 완전히 다른 행동.

### Sinkhorn iteration 구현

```python
def sinkhorn(a, b, C, eps=0.1, n_iter=100):
    # a, b 는 marginals (prob vectors)
    K = np.exp(-C / eps)
    u = np.ones_like(a)
    for _ in range(n_iter):
        v = b / (K.T @ u + 1e-20)
        u = a / (K @ v + 1e-20)
    gamma = np.diag(u) @ K @ np.diag(v)
    return gamma, np.sum(gamma * C)

n = 20
a = np.ones(n) / n; b = np.ones(n) / n
xs = np.linspace(0, 1, n); ys = np.linspace(0.3, 1.3, n)
C = (xs[:, None] - ys[None, :])**2
gamma, cost = sinkhorn(a, b, C, eps=0.001, n_iter=500)
print(f"Entropic OT cost = {cost:.4f}, 이론 W_2^2 ≈ 0.09")
```

### Kantorovich–Rubinstein 1-Lipschitz 추정 (간단)

```python
# 두 샘플 X, Y 에서 W1 ≈ sup_{1-Lip f} E[f(X)] - E[f(Y)]
# 1D에서는 CDF로 바로
xs = rng.normal(0, 1, 2000); ys = rng.normal(1.5, 1, 2000)
emp_w1 = wasserstein_distance(xs, ys)
# 1-Lipschitz 테스트 함수 몇 개:
for f in [lambda z: z, lambda z: np.tanh(z), lambda z: np.clip(z, -2, 2)]:
    v = f(xs).mean() - f(ys).mean()
    print(f"lower bound = {abs(v):.3f}    (W1 = {emp_w1:.3f})")
```

---

## 🔗 AI/ML 연결고리

| 응용 | 역할 |
|---|---|
| **WGAN** | Critic = 1-Lipschitz, 판별기 loss = $\mathbb{E}_p[f] - \mathbb{E}_{p_g}[f]$ |
| **WGAN-GP** | Lipschitz 제약을 gradient penalty 로 soft 적용 |
| **Sinkhorn-GAN** | Entropic OT 를 cost로 |
| **FID** | Inception feature Gaussian $W_2^2$ |
| **DRO (Wasserstein)** | $\max_{q: W(q, p_{\mathrm{data}}) \le \epsilon} \mathbb{E}_q[\ell]$ |
| **Domain adaptation** | 소스·타깃 feature 분포간 OT |
| **Color / Style transfer** | Color histogram 또는 feature distributions 에 OT |
| **Diffusion flow matching** | Continuous normalizing flow 가 근사적으로 OT flow |
| **Trust region policy (TRPO)** | 일부 variant 는 Wasserstein trust region |
| **Fairness metric** | 보호그룹 간 예측분포 Wasserstein 거리 |

### WGAN 이 왜 더 안정한가 (Arjovsky et al. 2017)

- Theorem 2.5.3: $W_1$ 은 weak convergence 에 대해 연속 → generator 가 $p_{\mathrm{data}}$ 로 점진적 수렴할 때 loss 가 **smoothly 감소**.
- JSD 는 support 안 겹치면 상수 → gradient = 0.
- WGAN 실험적으로 mode coverage 개선, hyperparam 덜 민감.

### Wasserstein vs f-divergence: **IPM vs $\phi$-divergence**

- **f-divergence**: $\int q f(p/q)$ — density ratio 기반. support 겹침 필요.
- **IPM** (Integral Probability Metric): $\sup_{f \in \mathcal{F}} |\mathbb{E}_p f - \mathbb{E}_q f|$ — 테스트 함수 class 기반. Wasserstein($\mathcal{F}$=1-Lip), MMD($\mathcal{F}$=RKHS ball), TV($\mathcal{F}=\{f: |f|\le 1\}$).
- TV 는 **둘 다** (f-div and IPM). 유일한 교집합. (Sriperumbudur 2009)

---

## ⚖️ 가정·한계·함정

1. **계산 비용** — 일반 $n$-sample OT 는 $O(n^3 \log n)$ (네트워크 플로). Sinkhorn으로 $O(n^2)$ 까지. 샘플이 많으면 비쌈.
2. **curse of dimensionality** — empirical $W_1$ 의 샘플 수렴율 $\sim n^{-1/d}$ → 고차원에선 샘플 수가 기하급수적 필요. Sliced Wasserstein 등으로 완화.
3. **Lipschitz 강제** — WGAN 에서 weight clipping 은 편법. WGAN-GP, spectral norm 이 개선.
4. **"진짜 metric"이지만 Fisher 가 다름** — Wasserstein 은 $d$-공간의 geometry 를 반영, f-div 처럼 파라미터 공간의 Fisher metric 과는 다른 구조.
5. **Cost 함수 선택** — $d$ 를 어떻게 잡느냐가 결과를 완전히 바꿈. 고차원 이미지에서 raw pixel L2 는 의미가 없어서 Feature embedding 위의 OT 가 관행 (FID).

---

## 📌 핵심 정리

1. $W_p(p, q) = \left(\inf_\gamma \int d^p d\gamma\right)^{1/p}$ — Optimal Transport 비용.
2. **진짜 metric** on $\mathcal{P}_p(\mathcal{X})$; weak convergence 와 거의 동등.
3. **Kantorovich–Rubinstein**: $W_1 = \sup_{1\text{-Lip } f}\{\mathbb{E}_p f - \mathbb{E}_q f\}$.
4. **Support mismatch에도 연속** — JSD/KL의 치명적 결함 해결.
5. **WGAN**: critic = 1-Lipschitz 근사, $W_1$ minimize.
6. **Gaussian $W_2^2$** 가 FID의 수식.
7. **Sinkhorn (entropic OT)** — 대규모 OT 의 실용적 해결책.
8. f-divergence 와 **상호보완**: density ratio 관점 vs 지리적 관점.

---

## 🤔 생각해볼 문제

### 문제 1. Point masses 의 $W_1$
$p = \sum_i \alpha_i \delta_{x_i}$, $q = \sum_j \beta_j \delta_{y_j}$ 간 $W_1$ 을 LP(선형계획) 로 쓰고, 1D에서 CDF 기반 공식과의 등가성을 보여라.

<details>
<summary>해설</summary>

LP: $\min \sum_{ij} \gamma_{ij} d(x_i, y_j)$ s.t. $\sum_j \gamma_{ij} = \alpha_i, \sum_i \gamma_{ij} = \beta_j, \gamma \ge 0$. 1D 에서 점을 정렬하고 "왼쪽부터 greedy" 로 최적 → CDF 역함수 공식 $W_1 = \int_0^1 |F_p^{-1}(u)-F_q^{-1}(u)| du$.
</details>

### 문제 2. WGAN의 Lipschitz 강제
Weight clipping, gradient penalty, spectral norm 의 장단점을 써라.

<details>
<summary>해설</summary>

- **Clipping**: 간단하지만 capacity 제약 심하고 gradient vanishing.
- **GP**: Gradient of critic을 [0,1]에서 1로 제약 (soft); 효과적이지만 추가 forward/backward 필요.
- **Spectral norm**: 각 layer의 spectral norm 을 1로 normalize; 결정적이고 빠름 but strictly 1-Lipschitz 보장 아님(layer 합성 이후).
</details>

### 문제 3. $W_2$ geodesic
두 Gaussian 간 $W_2$-geodesic 은 $\mathcal{N}(\mu_t, \Sigma_t)$ 형태. $\mu_t = (1-t)\mu_1 + t\mu_2$. $\Sigma_t$ 는 어떤 식?

<details>
<summary>해설</summary>

$\Sigma_t^{1/2} = (1-t) \Sigma_1^{1/2} + t \Sigma_2^{1/2}$ (simplification for commuting case). 일반 Gaussian 의 displacement interpolation. Diffusion 의 intermediate distribution 과 연결.
</details>

### 문제 4. MMD vs Wasserstein
Maximum Mean Discrepancy (MMD) = $\sup_{\|f\|_\mathcal{H}\le 1}|\mathbb{E}_p f - \mathbb{E}_q f|$ 은 RKHS-based IPM. $W_1$ 과의 차이점과 실무적 장단점.

<details>
<summary>해설</summary>

MMD: 닫힌 형 추정량($U$-statistic) 가능, $O(n^2)$ with quadratic kernel, convergence rate $n^{-1/2}$ (dimension-independent). $W_1$: geometrically meaningful but $n^{-1/d}$ 수렴. 고차원에서는 MMD가 통계적으로 더 안정, 시각화는 $W_1$이 직관적.
</details>

### 문제 5. Sliced Wasserstein
$SW_p(p, q) = \left(\mathbb{E}_\theta W_p^p(\theta_\sharp p, \theta_\sharp q)\right)^{1/p}$ ($\theta$ random projection). 고차원에서 왜 유용한가?

<details>
<summary>해설</summary>

1D projection 후 $W_p$ 는 sorting 기반 $O(n \log n)$ 으로 계산. 여러 projection 평균화 → 고차원 curse 완화. MMD와 유사한 computational 이점, OT geometry 의 일부 유지. Generative 모델 evaluation, style transfer 등에 실용.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [2.4 f-divergence](./04-f-divergence.md) | [2.6 어떤 발산을 쓸 것인가](./06-choosing-divergence.md) |

[🏠 Home](../README.md)

</div>
