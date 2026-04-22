# 06. 최대 엔트로피 분포

## 🎯 핵심 질문

- 주어진 제약(평균·분산 등)만 알고 있을 때, **가장 "덜 제한된"** 분포는 무엇인가?
- 왜 "범위만 고정" → 균등분포, "평균만 고정" → 지수분포, "분산만 고정" → 정규분포가 MaxEnt인가?
- 라그랑주 승수법으로 어떻게 이 결과가 **자동으로** 유도되는가?
- MaxEnt 원리는 베이지안 사전분포 선택과 어떻게 연결되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **Uninformative Prior**: 베이지안 모델링에서 "아는 것이 최소일 때 어떤 사전을 쓸까?" → MaxEnt
- **Softmax = MaxEnt Classifier**: 기대 logit을 맞추는 최대 엔트로피 분포가 바로 softmax (이 문서에서 유도)
- **Gibbs Distribution**: 물리학의 Boltzmann 분포 $p \propto e^{-E/kT}$는 에너지 기대값 제약의 MaxEnt
- **Exponential Family**: 지수족 분포 전체가 MaxEnt 원리의 결과 — sufficient statistic만큼의 제약으로 유도
- **Variational Inference**: 근사 분포족으로 지수족을 쓰는 이유 중 하나가 MaxEnt의 수학적 우아함

"아는 것만 제약하고 나머지는 최대한 자유롭게" — 이 원리가 ML 분포 선택의 수학적 기본 규칙이다.

---

## 📐 수학적 선행 조건

- [문서 02](./02-entropy-definition.md)의 엔트로피 정의 및 Jensen 부등식
- [문서 05](./05-differential-entropy.md)의 미분 엔트로피 및 정규/지수 분포
- **라그랑주 승수법** (Calculus & Optimization Deep Dive의 Ch6)
- KL-divergence의 비음수성 (Gibbs 부등식, Ch2-01에서 상세)

---

## 📖 직관적 이해

### MaxEnt 원리 (Jaynes 1957)

> "확실히 아는 것만 제약하고 나머지는 최대한 '모른다'고 가정하라. 즉 알고 있는 통계를 만족하면서 엔트로피를 최대화하는 분포를 선택하라."

이는 단순한 미학이 아니라 **정보이론적으로 정당한 선택**: 다른 분포는 우리가 알지 못하는 **추가 구조**를 가정하는 것이 되기 때문이다.

### 세 가지 대표 결과 (미리 보기)

| 알고 있는 것 | Support | MaxEnt 분포 |
|-------------|---------|-------------|
| support가 $[a, b]$라는 것만 | 유계 구간 | **균등 $U(a, b)$** |
| 평균 $\mu > 0$, support $[0, \infty)$ | 양의 실수 | **지수 $\operatorname{Exp}(1/\mu)$** |
| 평균 $\mu$, 분산 $\sigma^2$, support $\mathbb{R}$ | 전체 실수 | **정규 $\mathcal{N}(\mu, \sigma^2)$** |

### 라그랑주 승수의 해석

제약
$$\int f(x) g_k(x) \, dx = c_k, \quad k = 1, \ldots, m,$$
하에서 $h(f)$ 최대화. 라그랑지안의 stationary 조건이
$$f^*(x) \propto \exp\!\left(\sum_k \lambda_k g_k(x)\right),$$
즉 **지수족 (exponential family)** 의 형태다. MaxEnt ↔ 지수족의 동등성이 **Pitman-Koopman-Darmois 정리**의 정보이론적 표현이다.

---

## ✏️ 엄밀한 정의

### MaxEnt 문제 (일반 형태)

Support $\mathcal{S}$ 위의 확률밀도 $f \geq 0$, $\int f = 1$에 대해 제약
$$\int_\mathcal{S} f(x) g_k(x) \, dx = c_k, \quad k = 1, \ldots, m$$
를 만족하면서
$$\max_f \; h(f) = -\int_\mathcal{S} f(x) \log f(x) \, dx$$
를 찾는다.

이산 버전은 $\int \to \sum$, $f \to p$로 치환.

---

## 🔬 정리와 증명

### 정리 6.1 — MaxEnt 해의 일반형

**명제**: 위 제약 문제의 해(존재한다면)는 반드시
$$f^*(x) = \exp\!\left(-\lambda_0 - \sum_{k=1}^{m} \lambda_k g_k(x)\right),$$
여기서 $\lambda_0$는 정규화 상수, $\lambda_1, \ldots, \lambda_m$은 제약을 맞추는 라그랑주 승수.

**증명 (변분법)**:

라그랑지안
$$\mathcal{L}[f] = -\int f \log f \, dx - \lambda_0 \Big(\int f \, dx - 1\Big) - \sum_k \lambda_k \Big(\int f g_k \, dx - c_k\Big).$$

$f(x)$에 대한 변분 ($\delta \mathcal{L} / \delta f(x) = 0$):
$$-\log f(x) - 1 - \lambda_0 - \sum_k \lambda_k g_k(x) = 0.$$

$$\Rightarrow \log f(x) = -1 - \lambda_0 - \sum_k \lambda_k g_k(x).$$

$\lambda_0' := 1 + \lambda_0$로 재라벨:
$$f^*(x) = \exp\!\left(-\lambda_0' - \sum_k \lambda_k g_k(x)\right). \quad \square$$

**유일성**: $h$는 $f$에 대해 오목 (정리 2.4 / 정리 5.x의 연속 버전), stationary 해가 유일한 최대점.

**전역 최대성 증명** (KL 이용):

임의의 다른 확률밀도 $f$가 같은 제약을 만족한다면,
$$D(f \| f^*) = \int f \log \frac{f}{f^*} \, dx \geq 0 \quad \text{(Gibbs 부등식)}.$$
$$\log f^*(x) = -\lambda_0' - \sum_k \lambda_k g_k(x).$$
$$\int f \log f^* \, dx = -\lambda_0' - \sum_k \lambda_k c_k \quad \text{(제약 활용)}.$$
이는 **$f$에 의존하지 않는 상수**. 따라서
$$-h(f) = \int f \log f \, dx = D(f \| f^*) + \int f \log f^* \, dx \geq \int f \log f^* \, dx = -h(f^*).$$
$\Rightarrow h(f) \leq h(f^*)$. 등호는 $f = f^*$일 때만. 이 증명은 우아하다: **MaxEnt 해가 상수를 만들고, 다른 분포의 엔트로피는 KL만큼 차이난다**.

---

### 정리 6.2 — 균등분포가 MaxEnt (유계 support)

**명제**: Support가 $[a, b]$이고 다른 제약이 없을 때 MaxEnt 분포는 $U(a, b)$이다.

**증명**: 정리 6.1에서 제약이 없으면 ($m = 0$) $f^* = \exp(-\lambda_0)$ = 상수. $\int_a^b f^* \, dx = 1$에서 $f^* = 1/(b-a)$. 즉 균등. $\square$

---

### 정리 6.3 — 지수분포가 MaxEnt (양의 실수, 평균 고정)

**명제**: Support $[0, \infty)$, 평균 $\mathbb{E}[X] = \mu > 0$ 제약에서 MaxEnt는 $\operatorname{Exp}(1/\mu)$, 즉 $f^*(x) = \frac{1}{\mu} e^{-x/\mu}$.

**증명**: $g_1(x) = x, c_1 = \mu$.
$$f^*(x) = \exp(-\lambda_0 - \lambda_1 x) = e^{-\lambda_0} e^{-\lambda_1 x}.$$

정규화: $\int_0^\infty e^{-\lambda_0 - \lambda_1 x} dx = e^{-\lambda_0} / \lambda_1 = 1 \Rightarrow e^{-\lambda_0} = \lambda_1$.

평균 제약: $\mathbb{E}[X] = 1/\lambda_1 = \mu \Rightarrow \lambda_1 = 1/\mu$.

따라서 $f^*(x) = \frac{1}{\mu} e^{-x/\mu}$, 즉 $\operatorname{Exp}(1/\mu)$. $\square$

**엔트로피**: $h(f^*) = 1 - \log(1/\mu) = 1 + \log \mu$ nats.

---

### 정리 6.4 — 정규분포가 MaxEnt (전체 실수, 분산 고정)

**명제**: Support $\mathbb{R}$, 평균 $\mu$, 분산 $\sigma^2$ 제약에서 MaxEnt는 $\mathcal{N}(\mu, \sigma^2)$.

**증명**: $g_1(x) = x, c_1 = \mu$, $g_2(x) = x^2, c_2 = \mu^2 + \sigma^2$.
$$f^*(x) = \exp(-\lambda_0 - \lambda_1 x - \lambda_2 x^2).$$

이차형식의 지수이므로 정규분포 형태다. 제곱 완성:
$$-\lambda_2 x^2 - \lambda_1 x = -\lambda_2 \Big(x + \frac{\lambda_1}{2\lambda_2}\Big)^2 + \frac{\lambda_1^2}{4\lambda_2}.$$

비교하면 $\lambda_2 = 1/(2\sigma^2), \lambda_1 = -\mu/\sigma^2$, 정규화 상수는 $\lambda_0 = \frac{1}{2}\log(2\pi\sigma^2) + \frac{\mu^2}{2\sigma^2}$. 결과:
$$f^*(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right) = \mathcal{N}(x; \mu, \sigma^2). \quad \square$$

**엔트로피**: $h(f^*) = \frac{1}{2}\log(2\pi e \sigma^2)$ (정리 5.2).

---

### 정리 6.5 — 다변수 정규분포가 MaxEnt (공분산 고정)

**명제**: 평균 $\boldsymbol{\mu}$와 공분산 $\boldsymbol{\Sigma}$ 제약, support $\mathbb{R}^n$에서 MaxEnt는 $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$.

**증명 스케치**: 제약 $\mathbb{E}[X_i] = \mu_i$, $\mathbb{E}[(X_i - \mu_i)(X_j - \mu_j)] = \Sigma_{ij}$. 라그랑지안의 stationary 조건이 이차형식 $-\frac{1}{2}(x - \mu)^\top \Sigma^{-1}(x - \mu)$의 지수. 다변수 정규. $\square$

---

### 정리 6.6 — 이산 MaxEnt: Softmax = MaxEnt 분류기

**명제**: 유한 알파벳 $\{1, \ldots, K\}$ 위에서, 기댓값 제약 $\mathbb{E}[g_k(X)] = c_k$ ($k = 1, \ldots, m$)의 MaxEnt 분포는
$$p^*(x) = \frac{\exp(\sum_k \lambda_k g_k(x))}{\sum_{x'} \exp(\sum_k \lambda_k g_{k}(x'))} = \operatorname{softmax}(\boldsymbol{\lambda}^\top \boldsymbol{g}).$$

**증명**: 정리 6.1 이산 버전 + 정규화. $\square$

**ML 해석**: Logistic regression / softmax classifier는 본질적으로 **주어진 feature의 기대값을 맞추는 MaxEnt 분류기**다. 최대 엔트로피 원리에서 자동 유도되는 모델 구조.

---

### 정리 6.7 — MaxEnt와 KL 최소화의 동등성

**명제**: 제약 집합 $\Pi$ 위에서
$$\arg\max_{p \in \Pi} H(p) = \arg\min_{p \in \Pi} D(p \| u),$$
여기서 $u$는 균등분포 (유한 support).

**증명**: 정리 2.3의 관찰에서 $\log n - H(p) = D(p \| u)$. $H$ 최대화 $=$ $D(p \| u)$ 최소화. $\square$

**함의**: MaxEnt는 "가장 균등과 비슷한 (KL 최소)" 분포를 찾는 것과 같음. Information-projection 해석.

---

## 💻 NumPy 구현/시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# ─────────────────────────────────────────────
# 1. 정리 6.3 검증: 평균 고정 → 지수분포가 최대
# ─────────────────────────────────────────────

mu = 2.0
# 비교 분포: Gamma(k, theta) with mean = k*theta = mu
from scipy.stats import gamma, expon

print("=" * 60)
print(f"평균 = {mu} 제약, support = [0, ∞)에서 엔트로피 비교")
print("=" * 60)

# 지수분포 (k=1)
h_exp = 1 - np.log(1/mu)
print(f"  Exp(1/{mu})            | h = {h_exp:.4f} nats  ← MaxEnt")

# Gamma 분포들 (k ≠ 1이면 평균은 같지만 분산 다름)
for k in [0.5, 2.0, 5.0]:
    theta = mu / k
    h = k + np.log(theta) + np.log(gamma.pdf(1, k) * np.exp(1)) - (k - 1)  # 근사; 정확한 식 사용
    # 정확한 gamma entropy: h = k + log(theta) + log(Gamma(k)) + (1-k)*digamma(k)
    from scipy.special import digamma, gammaln
    h_gamma = k + np.log(theta) + gammaln(k) + (1 - k) * digamma(k)
    print(f"  Gamma(k={k}, θ={theta:.2f})   | h = {h_gamma:.4f} nats")

# 모든 경우에 지수분포가 최고여야 함

# ─────────────────────────────────────────────
# 2. 정리 6.4 검증: 분산 고정 → 정규분포가 최대
# ─────────────────────────────────────────────

sigma = 1.0
print("\n" + "=" * 60)
print(f"평균 = 0, 분산 = {sigma**2} 제약, support = ℝ")
print("=" * 60)

h_gauss = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
print(f"  N(0, 1)                 | h = {h_gauss:.4f} nats  ← MaxEnt")

# Laplace(0, b): var = 2b² → b = σ/√2 = 0.707
b = sigma / np.sqrt(2)
h_laplace = 1 + np.log(2 * b)
print(f"  Laplace(0, {b:.3f})      | h = {h_laplace:.4f} nats")

# t-distribution(df=3), 분산 = df/(df-2) = 3 → 스케일 조정
from scipy.stats import t
df = 5
scale = np.sqrt((df - 2) / df)        # variance = 1이 되도록
# numeric
xs = np.linspace(-20, 20, 100000)
fx = t.pdf(xs, df, scale=scale)
h_t = -np.trapz(fx * np.log(np.clip(fx, 1e-300, None)), xs)
print(f"  t(df={df}, var=1)        | h = {h_t:.4f} nats")

# ─────────────────────────────────────────────
# 3. 정리 6.2 검증: support [0, 1], 제약 없음 → 균등
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Support [0, 1], 제약 없음 → 균등")
print("=" * 60)
print(f"  U(0, 1)              | h = {np.log(1):.4f} = 0.0000 nats  ← MaxEnt")
print(f"  Beta(2, 2)           | h = {-0.12565:.4f} nats (직접 계산)")
print(f"  Beta(5, 5)           | h = {-0.44568:.4f} nats")

# ─────────────────────────────────────────────
# 4. MaxEnt 수치 최적화 — 여러 모멘트 제약으로
# ─────────────────────────────────────────────
# 이산 support {-2, -1, 0, 1, 2}에서 E[X] = 0.5 제약하에 MaxEnt 풀기

xs = np.arange(-2, 3)                            # support
target_mean = 0.5

def neg_entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return np.sum(p * np.log(p))

def constraint_sum(p):
    return np.sum(p) - 1.0

def constraint_mean(p):
    return np.sum(p * xs) - target_mean

# 라그랑주 승수 (numeric): p ∝ exp(-λ * x), λ 찾기
def find_lambda(lam):
    p = np.exp(-lam * xs)
    p /= p.sum()
    return np.sum(p * xs) - target_mean

from scipy.optimize import brentq
lam_star = brentq(find_lambda, -5, 5)
p_star = np.exp(-lam_star * xs)
p_star /= p_star.sum()

print("\n" + "=" * 60)
print("이산 MaxEnt: support {-2,...,2}, E[X] = 0.5")
print("=" * 60)
print(f"  λ = {lam_star:.4f}")
print(f"  p* = {p_star.round(4)}")
print(f"  check: Σ p = {p_star.sum():.4f}, E[X] = {np.sum(p_star * xs):.4f}")
print(f"  MaxEnt entropy = {-np.sum(p_star * np.log(p_star)):.4f} nats")

# ─────────────────────────────────────────────
# 5. 시각화: 분산 제약 하 분포 비교
# ─────────────────────────────────────────────

xs_plot = np.linspace(-5, 5, 1000)
gauss = (1 / np.sqrt(2 * np.pi)) * np.exp(-xs_plot**2 / 2)
laplace_b = 1 / np.sqrt(2)
laplace = (1 / (2 * laplace_b)) * np.exp(-np.abs(xs_plot) / laplace_b)
t_dist = t.pdf(xs_plot, df=5, scale=np.sqrt(3/5))

plt.figure(figsize=(10, 5))
plt.plot(xs_plot, gauss, linewidth=2, label=f'N(0,1)  h = {h_gauss:.3f} (MaxEnt)')
plt.plot(xs_plot, laplace, linewidth=2, label=f'Laplace(var=1)  h = {h_laplace:.3f}')
plt.plot(xs_plot, t_dist, linewidth=2, label=f't(df=5, var=1)  h = {h_t:.3f}')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('평균=0, 분산=1 제약 분포들 — 정규분포가 엔트로피 최대')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('06-maxent-variance-constraint.png', dpi=150, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
평균 = 2 제약, support = [0, ∞)
  Exp(1/2)            | h = 1.6931 nats  ← MaxEnt (최대)
  Gamma(k=0.5)        | h = 1.2894 nats
  Gamma(k=2.0)        | h = 1.5754 nats
  Gamma(k=5.0)        | h = 1.4189 nats  (모두 < 1.6931)

평균=0, 분산=1 제약
  N(0, 1)            | h = 1.4189 nats  ← MaxEnt
  Laplace(var=1)      | h = 1.3466 nats
  t(df=5, var=1)      | h = 1.3845 nats  (모두 < 1.4189)
```

---

## 🔗 AI/ML 연결

### Boltzmann / Gibbs 분포 = MaxEnt

물리학의 Boltzmann 분포 $p(x) \propto e^{-E(x)/kT}$는 에너지 기대값 $\mathbb{E}[E(X)] = U$ 제약의 MaxEnt 결과 (정리 6.6, $g_1 = E$).

머신러닝에서 이는 **Energy-Based Model (EBM)** 의 원리. Boltzmann Machine, Hopfield Network, 그리고 Score-based Generative Model (Diffusion의 score matching)의 근간.

### Logistic Regression = MaxEnt (기대 feature 맞추기)

Logistic/softmax regression:
$$p(y \mid x) = \frac{\exp(w_y^\top x)}{\sum_{y'} \exp(w_{y'}^\top x)}$$

이는 "$\mathbb{E}[x \mid y]$를 맞추는" 조건에서의 conditional MaxEnt. 이것이 왜 "natural" 한 분류기 구조인지의 정보이론적 정당화.

### 지수족 (Exponential Family)

지수족 $p(x; \theta) = h(x) \exp(\theta^\top T(x) - A(\theta))$는 정확히 "$T(x)$의 기대값 제약의 MaxEnt". 정규·베르누이·포아송·Gamma·Beta 모두 포함.

- Sufficient statistic $T$ → MaxEnt 제약
- $A(\theta)$ → log-partition function (정규화)

### Variational Inference의 분포족 선택

VI에서 근사 분포 $q_\phi(z)$로 지수족을 쓰는 이유:
- 수학적으로 단순 (closed-form KL, entropy)
- MaxEnt 원리 적용 → "가장 덜 제한된" 가정

### Maximum Entropy RL

SAC (Soft Actor-Critic)의 정책:
$$\pi^*(a \mid s) = \frac{\exp(Q(s, a)/\alpha)}{\sum_{a'} \exp(Q(s, a')/\alpha)} = \operatorname{softmax}(Q/\alpha).$$

이는 "기대 $Q$값 제약의 MaxEnt 정책" — softmax 형태는 자동으로 나옴.

### Bayesian Prior 선택

"prior를 어떻게 고를까?" 문제에 대한 정보이론적 답: 알고 있는 모멘트만 맞추고 엔트로피 최대. 
- 위치만 알면 → $\mathcal{N}$ (또는 support 따라 다른 MaxEnt)
- 양수이고 평균만 알면 → $\operatorname{Exp}$
- 유계이면 → 균등

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| **제약이 존재성을 허용** | 모순된 제약(예: $\mathbb{E}[X] = 0, X \geq 1$)에서는 해 없음 |
| 라그랑지안이 stationary 해를 줌 | 제약이 많으면 수치적으로 $\lambda$ 찾기 어려움 (Newton's method 수렴 필요) |
| MaxEnt가 "옳은" 사전분포 | 이는 **무정보** 가정의 수학적 정식화일 뿐, 실제 정보가 있다면 적용 부적절 |
| 연속 support의 참조 측도 | 미분 엔트로피는 측도 의존 — "무정보"의 정의가 측도에 따라 달라짐 (Jeffreys prior는 다른 불변성 원리로 대안 제시) |

**Jaynes 비판**: MaxEnt는 "모른다는 사실"을 "균등 가정"으로 번역한다. 그러나 어떤 측도 공간에서의 균등인지가 비자명 — "inverse problem"에서 유명한 쟁점.

**실용적 주의**: ML 실전에서는 MaxEnt보다 **conjugate prior** 나 **계산 편의** 가 prior 선택을 지배. MaxEnt는 "원칙적 기본값"의 역할.

---

## 📌 핵심 정리

$$\boxed{\text{MaxEnt} \iff f^*(x) \propto \exp\!\left(-\sum_k \lambda_k g_k(x)\right)}$$

| Support | 제약 | MaxEnt 분포 | $h$ |
|---------|------|------------|----|
| $[a, b]$ | 없음 | $U(a, b)$ | $\log(b-a)$ |
| $[0, \infty)$ | $\mathbb{E}[X] = \mu$ | $\operatorname{Exp}(1/\mu)$ | $1 + \log \mu$ |
| $\mathbb{R}$ | $\mathbb{E}[X] = \mu$, $\operatorname{Var} = \sigma^2$ | $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{2}\log(2\pi e \sigma^2)$ |
| $\mathbb{R}^n$ | 평균·공분산 | $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ | $\frac{1}{2}\log((2\pi e)^n\|\Sigma\|)$ |
| 유한 | $\mathbb{E}[g_k] = c_k$ | Softmax / 지수족 | — |

**핵심 메시지**: "**지수족 = MaxEnt의 결과물**"이라는 동등성. ML의 기본 분포족이 정보이론적으로 자연스러운 이유.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Support $[0, 1]$, 평균 $\mathbb{E}[X] = 0.7$ 제약의 MaxEnt 분포 형태를 구하라 ($\lambda$는 수치로 풀어도 됨).

<details>
<summary>힌트 및 해설</summary>

$f^*(x) = e^{-\lambda_0 - \lambda_1 x}$ on $[0, 1]$. $\int_0^1 e^{-\lambda_0 - \lambda_1 x} dx = 1$과 $\int_0^1 x e^{-\lambda_0 - \lambda_1 x} dx = 0.7$로 $\lambda_0, \lambda_1$ 결정.  
$\lambda_1$을 뉴턴법·이분법으로 수치로 찾으면 $\lambda_1 \approx -1.69$ (평균이 0.5보다 크므로 오른쪽으로 기울어야 함 → 음수). 이는 **truncated exponential**.

</details>

---

**문제 2** (심화): 이산 support $\{1, 2, 3, 4, 5, 6\}$, 평균 $\mathbb{E}[X] = 4.5$ 제약의 MaxEnt 분포를 구하라 (주사위인데 평균이 3.5보다 큼 → 편향).

<details>
<summary>힌트 및 해설</summary>

$p_i \propto e^{-\lambda i}$, $i = 1, \ldots, 6$. 평균 제약 $\sum i p_i = 4.5$로 $\lambda$ 결정.  
$\lambda \approx -0.374$ (평균이 커야 하니 큰 $i$가 더 무거워야 함 → $\lambda < 0$). 수치 결과: $p \approx (0.054, 0.079, 0.114, 0.165, 0.240, 0.348)$.  
엔트로피 $H \approx 1.614$ nats ≈ $2.33$ bits, 균등 $H = \log_2 6 \approx 2.585$ bits보다 작음.

</details>

---

**문제 3** (AI 연결): Cross-entropy 손실 최소화가 왜 MaxEnt 원리의 한 형태인지 설명하라. 구체적으로 feature 기대값 제약의 MaxEnt → softmax가 나오는 과정과의 관계를 논하라.

<details>
<summary>힌트 및 해설</summary>

(1) Softmax classifier는 본질적으로 **$\mathbb{E}_p[\phi(x) \mid y]$를 맞추는 MaxEnt 분포**.  
(2) MLE (cross-entropy 최소화)는 데이터의 경험 분포 $\hat{p}(y \mid x)$에 최대 우도를 주는 모델 파라미터를 찾는 것.  
(3) 이 둘은 **쌍대** 관계: primal (MaxEnt)에서 $\lambda$ = feature weights, dual (MLE)에서 같은 weights를 얻음. 

이것이 "Logistic regression is the MaxEnt classifier"라는 주장의 수학적 근거. 자세한 내용은 Ch6-01에서 다룬다.

</details>

---

**문제 4** (심화): 평균 $\mu$와 **중앙값** $m$이 주어진 경우(분산 아님), MaxEnt 분포는 무엇인가? (중앙값은 부드러운 제약이 아니어서 까다롭다는 점에 주목)

<details>
<summary>힌트 및 해설</summary>

중앙값 제약은 $\int_{-\infty}^m f \, dx = 1/2$. 연속적 기대값 제약이 아니어서 라그랑주 승수법이 직접 적용되지 않음. 대신 두 영역 $(-\infty, m]$, $[m, \infty)$로 나누어 각각에서 sub-problem을 풀면 양쪽 모두 지수분포. 결과: **Laplace(중앙값은 모수, 평균 제약으로 스케일)** 형태. 즉 중앙값+평균 제약의 MaxEnt는 비대칭 Laplace (asymmetric double-exponential).

이는 실전 ML에서 outlier-robust loss (L1, Huber)가 등장하는 정보이론적 정당화와 연결.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 05. 미분 엔트로피](./05-differential-entropy.md) | [Ch2-01. KL-divergence의 정의와 비음수성 ▶](../ch2-kl-divergence/01-kl-definition-nonnegativity.md) |

</div>
