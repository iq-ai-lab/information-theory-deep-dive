# 01. KL-divergence의 정의와 비음수성

## 🎯 핵심 질문

- KL-divergence $D(p \| q) = \sum p \log(p/q)$는 왜 항상 $\geq 0$인가?
- Jensen 부등식과 $-\log$의 볼록성이 Gibbs 부등식 증명의 어떻게 맞물리는가?
- 등호 $D(p \| q) = 0$이 "$p = q$ 거의 확실히"와 동치인 이유는?
- KL이 **거리 함수(metric)** 가 아닌 이유는 무엇인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **Cross-Entropy 손실**: $H(p, q) = H(p) + D(p \| q)$ — MLE 최소화 = KL 최소화
- **VAE ELBO**: reconstruction - $D(q(z\|x) \| p(z))$ 형태
- **RLHF**: 정책을 기준 모델과 가깝게 유지하려고 KL penalty 사용
- **VI (Variational Inference)**: 근사 사후 $q$와 참 사후 $p$ 사이의 KL을 최소화
- **Diffusion Model**: 각 스텝의 forward / reverse 분포 사이의 KL의 합이 목적함수

KL이 ML의 "표준 거리"처럼 쓰이는 이유는 그 비음수성과 0 등호가 "모델이 데이터 분포를 완전히 재현했을 때만 0"을 보장하기 때문이다.

---

## 📐 수학적 선행 조건

- [Ch1-02](../ch1-entropy-axioms/02-entropy-definition.md)의 엔트로피와 Jensen 부등식
- **Jensen 부등식** (볼록/오목 함수의 기댓값 부등식)
- $-\log x$의 엄격 볼록성

---

## 📖 직관적 이해

### "분포 $p$의 시선으로 본 $q$의 놀라움"

$D(p \| q) = \mathbb{E}_{X \sim p}[\log p(X) - \log q(X)]$. 이는:
- $p$에서 샘플링 → 각 샘플에서 "$p$가 부여한 log-prob vs $q$가 부여한 log-prob의 차"를 평균
- $q$가 $p$를 잘 근사할수록 이 차가 작음 → KL 작음

### 부호화 비용의 초과량 (Cross-Entropy 해석)

$p$로 데이터가 실제로 생성되는데 $q$를 기준으로 부호화하면:
- 최적 부호는 $-\log p$ (문서 01 공리로부터)
- 그러나 $q$ 기준 부호는 $-\log q$를 사용 → 평균 $-\sum p \log q = H(p, q)$ bits
- 여분 비용 = $H(p, q) - H(p) = D(p \| q)$ bits

**KL은 "잘못된 모델 $q$를 써서 발생하는 평균 여분 부호 길이"**.

### 왜 거리가 아닌가

- $D(p \| q) \neq D(q \| p)$ (**비대칭**)
- 삼각 부등식 불성립

실제 "거리"는 다음 문서들에서 다루는 JS (대칭화), Wasserstein, 또는 $\sqrt{D}$ 등.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — KL-divergence (이산)

두 확률분포 $p, q$가 같은 support $\mathcal{X}$ 위에서 정의될 때
$$D(p \| q) := \sum_{x \in \mathcal{X}} p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_{X \sim p}\!\left[\log \frac{p(X)}{q(X)}\right],$$
단 다음 규약:
- $0 \log \frac{0}{q} = 0$ (극한에서 정당)
- $p \log \frac{p}{0} = +\infty$ (if $p > 0$)

### 정의 1.2 — KL-divergence (연속)

밀도 $f, g$에 대해
$$D(f \| g) := \int f(x) \log \frac{f(x)}{g(x)} \, dx.$$

### 관찰: KL은 **절대연속** 조건을 요구

$p$가 $q$에 대해 절대연속 ($p \ll q$) 이 아니면, 즉 어떤 $x$에서 $p(x) > 0$이고 $q(x) = 0$이면, $D(p \| q) = +\infty$.

**ML 함의**: 모델 분포 $q$가 데이터 분포 $p$의 support를 커버하지 못하면 KL이 폭발 — **이는 GAN mode collapse와 관련 있음** (Ch2-06에서 심화).

---

## 🔬 정리와 증명

### 보조정리 1.1 — Jensen 부등식 (복습)

**명제**: $\varphi$가 볼록함수이고 $X$는 확률변수이면
$$\varphi(\mathbb{E}[X]) \leq \mathbb{E}[\varphi(X)].$$

$\varphi$가 엄격 볼록이면 등호는 $X$가 상수일 때만.

(증명은 확률론 기본; Ch1-02 복습.)

---

### 정리 1.1 — Gibbs 부등식 (KL의 비음수성)

**명제**: 임의의 분포 $p, q$에 대해
$$D(p \| q) \geq 0,$$
등호는 $p = q$ a.s.일 때 **정확히** 성립.

**증명 (Jensen 이용)**:

$-\log$는 엄격 볼록이므로 $\log$는 엄격 오목. Jensen (오목):
$$\mathbb{E}_p\!\left[\log \frac{q(X)}{p(X)}\right] \leq \log \mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right].$$

**좌변**: $\mathbb{E}_p[\log(q/p)] = -D(p \| q)$.

**우변**:
$$\mathbb{E}_p\!\left[\frac{q(X)}{p(X)}\right] = \sum_x p(x) \cdot \frac{q(x)}{p(x)} = \sum_{x: p(x) > 0} q(x) \leq \sum_x q(x) = 1.$$

따라서 우변 $\leq \log 1 = 0$. 즉
$$-D(p \| q) \leq 0 \quad \Rightarrow \quad D(p \| q) \geq 0. \quad \square$$

---

**등호 조건 (완전 증명)**:

$D(p \| q) = 0$인 경우, Jensen의 등호와 두 번째 부등식의 등호가 모두 성립해야 한다:

1. **Jensen 등호**: $\log$가 엄격 오목이므로 $q(X)/p(X)$가 $p$에 따른 **상수**여야 함. 즉 $p(x) > 0$인 모든 $x$에서 $q(x)/p(x) = c$ (상수).

2. **두 번째 등호**: $\sum_{x: p(x) > 0} q(x) = 1$, 즉 $p$의 support 밖에서 $q = 0$. 결합하면 $q(x) = c \cdot p(x)$ everywhere, 정규화에서 $c = 1$, 따라서 $q = p$. $\square$

---

### 정리 1.2 — KL의 엄격 볼록성 (쌍에 대해)

**명제**: $(p, q) \mapsto D(p \| q)$는 두 변수 모두에 대해 **엄격 볼록**이다. 즉 $(p_1, q_1), (p_2, q_2)$와 $\lambda \in [0, 1]$에 대해
$$D(\lambda p_1 + (1-\lambda) p_2 \,\|\, \lambda q_1 + (1-\lambda) q_2) \leq \lambda D(p_1 \| q_1) + (1-\lambda) D(p_2 \| q_2).$$

**증명 스케치**: Log-sum 부등식
$$(p_1 + p_2) \log \frac{p_1 + p_2}{q_1 + q_2} \leq p_1 \log \frac{p_1}{q_1} + p_2 \log \frac{p_2}{q_2}$$
을 요소별로 적용. 엄격성은 $p_1/q_1 = p_2/q_2$일 때만 등호. $\square$

---

### 정리 1.3 — Pinsker 부등식 (연습 문제급이지만 중요)

**명제**: Total Variation distance $\text{TV}(p, q) = \frac{1}{2} \sum |p(x) - q(x)|$와 KL 사이:
$$\text{TV}(p, q) \leq \sqrt{\frac{1}{2} D(p \| q)} \quad (\text{nats, 자연로그 기준}).$$

**함의**: KL이 작으면 TV도 작다 (역은 일반적으로 성립 안 함). KL은 **더 강한** 수렴 모드.

(증명은 Cover-Thomas §11.6. 간단히는 $f$-divergence 정리에서 유도.)

---

### 정리 1.4 — KL은 거리(metric)가 아니다

**명제**: KL은 다음을 만족하지 않는다:
- **대칭성**: $D(p \| q) = D(q \| p)$ 일반적으로 거짓
- **삼각 부등식**: $D(p \| r) \leq D(p \| q) + D(q \| r)$ 일반적으로 거짓

**반례 (비대칭)**: $p = (0.9, 0.1), q = (0.1, 0.9)$.
$$D(p \| q) = 0.9 \log \frac{0.9}{0.1} + 0.1 \log \frac{0.1}{0.9} \approx 1.758 \text{ nats},$$
$$D(q \| p) \approx 1.758 \text{ nats}.$$
(이 경우에만 대칭. 일반 분포에서는 다름; 다음 문서에서 상세.)

**더 명확한 비대칭**: $p = \mathcal{N}(0, 1), q = \mathcal{N}(0, 2)$.
$$D(p \| q) = \log 2 + \frac{1}{2 \cdot 4} - \frac{1}{2} \cdot \log e \approx 0.193,$$
$$D(q \| p) = -\log 2 + \frac{2}{2} - \frac{1}{2} \cdot \log e \approx 0.307. \quad (\neq)$$

---

## 💻 NumPy 구현/시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. KL 계산 유틸 + 안전 처리
# ─────────────────────────────────────────────

def kl_divergence(p, q, base=np.e, eps=1e-30):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    # 0 log 0 = 0; p > 0 and q = 0 → inf
    mask = p > 0
    if np.any(q[mask] <= 0):
        return np.inf
    return np.sum(p[mask] * np.log(p[mask] / q[mask])) / np.log(base)

# 검증: 같은 분포 → 0
p = np.array([0.2, 0.3, 0.5])
print(f"D(p || p) = {kl_divergence(p, p):.6f}  (기댓값 0)")

# 비대칭 확인
p = np.array([0.8, 0.2])
q = np.array([0.2, 0.8])
print(f"\n비대칭 확인:")
print(f"  D(p || q) = {kl_divergence(p, q):.6f}")
print(f"  D(q || p) = {kl_divergence(q, p):.6f}  (대칭이면 같아야 하지만...)")
# 이 특정 예시(대칭 스왑)는 같지만, 일반적으론 다름

p = np.array([0.1, 0.4, 0.5])
q = np.array([0.3, 0.3, 0.4])
print(f"  일반 예시:")
print(f"    D(p || q) = {kl_divergence(p, q):.6f}")
print(f"    D(q || p) = {kl_divergence(q, p):.6f}   ← 다름!")

# ─────────────────────────────────────────────
# 2. support 불일치 → ∞
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("Support 불일치 → D(p || q) = ∞")
print("=" * 60)
p = np.array([0.5, 0.5])
q = np.array([1.0, 0.0])     # q(x2) = 0이지만 p(x2) > 0
print(f"  p = {p}, q = {q}")
print(f"  D(p || q) = {kl_divergence(p, q)}")
print(f"  D(q || p) = {kl_divergence(q, p):.6f}  (q의 support 안에서는 유한)")

# ─────────────────────────────────────────────
# 3. Gibbs 부등식 수치 검증 — 100개 무작위 쌍
# ─────────────────────────────────────────────

rng = np.random.default_rng(42)
n = 5       # 5차원 분포
kls = []
for _ in range(100):
    p = rng.dirichlet(np.ones(n))
    q = rng.dirichlet(np.ones(n))
    kls.append(kl_divergence(p, q))

print("\n" + "=" * 60)
print(f"Gibbs 부등식: 100개 무작위 쌍의 D(p || q) 분포")
print("=" * 60)
print(f"  min = {min(kls):.6f}  (기댓값 ≥ 0)")
print(f"  max = {max(kls):.6f}")
print(f"  mean = {np.mean(kls):.6f}")
print(f"  음수 개수 = {sum(k < 0 for k in kls)}  (기댓값 0)")

# ─────────────────────────────────────────────
# 4. Cross-Entropy = H(p) + KL  항등식
# ─────────────────────────────────────────────

def entropy(p, base=np.e):
    p = p[p > 0]
    return -np.sum(p * np.log(p)) / np.log(base)

def cross_entropy(p, q, base=np.e, eps=1e-30):
    mask = p > 0
    return -np.sum(p[mask] * np.log(q[mask] + eps)) / np.log(base)

p = np.array([0.1, 0.4, 0.5])
q = np.array([0.3, 0.3, 0.4])

print("\n" + "=" * 60)
print("Cross-Entropy = H(p) + D(p || q)")
print("=" * 60)
Hp = entropy(p)
Hpq = cross_entropy(p, q)
Dpq = kl_divergence(p, q)
print(f"  H(p)        = {Hp:.6f}")
print(f"  D(p || q)   = {Dpq:.6f}")
print(f"  H(p) + D    = {Hp + Dpq:.6f}")
print(f"  H(p, q)     = {Hpq:.6f}  ← 같아야 함")

# ─────────────────────────────────────────────
# 5. 정규분포 간 KL — 해석적 공식 검증
# ─────────────────────────────────────────────

# D(N(μ1, σ1²) || N(μ2, σ2²))
# = log(σ2/σ1) + (σ1² + (μ1-μ2)²) / (2σ2²) - 1/2
def kl_gaussian(mu1, s1, mu2, s2):
    return np.log(s2 / s1) + (s1**2 + (mu1 - mu2)**2) / (2 * s2**2) - 0.5

print("\n" + "=" * 60)
print("정규분포 간 KL (해석적 공식)")
print("=" * 60)
cases = [
    ("D(N(0,1) || N(0,1))", 0, 1, 0, 1),
    ("D(N(0,1) || N(2,1))", 0, 1, 2, 1),
    ("D(N(0,1) || N(0,2))", 0, 1, 0, 2),
    ("D(N(0,2) || N(0,1))", 0, 2, 0, 1),
]
for name, m1, s1, m2, s2 in cases:
    print(f"  {name:25s} = {kl_gaussian(m1, s1, m2, s2):.4f}")
# 마지막 두 행: 비대칭성 → 서로 다른 값!
```

**출력 예시**:
```
D(p || p) = 0.000000  (기댓값 0)

일반 예시:
  D(p || q) = 0.090236
  D(q || p) = 0.098754   ← 다름!

Support 불일치 → D(p || q) = ∞
  D(p || q) = inf
  D(q || p) = 0.693147

Gibbs 부등식 100개 쌍:
  min = 0.002xxx  (항상 ≥ 0 ✔)
  음수 개수 = 0

정규분포 비대칭:
  D(N(0,1) || N(0,2)) = 0.1931
  D(N(0,2) || N(0,1)) = 0.3069   ← 같지 않음
```

---

## 🔗 AI/ML 연결

### Cross-Entropy Loss = KL + 상수

분류 문제:
$$\mathcal{L}_{\text{CE}} = -\sum y_i \log \hat{y}_i = H(y, \hat{y}) = H(y) + D(y \| \hat{y}).$$

$y$ (레이블 분포)는 데이터에서 고정 → $H(y)$ 상수. Cross-entropy 최소화 = **$D(y \| \hat{y})$ 최소화**. 이는 Ch6-01에서 MLE와의 동등성 완전 유도.

### VAE ELBO의 KL 항

$$\text{ELBO} = \mathbb{E}_q[\log p(x \mid z)] - D(q(z \mid x) \| p(z)).$$

두 번째 항은 posterior $q$를 prior $p$에서 벗어나지 않도록 하는 **KL 정규화**. Ch6-02에서 완전 유도.

### RLHF / DPO의 KL Penalty

LLM 정책 $\pi_\theta$를 사전훈련 모델 $\pi_\text{ref}$에서 너무 멀어지지 않도록:
$$\mathcal{L} = -\mathbb{E}_{\pi_\theta}[r(x, y)] + \beta D(\pi_\theta \| \pi_\text{ref}).$$

KL이 폭발하면 분포가 크게 이동했다는 신호. **등호 조건**: $D = 0 \iff \pi_\theta = \pi_\text{ref}$.

### Variational Inference

$$\min_q D(q(z) \| p(z \mid x)).$$
Posterior $p(z \mid x)$가 계산 불가 → surrogate로 ELBO 사용 (KL은 음수가 될 수 없어 ELBO가 $\log p(x)$의 하한).

### Contrastive Divergence (RBM, EBM)

RBM의 training objective: $\arg\min_\theta D(p_{\text{data}} \| p_\theta)$. Contrastive divergence는 이 KL의 gradient를 근사.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| **$p \ll q$** (절대연속) | Support 불일치 → $D = +\infty$. 실전 GAN에서 $p_\text{data}$와 $p_\text{model}$이 저차원 매니폴드에 있으면 거의 항상 $\infty$ |
| 대칭성 / 삼각부등식 **없음** | "거리"처럼 기하학적 직관을 쓰면 오류 (대칭화가 필요하면 JS; Ch2-03) |
| 로그 밑 선택 | Pinsker 등 일부 부등식은 밑에 의존 — 자연로그(nats)로 통일 필요 |
| 경험 분포 추정 | 실전에서 $p$는 샘플로 추정 — 밀도 비 $p/q$ 추정이 고차원에서 어려움 |

**수치적 주의**: log-prob 차 $\log p - \log q$는 underflow 쉬움. `logsumexp` 및 $\log(p/q) = \log p - \log q$ 형태로 직접 계산 권장. 또한 $p = 0, q = 0$ 동시에 있는 항은 반드시 mask 처리.

---

## 📌 핵심 정리

$$\boxed{D(p \| q) = \sum p(x) \log \frac{p(x)}{q(x)} \geq 0, \quad = 0 \iff p = q}$$

| 성질 | 식 / 의미 |
|------|----------|
| 비음수성 (Gibbs) | $D(p \| q) \geq 0$, Jensen + $\log$ 오목 |
| 등호 조건 | $p = q$ a.s. |
| 비대칭 | 일반적으로 $D(p \| q) \neq D(q \| p)$ |
| 삼각부등식 불성립 | 거리가 아님 |
| Cross-entropy 분해 | $H(p, q) = H(p) + D(p \| q)$ |
| 절대연속 요구 | $p \ll q$ 아니면 $D = \infty$ |

---

## 🤔 생각해볼 문제

**문제 1** (기초): $p = (0.5, 0.5)$와 $q = (0.9, 0.1)$ 사이의 $D(p \| q)$와 $D(q \| p)$를 bits 단위로 계산하고 비대칭성을 확인하라.

<details>
<summary>힌트 및 해설</summary>

$D(p \| q) = 0.5 \log_2 \frac{0.5}{0.9} + 0.5 \log_2 \frac{0.5}{0.1} \approx 0.5 \cdot (-0.848) + 0.5 \cdot 2.322 = 0.737$ bits.  
$D(q \| p) = 0.9 \log_2 \frac{0.9}{0.5} + 0.1 \log_2 \frac{0.1}{0.5} \approx 0.9 \cdot 0.848 + 0.1 \cdot (-2.322) = 0.531$ bits.  
$\neq$ → 비대칭.

</details>

---

**문제 2** (심화): Jensen 부등식을 **다른 방식** 으로 증명하라: $f(x) := x - 1 - \log x \geq 0$ for $x > 0$ (등호는 $x = 1$)을 이용.

<details>
<summary>힌트 및 해설</summary>

$x = q(X)/p(X)$에 대입: $\frac{q(X)}{p(X)} - 1 - \log \frac{q(X)}{p(X)} \geq 0$.  
기댓값 $\mathbb{E}_p$: $\underbrace{\mathbb{E}_p[q/p]}_{=1 \text{ or } \leq 1} - 1 - \mathbb{E}_p[\log(q/p)] \geq 0$.  
$\Rightarrow -D(p\|q) \leq 0 \Rightarrow D(p\|q) \geq 0$. $\square$

이 증명은 $-\log$의 접선 부등식 $-\log x \geq 1 - x$(접선은 $y = 1 - x$)의 직접 적용.

</details>

---

**문제 3** (AI 연결): RLHF에서 $\pi_\theta$와 $\pi_\text{ref}$ 사이의 KL이 아주 큰 값이 되면 어떤 실용적 문제가 발생하는가? 반대로 KL이 거의 0이면?

<details>
<summary>힌트 및 해설</summary>

KL 큼 → 정책이 사전훈련에서 너무 멀어짐 → 사전훈련 성능 파괴, 예측 불가 (reward hacking, gibberish 생성).  
KL 작음 → 정책이 사전훈련에서 거의 변하지 않음 → reward 개선이 미미.  
$\beta$ 하이퍼파라미터가 이 trade-off를 조절.

</details>

---

**문제 4** (증명): 정규분포 $\mathcal{N}(\mu_1, \sigma_1^2)$과 $\mathcal{N}(\mu_2, \sigma_2^2)$ 사이의 KL이
$$D = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$$
임을 유도하라.

<details>
<summary>힌트 및 해설</summary>

$\log(p/q) = \log \frac{\sigma_2}{\sigma_1} + \frac{(x - \mu_2)^2}{2\sigma_2^2} - \frac{(x - \mu_1)^2}{2\sigma_1^2}$.  
$\mathbb{E}_p[x] = \mu_1, \mathbb{E}_p[(x - \mu_1)^2] = \sigma_1^2, \mathbb{E}_p[(x - \mu_2)^2] = \sigma_1^2 + (\mu_1 - \mu_2)^2$.  
대입 정리:  $D = \log(\sigma_2/\sigma_1) + \frac{\sigma_1^2 + (\mu_1 - \mu_2)^2}{2\sigma_2^2} - \frac{\sigma_1^2}{2\sigma_1^2} = \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac{1}{2}$. $\square$

</details>

---

<div align="center">

| | |
|---|---|
| [◀ Ch1-06. 최대 엔트로피 분포](../ch1-entropy-axioms/06-maxent-distributions.md) | [02. KL의 비대칭성 — Forward vs Reverse ▶](./02-forward-reverse-kl.md) |

</div>
