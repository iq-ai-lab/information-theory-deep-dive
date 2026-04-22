# 05. 미분 엔트로피(Differential Entropy)

## 🎯 핵심 질문

- 연속 확률변수의 엔트로피를 어떻게 정의하는가?
- 왜 미분 엔트로피는 **음수가 될 수 있는가**?
- 이산 엔트로피와 달리 미분 엔트로피가 **좌표 변환(스케일 변화)** 에 불변이 아닌 이유는?
- 정규분포의 미분 엔트로피 $\frac{1}{2} \log(2\pi e \sigma^2)$은 어떻게 유도되는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **Normalizing Flow / VAE**: 연속 잠재 변수 $z$의 엔트로피·KL 계산이 모두 미분 엔트로피 기반
- **Gaussian likelihood**: 회귀 모델의 NLL $-\log p(y \mid x)$는 정규분포 미분 엔트로피와 직접 연결
- **Continuous MI**: 표현 학습 (InfoNCE, MINE)은 연속 변수 MI를 추정 — 미분 엔트로피의 **차**
- **Reparameterization Trick**: $z = \mu + \sigma \odot \epsilon$의 Jacobian 보정이 필요한 이유 — 미분 엔트로피는 좌표에 의존

"연속 변수의 엔트로피는 이산과 다르다"는 사실을 놓치면 VAE ELBO의 부호·수치가 이상해진다.

---

## 📐 수학적 선행 조건

- 적분과 밀도 함수: $\int f(x) dx = 1$, $f(x) \geq 0$
- 가우시안 밀도: $\mathcal{N}(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\!\big(-\frac{(x-\mu)^2}{2\sigma^2}\big)$
- 다변수 정규분포, 공분산 행렬, 행렬식

> 공분산 행렬과 행렬식이 필요하므로 [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive)가 도움이 됩니다.

---

## 📖 직관적 이해

### 연속으로 넘어갈 때의 문제

이산 엔트로피는 $H = -\sum p \log p \geq 0$. 연속으로 $p \to f$로 바꾸고 적분으로 바꾸면:
$$h(X) = -\int f(x) \log f(x) \, dx.$$

그러나 밀도 $f$는 확률이 아닌 **밀도** (단위: $1/\text{길이}$). 따라서 $\log f$는 음수가 될 수도, 매우 클 수도 있다 — 엔트로피의 부호 제약이 사라진다.

**예**: 균등분포 $U(0, 0.1)$의 밀도는 $f = 10$. 엔트로피 $h = -\int 10 \log 10 \, dx = -\log 10 < 0$. **음수다!**

### 미분 엔트로피는 "상대적" 양

이산 엔트로피가 "절대 정보량 (bits/nats)"이라면, 미분 엔트로피는 **좌표계 기준의 상대적 불확실성**이다. 좌표를 바꾸면 값이 바뀐다. 그러나 두 분포의 미분 엔트로피 **차이**(KL이나 MI)는 좌표에 무관한 본질적 양으로 남는다.

### 왜 정규분포가 특별한가

고정된 분산을 가지는 연속 분포 중에서 **정규분포가 최대 미분 엔트로피**를 가진다 (문서 06). 이는 "가장 덜 제한된 분포"라는 Bayesian uninformative prior 선택의 정보이론적 정당화.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — 미분 엔트로피

연속 확률변수 $X$가 밀도 $f: \mathbb{R} \to \mathbb{R}_{\geq 0}$를 가질 때, 미분 엔트로피는
$$h(X) := -\int_{\mathcal{S}} f(x) \log f(x) \, dx = \mathbb{E}_{X \sim f}\!\big[-\log f(X)\big],$$
여기서 $\mathcal{S} = \{x : f(x) > 0\}$ (support), 적분이 존재한다고 가정.

### 정의 5.2 — 결합 / 조건부 미분 엔트로피

$$h(X, Y) = -\int f(x, y) \log f(x, y) \, dx \, dy, \quad h(X \mid Y) = -\int f(x, y) \log f(x \mid y) \, dx \, dy.$$

Chain rule $h(X, Y) = h(X) + h(Y \mid X)$가 그대로 성립.

### 정의 5.3 — 연속 상호정보량

$$I(X; Y) = \int f(x, y) \log \frac{f(x, y)}{f(x) f(y)} \, dx \, dy = D(f_{XY} \| f_X f_Y).$$

**중요**: 연속 MI는 **비음수이고 좌표 변환에 불변** — 미분 엔트로피의 단점을 물려받지 않는다.

---

## 🔬 정리와 증명

### 정리 5.1 — 균등 분포의 미분 엔트로피

**명제**: $X \sim U(a, b)$이면
$$h(X) = \log(b - a).$$

**증명**:
$$h(X) = -\int_a^b \frac{1}{b-a} \log \frac{1}{b-a} \, dx = -\log \frac{1}{b-a} = \log(b-a). \quad \square$$

**함의**: $b - a < 1$이면 $h < 0$. 미분 엔트로피가 음수가 될 수 있는 **명시적 반례**.

---

### 정리 5.2 — 정규분포의 미분 엔트로피

**명제**: $X \sim \mathcal{N}(\mu, \sigma^2)$이면
$$h(X) = \frac{1}{2} \log(2\pi e \sigma^2) = \frac{1}{2} \log(2\pi \sigma^2) + \frac{1}{2}.$$

**증명**:
$$-\log f(x) = \frac{1}{2} \log(2\pi \sigma^2) + \frac{(x - \mu)^2}{2\sigma^2}.$$

기댓값:
$$h(X) = \mathbb{E}[-\log f(X)] = \frac{1}{2} \log(2\pi \sigma^2) + \frac{\mathbb{E}[(X - \mu)^2]}{2\sigma^2} = \frac{1}{2} \log(2\pi \sigma^2) + \frac{\sigma^2}{2\sigma^2} = \frac{1}{2} \log(2\pi \sigma^2) + \frac{1}{2}.$$

$\frac{1}{2} = \frac{1}{2} \log e$ (자연로그)이므로 $h = \frac{1}{2} \log(2\pi e \sigma^2)$. $\square$

**해석**: $\sigma$가 클수록 엔트로피 증가 (선형 in $\log \sigma$). 분산이 작을수록 예측 가능.

---

### 정리 5.3 — 다변수 정규분포의 미분 엔트로피

**명제**: $\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, $\dim = n$이면
$$h(\mathbf{X}) = \frac{1}{2} \log\big((2\pi e)^n |\boldsymbol{\Sigma}|\big) = \frac{n}{2} \log(2\pi e) + \frac{1}{2} \log |\boldsymbol{\Sigma}|.$$

**증명 스케치**: 정리 5.2의 다변수 버전. 밀도의 로그:
$$-\log f(\mathbf{x}) = \frac{n}{2} \log(2\pi) + \frac{1}{2} \log |\boldsymbol{\Sigma}| + \frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}).$$

$\mathbb{E}[(X - \mu)^\top \Sigma^{-1} (X - \mu)] = \operatorname{tr}(\Sigma^{-1} \Sigma) = n$ (trace trick).

$$h = \frac{n}{2} \log(2\pi) + \frac{1}{2} \log |\Sigma| + \frac{n}{2} = \frac{1}{2} \log\big((2\pi e)^n |\Sigma|\big). \quad \square$$

**함의**: 공분산 행렬의 **행렬식**(체적)이 엔트로피를 결정. 상관이 강해 $|\Sigma|$가 작으면 엔트로피가 낮음 (분포가 저차원 부분공간에 "뭉쳐" 있음).

---

### 정리 5.4 — 스케일 변환 공식

**명제**: $Y = aX + b$ ($a \neq 0$)이면
$$h(Y) = h(X) + \log |a|.$$

**증명**: 변수 변환으로 $f_Y(y) = f_X((y - b)/a) / |a|$.
$$h(Y) = -\int f_Y(y) \log f_Y(y) \, dy = -\int f_Y(y) [\log f_X((y-b)/a) - \log |a|] \, dy$$
$$= -\mathbb{E}_Y[\log f_X((Y-b)/a)] + \log|a| = h(X) + \log |a|.$$

(마지막 동등: $X = (Y - b)/a$의 분포는 원래 $X$이므로 $\mathbb{E}_Y[\log f_X((Y-b)/a)] = \mathbb{E}_X[\log f_X(X)] = -h(X)$.) $\square$

**함의**: 단위 변경(예: m → cm, 스케일 100배)만으로 $h$가 $\log 100$만큼 바뀐다. 이것이 미분 엔트로피가 **좌표 의존적**이라는 핵심 증거.

**다변수 일반화**: $\mathbf{Y} = A \mathbf{X} + \mathbf{b}$, $A$ 가역이면 $h(\mathbf{Y}) = h(\mathbf{X}) + \log |\det A|$.

---

### 정리 5.5 — 평행이동 불변

**명제**: $Y = X + b$이면 $h(Y) = h(X)$.

**증명**: 정리 5.4에서 $a = 1$, $\log 1 = 0$. $\square$

---

### 정리 5.6 — KL과 MI는 좌표 불변 (미분 엔트로피의 단점을 보완)

**명제**: 가역 매핑 $\mathbf{Y} = g(\mathbf{X})$에 대해
$$D(f_Y \| f'_Y) = D(f_X \| f'_X), \quad I(Y_1; Y_2) = I(X_1; X_2).$$

(여기서 $f_Y, f'_Y$는 $\mathbf{X}, \mathbf{X}'$가 각각 $f_X, f'_X$를 가질 때의 $g$에 의한 push-forward)

**증명 스케치**: 변수 변환에서 두 분포의 Jacobian 보정이 같은 크기로 발생하여 **상쇄**. 자세한 증명은 Cover-Thomas §8. 5.

**의미**: 스케일 변환 문제로 미분 엔트로피는 "절대 정보량"이 아니지만, **KL-divergence와 MI는 좌표 불변의 물리적 양**. ML의 실제 손실(KL, MI)은 이 불변성을 유지한다.

---

## 💻 NumPy 구현/시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, uniform, expon

# ─────────────────────────────────────────────
# 1. 균등·정규·지수 분포의 미분 엔트로피 (이론 + 수치)
# ─────────────────────────────────────────────

# 수치 적분 유틸 (자연로그 기준, nats)
def h_numeric(pdf, a, b, n=10000):
    xs = np.linspace(a, b, n)
    f = pdf(xs)
    mask = f > 0
    integrand = -f[mask] * np.log(f[mask])
    return np.trapz(integrand, xs[mask])

print("=" * 60)
print("미분 엔트로피: 이론값 vs 수치 적분 (nats)")
print("=" * 60)

# (1) 균등 U(0, 2) — h = log(2)
h_theory = np.log(2)
h_num = h_numeric(lambda x: uniform.pdf(x, 0, 2), -1, 3)
print(f"  U(0, 2)           | theory = {h_theory:.4f}  | numeric = {h_num:.4f}")

# (2) 균등 U(0, 0.1) — h = log(0.1) = -log(10) ← 음수!
h_theory = np.log(0.1)
h_num = h_numeric(lambda x: uniform.pdf(x, 0, 0.1), -0.05, 0.15)
print(f"  U(0, 0.1)         | theory = {h_theory:.4f}  | numeric = {h_num:.4f}  ← 음수 확인")

# (3) 정규 N(0, 1)
h_theory = 0.5 * np.log(2 * np.pi * np.e * 1.0)
h_num = h_numeric(lambda x: norm.pdf(x, 0, 1), -10, 10)
print(f"  N(0, 1)           | theory = {h_theory:.4f}  | numeric = {h_num:.4f}")

# (4) 정규 N(0, σ²) σ = 0.1 — 매우 좁음 → 음수
sigma = 0.1
h_theory = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
h_num = h_numeric(lambda x: norm.pdf(x, 0, sigma), -1, 1)
print(f"  N(0, 0.1²)        | theory = {h_theory:.4f}  | numeric = {h_num:.4f}  ← 음수")

# (5) 지수 Exp(λ=1) — h = 1 - log(λ) = 1
h_theory = 1 - np.log(1.0)
h_num = h_numeric(lambda x: expon.pdf(x, scale=1.0), 0, 30)
print(f"  Exp(λ=1)          | theory = {h_theory:.4f}  | numeric = {h_num:.4f}")

# ─────────────────────────────────────────────
# 2. 스케일 변환 공식 h(aX) = h(X) + log|a| 검증
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("정리 5.4 검증:  h(aX) = h(X) + log|a|")
print("=" * 60)
sigma = 1.0
h_base = 0.5 * np.log(2 * np.pi * np.e * sigma**2)
print(f"  h(X), X ~ N(0, 1)  = {h_base:.4f}")
for a in [0.5, 2.0, 10.0]:
    sigma_new = sigma * a
    h_scaled = 0.5 * np.log(2 * np.pi * np.e * sigma_new**2)
    expected = h_base + np.log(abs(a))
    print(f"  h(aX), a={a:4.1f}    = {h_scaled:.4f}  (예상: {expected:.4f})  diff = {abs(h_scaled-expected):.2e}")

# ─────────────────────────────────────────────
# 3. 시각화: σ 변화에 따른 h(X) (정규분포)
# ─────────────────────────────────────────────

sigmas = np.linspace(0.05, 5, 200)
h_vals = 0.5 * np.log(2 * np.pi * np.e * sigmas**2)

plt.figure(figsize=(8, 5))
plt.plot(sigmas, h_vals, linewidth=2)
plt.axhline(0, color='k', linestyle='--', alpha=0.5, label='h = 0')
plt.fill_between(sigmas, h_vals, 0, where=(h_vals < 0), alpha=0.2, color='red', label='h < 0 (음수 영역)')
plt.fill_between(sigmas, h_vals, 0, where=(h_vals > 0), alpha=0.2, color='green')
plt.xlabel(r'$\sigma$')
plt.ylabel(r'$h(X)$ (nats)')
plt.title(r'정규분포 미분 엔트로피: $h = \frac{1}{2}\log(2\pi e \sigma^2)$ — $\sigma < 1/\sqrt{2\pi e}$에서 음수')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('05-differential-entropy-sigma.png', dpi=150, bbox_inches='tight')
plt.show()

# h(N(0, σ²)) = 0이 되는 σ 값
sigma_zero = 1 / np.sqrt(2 * np.pi * np.e)
print(f"\n  h = 0 되는 σ = 1/sqrt(2πe) ≈ {sigma_zero:.4f}")
print(f"  σ < {sigma_zero:.4f}  →  h < 0  (좁은 정규분포)")

# ─────────────────────────────────────────────
# 4. 다변수 정규분포: h = (n/2) log(2πe) + (1/2) log|Σ|
# ─────────────────────────────────────────────

n = 3
Sigma = np.array([[2.0, 0.5, 0.1],
                  [0.5, 1.0, 0.3],
                  [0.1, 0.3, 1.5]])
det = np.linalg.det(Sigma)
h_mvn = 0.5 * np.log((2 * np.pi * np.e)**n * det)

print("\n" + "=" * 60)
print(f"다변수 정규분포 N(0, Σ), n={n}")
print("=" * 60)
print(f"  |Σ| = {det:.4f}")
print(f"  h(X) = {h_mvn:.4f} nats")
print(f"        = {h_mvn / np.log(2):.4f} bits")
```

**출력 예시**:
```
미분 엔트로피: 이론값 vs 수치 적분 (nats)
  U(0, 2)          | theory = 0.6931  | numeric = 0.6931
  U(0, 0.1)        | theory = -2.3026 | numeric = -2.3026  ← 음수 확인
  N(0, 1)          | theory = 1.4189  | numeric = 1.4189
  N(0, 0.1²)       | theory = -0.8837 | numeric = -0.8837  ← 음수
  Exp(λ=1)         | theory = 1.0000  | numeric = 1.0000

정리 5.4 검증
  h(X), X ~ N(0,1) = 1.4189
  h(0.5·X)         = 0.7258  (예상: 0.7258) diff = 1.1e-16
  h(2·X)           = 2.1120  (예상: 2.1120) diff = 2.2e-16
  h(10·X)          = 3.7216  (예상: 3.7216) diff = 0.0
```

---

## 🔗 AI/ML 연결

### VAE의 KL 항 (정규분포 간 KL)

VAE ELBO에서
$$D\big(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, 1)\big) = \frac{1}{2}\big[\mu^2 + \sigma^2 - 1 - \log \sigma^2\big].$$

이 공식은 두 정규분포의 미분 엔트로피 차 + 교차엔트로피의 직접 계산에서 유도. $\log \sigma^2$ 항의 출처가 바로 미분 엔트로피 $h = \frac{1}{2} \log(2\pi e \sigma^2)$의 일부.

### Normalizing Flow의 Jacobian 항

$\mathbf{z} = g_\theta(\mathbf{x})$ 가역 변환 → 밀도 변화:
$$\log p_X(\mathbf{x}) = \log p_Z(g(\mathbf{x})) + \log |\det J_g(\mathbf{x})|.$$

Jacobian 보정은 정리 5.4 (다변수 버전) 직접 적용 — **미분 엔트로피가 좌표에 의존하므로 매핑의 체적 변화를 보정해야** 정상적 확률 밀도가 됨.

### Gaussian Likelihood Regression

회귀 모델 $y \sim \mathcal{N}(f_\theta(x), \sigma^2)$의 NLL:
$$-\log p(y \mid x) = \frac{(y - f_\theta(x))^2}{2\sigma^2} + \frac{1}{2} \log(2\pi \sigma^2).$$

두 번째 항이 정규분포 미분 엔트로피 $h$. $\sigma$가 학습 가능하면 엔트로피 항이 자동으로 적절히 조정 (heteroscedastic regression).

### Continuous MI Estimation (MINE, InfoNCE)

연속 변수의 MI는
$$I(X; Z) = h(X) + h(Z) - h(X, Z),$$
각 미분 엔트로피의 **차로**만 나타남 → 좌표 불변 (정리 5.6). MINE은 Donsker-Varadhan 변분 표현으로 이 MI를 직접 추정 (Ch3-04).

### Reparameterization Trick

$z = \mu + \sigma \odot \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$. Jacobian $|\det \operatorname{diag}(\sigma)| = \prod_i \sigma_i$.  
VAE의 gradient가 올바르게 흘러가려면 Jacobian 보정이 필요 — 이는 정리 5.4의 다변수 버전.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 밀도 $f$가 존재 | 이산·연속 혼합 분포는 미분 엔트로피로 다룰 수 없음 |
| **좌표에 의존** | $h(X)$는 "절대 정보량"이 아님 — 단위/스케일에 따라 값 달라짐 |
| **음수 가능** | 이산 엔트로피와 달리 $h < 0$ 가능 — 해석 시 주의 |
| 적분의 유한성 | $h(X) = \pm \infty$인 분포 존재 (예: Cauchy는 $\int f \log f$가 수렴하지만 평균은 무정의) |

**핵심 경고**: 미분 엔트로피를 "분포가 얼마나 혼란스러운가"의 **절대 지표**로 쓰면 안 된다. 두 분포의 KL 또는 MI처럼 **차이·상대량**으로만 의미가 있다.

**측도-이론적 의견**: 미분 엔트로피는 Lebesgue 측도 기준. 다른 기준 측도 $\mu$에 대한 **상대 엔트로피** $-\int f \log(f/\mu) \, d\mu$로 일반화하면 좌표 불변이 회복됨. 이것이 KL의 일반적 정의다.

---

## 📌 핵심 정리

$$\boxed{h(X) = -\int f(x) \log f(x) \, dx}$$

| 분포 | $h(X)$ |
|------|-------|
| $U(a, b)$ | $\log(b - a)$ |
| $\mathcal{N}(\mu, \sigma^2)$ | $\frac{1}{2} \log(2\pi e \sigma^2)$ |
| $\mathcal{N}(\mu, \Sigma), \dim n$ | $\frac{n}{2} \log(2\pi e) + \frac{1}{2} \log \|\Sigma\|$ |
| $\operatorname{Exp}(\lambda)$ | $1 - \log \lambda$ |
| $\operatorname{Laplace}(0, b)$ | $1 + \log(2b)$ |

**핵심 성질**:
- 음수 가능
- $h(aX + b) = h(X) + \log |a|$ (스케일에 의존, 평행이동에 불변)
- KL과 MI는 좌표 불변 (미분 엔트로피의 단점을 보상)

---

## 🤔 생각해볼 문제

**문제 1** (기초): $X \sim U(0, L)$일 때 $h(X)$를 구하고, $L$이 커질수록 $h$가 어떻게 변하는지 설명하라. $h = 0$이 되는 $L$ 값은?

<details>
<summary>힌트 및 해설</summary>

$h(X) = \log L$ nats.  
$L$이 커지면 $h$ 증가 (분포가 넓어질수록 불확실).  
$L = 1 \Rightarrow h = 0$.  
$L < 1 \Rightarrow h < 0$ (좁은 분포는 음의 엔트로피).

</details>

---

**문제 2** (심화): 두 정규분포 $p = \mathcal{N}(0, \sigma_1^2)$, $q = \mathcal{N}(0, \sigma_2^2)$ 사이의 $D(p\|q)$를 계산하고, $\sigma_1 = \sigma_2$일 때 $D = 0$임을 확인하라.

<details>
<summary>힌트 및 해설</summary>

$$D(p \| q) = \mathbb{E}_p\!\left[\log \frac{p(X)}{q(X)}\right] = \mathbb{E}_p\!\left[-\frac{X^2}{2\sigma_1^2} + \frac{X^2}{2\sigma_2^2} + \log \frac{\sigma_2}{\sigma_1}\right]$$
$$= \log \frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2}{2\sigma_2^2} - \frac{1}{2}.$$
$\sigma_1 = \sigma_2$일 때 $= 0 + \frac{1}{2} - \frac{1}{2} = 0$. ✔

</details>

---

**문제 3** (AI 연결): VAE에서 $q(z \mid x) = \mathcal{N}(\mu, \sigma^2)$와 $p(z) = \mathcal{N}(0, 1)$ 사이의 KL이
$$D = \frac{1}{2}[\mu^2 + \sigma^2 - 1 - \log \sigma^2]$$
임을 미분 엔트로피 공식을 이용해 유도하라.

<details>
<summary>힌트 및 해설</summary>

$D(q \| p) = \mathbb{E}_q[\log q - \log p]$.  
$\mathbb{E}_q[\log q] = -h(q) = -\frac{1}{2} \log(2\pi e \sigma^2)$.  
$\mathbb{E}_q[-\log p] = \mathbb{E}_q\!\left[\frac{1}{2} \log(2\pi) + \frac{Z^2}{2}\right] = \frac{1}{2}\log(2\pi) + \frac{\mu^2 + \sigma^2}{2}$.  
합치면 $D = \frac{1}{2}\log(2\pi) + \frac{\mu^2+\sigma^2}{2} - \frac{1}{2}\log(2\pi e \sigma^2) = \frac{1}{2}[\mu^2 + \sigma^2 - 1 - \log\sigma^2]$. ✔

</details>

---

**문제 4** (증명): $Y = aX + b$에 대해 $I(Y; Z) = I(X; Z)$ ($a \neq 0$)임을 정리 5.4와 MI의 정의를 이용해 증명하라. 이것이 왜 ML에서 중요한 사실인지 설명하라.

<details>
<summary>힌트 및 해설</summary>

$I(Y; Z) = h(Y) + h(Z) - h(Y, Z)$. 정리 5.4에서 $h(Y) = h(X) + \log|a|$, $h(Y, Z) = h(X, Z) + \log|a|$ (Jacobian은 $X$ 축에만 적용). $\log|a|$가 두 항에서 상쇄되어 $I(Y; Z) = I(X; Z)$. ✔

**ML 중요성**: feature를 스케일링해도 MI는 변하지 않음 → MI-based feature selection은 스케일에 robust. 이는 편미분이 스케일에 의존하는 것과 대조적이다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 04. Chain Rule과 정보의 계층 구조](./04-chain-rule-hierarchy.md) | [06. 최대 엔트로피 분포 ▶](./06-maxent-distributions.md) |

</div>
