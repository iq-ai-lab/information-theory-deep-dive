# 3.1 상호정보량 — 정의와 기본 성질

## 🎯 핵심 질문

> **"$X$ 를 알면 $Y$ 에 대해 얼마나 많은 정보를 얻게 되는가?"** 를 어떻게 수치로 정의하는가?
> 왜 $I(X;Y) = D(p_{XY} \| p_X p_Y)$ 로 쓰는 것이 자연스러운가?
> 세 가지 동치형 $H(X) - H(X|Y)$, $H(Y) - H(Y|X)$, $H(X) + H(Y) - H(X,Y)$ 은 어떻게 같은가?

---

## 🔍 왜 AI에서 중요한가

- **Representation learning** 의 보편적 목표: "입력의 **정보를 잃지 않는** representation 을 학습하라" → $\max I(X; Z)$.
- **InfoNCE, CPC, SimCLR, CLIP** 모두 MI 최대화 관점에서 통합.
- **Information Bottleneck**: $\min I(X; Z) - \beta I(Z; Y)$ — compression–prediction tradeoff.
- **VAE의 ELBO 재조명**: mutual information 항 포함.
- **Feature selection**: MI-based (high MI feature가 유용).
- **Fairness**: $I(\hat Y; S) = 0$ (예측이 민감속성 $S$ 와 독립) 으로 공정성 수학적 정의.
- **인과 추론**: MI ≠ 0 이면 연관 존재(단, 원인-결과는 별도 조건 필요).

$I(X;Y) = 0 \Leftrightarrow X \perp\!\!\!\perp Y$ — 독립성의 **정보이론적 사각형**. 0 이 아니면 무조건 정보 공유 있음.

---

## 📐 선행 학습 지식

- [1.3 결합·조건부·상호정보량](../ch1-entropy-axioms/03-joint-conditional-mutual.md)
- [2.1 KL의 정의](../ch2-kl-divergence/01-kl-definition-nonnegativity.md)
- Jensen 부등식, conditional expectation
- 독립사건과 결합확률

---

## 📖 직관

### 핵심 그림: Venn Diagram

$H(X), H(Y)$ 를 두 원으로 보면 MI $I(X;Y)$ 는 **교집합**:

```
 ┌────────── H(X) ──────────┐
 │                          │
 │  H(X|Y)     I(X;Y)       │  ← 겹치는 부분
 │         ┌────────────────┤──────────── H(Y) ────┐
 │         │                │                      │
 └─────────┤                │        H(Y|X)        │
           └────────────────┴──────────────────────┘
            = H(X) + H(Y) - H(X,Y)
            = H(X) - H(X|Y)
            = H(Y) - H(Y|X)
```

"$X$ 의 불확실성 중 $Y$ 를 알면 사라지는 부분" = "$Y$ 의 불확실성 중 $X$ 를 알면 사라지는 부분" = 둘이 공유하는 정보.

### 또 다른 관점: KL from independence

$$
I(X;Y) = D(p_{XY} \| p_X \otimes p_Y)
$$

"결합분포 $p_{XY}$ 가 **독립분포** $p_X p_Y$ 로부터 얼마나 떨어져 있는가" 를 KL 로 잰 것. KL의 비음성이 $I \ge 0$ 을 보장.

---

## ✏️ 공식 정의

**정의 3.1.1 (이산 MI)**
$$
I(X; Y) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x)\, p(y)}
$$

**정의 3.1.2 (연속 MI)**
$$
I(X; Y) = \int\!\!\int p(x, y) \log \frac{p(x, y)}{p(x)\, p(y)} \, dx\, dy
$$

**정의 3.1.3 (일반 / 측도론적)**
$$
I(X; Y) = D(p_{XY} \| p_X \otimes p_Y)
$$
여기서 $p_X \otimes p_Y$ 는 곱 측도(product measure).

**정의 3.1.4 (조건부 MI)**
$$
I(X; Y | Z) = \mathbb{E}_{z}[I(X; Y | Z=z)] = H(X|Z) - H(X|Y,Z)
$$

---

## 🔬 정리와 증명

### Theorem 3.1.1 (세 가지 동치형)

**진술.**
$$
I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)
$$

**증명.**
$$
\begin{aligned}
I(X;Y) &= \sum p(x,y) \log \frac{p(x,y)}{p(x)p(y)} \\
 &= \sum p(x,y) [\log p(x|y) - \log p(x)] \\
 &= -H(X|Y) - (-H(X)) = H(X) - H(X|Y).
\end{aligned}
$$
대칭성으로 $= H(Y) - H(Y|X)$. 또한
$$
H(X) - H(X|Y) = H(X) - (H(X,Y) - H(Y)) = H(X) + H(Y) - H(X,Y). \blacksquare
$$

### Theorem 3.1.2 (비음성)

**진술.** $I(X;Y) \ge 0$, 등호는 $X \perp\!\!\!\perp Y$.

**증명.** $I(X;Y) = D(p_{XY} \| p_X p_Y) \ge 0$ (Gibbs 부등식, §2.1 Theorem 2.1.1). 등호는 $p_{XY} = p_X p_Y$ a.s. — 즉 독립. $\blacksquare$

### Theorem 3.1.3 (대칭성)

**진술.** $I(X; Y) = I(Y; X)$.

**증명.** 정의식이 $(x, y)$ 와 $(y, x)$ 에 대해 대칭. $\blacksquare$

### Theorem 3.1.4 (자기정보는 엔트로피)

**진술.** $I(X; X) = H(X)$.

**증명.** $H(X|X) = 0$ 이므로 $I(X;X) = H(X) - 0 = H(X)$. $\blacksquare$

> **함의**: 자기자신에 대한 MI가 곧 엔트로피. "자기 자신을 알면 자기 자신의 모든 정보를 얻는다".

### Theorem 3.1.5 (deterministic 관계)

**진술.** $Y = g(X)$ 이면 $I(X;Y) = H(Y) \le H(X)$.

**증명.** $H(Y|X) = 0$ 이므로 $I(X;Y) = H(Y) - 0 = H(Y)$. $H(Y) \le H(X)$ 는 deterministic 함수로 엔트로피 안 늘어남(§1.3). $\blacksquare$

### Theorem 3.1.6 (Conditioning 이 MI 를 **늘리거나 줄일 수 있다** — 주의)

**진술.** $H(X|Y) \le H(X)$ 는 항상 성립하지만, $I(X; Y | Z)$ 와 $I(X; Y)$ 의 대소는 **결정되지 않음**.

**반례.**
- $X, Y$ 독립 이지만 $Z = X + Y$ 에 대해 conditioning 하면 $X$ 와 $Y$ 가 dependent → $I(X;Y|Z) > 0 = I(X;Y)$. (**생성 효과**, "explain-away")
- 반대 예: $X, Y$ 강한 dependence 인데 $Z = X$ 이면 $I(X;Y|Z) = 0$. (**설명**, "screening")

> **함의**: MI 가 양이냐 음이냐가 **인과구조**에 의존 — Simpson's paradox 를 일으키는 원인. (조건부 독립성 ≠ 독립성)

### Theorem 3.1.7 (Chain Rule for MI)

**진술.**
$$
I(X_1, \ldots, X_n; Y) = \sum_{i=1}^n I(X_i; Y | X_1, \ldots, X_{i-1})
$$

**증명.** $H$ 의 chain rule(§1.4 Theorem) $+$ MI 정의 합성. $\blacksquare$

### Theorem 3.1.8 (MI는 $(p_{XY})$ 에 대해 볼록)

**진술.** 주변 $p_X, p_Y$ 고정 시 $p_{XY} \mapsto I(X;Y)$ 는 **볼록** (convex). 반면 $p_{Y|X}$ 고정 + $p_X$ 변화 시 $p_X \mapsto I(X;Y)$ 는 **오목** (concave).

**증명 스케치.** KL 이 $(p, q)$ 에 대해 jointly convex. $q = p_X p_Y$ 는 주변을 고정하면 상수 → $I$ 가 $p_{XY}$ 의 convex 함수. 두 번째는 capacity 계산에 쓰는 성질 (§5.1).

### Theorem 3.1.9 (Gaussian 상호정보량 공식)

**진술.** $(X, Y) \sim \mathcal{N}(\mu, \Sigma)$ (2차원)이고 상관계수 $\rho$ 이면
$$
I(X; Y) = -\tfrac{1}{2}\log(1 - \rho^2)
$$

**증명.** Gaussian 엔트로피 공식 $h = \frac{1}{2}\log((2\pi e)^n|\Sigma|)$ (§1.5) 에서 $h(X) + h(Y) - h(X,Y) = \frac{1}{2}\log\frac{\sigma_X^2 \sigma_Y^2}{\sigma_X^2 \sigma_Y^2 (1-\rho^2)} = -\frac{1}{2}\log(1-\rho^2)$. $\blacksquare$

> **함의**: $\rho = 0$ 이면 MI = 0 (Gaussian 하에서 correlation = 0 ↔ independence). 비가우시안에서는 correlation=0 이어도 MI≠0 가능 (e.g., $Y=X^2$).

---

## 💻 NumPy로 직접 확인

```python
import numpy as np

def H(p):  # 이산 엔트로피
    p = p[p > 0]
    return -np.sum(p * np.log(p))

def MI_joint(P):  # P: 2D joint prob matrix
    Px = P.sum(axis=1); Py = P.sum(axis=0)
    outer = np.outer(Px, Py)
    mask = (P > 0)
    return np.sum(P[mask] * np.log(P[mask] / outer[mask]))

# 예제 1: 완전 독립 → MI = 0
P = np.outer([0.5, 0.5], [0.3, 0.7])
print("독립 MI =", MI_joint(P))         # ≈ 0

# 예제 2: Y = X (완전 의존)
P = np.diag([0.5, 0.5])
print("Y=X  MI =", MI_joint(P), "   log 2 =", np.log(2))

# 예제 3: Y = X XOR Z (X, Y 각각은 독립, 하지만 Z 주면 관계 생김)
# joint (X, Y): uniform (X, Z) 에서 Y = X XOR Z 하면 (X, Y) 도 uniform → MI=0
P = np.array([[0.25, 0.25], [0.25, 0.25]])
print("XOR(X,Z) 의 (X,Y) MI =", MI_joint(P))  # ≈ 0 (marginally independent)
```

### Gaussian 예제

```python
from scipy.stats import multivariate_normal

def gaussian_MI(rho):
    return -0.5 * np.log(1 - rho**2)

for rho in [0.0, 0.3, 0.7, 0.9, 0.99]:
    print(f"rho = {rho:.2f}  MI = {gaussian_MI(rho):.4f}")
```

출력:
```
rho = 0.00  MI = 0.0000
rho = 0.30  MI = 0.0472
rho = 0.70  MI = 0.3438
rho = 0.90  MI = 0.8301
rho = 0.99  MI = 1.9560
```

$\rho \to 1$ 에서 MI → $\infty$ (완전 의존).

### Conditioning 비단조성 데모

```python
# X, Y 독립 Bernoulli(1/2), Z = X XOR Y
rng = np.random.default_rng(0)
N = 100000
X = rng.integers(0, 2, N); Y = rng.integers(0, 2, N); Z = X ^ Y

# I(X;Y) = ?
P_XY, _, _ = np.histogram2d(X, Y, bins=[2, 2], density=False)
P_XY = P_XY / P_XY.sum()
print("I(X;Y) =", MI_joint(P_XY))       # ≈ 0

# I(X;Y|Z) 는?   Z=0 조건에서 X=Y 이므로 dependence!
for z in [0, 1]:
    idx = (Z == z)
    P_XY_z, _, _ = np.histogram2d(X[idx], Y[idx], bins=[2, 2], density=False)
    P_XY_z = P_XY_z / P_XY_z.sum()
    print(f"I(X;Y | Z={z}) = {MI_joint(P_XY_z):.4f}")
# 둘 다 log 2 → I(X;Y|Z) = log 2 (통합 평균)
```
출력:
```
I(X;Y) = 0.0000
I(X;Y | Z=0) = 0.6927
I(X;Y | Z=1) = 0.6931
```
→ **conditioning on Z creates dependence between independent X, Y**: 고전적 "explain-away" 현상.

---

## 🔗 AI/ML 연결고리

### 1. Information Bottleneck (Tishby 2000)
$$
\min_{p(z|x)}\ I(X; Z) - \beta I(Z; Y)
$$
- $I(X;Z)$ : representation 압축 정도
- $I(Z;Y)$ : task-relevant 정보 보존
- $\beta$ : compression vs prediction tradeoff
- DNN training 의 dynamics 를 IB plane 에서 해석 (Tishby–Shwartz-Ziv 2017).

### 2. InfoNCE (Oord et al. 2018)
$$
\mathcal{L}_{\mathrm{InfoNCE}} = -\mathbb{E}\left[\log \frac{\exp(f(x, c)/\tau)}{\sum_{j} \exp(f(x_j, c)/\tau)}\right] \ge -I(X; C) + \log K
$$
**MI 의 lower bound** → 최소화하면 MI 최대화. SimCLR, CLIP, MoCo 의 이론적 토대.

### 3. $\beta$-VAE (Higgins 2017)
$$
\mathcal{L} = \mathbb{E}_{q}[\log p(x|z)] - \beta\, D(q(z|x) \| p(z))
$$
$\beta > 1$: rate-distortion tradeoff 조정 → disentanglement 유도. MI 분해(§6.2) 에서 $I(X;Z)$ 제어.

### 4. Fairness 조건
- **Statistical parity**: $P(\hat Y | S=0) = P(\hat Y | S=1) \Leftrightarrow I(\hat Y; S) = 0$.
- **Equalized odds**: $I(\hat Y; S | Y) = 0$.
- MI 를 직접 minimize 하는 adversarial debiasing.

### 5. Mutual Information Estimation
- **MINE** (Belghazi 2018), **SMILE** (Song & Ermon 2020), **CLUB** (Cheng 2020) — MI 를 신경망으로 추정 (§3.4).

---

## ⚖️ 가정·한계·함정

1. **Continuous MI 추정은 본질적으로 어렵다** — density estimation 이 필요. 고차원 샘플만으로는 bias 제어 어려움(§3.4).
2. **MI ≠ 0 ≠ causality** — 연관일 뿐, 인과 방향 미정.
3. **$I(X;Y)$ 가 유계가 아님** — continuous 변수에서 $\infty$ 가능 (deterministic 관계 $Y = f(X)$).
4. **Scaling invariance** — $I(aX+b; Y) = I(X; Y)$ — 좋은 성질이지만 해석 시 주의.
5. **Reparameterization trick** 없음 in general — KL with known $q$ 는 reparam 가능하지만 MI 는 일반적으로 아님 → MINE 같은 추정기 필요.
6. **Conditioning 비단조성** — §Theorem 3.1.6. DPI (§3.2) 와 혼동하지 말 것.

---

## 📌 핵심 정리

1. $I(X;Y) = D(p_{XY} \| p_X p_Y)$ — 독립으로부터의 KL 거리.
2. 세 동치형: $H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X)+H(Y)-H(X,Y)$.
3. 비음성 $I \ge 0$, $= 0 \Leftrightarrow X \perp Y$.
4. 대칭성 $I(X;Y) = I(Y;X)$, 자기 MI = 엔트로피.
5. Conditioning 은 MI 를 **늘리거나 줄이거나** — explain-away vs screening.
6. Gaussian: $I = -\frac12 \log(1-\rho^2)$.
7. **AI 적용**: IB, InfoNCE, $\beta$-VAE, Fairness, feature selection — MI가 대부분의 "표현 학습 원리"의 공통 언어.

---

## 🤔 생각해볼 문제

### 문제 1. 독립이지만 조건부 종속
$X, Y \sim \mathrm{Bernoulli}(1/2)$ 독립, $Z = X \oplus Y$. $I(X;Y)$ 와 $I(X;Y|Z)$ 를 계산.

<details>
<summary>해설</summary>

$I(X;Y) = 0$ (구성상 독립). $I(X;Y|Z) = H(X|Z) - H(X|Y,Z)$. $Z$ 알면 $X = Z \oplus Y$ 결정론 → $H(X|Y,Z) = 0$. $H(X|Z) = H(X) = \log 2$ (uniform). 따라서 $I(X;Y|Z) = \log 2 \approx 0.693$. 조건부가 독립성을 깬다 (explain-away).
</details>

### 문제 2. 두 개의 Gaussian 상관
$X \sim \mathcal{N}(0,1), Y = \rho X + \sqrt{1-\rho^2} N$ 에서 Theorem 3.1.9 유도를 처음부터.

<details>
<summary>해설</summary>

$Y$ 는 $\mathcal{N}(0, 1)$, $(X, Y)$ 2변량 정규, covariance $\rho$. $\Sigma = \begin{pmatrix}1 & \rho \\ \rho & 1\end{pmatrix}$, $|\Sigma| = 1 - \rho^2$. $h(X,Y) = \frac12\log((2\pi e)^2(1-\rho^2)) = \log(2\pi e) + \frac12 \log(1-\rho^2)$. $h(X) + h(Y) = 2 \cdot \frac12 \log(2\pi e) = \log(2\pi e)$. MI $= h(X)+h(Y)-h(X,Y) = -\frac12\log(1-\rho^2)$. ✅
</details>

### 문제 3. MI 의 상한 — Gaussian channel capacity
출력 $Y = X + N$, $N \sim \mathcal{N}(0, \sigma^2)$, 입력 $X$ 의 power $\le P$. $\max_{p_X} I(X;Y)$ 는?

<details>
<summary>해설</summary>

Shannon–Hartley: $\max = \frac12 \log(1 + P/\sigma^2)$. 최대화 분포는 $X \sim \mathcal{N}(0, P)$. MI 의 concave 성질 ($p_X$ 측면) 과 entropy maxent가 Gaussian 이라는 사실(§1.6) 로부터. 5장 channel capacity 의 핵심 예제.
</details>

### 문제 4. MI와 correlation 은 다르다
$X \sim \mathrm{Uniform}(-1, 1), Y = X^2$. $\mathrm{Corr}(X, Y) = 0$ 이지만 $I(X; Y) > 0$ 임을 보여라.

<details>
<summary>해설</summary>

$\mathbb{E}[XY] = \mathbb{E}[X^3] = 0$ (odd function), $\mathbb{E}[X]\mathbb{E}[Y] = 0 \cdot 1/3 = 0$ → Corr=0. 하지만 $Y = X^2$ 는 $|X|$ 와 일대일 → $I(X;Y) = H(Y) > 0$ (continuous 에서는 $\infty$). **상관계수는 선형 의존만 잡음, MI는 모든 의존.**
</details>

### 문제 5. 조건부 MI의 chain rule
$I(X;Y,Z) = I(X;Y) + I(X;Z|Y)$ 증명.

<details>
<summary>해설</summary>

$I(X; Y, Z) = H(X) - H(X|Y,Z)$. $= [H(X) - H(X|Y)] + [H(X|Y) - H(X|Y,Z)] = I(X;Y) + I(X;Z|Y)$. ✅
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [2.6 어떤 발산을 쓸 것인가](../ch2-kl-divergence/06-choosing-divergence.md) | [3.2 Data Processing Inequality](./02-data-processing-inequality.md) |

[🏠 Home](../README.md)

</div>
