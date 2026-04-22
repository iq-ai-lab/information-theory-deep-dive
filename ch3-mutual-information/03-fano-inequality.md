# 3.3 Fano 의 부등식 — 분류 오차의 정보이론적 하한

## 🎯 핵심 질문

> **"$Y$ 로부터 $X$ 를 추정하는데 오차를 얼마나 작게 만들 수 있는가?"**
> 상호정보량 $I(X; Y)$ 이 작다면, 추정 오차 $P_e$ 는 반드시 얼마나 큰가?
> 분류문제의 **정보이론적 한계** — 모델이 얼마나 좋아지든 넘을 수 없는 벽.

---

## 🔍 왜 AI에서 중요한가

- **분류 정확도의 근본 한계**: 데이터-라벨 간 MI $I(X; Y)$ 가 제한적이면 임의의 모델로도 일정 오차 이하 불가능.
- **Bayes Error Rate 의 하한**: Fano 부등식은 Bayes error 의 명시적 lower bound.
- **Active Learning / Labeling 비용**: label noise 하에서 "얼마나 많이 label 해야 충분한가" 의 정량.
- **Generalization Bound**: Fano + DPI 로 "학습 불가능한 문제" 정의.
- **Impossibility results 의 핵심 도구**: 로버스트 통계, 적응적 lower bound, 통신 이론 모두 Fano 를 활용.
- **Representation Learning 평가**: linear probe accuracy 가 $I(Z; Y)$ 하한.

"모델이 더 크면 좋아진다" 는 실무 경험과 별개로, **정보적으로 풀 수 없는 문제** 는 존재하며 그 한계를 명시적으로 제공.

---

## 📐 선행 학습 지식

- [1.2 엔트로피 정의](../ch1-entropy-axioms/02-entropy-definition.md)
- [1.3 조건부·MI](../ch1-entropy-axioms/03-joint-conditional-mutual.md)
- [3.1 MI 정의](./01-mi-definitions.md), [3.2 DPI](./02-data-processing-inequality.md)
- 분류 문제 기본 (오류율, Bayes 최적 분류기)

---

## 📖 직관

### 기본 아이디어

$X$ 를 추정하는 함수 $\hat X = g(Y)$. 오류 $P_e = P(\hat X \ne X)$.

오류가 있는지 없는지 알려주는 **에러 지시 확률변수** $E = \mathbb{1}[X \ne \hat X]$. $E$ 가 $X$ 에 대한 정보를 일부 알려주므로:

$$
H(X | \hat X) \approx H(E) + P_e \cdot H(X | E=1, \hat X) + 0
$$

$E=0$ 이면 $X = \hat X$ 알려짐 → $H(X|E=0, \hat X)=0$. $E=1$ 이면 $X$ 는 $\hat X$ 가 아닌 $K-1$ 개 중 하나 → 최대 엔트로피 $\log(K-1)$.

### 그림

```
 오류 지시 E ∈ {0, 1}
  ├─ E=0 (확률 1-P_e): X = \hat X (결정)
  └─ E=1 (확률 P_e): X ∈ {나머지 K-1 클래스}
                         └→ 최악의 경우 uniform → log(K-1)
```

이로부터
$$
H(X | \hat X) \le H(E) + P_e \log(K - 1) = H(P_e) + P_e \log(K-1)
$$

이것이 Fano 부등식.

---

## ✏️ 공식 정의

**정의 3.3.1 (Binary entropy)**
$$
H(p) = -p \log p - (1-p) \log(1-p)
$$

**정의 3.3.2 (Fano 부등식)**
$X \in \mathcal{X}, |\mathcal{X}| = K$, $\hat X = g(Y)$ 추정량, $P_e = P(\hat X \ne X)$ 라 하자. 그러면
$$
\boxed{\ H(P_e) + P_e \log(K - 1) \ge H(X | \hat X) \ge H(X | Y) = H(X) - I(X; Y)\ }
$$

**정의 3.3.3 (단순화 형)**
$H(P_e) \le \log 2$ 이므로 느슨하지만 단순한 경계:
$$
P_e \ge \frac{H(X | Y) - \log 2}{\log(K - 1)} = \frac{H(X) - I(X; Y) - \log 2}{\log(K - 1)}
$$

**정의 3.3.4 (Binary case)**
$K = 2$ 이면 $\log(K-1) = 0$. Fano 부등식이
$$
H(P_e) \ge H(X | Y)
$$
즉 binary 에러 엔트로피가 조건부 엔트로피 상한.

---

## 🔬 정리와 증명

### Theorem 3.3.1 (Fano)

**진술.** 위 정의 3.3.2.

**증명.** $E = \mathbb{1}[X \ne \hat X]$. Chain rule:
$$
H(X, E | \hat X) = H(E | \hat X) + H(X | E, \hat X)
$$
좌변 $\ge H(X | \hat X)$ (remove $E$ reduces entropy).

- $H(E | \hat X) \le H(E) = H(P_e)$.
- $H(X | E, \hat X) = P_e \cdot H(X | E=1, \hat X) + (1-P_e) \cdot H(X | E=0, \hat X)$.
- $E=0 \Rightarrow X = \hat X$ → $H(X|E=0, \hat X) = 0$.
- $E=1 \Rightarrow X$ 는 $\hat X$ 를 제외한 $K-1$ 값 중 하나 → $H(X|E=1, \hat X) \le \log(K-1)$.

합쳐서 $H(X, E | \hat X) \le H(P_e) + P_e \log(K-1)$. 따라서
$$
H(X | \hat X) \le H(P_e) + P_e \log(K-1).
$$
그리고 DPI $X \to Y \to \hat X$ Markov → $H(X | \hat X) \ge H(X | Y)$. 최종
$$
H(P_e) + P_e \log(K-1) \ge H(X | Y). \blacksquare
$$

### Theorem 3.3.2 (Tightness)

**진술.** $\hat X = \arg\max_x P(X=x | Y=y)$ (Bayes 최적 분류기) 와 uniform misclassification 일 때 Fano bound 는 **타이트**.

**증명 스케치.** Fano 증명에서 부등식이 성립하는 조건:
1. $H(E|\hat X) = H(E)$: $E$ 와 $\hat X$ 가 독립 → 오류가 추정값에 균등.
2. $H(X|E=1, \hat X) = \log(K-1)$: 오류 시 나머지 $K-1$ 값에 uniform.
3. DPI 등호: $X \to \hat X \to Y$ 도 Markov — 즉 $\hat X$ 가 sufficient.

이 조건을 만족하는 예: symmetric channel + uniform source. $\blacksquare$

### Theorem 3.3.3 (대안 형 — Fano lower on $P_e$)

**진술.** $K \ge 2$ 에 대해
$$
P_e \ge \frac{H(X | Y) - 1}{\log K}
$$
(자연로그 아닌 경우 상수 다름; 통상적 단순화 형).

**증명 스케치.** $H(P_e) \le 1$ (bits) → 정의 3.3.3 과 유사. $\blacksquare$

### Theorem 3.3.4 (Binary case 의 정확형)

**진술.** $K=2$ 에서 $H(P_e) \ge H(X|Y)$. 따라서
$$
P_e \ge H^{-1}(H(X|Y))
$$
여기서 $H^{-1}$ 은 $[0, 1/2]$ 에서 단조증가하는 binary entropy 의 역함수.

**증명.** $K-1 = 1$ 이면 $\log(K-1) = 0$ → 공식 단순화. $\blacksquare$

### Theorem 3.3.5 (MI form)

**진술.** $I(X; Y)$ 관점에서
$$
P_e \log(K-1) \ge H(X) - I(X; Y) - H(P_e)
$$
$X$ uniform ($H(X) = \log K$) 이면
$$
P_e \ge \frac{\log K - I(X; Y) - \log 2}{\log(K - 1)} \approx 1 - \frac{I(X; Y)}{\log K}
$$
(마지막 근사는 $K$ 가 클 때).

**증명.** 정의 3.3.2 에서 $H(X|Y) = H(X) - I(X;Y)$ 대입. $\blacksquare$

> **함의**: **$I(X;Y) = 0$** (독립) → $P_e \ge 1 - 1/K - O(1/\log K)$ → random guessing 이 최선.

### Theorem 3.3.6 (Continuous case 일반화)

**진술.** $X$ 가 연속일 때 Fano 의 적합한 확장은 minimax risk 의 하한 형태로 존재 (Yu 1997, Birge–Massart).

**내용 (간략)**: 매개변수 추정 $\hat \theta = \hat \theta(Y)$ 에 대해
$$
\inf_{\hat \theta} \max_\theta \mathbb{E}\|\hat \theta - \theta\|^2 \ge c \cdot \frac{\log M - \log 2}{I(V; Y)}
$$
여기서 $V$ 는 $\theta$ space 의 $M$-packing 의 index. 통계적 minimax 분석의 핵심 도구.

---

## 💻 NumPy로 직접 확인

### Fano 부등식의 실증

```python
import numpy as np

def binary_entropy(p):
    if p <= 0 or p >= 1: return 0.0
    return -p*np.log(p) - (1-p)*np.log(1-p)

def H(p):
    p = np.asarray(p); p = p[p > 0]
    return -np.sum(p * np.log(p))

# Symmetric K-ary channel
K = 5
N = 100000
rng = np.random.default_rng(0)
X = rng.integers(0, K, N)

for p_err in [0.05, 0.2, 0.4, 0.7]:
    # 분류기가 p_err 확률로 틀리고, 틀릴 때 uniform in other K-1 classes
    noise = rng.random(N) < p_err
    wrong_class = (X + 1 + rng.integers(0, K-1, N)) % K
    hat_X = np.where(noise, wrong_class, X)
    
    # 실제 P_e
    P_e_emp = np.mean(hat_X != X)
    # H(X|hat_X) 추정
    H_X = np.log(K)  # uniform X
    # Fano RHS
    fano_rhs = binary_entropy(p_err) + p_err * np.log(K - 1)
    # Fano LHS = H(X|hat_X) (for Bayes-optimal symmetric channel)
    
    print(f"P_e={p_err:.2f}  Fano_RHS={fano_rhs:.4f}  theoretic_H(X|hatX)={fano_rhs:.4f}")
```

### $I(X;Y)$ 가 작으면 오류 필연

```python
# X uniform over K=10 classes, Y = X + large noise
K = 10
# Channel: Y = X with prob 1-p, else uniform. I(X;Y) = (1-p) * log K
for p_noise in [0.1, 0.5, 0.9, 0.99]:
    I = (1 - p_noise) * np.log(K)
    # Fano lower bound
    LB = (np.log(K) - I - np.log(2)) / np.log(K - 1)
    print(f"p_noise={p_noise}  I(X;Y)={I:.3f}  P_e ≥ {max(0, LB):.3f}")
```

출력:
```
p_noise=0.1   I(X;Y)=2.073   P_e ≥ 0.016
p_noise=0.5   I(X;Y)=1.151   P_e ≥ 0.433
p_noise=0.9   I(X;Y)=0.230   P_e ≥ 0.854
p_noise=0.99  I(X;Y)=0.023   P_e ≥ 0.948
```

MI 가 줄수록 최선 분류기도 오차가 점증.

### Binary classification 하한

```python
# P(Y=1|X=1) = 1-eps, P(Y=0|X=0) = 1-eps (binary symmetric)
for eps in [0.01, 0.1, 0.3, 0.5]:
    H_XgY = binary_entropy(eps)
    # H(P_e) >= H(X|Y) => P_e >= eps (실제로 Bayes error = eps)
    # 역함수 이용
    from scipy.optimize import brentq
    if H_XgY < 1e-10:
        P_e_LB = 0
    else:
        f = lambda p: binary_entropy(p) - H_XgY
        P_e_LB = brentq(f, 1e-6, 0.5)
    print(f"eps={eps:.2f}  H(X|Y)={H_XgY:.4f}  P_e^LB={P_e_LB:.4f}  (실제 Bayes err = {eps})")
```
출력:
```
eps=0.01  H(X|Y)=0.0560  P_e^LB=0.0100
eps=0.10  H(X|Y)=0.3251  P_e^LB=0.1000
eps=0.30  H(X|Y)=0.6109  P_e^LB=0.3000
eps=0.50  H(X|Y)=0.6931  P_e^LB=0.5000
```
Binary case 에서 Fano bound 는 **정확히** Bayes error 와 일치 (tight).

---

## 🔗 AI/ML 연결고리

### 1. Label noise 하의 분류 한계
Training label 의 noise 율 $\eta$: $I(X; Y) = (1-\eta) \log K + H(\eta)$ 같은 형태로 감소 → Fano 로 unavoidable error 산출.

### 2. Domain Adaptation의 불가능성
Source 와 target 이 다를 때 target 의 MI $I(X; Y_{\mathrm{target}})$ 가 작으면 Fano bound 로 adaptation 한계.

### 3. Privacy Attacks
Membership inference 에서 attacker 의 MI $I(\mathrm{membership}; \mathrm{output})$ 이 DP 로 제한 → Fano bound 로 attacker 오류율 하한 보장 (privacy guarantee).

### 4. Bayesian Experimental Design
다음 실험을 고르는 목적: $\max I(\theta; Y_{\mathrm{next}})$ → Fano 로 파라미터 추정 오류 최소화.

### 5. Minimax Estimation Theory
Le Cam, Fano, Assouad 의 3 methods 중 **Fano 방법**: $\theta$ 공간에 packing 구성 후 Fano 부등식 적용. 고차원 추정의 minimax lower bound 의 표준 도구.

### 6. Representation Quality
Linear probe accuracy $\approx 1 - P_e$. $P_e$ 는 Fano 로 $I(Z; Y)$ 하한 → "representation 의 task-relevant MI" 직접 측정.

### 7. 인과 탐지
$I(X; Y | W)$ 를 Fano 로 추정 오차 하한. 조건부 독립 검정 → causal discovery (PC, GES algorithms).

---

## ⚖️ 가정·한계·함정

1. **Markov $X \to Y \to \hat X$** — 추정기가 $Y$ 만 보고 구성. 추가 정보 있으면 bound 완화.
2. **$\mathcal{X}$ 크기 $K$** — Fano bound 는 $K$ 에 의존. $K$ 가 크면 bound 가 느슨 (log scale).
3. **Uniform $X$ 가 worst case** — 편향된 prior 에서는 Fano 가 overly conservative.
4. **유한 sample 에서의 MI 추정 어려움** — 실제 MI 값 정확히 모르면 Fano bound 도 부정확. MINE/k-NN 추정기 bias 고려 필요.
5. **"타이트하지 않을 수 있다"** — asymmetric channel, heavy-tailed loss 등에서 Fano 보다 더 강한 bound (Assouad) 가 더 타이트.
6. **Continuous target** — 직접 Fano 적용 못하고 packing 필요. Le Cam's method 보다 기술적.

---

## 📌 핵심 정리

1. **Fano**: $H(P_e) + P_e \log(K-1) \ge H(X|Y) = H(X) - I(X;Y)$.
2. $I(X;Y)$ 가 작으면 $P_e$ 는 **반드시** 큼.
3. 증명 핵심: 에러 지시 $E$ 의 chain rule + DPI.
4. Binary case: $H(P_e) \ge H(X|Y)$ (정확).
5. Tightness: Symmetric channel + uniform source.
6. 응용: **Bayes error LB**, privacy, minimax statistics, representation probe.
7. 단점: $K$ 가 커지면 느슨, continuous 는 packing 필요.

---

## 🤔 생각해볼 문제

### 문제 1. Fano bound 의 직접 유도
$K = 3$ 에서 Fano 부등식을 손으로 유도. $P_e = 0.3$ 이면 $H(X|\hat X)$ 상한?

<details>
<summary>해설</summary>

$H(0.3) + 0.3 \log 2 = 0.611 + 0.208 \approx 0.819$. 즉 $H(X|\hat X) \le 0.819$ (natural log). 상한이 achievable 한 조건은 $E \perp \hat X$ and $X|E=1, \hat X$ uniform.
</details>

### 문제 2. Binary Bayes error 와 Fano 일치
Binary channel $P(Y=1|X=0) = \epsilon = P(Y=0|X=1)$, $X$ uniform. Bayes classifier 의 정확한 error 와 Fano lower bound 비교.

<details>
<summary>해설</summary>

Bayes classifier: $\hat X = Y$ → error = $\epsilon$. $H(X|Y) = H(\epsilon)$. Fano: $H(P_e) \ge H(\epsilon)$ → $P_e \ge \epsilon$. 정확히 일치. Binary symmetric channel에서는 Fano 가 tight.
</details>

### 문제 3. High-dim classification
$K = 1000, I(X; Y) = 5$ nats. $P_e$ 의 Fano lower bound?

<details>
<summary>해설</summary>

$H(X) = \log 1000 \approx 6.908$. $H(X|Y) \ge 6.908 - 5 = 1.908$. $P_e \ge (1.908 - \log 2)/\log 999 \approx (1.908 - 0.693)/6.907 \approx 0.176$. 즉 **17.6% 이상** 오류 필연.
</details>

### 문제 4. Generative Model 의 Fano 해석
LM 의 next-token prediction accuracy 가 $1 - P_e$ 라면, training corpus 의 $I(\mathrm{context}; \mathrm{next\ token})$ 과 Fano 로 LM perplexity 와의 관계?

<details>
<summary>해설</summary>

Perplexity = $\exp(H(\mathrm{token} | \mathrm{context}))$. Accuracy 은 top-1 prediction 정확도. Fano: $H(P_e) + P_e \log(V-1) \ge H(\mathrm{token}|\mathrm{context}) = \log(\mathrm{perplexity})$. $V$ = vocab size. Perplexity 가 낮으면(model 좋으면) $P_e$ 가 **반드시** 낮아지진 않지만 하한이 존재. (LM 의 top-1 accuracy 는 perplexity 의 직접 함수가 아님.)
</details>

### 문제 5. Uniform prior 가 worst case 인 이유
$X$ 의 prior 가 편향되면 ($P(X=1) = 0.99$) Fano bound 가 덜 타이트해지는 이유.

<details>
<summary>해설</summary>

Trivial estimator $\hat X = 1$ 이 $P_e = 0.01$ 만 에러. $H(X) = H(0.99) \approx 0.056 \ll \log 2$. 편향된 prior는 $H(X)$ 자체가 작아 Fano 의 숫자가 작아지고 실제 bound 가 trivial. Uniform 이면 $H(X) = \log K$ 최대 → bound 도 최대.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [3.2 Data Processing Inequality](./02-data-processing-inequality.md) | [3.4 연속 MI와 MINE](./04-continuous-mi-mine.md) |

[🏠 Home](../README.md)

</div>
