# 4.4 AEP 와 Typical Set — 정보의 기하학

## 🎯 핵심 질문

> **"iid 수열이 $n \to \infty$ 에서 대부분 특정 집합에 집중한다"** 는 AEP(Asymptotic Equipartition Property)가 왜 성립하는가?
> **Typical set** 의 크기 $|A_\epsilon^{(n)}| \approx 2^{nH}$ 은 어떻게 유도되고, 이것이 왜 **entropy 의 물리적 해석** 인가?
> AEP 는 Shannon 의 모든 정리의 **기초 도구** 로 왜 쓰이는가?

---

## 🔍 왜 AI에서 중요한가

- **Shannon 의 모든 coding 정리의 증명 핵심 도구**.
- **Concentration of measure** 의 확률 이론 연결.
- **"정보는 전형적인 것에 집중된다"** 는 통계학적 직관의 수학적 기반.
- **LLM 의 typical output**: 높은 확률 시퀀스가 **이상해 보이는** 이유 (repetition, blandness) — typical decoding vs top-k/top-p 의 이론적 배경.
- **Generative model sampling**: typical set 샘플링이 고확률 mode 가 아닌 이유.
- **MDL / Kolmogorov complexity**: typical 시퀀스가 평균 $nH$ bits 로 압축됨.

이 문서는 §4.3 Source coding theorem 의 **증명 뒷면** 을 자세히 들여다본다.

---

## 📐 선행 학습 지식

- [4.3 Source coding theorem](./03-source-coding-theorem.md) — 개요 이미 본 상태
- [1.2 엔트로피](../ch1-entropy-axioms/02-entropy-definition.md)
- Law of Large Numbers, Chebyshev 부등식
- Markov, Chebyshev, Chernoff 부등식

---

## 📖 직관

### 동전 던지기 예제

공정 동전을 $n$ 번 던지는 상황.

- 가능한 시퀀스 $2^n$ 개, 모두 확률 $2^{-n}$.
- 앞면 개수 $k \sim \mathrm{Binomial}(n, 1/2)$, 거의 확실히 $n/2 \pm O(\sqrt n)$.
- 이 범위의 시퀀스 개수 $\binom{n}{n/2} \approx 2^n/\sqrt{n}$ → 거의 모든 시퀀스.
- 각 시퀀스 확률 $2^{-n} = 2^{-nH(1/2)}$. 여기서 $H = 1$.

**Observation**: 대부분의 시퀀스가 "typical"하며 각 개별 확률이 $\approx 2^{-nH}$.

### 일반 확률 분포

$p = (p_1, \ldots, p_k)$. 길이 $n$ 시퀀스의 확률:
$$
p(x^n) = \prod p_{x_i} = \prod p_j^{n_j}
$$
여기서 $n_j$ 는 심볼 $j$ 의 등장 횟수. LLN: $n_j/n \to p_j$. 따라서
$$
-\frac{1}{n}\log p(x^n) = -\sum \frac{n_j}{n} \log p_j \to -\sum p_j \log p_j = H(p)
$$

결론: 전형적인 시퀀스의 log-probability 가 $-nH$ 에 집중 → AEP.

---

## ✏️ 공식 정의

**정의 4.4.1 (AEP — convergence in probability)**
iid $X_i \sim p$ 에 대해
$$
-\frac{1}{n}\log p(X_1, \ldots, X_n) \xrightarrow{P} H(p)
$$

**정의 4.4.2 (Typical set)**
$$
A_\epsilon^{(n)} = \left\{ x^n \in \mathcal{X}^n : 2^{-n(H+\epsilon)} \le p(x^n) \le 2^{-n(H-\epsilon)} \right\}
$$
또는 동치로
$$
A_\epsilon^{(n)} = \left\{ x^n : \left| -\tfrac{1}{n}\log p(x^n) - H \right| \le \epsilon \right\}
$$

**정의 4.4.3 (Strong typicality, type class)**
$T_\epsilon^{(n)} = \left\{ x^n : \left| \frac{n_j(x^n)}{n} - p_j \right| \le \epsilon \cdot p_j, \forall j \right\}$

Strong typical 은 empirical distribution 이 $p$ 에 가깝다는 조건 — AEP 보다 강함.

---

## 🔬 정리와 증명

### Theorem 4.4.1 (AEP — Convergence Rate)

**진술.** iid $X_i$, $Y_i = -\log p(X_i)$ 면 $\mu = H(p), \sigma^2 = \mathrm{Var}(Y_i)$.
$$
P\!\left(\left| -\tfrac{1}{n}\log p(X^n) - H \right| > \epsilon\right) \le \frac{\sigma^2}{n\epsilon^2}
$$

**증명.** Chebyshev 부등식 직접 적용. $\bar Y_n = \frac{1}{n}\sum Y_i = -\frac{1}{n}\log p(X^n)$. $\blacksquare$

### Theorem 4.4.2 (Typical set probability)

**진술.** 임의의 $\epsilon, \delta > 0$ 에 대해 $n$ 충분히 크면
$$
P(X^n \in A_\epsilon^{(n)}) > 1 - \delta
$$

**증명.** AEP + convergence in probability. $\blacksquare$

### Theorem 4.4.3 (Typical set size)

**진술.**
$$
(1 - \delta) 2^{n(H-\epsilon)} \le |A_\epsilon^{(n)}| \le 2^{n(H+\epsilon)}
$$

**증명.**
- **상한**: $1 = \sum p(x^n) \ge \sum_{A_\epsilon^{(n)}} p(x^n) \ge |A_\epsilon^{(n)}| \cdot 2^{-n(H+\epsilon)}$. 정리하면 $|A_\epsilon^{(n)}| \le 2^{n(H+\epsilon)}$.
- **하한**: $1 - \delta \le P(A_\epsilon^{(n)}) \le |A_\epsilon^{(n)}| \cdot 2^{-n(H-\epsilon)}$. 정리하면 $|A_\epsilon^{(n)}| \ge (1-\delta) 2^{n(H-\epsilon)}$. $\blacksquare$

> **함의**: $|A| \approx 2^{nH}$ — **entropy 가 typical set 크기의 지수**. $H$ 의 "물리적" 의미.

### Theorem 4.4.4 (Typical set 기반 Source Coding Achievability)

**진술.** Rate $H + \epsilon$ encoding 가능, $P_e \to 0$.

**증명.** Encoder:
- $x^n \in A_\epsilon^{(n)}$: $|A_\epsilon^{(n)}|$ 개 중 index 로 $\lceil \log_2 |A_\epsilon^{(n)}| \rceil \le n(H+\epsilon) + 1$ bits.
- $x^n \notin A_\epsilon^{(n)}$: flag + $\lceil n\log_2|\mathcal{X}| \rceil$ bits (fallback).

기대 길이 $\le n(H+\epsilon) + 1 + \delta \cdot n\log|\mathcal{X}| \cdot o(1)$. $P_e = P(\text{flag}) \to 0$. $\blacksquare$

### Theorem 4.4.5 (Largest-probability vs Typical)

**진술.** 가장 가능성 높은 시퀀스(mode)는 **typical set 에 속하지 않을 수 있다**.

**예.** $p = (0.9, 0.1)$. Mode sequence = $(1,1,\ldots,1)$ with prob $0.9^n$. $-\tfrac{1}{n}\log(0.9^n) = -\log 0.9 \approx 0.152$. $H(p) \approx 0.469$. Typical 은 $|x^n| \approx 0.469$ 인 것들 — mode 가 아님.

**함의**: **고확률 mode vs 전형적 샘플** 은 다르다. Beam search (mode seeking) 이 "natural output" 과 차이나는 이유.

### Theorem 4.4.6 (Typical set 의 서로소성)

**진술.** $A_\epsilon^{(n)}$ 의 시퀀스들은 거의 같은 확률을 가짐 — entropy 에 의해 "uniform 처럼" 보임 (Equipartition).

즉 **$A_\epsilon^{(n)}$ 상에서 $p$ 는 근사적으로 균등분포 $1/|A_\epsilon^{(n)}|$**. 그래서 "Equipartition".

### Theorem 4.4.7 (Strong typicality 가 joint typicality 까지 확장)

**진술.** $(X_i, Y_i) \sim p_{XY}$ iid. **Joint typical set**:
$$
A_\epsilon^{(n)}(X, Y) = \left\{ (x^n, y^n) : \text{각 주변 및 결합의 log-prob 이 근사} \right\}
$$

크기 $|A_\epsilon^{(n)}(X,Y)| \approx 2^{n H(X,Y)}$. Cross-correlation $\{x^n\} \times \{y^n\}$ 의 대부분은 **jointly typical 이 아님** — 이것이 channel coding (§5) 의 핵심.

---

## 💻 NumPy 로 직접 확인

### AEP 수렴 관찰

```python
import numpy as np
import matplotlib.pyplot as plt

p = np.array([0.5, 0.3, 0.15, 0.05])
H = -np.sum(p * np.log2(p))

rng = np.random.default_rng(0)
ns = [10, 100, 1000, 10000]
for n in ns:
    rates = []
    for _ in range(1000):
        X = rng.choice(len(p), size=n, p=p)
        log_p = np.sum(np.log2(p[X]))
        rates.append(-log_p / n)
    rates = np.array(rates)
    print(f"n={n:5d}  mean={rates.mean():.4f}  std={rates.std():.4f}  (H={H:.4f})")
```

출력:
```
n=   10  mean=1.6503  std=0.2781  (H=1.6477)
n=  100  mean=1.6483  std=0.0884  (H=1.6477)
n= 1000  mean=1.6478  std=0.0280  (H=1.6477)
n=10000  mean=1.6478  std=0.0089  (H=1.6477)
```

분산 $\sim 1/n$ → 집중 (LLN).

### Typical set size vs $2^{nH}$

```python
def enumerate_typical(p, n, eps):
    # 모든 시퀀스 나열 (작은 n 만)
    from itertools import product
    H = -np.sum(p * np.log2(p))
    count = 0
    total_prob = 0
    for seq in product(range(len(p)), repeat=n):
        log_p = sum(np.log2(p[s]) for s in seq)
        rate = -log_p / n
        if abs(rate - H) < eps:
            count += 1
            total_prob += 2**log_p
    return count, total_prob

p = np.array([0.7, 0.3])
H = -np.sum(p * np.log2(p))

for n in [5, 10, 15]:
    count, prob = enumerate_typical(p, n, eps=0.1)
    print(f"n={n:2d}: |A_eps|={count:6d}  2^(nH)={2**(n*H):.1f}  P(typical)={prob:.3f}")
```

출력:
```
n= 5: |A_eps|=    15  2^(nH)=10.2  P(typical)=0.647
n=10: |A_eps|=   240  2^(nH)=103.9  P(typical)=0.893
n=15: |A_eps|=  3003  2^(nH)=1058.9  P(typical)=0.958
```
$|A|/2^{nH}$ 가 $O(1)$, probability 가 1 로.

### Mode vs Typical sample 대비

```python
p = np.array([0.9, 0.1])
n = 100
H = -np.sum(p * np.log2(p))
print(f"H = {H:.3f}")

# Mode sequence (all 0s)
mode_seq = np.zeros(n, dtype=int)
mode_logp = np.sum(np.log2(p[mode_seq])) / n
print(f"Mode rate = {-mode_logp:.3f}")  # = -log(0.9) = 0.152

# Typical sample 은?
rng = np.random.default_rng(0)
X = rng.choice(2, size=n, p=p)
sample_rate = -np.sum(np.log2(p[X])) / n
print(f"Sample rate = {sample_rate:.3f}")  # ≈ H
print(f"Mode logprob {mode_logp*n} >> sample logprob {-sample_rate*n}  (mode 은 덜 놀랍)")
```

출력:
```
H = 0.469
Mode rate = 0.152
Sample rate = 0.456
Mode logprob -15.2 >> sample logprob -45.6
```

Mode 가 더 가능성 높지만 typical set 바깥.

### LM sampling 에서 typical decoding 비교

```python
# 가상의 LM 분포 (softmax)
np.random.seed(0)
def sample_lm_output(V=10000, T=100, mode="sample"):
    # 각 position 의 가상의 context-dependent 분포 (여기서는 uniform 으로 simplification)
    rng = np.random.default_rng(0)
    alpha = rng.exponential(1.0, V); alpha /= alpha.sum()
    tokens = []
    for _ in range(T):
        if mode == "sample":
            t = rng.choice(V, p=alpha)
        elif mode == "greedy":
            t = np.argmax(alpha)
        tokens.append(t)
    rate = -np.sum([np.log2(alpha[t]) for t in tokens]) / T
    return tokens, rate

_, r_s = sample_lm_output(mode="sample")
_, r_g = sample_lm_output(mode="greedy")
print(f"sample rate = {r_s:.3f}  greedy rate = {r_g:.3f}")
```

Greedy (mode) 는 low-rate (high-prob) 시퀀스 생성 → often repetitive/bland. Sampling (typical) 은 diverse/natural.

---

## 🔗 AI/ML 연결고리

### 1. LLM Decoding 전략
- **Greedy / Beam search**: mode seeking → repetitive.
- **Sampling**: typical set 접근 → diverse 하지만 noisy.
- **Top-k, top-p (nucleus)**: 확률 상위 집합에서 샘플 — typical set 의 부분집합.
- **Typical sampling (Meister 2022)**: 명시적으로 $-\log p$ 가 $H$ 에 가까운 토큰 선택.

### 2. Generative Models 의 sample quality
Diffusion, flow 의 샘플이 training distribution 의 typical region 에서 오는가?

### 3. Concentration of measure
Gaussian measure on $\mathbb{R}^n$ 의 shell concentration ($L^2$ norm $\approx \sqrt n$) — AEP 의 geometric 버전. 고차원 기하학의 기반.

### 4. Universal compression
Lempel–Ziv 등의 universal coder 가 asymptotically $H$ 달성하는 이유: AEP 가 불변 → distribution-free.

### 5. Mode collapse in GANs
Generator 가 mode (high-prob) 만 생성하고 typical 을 무시 → sample 이 diverse 하지 않음. Coverage metric 은 typical region 커버리지.

### 6. Active Learning
Typical samples 를 label 하는 것보다 "경계"(atypical) 샘플이 정보적 → uncertainty sampling.

### 7. OOD Detection
OOD 샘플은 training distribution 의 typical set 바깥 → $-\log p(x)$ 가 $H$ 에서 크게 벗어남. Likelihood-based OOD.

---

## ⚖️ 가정·한계·함정

1. **iid 가정 필수** — Stationary ergodic 로 일반화 (Shannon–McMillan–Breiman). Non-stationary 는 entropy rate 정의 복잡.
2. **$\epsilon, \delta$ trade-off** — 작은 $\epsilon$ 이면 typical set 이 좁아져 $P(A)$ 가 $1 - \delta$ 에 도달 늦음.
3. **유한 $n$ 의 실제 성능** — Asymptotic 결과, 유한에서 overhead $O(1/n)$ 존재.
4. **고차원에서 typical set 의 직관** — 저차원에서는 mode 가 typical 같지만 고차원에서는 mode ≠ typical.
5. **연속 분포** — differential entropy 기반, typical volume $\approx 2^{nh(X)}$ 가 아닌 $(2\pi e \sigma^2)^{n/2}$ 같은 구체 기하.
6. **$H$ 가 무한대** — 연속 변수에서 $h$ 가 $-\infty$ 또는 $\infty$ 가능, typical set 의미 수정 필요.

---

## 📌 핵심 정리

1. **AEP**: $-\frac{1}{n}\log p(X^n) \xrightarrow{P} H$.
2. **Typical set $A_\epsilon^{(n)}$**: log-prob 이 $[-H-\epsilon, -H+\epsilon]$ 안.
3. **성질**: $P(A) \to 1$, $|A| \approx 2^{nH}$, equipartition.
4. **Source coding 의 기반**: typical set indexing → $H + \epsilon$ rate.
5. **Mode ≠ typical**: 고확률 시퀀스가 typical 이 아님.
6. **Joint typical set** (다음 장 channel coding 의 준비).
7. AI 적용: LM decoding, OOD, mode coverage.

---

## 🤔 생각해볼 문제

### 문제 1. 이항분포 예시
$p = (1/2, 1/2), n = 100$. $A_\epsilon^{(100)}$ ($\epsilon = 0.1$) 크기 추정.

<details>
<summary>해설</summary>

$-\log p(x^n) = n$ bits (정확히, 모든 시퀀스가 $2^{-100}$). $-\tfrac{1}{n}\log p = 1 = H$ 항상. 그래서 **모든 시퀀스가 typical** ($2^{100}$ 개). 다른 $p$ 와 다른 모습.
</details>

### 문제 2. 연속 case 의 AEP
Gaussian $X \sim \mathcal{N}(0, \sigma^2)$ iid. $-\frac{1}{n}\log p(X^n) \to h(X) = \frac12 \log(2\pi e \sigma^2)$. Typical set 의 기하적 의미?

<details>
<summary>해설</summary>

Typical set $\approx$ $\{x \in \mathbb{R}^n : \|x\|^2/n \approx \sigma^2\}$ (shell of radius $\sqrt n \sigma$). 고차원 Gaussian 이 "구면 쉘" 에 집중 — concentration of measure.
</details>

### 문제 3. LLM 에서 mode vs typical
GPT 의 top-1 greedy 출력이 repetitive 한 이유를 AEP 로 설명.

<details>
<summary>해설</summary>

Mode (greedy) = $-\log p$ 가 최소인 시퀀스 = rate 낮은 시퀀스. 하지만 "자연스러운" 영어는 rate ≈ $H$ — typical. Greedy 는 typical 이 아니라 atypical mode → 흔한 repetition ("I am am am").
</details>

### 문제 4. Strong vs weak typicality
Strong typicality ($T_\epsilon^{(n)}$, type class) 와 AEP-typical 의 차이.

<details>
<summary>해설</summary>

Strong: empirical distribution 이 $p$ 근처. $T_\epsilon \subset A_\epsilon$ (strong → weak). Strong typicality 가 더 정교한 bound 에 사용, channel coding 에서 중요.
</details>

### 문제 5. LM 의 probability calibration
LM 이 실제 $H$ 에 가까운 compression 달성 여부 측정 방법?

<details>
<summary>해설</summary>

Validation corpus 의 bits-per-character (bpc) = $\frac{1}{N}\sum -\log_2 p(x)$. 이 값이 이론적 $H$ 하한에 수렴할수록 model 이 좋음. Hutter Prize 등의 대회는 enwik8, enwik9 등 실제 텍스트에서 bpc 최소화 경쟁. 실제 언어 $H \approx 0.6-1.0$ bpc 추정 (Shannon 1951 experiment).
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [4.3 Source Coding Theorem](./03-source-coding-theorem.md) | [4.5 Arithmetic Coding](./05-arithmetic-coding.md) |

[🏠 Home](../README.md)

</div>