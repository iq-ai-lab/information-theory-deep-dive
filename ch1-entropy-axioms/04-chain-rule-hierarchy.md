# 04. Chain Rule과 정보의 계층 구조

## 🎯 핵심 질문

- $n$개의 확률변수 $X_1, \ldots, X_n$의 엔트로피는 어떻게 **순차적 조건부 엔트로피의 합** 으로 분해되는가?
- "조건은 엔트로피를 감소시킨다"가 여러 변수 상황에서도 성립하는가?
- 언어 모델의 **Perplexity** 가 왜 평균 조건부 엔트로피의 지수인가?
- Chain Rule은 autoregressive 모델링 (GPT)의 수학적 근간인가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **Autoregressive 모델**: GPT, WaveNet, PixelCNN 등은 $p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i \mid x_{<i})$로 분해 — 이 곱을 로그 취하면 정확히 **chain rule of entropy**
- **Perplexity**: 언어 모델 평가 지표는 조건부 엔트로피의 기하평균의 형태
- **Markov 가정**: 조건부 엔트로피의 계층에서 어느 이전 항까지 보는가의 선택
- **Causal Attention**: 현재 토큰이 이전 토큰에만 조건 → chain rule 구조와 일치
- **Diffusion Model**: 시간축으로 chain rule 전개 — 각 시간 step의 조건부 확률의 로그 합

실제 ML에서 "loss.mean()"으로 평균되는 token-level cross-entropy는 chain rule 덕분에 **전체 시퀀스 엔트로피의 추정치** 로 해석된다.

---

## 📐 수학적 선행 조건

- [문서 03](./03-joint-conditional-mutual.md)의 결합·조건부 엔트로피와 MI
- 다변수 곱셈 법칙: $p(x_1, \ldots, x_n) = \prod_i p(x_i \mid x_1, \ldots, x_{i-1})$
- 귀납법 증명에 대한 익숙함

> 다변수 확률의 chain rule은 [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-statistics-deep-dive)에서 기본 법칙으로 다룹니다.

---

## 📖 직관적 이해

### "한 번에 하나씩"의 사고

$n$장의 카드를 한 번에 뽑는 것 vs 한 장씩 순차적으로 뽑아 각 카드의 놀라움을 누적하는 것. Chain rule은 두 방식이 **같은 총 정보량**을 준다는 법칙이다:

$$\underbrace{H(X_1, \ldots, X_n)}_{\text{전체를 한 번에}} = \underbrace{H(X_1) + H(X_2 \mid X_1) + \cdots + H(X_n \mid X_1, \ldots, X_{n-1})}_{\text{순차적 누적}}.$$

### 언어 모델이 단어를 생성하는 법

문장 $w_1, w_2, \ldots, w_T$의 확률을
$$p(w_1, w_2, \ldots, w_T) = p(w_1) \cdot p(w_2 \mid w_1) \cdot p(w_3 \mid w_1, w_2) \cdots$$
로 분해한다. 음의 로그를 취하면
$$-\log p(w_{1:T}) = \sum_{t=1}^T -\log p(w_t \mid w_{<t}).$$
이것이 **token-level NLL(Negative Log-Likelihood) 손실**의 합이다. 기댓값을 취하면 entropy chain rule과 일치.

### Markov 사슬과 조건의 축소

$X_1 \to X_2 \to X_3$이 Markov 사슬이면 $H(X_3 \mid X_1, X_2) = H(X_3 \mid X_2)$. 이는 "이전 정보 중 바로 직전의 것만 중요"라는 Markov 성질의 엔트로피 버전.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — 조건부 결합 엔트로피

변수 집합 $X_1, \ldots, X_k$ 및 조건 변수 집합 $Y_1, \ldots, Y_m$에 대해
$$H(X_1, \ldots, X_k \mid Y_1, \ldots, Y_m) := -\sum p(x_1, \ldots, x_k, y_1, \ldots, y_m) \log p(x_1, \ldots, x_k \mid y_1, \ldots, y_m).$$

### 정의 4.2 — 조건부 상호정보량

$$I(X; Y \mid Z) := H(X \mid Z) - H(X \mid Y, Z) = \mathbb{E}_Z\!\left[I(X; Y \mid Z = z)\right].$$

$Z$를 이미 안 상태에서 $Y$를 추가로 알았을 때 $X$에 대해 새로 얻는 정보.

---

## 🔬 정리와 증명

### 정리 4.1 — Chain Rule of Entropy

**명제**: 임의의 결합 분포를 가진 $X_1, \ldots, X_n$에 대해
$$H(X_1, X_2, \ldots, X_n) = \sum_{i=1}^{n} H(X_i \mid X_1, \ldots, X_{i-1}),$$
여기서 $i = 1$의 경우 조건이 비어있으므로 $H(X_1 \mid \emptyset) := H(X_1)$.

**증명 (귀납)**:

**기저 ($n = 2$)**: 정리 3.1에서 $H(X_1, X_2) = H(X_1) + H(X_2 \mid X_1)$. 성립.

**귀납 단계**: $n$에서 성립한다고 가정. $n + 1$에 대해:

$n + 1$개를 $(X_{1:n}, X_{n+1})$로 묶어 두 변수 결합으로 본다. 정리 3.1 적용:
$$H(X_{1:n}, X_{n+1}) = H(X_{1:n}) + H(X_{n+1} \mid X_{1:n}).$$

귀납 가정으로 $H(X_{1:n}) = \sum_{i=1}^n H(X_i \mid X_{<i})$ 대입:
$$H(X_{1:n+1}) = \sum_{i=1}^n H(X_i \mid X_{<i}) + H(X_{n+1} \mid X_{1:n}) = \sum_{i=1}^{n+1} H(X_i \mid X_{<i}). \quad \square$$

---

### 정리 4.2 — Chain Rule of Mutual Information

**명제**:
$$I(X_1, \ldots, X_n; Y) = \sum_{i=1}^{n} I(X_i; Y \mid X_1, \ldots, X_{i-1}).$$

**증명**:

$I(X_{1:n}; Y) = H(X_{1:n}) - H(X_{1:n} \mid Y)$.  
Chain rule을 두 엔트로피에 각각 적용:
$$H(X_{1:n}) = \sum_i H(X_i \mid X_{<i}), \quad H(X_{1:n} \mid Y) = \sum_i H(X_i \mid X_{<i}, Y).$$

따라서
$$I(X_{1:n}; Y) = \sum_i [H(X_i \mid X_{<i}) - H(X_i \mid X_{<i}, Y)] = \sum_i I(X_i; Y \mid X_{<i}). \quad \square$$

---

### 정리 4.3 — 조건은 엔트로피를 감소시킨다 (다변수)

**명제**: 임의의 변수 집합 $\mathcal{Z}$에 대해
$$H(X \mid Y, \mathcal{Z}) \leq H(X \mid \mathcal{Z}),$$
등호는 $X \perp Y \mid \mathcal{Z}$일 때만 (조건부 독립).

**증명**: $\mathcal{Z}$로 조건부를 잡은 상태에서 문서 03의 정리 3.4를 적용:
$$I(X; Y \mid \mathcal{Z}) = H(X \mid \mathcal{Z}) - H(X \mid Y, \mathcal{Z}) \geq 0,$$
$I \geq 0$은 조건부 KL의 비음수성 (Gibbs). $\square$

---

### 정리 4.4 — 독립이면 엔트로피는 합

**명제**: $X_1, \ldots, X_n$가 **상호 독립** 이면
$$H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i).$$

**증명**: 독립이면 $H(X_i \mid X_{<i}) = H(X_i)$ (정리 4.3의 등호 조건). Chain rule에 대입. $\square$

---

### 정리 4.5 — 결합 엔트로피의 일반 상한

**명제**:
$$H(X_1, \ldots, X_n) \leq \sum_{i=1}^n H(X_i),$$
등호는 **상호 독립** 일 때 정확히 성립.

**증명**: Chain rule + 정리 4.3으로
$$H(X_{1:n}) = \sum_i H(X_i \mid X_{<i}) \leq \sum_i H(X_i). \quad \square$$

**의미**: "여러 변수를 함께 보는 것이 각 변수의 엔트로피의 합보다 클 수 없다." 상호 정보량이 양수인 만큼 결합은 "덜 불확실".

---

### 정리 4.6 — 조건부 엔트로피의 오목성 (in $p_Y$)

**명제**: $p(y)$에 대해 $H(X \mid Y)$는 오목 함수다. 즉 두 주변 분포 $p_Y, q_Y$ (같은 조건부 $p(x \mid y)$)와 $\lambda \in [0, 1]$에 대해,
$$H(X \mid Y)_{\lambda p_Y + (1-\lambda) q_Y} \geq \lambda H(X \mid Y)_{p_Y} + (1-\lambda) H(X \mid Y)_{q_Y}.$$

(증명은 문서 02 정리 2.4와 유사하므로 생략; Cover-Thomas 정리 2.7.3 참조.)

---

## 💻 NumPy 구현/시뮬레이션

```python
import numpy as np

# ─────────────────────────────────────────────
# 1. 3변수 결합 분포에서 chain rule 수치 검증
# ─────────────────────────────────────────────

rng = np.random.default_rng(7)

# p(x1, x2, x3): 3×3×3 분포 (임의)
p = rng.dirichlet(np.ones(27)).reshape(3, 3, 3)

def H(p, base=2):
    p = p[p > 0]
    return -np.sum(p * np.log(p)) / np.log(base)

def H_cond(p_xy, base=2):
    """H(X|Y) = H(X, Y) - H(Y). p_xy shape: (|X|, |Y|)"""
    p_y = p_xy.sum(axis=0)
    return H(p_xy, base) - H(p_y, base)

# 각 주변/조건부 엔트로피
p1 = p.sum(axis=(1, 2))                        # p(x1)
p12 = p.sum(axis=2)                            # p(x1, x2)
p123 = p                                       # p(x1, x2, x3)

H1 = H(p1)
H12 = H(p12)
H123 = H(p123)

# Chain rule: H(X1, X2, X3) = H(X1) + H(X2|X1) + H(X3|X1, X2)
H_2_given_1 = H12 - H1
H_3_given_12 = H123 - H12

lhs = H123
rhs = H1 + H_2_given_1 + H_3_given_12

print("Chain Rule of Entropy 수치 검증")
print(f"  H(X1)          = {H1:.6f}")
print(f"  H(X2 | X1)     = {H_2_given_1:.6f}")
print(f"  H(X3 | X1, X2) = {H_3_given_12:.6f}")
print(f"  Sum            = {rhs:.6f}")
print(f"  H(X1, X2, X3)  = {lhs:.6f}")
print(f"  ⇒ diff         = {abs(lhs - rhs):.2e}   (should be ~0)")

# ─────────────────────────────────────────────
# 2. 독립 vs 의존 결합 엔트로피 비교
# ─────────────────────────────────────────────

print("\n" + "=" * 60)
print("독립 분포 vs 상관 분포의 결합 엔트로피")
print("=" * 60)

# 독립 분포
p_indep = np.outer(p1, p12.sum(axis=0) / p12.sum(axis=0).sum())  # p(x1) * p(x2)
H_sum = H(p1) + H(p12.sum(axis=0))
H_joint_indep = H(p_indep)
print(f"독립:  H(X1) + H(X2) = {H_sum:.4f},  H(X1, X2) = {H_joint_indep:.4f}  (같음)")

# 실제 (상관 있음)
H_joint_true = H(p12)
print(f"실제:  H(X1) + H(X2) = {H_sum:.4f},  H(X1, X2) = {H_joint_true:.4f}")
print(f"       차이 = I(X1; X2) = {H_sum - H_joint_true:.4f} bits")

# ─────────────────────────────────────────────
# 3. Markov 사슬 X1 → X2 → X3에서 H(X3|X1,X2) = H(X3|X2)
# ─────────────────────────────────────────────

# Markov: p(x3 | x1, x2) = p(x3 | x2) 강제
p_x1 = rng.dirichlet(np.ones(3))
p_x2_given_x1 = rng.dirichlet(np.ones(3), size=3)        # (x1, x2)
p_x3_given_x2 = rng.dirichlet(np.ones(3), size=3)        # (x2, x3)

p_markov = np.zeros((3, 3, 3))
for i in range(3):
    for j in range(3):
        for k in range(3):
            p_markov[i, j, k] = p_x1[i] * p_x2_given_x1[i, j] * p_x3_given_x2[j, k]

# H(X3 | X1, X2)
p_123 = p_markov
H_123 = H(p_123)
H_12 = H(p_123.sum(axis=2))
H_3_given_12_mk = H_123 - H_12

# H(X3 | X2)
p_23 = p_markov.sum(axis=0)
H_23 = H(p_23)
H_2 = H(p_23.sum(axis=1))
H_3_given_2 = H_23 - H_2

print("\n" + "=" * 60)
print("Markov 성질 확인:  X1 → X2 → X3 에서 H(X3|X1,X2) = H(X3|X2)")
print("=" * 60)
print(f"  H(X3 | X1, X2) = {H_3_given_12_mk:.6f}")
print(f"  H(X3 | X2)     = {H_3_given_2:.6f}")
print(f"  diff           = {abs(H_3_given_12_mk - H_3_given_2):.2e}")

# ─────────────────────────────────────────────
# 4. Perplexity 계산 예시
# ─────────────────────────────────────────────

# 가짜 언어 모델의 token log-probs (nats)
log_probs = np.array([-1.2, -4.5, -0.3, -2.1, -1.0, -3.7])  # 6 tokens
T = len(log_probs)

nll = -log_probs.mean()                              # 평균 -log p(w|context)
ppl = np.exp(nll)

print("\n" + "=" * 60)
print("Perplexity 계산 (언어 모델 평가)")
print("=" * 60)
print(f"  Sequence token count: {T}")
print(f"  Avg NLL (≈ H_cond est): {nll:.4f} nats/token")
print(f"  Perplexity = exp(NLL)  = {ppl:.4f}")
print(f"  해석: 매 토큰마다 약 {ppl:.1f}개의 동등 후보에서 선택하는 정도의 불확실성")
```

**출력 예시**:
```
Chain Rule of Entropy 수치 검증
  H(X1)          = 1.548...
  H(X2 | X1)     = 1.501...
  H(X3 | X1, X2) = 1.472...
  Sum            = 4.521...
  H(X1, X2, X3)  = 4.521...
  ⇒ diff         = 1.77e-15

Markov 성질 확인
  H(X3 | X1, X2) = 1.502...
  H(X3 | X2)     = 1.502...
  diff           = 2.22e-16
```

---

## 🔗 AI/ML 연결

### Autoregressive 언어 모델링

GPT 같은 모델은
$$p_\theta(w_{1:T}) = \prod_{t=1}^T p_\theta(w_t \mid w_{<t})$$
로 결합 분포를 모델링. Cross-entropy 손실:
$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^T \log p_\theta(w_t \mid w_{<t}) \approx \frac{1}{T} H(W_{1:T}) \text{ (Chain rule)}.$$

즉 token-level NLL의 평균은 데이터의 **평균 조건부 엔트로피** 를 추정하는 것. 완벽한 LM은 $\mathcal{L} \to H_\text{true}$ (시퀀스의 진짜 엔트로피 rate).

### Perplexity의 수학적 의미

$$\text{PPL} = \exp\!\left(\frac{1}{T} \sum_{t=1}^T -\log p_\theta(w_t \mid w_{<t})\right) = \exp(H_\text{cond}).$$

Chain rule + 20 questions 해석:
- PPL = 10 → 평균적으로 매 스텝마다 "10개의 동등 가능 후보"에서 선택
- PPL이 작을수록 좋은 LM (혼란이 적음)

### Markov 가정과 n-gram 모델

$k$-차 Markov 근사:
$$p(w_t \mid w_{<t}) \approx p(w_t \mid w_{t-k:t-1}).$$

$n$-gram LM은 $k = n - 1$. 근대 Transformer는 사실상 $k = \text{context length}$의 매우 큰 Markov 근사.

### Diffusion Model의 ELBO 분해

DDPM의 손실 (Ch6-05 참조):
$$\mathcal{L} = \mathbb{E}\!\left[\sum_{t=1}^T -\log p_\theta(x_{t-1} \mid x_t)\right] + \text{상수}.$$

Chain rule 구조가 시간축에 그대로. Diffusion은 time-series Markov 사슬로 볼 수 있음.

### Causal Mask = Chain Rule의 구현

Transformer의 causal mask는
$$p(w_t \mid w_{<t}) \text{ 만 의존, } w_{>t} \text{는 미래라 보지 않음}$$
을 구현. 이는 chain rule의 인덱스 순서를 architecture 수준에서 강제.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| **결합 분포 정확히 알고 있음** | 실전에서는 샘플로 추정 — 조건부 분포 추정이 고차원에서 어려움 |
| 유한 시퀀스 | 무한 스트림은 **entropy rate** $\bar{H} = \lim_{n \to \infty} H(X_n \mid X_{<n})$로 확장 (Cover-Thomas §4) |
| Chain Rule의 인덱스 순서 | 어느 순서로 분해해도 값은 동일 (대칭성), 하지만 각 항 $H(X_i \mid X_{<i})$은 순서에 따라 다름 — autoregressive 모델이 **어떤 순서로 decoding** 할지가 실제 성능에 영향 (예: PixelCNN raster vs diagonal) |
| Markov 성질의 엄격함 | 현실 언어는 long-range dependency가 있어 $k$-차 근사로는 한계 — Transformer의 긴 context가 필요한 이유 |

**수치적 주의**: 매우 긴 시퀀스에서 token log-prob의 합은 극도로 작은 수가 될 수 있음. 합을 먼저 계산하고 마지막에 `exp`하는 형태로 underflow 회피 필수 (log-sum-exp 패턴).

---

## 📌 핵심 정리

$$\boxed{H(X_1, \ldots, X_n) = \sum_{i=1}^{n} H(X_i \mid X_1, \ldots, X_{i-1})}$$

| 성질 | 수식 | 해석 |
|------|------|------|
| Chain rule (entropy) | $H(X_{1:n}) = \sum_i H(X_i \mid X_{<i})$ | 순차 누적의 정보 |
| Chain rule (MI) | $I(X_{1:n}; Y) = \sum_i I(X_i; Y \mid X_{<i})$ | 조건부 MI의 합 |
| 조건은 감소시킨다 | $H(X \mid Y, \mathcal{Z}) \leq H(X \mid \mathcal{Z})$ | 더 많이 알수록 덜 혼란 |
| 독립이면 합 | $H(X_{1:n}) = \sum_i H(X_i)$ if 상호 독립 | 상호 정보 없음 |
| 상한 | $H(X_{1:n}) \leq \sum_i H(X_i)$ | 독립이 최대 |

**ML에서의 의미**: autoregressive loss의 평균 = 조건부 엔트로피의 추정, exp → perplexity.

---

## 🤔 생각해볼 문제

**문제 1** (기초): $X_1, X_2, X_3$이 상호 독립 Bernoulli(0.5)일 때 $H(X_1, X_2, X_3)$을 계산하고, chain rule의 각 항을 명시하라.

<details>
<summary>힌트 및 해설</summary>

독립이므로 각 $H(X_i \mid X_{<i}) = H(X_i) = 1$ bit.  
$H(X_1, X_2, X_3) = 1 + 1 + 1 = 3$ bits = $\log_2 8$. 8개 동등 가능 outcome과 일치.

</details>

---

**문제 2** (심화): $Y = f(X_1, X_2)$이 결정적 함수일 때 $H(X_1, X_2, Y) = H(X_1, X_2)$임을 chain rule로 증명하라.

<details>
<summary>힌트 및 해설</summary>

$H(X_1, X_2, Y) = H(X_1, X_2) + H(Y \mid X_1, X_2) = H(X_1, X_2) + 0 = H(X_1, X_2)$.  
$(X_1, X_2)$가 $Y$를 결정하므로 $H(Y \mid X_1, X_2) = 0$ (문서 03의 정리 3.6 방향 적용). $\square$

</details>

---

**문제 3** (AI 연결): 언어 모델이 문장 "I love cats"에 부여한 token NLL이 $[-0.3, -2.1, -1.2]$ (nats)라고 하자. 이 문장의 total NLL, per-token NLL, perplexity를 계산하라.

<details>
<summary>힌트 및 해설</summary>

Total NLL = $0.3 + 2.1 + 1.2 = 3.6$ nats (chain rule 합).  
Per-token NLL = $3.6 / 3 = 1.2$ nats/token.  
Perplexity = $e^{1.2} \approx 3.32$.  
해석: 매 토큰에 약 3.3개의 동등 후보 중 선택하는 정도의 불확실성. 좋은 영어 LM 수준은 PPL ≈ 10~30 (정의 데이터셋에 따라).

</details>

---

**문제 4** (증명): $X \to Y \to Z$가 Markov 사슬일 때 $I(X; Y, Z) = I(X; Y)$임을 Chain rule of MI로 보여라.

<details>
<summary>힌트 및 해설</summary>

Chain rule: $I(X; Y, Z) = I(X; Y) + I(X; Z \mid Y)$.  
Markov 성질 $X \perp Z \mid Y$에서 $I(X; Z \mid Y) = 0$.  
따라서 $I(X; Y, Z) = I(X; Y)$. $\square$  
해석: "Y를 알면 Z로부터 X에 대한 새로운 정보를 얻을 수 없다" — DPI (Ch3-02)의 첫걸음.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 03. 결합·조건부·상호정보량](./03-joint-conditional-mutual.md) | [05. 미분 엔트로피 ▶](./05-differential-entropy.md) |

</div>
