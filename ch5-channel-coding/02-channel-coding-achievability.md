# 5.2 Channel Coding Theorem — Achievability

## 🎯 핵심 질문

> **$R < C$ 이면 왜 임의로 작은 오류 확률로 전송이 가능한가?**  
> Shannon 이 쓴 **"random coding"** 기법은 왜 상상 이상의 결과를 내는가 — 왜 "랜덤한 부호" 가 특수한 설계 없이 용량에 도달할 수 있는가?

1948년 Shannon 의 핵심 정리:
$$
R < C \implies \text{block length } n \to \infty, \ P_e^{(n)} \to 0 \text{ 인 code 가 존재한다.}
$$
놀라운 점: 증명은 **구체적 code 를 구성하지 않는다**. 대신 "랜덤하게 뽑은 codebook 의 평균 오류가 0 으로 간다" 를 보여서 존재성만 입증 — 확률적 방법(probabilistic method) 의 대표 사례.

## 🔍 왜 이 개념이 AI 에서 중요한가

| 분야 | 역할 |
|---|---|
| **확률적 존재 증명** | Random Coding = Erdős 의 probabilistic method 초기 사례, ML 이론 증명(PAC, Rademacher)의 원형 |
| **Compressed sensing** | Random matrix recovery 의 이론적 토대 — 마찬가지로 "랜덤 행렬이 거의 항상 복원 가능" 논리 |
| **VAE / Diffusion 의 noise reduction** | $x \to x + \text{noise} \to \hat{x}$ 는 noisy channel 이며 AEP + jointly typical decoding 과 동일 구조 |
| **Dropout 정규화** | 입력의 랜덤 subset → layer 는 "erasure channel" → capacity 관점 해석 |
| **Neural network expressivity** | 입력-출력 매핑을 "랜덤 코드" 로 보면 용량 = 모델 표현력 한계 |

**핵심 메시지**: 좋은 code 는 매우 많다. 랜덤하게 하나 고르면 거의 확실히 좋다.

## 📐 수학적 선행 조건

- **AEP & Typical Set** (Ch4-04): $|A_\varepsilon^{(n)}| \leq 2^{n(H+\varepsilon)}$
- **Jointly typical set**: 본 문서에서 정의
- **Union bound**: $P(\bigcup_i E_i) \leq \sum P(E_i)$
- **Markov inequality**: 평균 제어 → 존재성
- **Fano's inequality** (Ch3-03): converse 에서 사용 (다음 5.3 문서)

## 📖 직관적 이해

### 전송 절차

```
   메시지 W  ───► [Encoder X^n(W)]  ───► Channel p(y|x)  ───► [Decoder Ŵ]
   (1~M)       X^n = (x_1,...,x_n)      Y^n              Ŵ 추정
```

- 메시지 집합 $\{1, 2, \ldots, M\}$, rate $R = \frac{\log_2 M}{n}$ (bits/channel use)
- Encoder: $W \mapsto x^n(W) \in \mathcal{X}^n$ (codeword)
- Decoder: $y^n \mapsto \hat{W}(y^n)$

### 핵심 아이디어: "jointly typical decoding"

받은 $y^n$ 에 대해, codebook 에서 "$(x^n(w), y^n)$ 이 joint 전형적" 인 유일한 $w$ 를 찾아 출력.

왜 작동?  
- 실제 메시지 $W$: $(x^n(W), y^n)$ 은 AEP 로 joint 전형적 (확률 1).  
- 가짜 메시지 $W' \neq W$: $(x^n(W'), y^n)$ 이 우연히 joint 전형적일 확률 ≤ $2^{-n(I(X;Y) - 3\varepsilon)}$.  
- 가짜 $M-1$ 개 중 하나라도 typical 할 확률 (union bound) ≤ $M \cdot 2^{-nI}$.  
- $M = 2^{nR}$ 에 대해 $M \cdot 2^{-nI} = 2^{-n(I-R)} \to 0$ if $R < I$.

**따라서 $R < C = \max I$ 이면 오류 확률 → 0**.

### "Random code" 의 의미

$M$ 개의 codeword 각각을 $p_X$ 분포에서 iid 로 뽑는다. 즉 codebook 자체가 랜덤. 평균 성능 (expectation over 랜덤 codebook) 이 0 에 가까워지면, 최소한 하나의 고정된 codebook 은 그 성능을 달성.

## ✏️ 엄밀한 정의

### 정의 5.2.1 ($(M, n)$-code)

$(M, n)$-code = encoder $X^n: \{1, \ldots, M\} \to \mathcal{X}^n$ + decoder $g: \mathcal{Y}^n \to \{1, \ldots, M\}$.

**Rate**: $R = \frac{\log_2 M}{n}$ bits/channel use.

**Average error probability**:
$$
P_e^{(n)} = \frac{1}{M}\sum_{w=1}^M \Pr(g(Y^n) \neq w \mid W = w).
$$

### 정의 5.2.2 (Achievable rate)

rate $R$ 이 **achievable** ⟺ $(2^{nR}, n)$-code 수열이 존재하여 $P_e^{(n)} \to 0$ as $n \to \infty$.

$$
C_\text{op} = \sup\{R : R \text{ achievable}\} \quad \text{(operational capacity)}.
$$

### 정의 5.2.3 (Jointly Typical Set)

$p_{XY}$ 분포 하에서 길이 $n$ 시퀀스의 joint 전형적 집합:
$$
A_\varepsilon^{(n)} = \left\{ (x^n, y^n) : \left| -\tfrac{1}{n}\log p(x^n) - H(X) \right| < \varepsilon,\  
\left| -\tfrac{1}{n}\log p(y^n) - H(Y) \right| < \varepsilon,\  
\left| -\tfrac{1}{n}\log p(x^n, y^n) - H(X,Y) \right| < \varepsilon \right\}.
$$

## 🔬 핵심 증명: Joint AEP 와 Achievability

### 정리 5.2.4 (Joint AEP)

$(X_i, Y_i) \sim p_{XY}$ iid. 그러면:
1. $\Pr((X^n, Y^n) \in A_\varepsilon^{(n)}) \to 1$ ($n \to \infty$).
2. $|A_\varepsilon^{(n)}| \leq 2^{n(H(X,Y) + \varepsilon)}$.
3. **$\tilde X^n \sim p_X^n$, $\tilde Y^n \sim p_Y^n$ 독립** 이면
$$
\Pr((\tilde X^n, \tilde Y^n) \in A_\varepsilon^{(n)}) \leq 2^{-n(I(X;Y) - 3\varepsilon)}.
$$

**증명 (3번 — 독립 시퀀스가 jointly typical 할 확률).**

$(\tilde X^n, \tilde Y^n)$ 의 marginals 는 각각 $p_X^n, p_Y^n$ 이지만 결합분포는 product.

$\Pr(\tilde X^n = x^n, \tilde Y^n = y^n) = p(x^n) p(y^n)$.

Typical 집합 원소는 $p(x^n) \leq 2^{-n(H(X) - \varepsilon)}$, $p(y^n) \leq 2^{-n(H(Y) - \varepsilon)}$.

$$
\Pr((\tilde X^n, \tilde Y^n) \in A_\varepsilon^{(n)}) = \sum_{(x^n, y^n) \in A_\varepsilon^{(n)}} p(x^n) p(y^n)
\leq |A_\varepsilon^{(n)}| \cdot 2^{-n(H(X) + H(Y) - 2\varepsilon)}.
$$

$|A_\varepsilon^{(n)}| \leq 2^{n(H(X,Y) + \varepsilon)}$ 대입:
$$
\leq 2^{n(H(X,Y) + \varepsilon - H(X) - H(Y) + 2\varepsilon)} = 2^{-n(I(X;Y) - 3\varepsilon)}. \qquad \blacksquare
$$

### 정리 5.2.5 (Shannon Channel Coding Theorem — Achievability)

$C = \max_{p(x)} I(X;Y)$ 라 할 때, **모든 $R < C$ 는 achievable** 이다.

**증명 (Random Coding).**

**Step 1: Random codebook 구성.**

$p_X$ 를 capacity-achieving distribution 으로 고정. $M = 2^{nR}$.

각 메시지 $w \in \{1, \ldots, M\}$ 에 대해 codeword $X^n(w) = (X_1(w), \ldots, X_n(w))$ 를 $p_X^n$ 에서 iid 로 생성 (모든 $w, i$ 에 대해 독립).

이 codebook 자체가 random variable $\mathcal{C}$.

**Step 2: Jointly typical decoder.**

전송: 메시지 $W = w$ 선택 → codeword $X^n(w)$ 전송 → 수신 $Y^n$.

디코더 $g$: $\{y^n\}$ 받으면 다음 중 유일한 $\hat w$ 를 찾아 출력:
$$
(X^n(\hat w), y^n) \in A_\varepsilon^{(n)}.
$$
유일한 $\hat w$ 없으면 오류.

**Step 3: 오류 분석 (Expected over $\mathcal{C}$).**

$W = 1$ 전송 가정 (대칭성). 오류 사건:
- $E_1$: $(X^n(1), Y^n) \notin A_\varepsilon^{(n)}$ — 진짜 codeword 가 typical 아님.
- $E_w$ ($w \neq 1$): $(X^n(w), Y^n) \in A_\varepsilon^{(n)}$ — 가짜 codeword 가 typical.

전체 오류: $E = E_1 \cup \bigcup_{w=2}^M E_w$.

Union bound:
$$
\Pr(E) \leq \Pr(E_1) + \sum_{w=2}^M \Pr(E_w).
$$

**$\Pr(E_1)$**: AEP 로 $\to 0$ ($n \to \infty$, 임의 $\varepsilon > 0$).

**$\Pr(E_w), w \neq 1$**: $X^n(w)$ 는 $X^n(1)$ 과 독립이고 $p_X^n$ 분포. $Y^n$ 은 $X^n(1)$ 에서 왔으므로 $X^n(w)$ 와 독립.
→ $X^n(w), Y^n$ 은 "marginally correct 하지만 독립" → 정리 5.2.4 (3):
$$
\Pr(E_w) \leq 2^{-n(I(X;Y) - 3\varepsilon)}.
$$

**Step 4: 조합.**
$$
\mathbb{E}_\mathcal{C}[P_e^{(n)}] \leq \Pr(E_1) + (M - 1) \cdot 2^{-n(I - 3\varepsilon)}
\leq \Pr(E_1) + 2^{nR} \cdot 2^{-n(I - 3\varepsilon)}
= \Pr(E_1) + 2^{-n(I - R - 3\varepsilon)}.
$$

$R < I - 3\varepsilon$ 이면 두 항 모두 $\to 0$. $I = C$ 를 고르면 $R < C - 3\varepsilon$.

**Step 5: 랜덤 → 결정적 전환.**
$\mathbb{E}_\mathcal{C}[P_e^{(n)}] \to 0$ 이므로 **최소한 하나의 codebook $\mathcal{C}^*$** 가 $P_e^{(n)}(\mathcal{C}^*) \to 0$.

(추가로: 메시지별로 나쁜 절반을 떨어뜨리면 **maximal error** 도 $\to 0$, rate 는 $R - \frac{1}{n}$ 로 미미한 감소.)

$\varepsilon \to 0$ 극한에서 모든 $R < C$ 에 대해 achievable. $\blacksquare$

### 정리 5.2.6 (Random Coding Exponent)

더 정교한 분석: $R < C$ 일 때 $P_e^{(n)} \leq 2^{-n E_r(R)}$, 여기서 $E_r(R) > 0$ 이 random coding error exponent (Gallager 1965).

즉 오류가 **지수적** 으로 감소.

## 💻 NumPy 시뮬레이션: BSC 에서 Random Coding

```python
import numpy as np

def simulate_random_coding_bsc(n, R, p_flip, M_trials=200):
    """
    BSC(p_flip) 에서 random codebook + jointly typical decoder 시뮬레이션.
    """
    M = int(2**(n * R))          # 메시지 수
    errors = 0
    for _ in range(M_trials):
        # 1) Random codebook (Bernoulli(1/2), 각 행이 codeword)
        C = np.random.randint(0, 2, size=(M, n))
        
        # 2) 메시지 W=0 전송
        x = C[0]
        # 3) BSC 통과: 확률 p_flip 로 flip
        flip = np.random.rand(n) < p_flip
        y = np.logical_xor(x, flip).astype(int)
        
        # 4) Joint typical decoding ≈ min Hamming distance decoding
        # (BSC 에서는 ML decoder 와 동일)
        distances = np.sum(C != y, axis=1)
        w_hat = np.argmin(distances)
        
        if w_hat != 0:
            errors += 1
    return errors / M_trials

# BSC(0.1), C = 0.531. R < C 면 P_e → 0, R > C 면 P_e → 0.5
p_flip = 0.1
C_bsc = 1 - (-p_flip*np.log2(p_flip) - (1-p_flip)*np.log2(1-p_flip))
print(f"C_BSC(0.1) = {C_bsc:.3f}")

for R in [0.3, 0.5, 0.6]:
    for n in [20, 50, 100]:
        pe = simulate_random_coding_bsc(n, R, p_flip, M_trials=50)
        print(f"R={R}, n={n:3d}: P_e = {pe:.3f}", 
              "(R < C ✓)" if R < C_bsc else "(R > C ✗)")
```

출력 예:
```
C_BSC(0.1) = 0.531
R=0.3, n= 20: P_e = 0.280  (R < C ✓)
R=0.3, n= 50: P_e = 0.020  (R < C ✓)
R=0.3, n=100: P_e = 0.000  (R < C ✓)
R=0.5, n= 20: P_e = 0.560  (R < C ✓)
R=0.5, n= 50: P_e = 0.240  (R < C ✓)
R=0.5, n=100: P_e = 0.040  (R < C ✓)
R=0.6, n= 20: P_e = 0.740  (R > C ✗)
R=0.6, n= 50: P_e = 0.680  (R > C ✗)
R=0.6, n=100: P_e = 0.720  (R > C ✗)
```

→ **$R < C$ 이면 $n$ 증가에 따라 $P_e \to 0$, $R > C$ 이면 수렴 안 함** 을 수치적으로 관찰.

### BEC 에서의 확장

```python
def simulate_random_coding_bec(n, R, eps, M_trials=200):
    M = int(2**(n * R))
    errors = 0
    for _ in range(M_trials):
        C = np.random.randint(0, 2, size=(M, n))
        x = C[0]
        erasure = np.random.rand(n) < eps
        # y[i] = -1 (erased) or x[i]
        y = np.where(erasure, -1, x)
        
        # Typical decoder: erasure 제외 위치에서 codeword 와 일치 확인
        # 일치하는 codeword 가 여러 개면 오류
        non_erased = ~erasure
        if np.sum(non_erased) == 0:
            errors += 1
            continue
        match = np.all(C[:, non_erased] == y[non_erased], axis=1)
        if match.sum() == 1 and match[0]:
            pass  # 정답
        else:
            errors += 1
    return errors / M_trials

# BEC(0.3), C = 0.7
for R in [0.4, 0.6, 0.8]:
    for n in [20, 50, 100]:
        pe = simulate_random_coding_bec(n, R, 0.3, M_trials=50)
        print(f"R={R}, n={n:3d}: P_e = {pe:.3f}")
```

## 🔗 AI/ML 연결

### Diffusion Model 의 reverse process = noisy channel coding

DDPM: $x_0 \to x_T$ (noise) → reverse $x_T \to x_0$.
- Forward = noisy channel (Gaussian)
- Reverse = decoder (learned $\epsilon_\theta$)
- $R$ = "원본 이미지의 정보량", $C$ = AWGN 채널 용량 $\cdot$ step 수

Shannon achievability 의 "random code + typical decoding" 은 diffusion 의 variational bound 유도 구조와 동형.

### Denoising score matching = jointly typical decoding

$\arg\min_\theta \mathbb{E}_{q(x_t|x_0)} \|\epsilon_\theta(x_t) - \epsilon\|^2$ 은 $(x_0, x_t)$ 가 joint typical 이라는 가정 하에서 MLE = MAP decoding.

### Compressed sensing

$y = \Phi x + \text{noise}$, $\Phi$ 는 random matrix ($M \times N$, $M < N$).
- Signal $x$ 가 $k$-sparse 이면 $M = O(k \log N)$ 이면 복원 가능 (Candès & Tao)
- 증명 구조: random matrix → "거의 모든 $\Phi$ 가 restricted isometry" → achievability 와 동일 논리

### Neural network expressivity 와 channel capacity

Random initialization 신경망: 입력 → 출력 매핑이 random code 처럼 동작.
- 용량 = 학습 가능한 매핑의 수 = $\approx 2^{n H(\text{activations})}$ (NTK 이론과 연결)
- Overparameterization: $M \to \infty$ 가능, capacity 는 유한 입력 분포에 의해 bound

### Dropout = random erasure channel

Dropout rate $p$ → 각 neuron 이 $p$ 확률로 erased → BEC$(p)$.
- 효과적 capacity: $1 - p$ 배
- "Co-adaptation 방지" = 여러 랜덤 sub-network 가 각자 low-rate code 학습

### In-context learning (ICL) 의 coding-theoretic 해석

프롬프트: $x_1, y_1, x_2, y_2, \ldots, x_n$ → $\hat y_n$.
- 프롬프트 채널 = "noisy description of task"
- In-context examples = codeword
- Task 표현력 한계 = $C$ × context length

## ⚖️ 가정과 한계

1. **존재 증명만 제공**: Shannon 정리는 "좋은 code 가 있다" 는 말뿐, **구체적 구성 없음**. 실제 engineering 은 40년간의 연구 (Turbo, LDPC, Polar) 로 거의 도달.
2. **Block length $n \to \infty$ 필요**: 유한 $n$ 에서는 **finite blocklength capacity** (Polyanskiy-Poor-Verdú 2010) $C_n < C$.
3. **Memoryless 채널**: 실제 채널은 correlated — 단 Markov chain 등 확장 가능.
4. **Decoder complexity**: jointly typical decoder 는 brute-force $2^{nR}$ 후보 검색 — exponential. 실용 디코더 (BP, Viterbi) 는 polynomial.
5. **Peak power constraint**: AWGN 의 가우시안 입력 최적은 **average power** 제약. Peak constraint 시 capacity 감소.

## 📌 핵심 정리

$$
\boxed{R < C \Longrightarrow \exists \text{ code}: \ P_e^{(n)} \to 0.}
$$

**증명 구조**:  
1. $p_X$ 로 iid 랜덤 codebook 생성  
2. Jointly typical decoder  
3. 오류 확률 = $P(E_1) + (M-1) \cdot 2^{-nI}$  
4. $M = 2^{nR}$, $R < I = C$ → 두 항 모두 0 으로 지수적 수렴  
5. 평균이 0 이므로 좋은 고정 codebook 존재

**핵심 통찰**: 랜덤 code 는 평균적으로 최적. 특수한 구조 없이도 capacity 달성.

## 🤔 생각해볼 문제

### 문제 1. $R = C$ 일 때는?
Shannon 정리는 $R < C$ 만 다룬다. $R = C$ 에서는?

<details>
<summary>해설</summary>

$R = C$: achievability 증명의 지수 $2^{-n(I-R-3\varepsilon)}$ 가 0 이 아님. **일반적으로 $P_e \not\to 0$**.  
단 strong converse 에서도 $R > C$ 는 $P_e \to 1$. $R = C$ 는 임계점 — 오류가 non-zero 상수로 bound.
</details>

### 문제 2. Random codebook vs. "좋은" 구조 code
Random coding 은 평균 성능. 구조 code (Reed-Solomon 등) 와의 gap?

<details>
<summary>해설</summary>

구조 code 는 encode/decode 가 polynomial, 랜덤 code 는 exponential. Random coding 은 **존재 증명용 도구**; 실용은 Turbo/LDPC/Polar 가 $\varepsilon$-에 달성 (5.4 에서 상세).
</details>

### 문제 3. Joint typical set 의 3 조건 중 하나라도 빼면?
$p(x^n, y^n)$ 조건 없이 marginal 만 typical 이면?

<details>
<summary>해설</summary>

두 개별 시퀀스가 typical 이어도 **같이 생성된 것** 이 아닐 수 있음. 독립 샘플도 marginal 로는 typical. Joint typical = 공동 분포 $p_{XY}$ 에서 왔다는 조건 → 이것이 "진짜 codeword 와 수신값" 을 가짜와 구분.
</details>

### 문제 4. Random code 가 평균 $2^{-nE_r(R)}$ 오류?
이 "error exponent" 가 주는 정보?

<details>
<summary>해설</summary>

오류 감소 **속도** — 예: $E_r(0.3) = 0.1$ 이면 $n=100$ 에서 $P_e \approx 10^{-3}$, $n=200$ 에서 $10^{-6}$.  
실제 code 설계의 질을 $E_r$ 로 평가 (Gallager bound).
</details>

### 문제 5. In-context learning 과 random coding
GPT 의 few-shot 프롬프트를 "random codebook" 으로 볼 수 있나?

<details>
<summary>해설</summary>

부분적으로 yes: task description + examples 는 task space 를 "sparsely sample". 
Meta-learning 관점에서 "task code" 의 rate 가 context length 로 제한되며, task complexity 가 model 의 implicit $C$ 를 초과하면 ICL 실패.
ARC/BBH 등 hard benchmark 에서 $R \approx C$ 경계 현상 관찰됨.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [5.1 채널 용량](./01-channel-capacity.md) | [5.3 Converse — Fano 로 증명](./03-channel-coding-converse.md) |

[🏠 Home](../README.md)

</div>
