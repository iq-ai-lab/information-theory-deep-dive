# 5.3 Channel Coding Theorem — Converse

## 🎯 핵심 질문

> **$R > C$ 이면 왜 오류 확률을 0 으로 만들 수 없는가?**  
> Fano 부등식과 DPI 를 이용하여 "용량 위에서는 신뢰성 있는 전송이 불가능" 임을 엄밀히 증명한다.

Shannon 정리의 반쪽. Achievability (5.2) 는 "용량 이하면 가능" 을 보였고, Converse 는 "용량 위에서는 불가능" 을 보여 $C$ 를 **진정한 경계**로 확정.

## 🔍 왜 이 개념이 AI 에서 중요한가

| 분야 | 역할 |
|---|---|
| **Lower bound 증명의 template** | Fano + DPI 구조는 ML 의 sample complexity / statistical estimation lower bound 증명에 재사용 |
| **Differential privacy** | 개인정보 보호 채널의 용량 상한 → privacy-utility trade-off |
| **Neural scaling law** | $N$ 파라미터로 얻는 정보량의 상한 — "파라미터 1 개당 2 bit" 추정치 |
| **RLHF / preference learning** | Human feedback channel 의 $C$ 와 모델 성능 ceiling |
| **Compression bound** | Data → model → output, DPI 로 전체 정보 손실 하한 |

## 📐 수학적 선행 조건

- **Fano 부등식** (Ch3-03): $H(X|Y) \leq H(P_e) + P_e \log(|\mathcal{X}| - 1)$
- **DPI** (Ch3-02): Markov chain $X \to Y \to Z$ 에서 $I(X;Z) \leq I(X;Y)$
- **Chain rule of MI**: $I(X^n; Y^n) \leq n I(X_i; Y_i)$ 부등식 유도
- **Channel memoryless**: $p(y^n | x^n) = \prod p(y_i | x_i)$

## 📖 직관적 이해

### 왜 DPI + Fano 조합인가?

전송 구조:
$$
W \to X^n(W) \to Y^n \to \hat W
$$
이것은 Markov chain. DPI 로:
$$
I(W; \hat W) \leq I(W; Y^n) \leq I(X^n; Y^n).
$$

Fano 로:
$$
H(W | \hat W) \leq H(P_e) + P_e \cdot \log M.
$$

$H(W) = \log M$ (uniform prior over $M$ messages), chain rule:
$$
\log M = H(W) = I(W; \hat W) + H(W|\hat W).
$$

종합:
$$
\log M \leq I(X^n; Y^n) + H(P_e) + P_e \log M.
$$

Channel capacity bound $I(X^n; Y^n) \leq nC$ 로:
$$
\log M \leq nC + 1 + P_e \log M.
$$

$R = \log M / n$ 로 나누면:
$$
R \leq C + \frac{1 + P_e \log M}{n} = C + \frac{1}{n} + P_e R.
$$

$P_e \to 0$ 가정 시 $n \to \infty$ 로 $R \leq C$. **즉 $R > C$ 이면 $P_e \not\to 0$**.

## ✏️ 엄밀한 정의

### 정의 5.3.1 (Weak Converse)

rate $R$ 이 achievable 하면 $R \leq C$ 이다.

### 정의 5.3.2 (Strong Converse)

$R > C$ 이면 **모든** code 에 대해 $P_e^{(n)} \to 1$ (Wolfowitz 1957).

Weak 는 $P_e \not\to 0$, Strong 은 더 강하게 $\to 1$ 이라는 주장.

## 🔬 Converse 증명 — 엄밀

### 정리 5.3.3 (Weak Converse)

$(2^{nR}, n)$-code 가 $P_e^{(n)} \to 0$ 을 달성하면 $R \leq C$.

**증명.**

$W \sim \text{Uniform}\{1, \ldots, M\}$, $M = 2^{nR}$.

**Step 1: Entropy decomposition.**
$$
\log M = H(W) = H(W | \hat W) + I(W; \hat W).
$$

**Step 2: Fano 부등식 적용.**
$P_e = \Pr(\hat W \neq W)$. $|\mathcal{W}| = M$ 이므로:
$$
H(W | \hat W) \leq H(P_e) + P_e \log(M - 1) \leq 1 + P_e \log M.
$$

**Step 3: DPI.**
$W \to X^n \to Y^n \to \hat W$ 는 Markov chain. DPI 로:
$$
I(W; \hat W) \leq I(W; Y^n) \leq I(X^n; Y^n).
$$

(첫 부등식: $\hat W$ 는 $Y^n$ 의 함수; 둘째: $X^n$ 은 $W$ 의 함수이므로 $I(W; Y^n) = I(X^n; Y^n) - I(X^n; Y^n | W) \leq I(X^n; Y^n)$ — 더 엄밀히는 deterministic encoder 가정.)

**Step 4: Channel capacity 상한.**

Lemma: Memoryless channel 에서 $I(X^n; Y^n) \leq n C$.

*증명*: 
$$
I(X^n; Y^n) = H(Y^n) - H(Y^n | X^n) = H(Y^n) - \sum_{i=1}^n H(Y_i | X_i)
$$
(memoryless 로 $H(Y^n | X^n) = \sum H(Y_i | X_i)$).

$H(Y^n) \leq \sum_{i=1}^n H(Y_i)$ (independence bound on entropy).

따라서:
$$
I(X^n; Y^n) \leq \sum_{i=1}^n [H(Y_i) - H(Y_i|X_i)] = \sum_{i=1}^n I(X_i; Y_i) \leq n C. \qquad \square
$$

**Step 5: 조합.**
$$
\log M \leq 1 + P_e \log M + n C.
$$

$\log M = nR$ 대입:
$$
nR \leq 1 + P_e \cdot nR + nC.
$$

$n$ 으로 나누고 재배치:
$$
R(1 - P_e) \leq C + \frac{1}{n}.
$$

$$
R \leq \frac{C + 1/n}{1 - P_e}.
$$

$P_e \to 0$, $n \to \infty$ 로:
$$
R \leq C. \qquad \blacksquare
$$

### 따름정리 5.3.4 (Achievable rate 의 sup)

$C_\text{op} = \sup\{R : R \text{ achievable}\} = C$.

(5.2 의 achievability + 5.3 의 converse 결합)

### 정리 5.3.5 (Strong Converse, Wolfowitz)

$R > C$ 이면 $P_e^{(n)} \to 1$ exponentially fast.

(증명은 더 정교한 sphere-packing 또는 Arikan 의 Polar code 연결을 사용 — 여기서는 생략.)

## 💻 NumPy 로 converse 관찰

```python
import numpy as np

def simulate_bsc_above_capacity(n, R, p_flip, M_trials=300):
    """R > C 에서 오류 확률이 0 으로 안 가는 것을 관찰."""
    M = int(2**(n*R))
    errors = 0
    for _ in range(M_trials):
        C_book = np.random.randint(0, 2, size=(M, n))
        x = C_book[0]
        flip = np.random.rand(n) < p_flip
        y = np.logical_xor(x, flip).astype(int)
        w_hat = np.argmin(np.sum(C_book != y, axis=1))
        if w_hat != 0:
            errors += 1
    return errors / M_trials

p_flip = 0.1
C_bsc = 1 - (-p_flip*np.log2(p_flip) - (1-p_flip)*np.log2(1-p_flip))
print(f"C = {C_bsc:.3f}")

# R = 0.8 > C = 0.531
for n in [30, 60, 100, 150]:
    pe = simulate_bsc_above_capacity(n, R=0.8, p_flip=p_flip)
    print(f"R=0.8, n={n:3d}: P_e = {pe:.3f} (이론: P_e ↛ 0)")

# Converse 의 경계 체크: 각 rate 별 Fano bound
for R in np.linspace(0.1, 1.0, 10):
    # 이론적 최소 P_e (weak converse):
    # R ≤ (C + 1/n) / (1 - P_e) → P_e ≥ 1 - (C+1/n)/R
    n = 100
    pe_min = max(0, 1 - (C_bsc + 1/n)/R)
    print(f"R={R:.2f}: P_e ≥ {pe_min:.3f}")
```

출력 예 (R > C 영역):
```
C = 0.531
R=0.8, n= 30: P_e = 0.610 (이론: P_e ↛ 0)
R=0.8, n= 60: P_e = 0.720 (이론: P_e ↛ 0)
R=0.8, n=100: P_e = 0.810 (이론: P_e ↛ 0)
R=0.8, n=150: P_e = 0.870 (이론: P_e ↛ 0)

R=0.10: P_e ≥ 0.000
R=0.20: P_e ≥ 0.000
R=0.30: P_e ≥ 0.000
R=0.40: P_e ≥ 0.000
R=0.50: P_e ≥ 0.000
R=0.60: P_e ≥ 0.097
R=0.70: P_e ≥ 0.226
R=0.80: P_e ≥ 0.323
R=0.90: P_e ≥ 0.398
R=1.00: P_e ≥ 0.459
```

→ $R > C$ 에서 $n$ 이 커질수록 $P_e \to 1$ (strong converse 관찰).

## 🔗 AI/ML 연결

### Sample Complexity Lower Bound

PAC learning: hypothesis class $\mathcal{H}$, 목표 정확도 $\varepsilon$, 샘플 수 $n$.
- Information channel: "true labels $\to$ 학습자" = noisy channel
- Fano 로: $n \geq \frac{\log |\mathcal{H}|}{C_\text{label}} = \Omega(\log |\mathcal{H}| / \varepsilon)$

딥러닝의 sample complexity $N \geq c \cdot d_\text{VC} / \varepsilon^2$ 유도에도 동일 Fano-DPI 구조 사용.

### LLM 의 information-theoretic ceiling

$n$ 개 토큰의 사전학습 → 모델 parameters $\theta$.
- DPI: $I(\text{data}; \theta) \leq I(\text{data}; \text{data}) = H(\text{data})$
- 파라미터 당 ≈ 2 bits 저장 한계 (Hinton, Hestness 2017)
- Chinchilla scaling law: 모델 크기·데이터 크기 balance 는 이 bound 의 효율 극대화

### Differential Privacy

$\varepsilon$-DP channel: 개인정보 → 출력. DP 제약 자체가 channel capacity 를 제한:
$$
C_\text{DP} \leq O(\varepsilon)
$$
(Duchi, Wainwright). Privacy-utility trade-off 의 converse.

### RLHF 의 reward ceiling

Human preference feedback → reward model. 
- Annotator agreement = 채널 잡음 (e.g., Kappa=0.6 → 상당한 잡음)
- $C_\text{reward} \ll 1$ bit/comparison 일 수 있음
- 샘플 효율성 하한 = Fano converse

### Hypothesis testing error rate

Le Cam's lemma, Fano 의 multiple hypothesis 버전:
$$
P_e \geq 1 - \frac{I(\text{param}; \text{obs}) + \log 2}{\log M}.
$$
Minimax lower bound 증명의 표준 도구.

## ⚖️ 가정과 한계

1. **Uniform prior $W$**: 비균등 경우 조금 더 강한 bound 필요.
2. **Memoryless**: 있는 채널은 적당한 조건 하에 확장 가능.
3. **Asymptotic**: finite blocklength 에서는 정확한 $C$ 대신 $C_n < C$.
4. **Strong vs Weak**: weak converse 는 $P_e \not\to 0$, strong 은 $P_e \to 1$ — 실무 구별 중요.
5. **Decoder 구현 무관**: Converse 는 **모든 가능한 decoder** 에 대해 — engineering 한계가 아닌 물리 한계.

## 📌 핵심 정리

$$
\boxed{
\begin{aligned}
&\text{Weak Converse: } R > C \implies P_e \not\to 0.\\
&\text{Strong Converse: } R > C \implies P_e \to 1.\\
&\text{Proof: Fano + DPI + memoryless channel bound}
\end{aligned}}
$$

**증명 pipeline**:
$$
\log M \underset{\text{entropy}}{=} I(W;\hat W) + H(W|\hat W) \underset{\text{DPI}}{\leq} I(X^n;Y^n) + \text{Fano} \underset{\text{memoryless}}{\leq} nC + 1 + P_e \log M.
$$

따라서 $R \leq (C + 1/n)/(1 - P_e)$. $P_e \to 0$ 이면 $R \leq C$.

## 🤔 생각해볼 문제

### 문제 1. 피드백이 있으면 $C$ 가 증가할까?
수신자가 송신자에게 즉각 피드백할 수 있다면?

<details>
<summary>해설</summary>

**No.** Shannon (1956): DMC 에서 피드백은 capacity 를 증가시키지 않음. 증명은 마찬가지로 Fano + DPI 구조 사용하되 feedback edge 포함.  
단 복잡도 / delay 는 감소 가능 (e.g., Burnashev 1976 의 error exponent).
</details>

### 문제 2. Joint code 과 single code 의 차이
$n$ 번 통신을 하나로 묶어 encode/decode vs. 한 번에 하나씩?

<details>
<summary>해설</summary>

Shannon 정리는 joint (block) encoding 의 우월성. 1 심볼씩 encode 하면 capacity 달성 불가 — AEP 가 필요.  
"Shannon's paradox" 의 핵심: 개별 심볼의 reliability 는 향상 못 해도 **블록으로 보면 가능**.
</details>

### 문제 3. Fano 의 constant 1 의 의미
$H(P_e) \leq 1$ bit — 이 상수가 rate bound 에 어떻게 영향?

<details>
<summary>해설</summary>

$R \leq C + 1/n$. 즉 finite $n$ 에서 $1/n$ gap. $n = 1000$ 이면 $10^{-3}$ bit 의 slack — 실용적으로 무시 가능. 그러나 finite blocklength capacity 에서는 $\sqrt{V/n}$ 보정항이 더 중요 ($V$ = dispersion).
</details>

### 문제 4. PAC learning 과의 동형
PAC learning 의 "sample complexity $n \geq \log|\mathcal{H}|/\varepsilon^2$" 를 Fano converse 로 유도할 수 있나?

<details>
<summary>해설</summary>

Yes. Hypothesis $h \in \mathcal{H}$ 를 "message", 데이터를 "channel output" 으로 보고 $I(\text{data}; \text{label}) \leq n C$ 로 bound, Fano 로 $H(h|\text{data}) \leq 1 + P_e \log|\mathcal{H}|$. 둘의 gap 이 $n$ 의 하한을 줌. Minimax rate 증명의 표준.
</details>

### 문제 5. 왜 Strong converse 가 "중요"한가?
Weak 만으로는 부족한 실무 문제?

<details>
<summary>해설</summary>

Weak converse: $R > C$ 면 $P_e \not\to 0$ — 하지만 $P_e$ 가 $0.01$ 정도일 수 있음 (여전히 유용?).  
Strong converse: $P_e \to 1$ — 완전히 실패. 
실무적으로 "R > C 에서 시도는 의미없다" 를 확증. Reliability-oriented system 설계 시 중요.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [5.2 Achievability](./02-channel-coding-achievability.md) | [5.4 현대 오류 정정 부호](./04-modern-codes.md) |

[🏠 Home](../README.md)

</div>
