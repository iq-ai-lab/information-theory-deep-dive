# 5.1 채널 용량 — $C = \max_{p(x)} I(X;Y)$

## 🎯 핵심 질문

> **잡음이 있는 채널은 "얼마나 많은 정보를 신뢰성 있게 전송할 수 있는가"?**  
> 그 답은 왜 **입력 분포에 대한 MI 의 최대값**이며, BSC·BEC·AWGN 에서 구체적으로 얼마인가?

1948년 Shannon 의 혁명적 통찰: 채널의 "성능"은 공학적 세부사항(변조·부호·디코더)이 아니라 오직 하나의 정보량 — **채널 용량 $C$** 로 결정된다.
전송률 $R < C$ 이면 오류 확률을 임의로 작게 만들 수 있고 (achievability — 다음 문서),  
$R > C$ 이면 그것이 불가능하다 (converse — 5.3 에서).

## 🔍 왜 이 개념이 AI 에서 중요한가

| 분야 | 역할 |
|---|---|
| **신경망 = 잡음 채널** | 각 layer 는 잡음 있는 채널. $I(X; Z_\ell)$ 의 "용량" 관점이 Information Bottleneck |
| **Attention 용량** | MHA 의 head 하나가 전달 가능한 정보량은 $I(\text{query}; \text{output})$ 의 상한 |
| **통신 효율 분산 학습** | Federated learning / gradient compression 의 이론 상한 |
| **Neural scaling law** | 데이터 $\to$ 모델로 전달 가능한 정보량 ≤ $C$. 데이터 부족 시 bottleneck |
| **RLHF reward channel** | 사람의 피드백 $\to$ 모델. Noisy label + limited bandwidth — Shannon 관점에서 본질 |

핵심 비유:
```
데이터  ──►  [encoder]  ──►  [noisy channel]  ──►  [decoder]  ──►  추정
  X            codeword         Y = X+noise         decode          X̂
```
딥러닝에서 encoder/decoder 는 신경망, channel 은 "잠재공간을 거치는 정보 손실" 자체.

## 📐 수학적 선행 조건

- **조건부 확률 분포** $p(y|x)$: 채널의 수학적 정의 그 자체
- **상호정보량 $I(X;Y)$**: Chapter 3 전체 (특히 3.1 세 가지 정의)
- **Jensen 부등식 / 볼록성**: $I(X;Y)$ 는 $p(x)$ 에 대해 concave — 최대값이 unique
- **Lagrange 승수법**: KKT 로 BSC capacity 유도

## 📖 직관적 이해

### "잡음 = 정보 누설"

결정적 채널 $Y = X$: 완벽한 전달 → $I(X;Y) = H(X)$. 용량 = $\log|\mathcal{X}|$ (입력 공간 전체).

완전히 잡음: $Y \perp X$ → $I(X;Y) = 0$. 용량 = 0.

**중간 상태**: 입력이 출력에서 얼마나 "복원 가능"한가 = $I(X;Y)$.

### 왜 $\max_{p(x)}$?

채널 $p(y|x)$ 은 주어진 자연. 우리가 선택할 수 있는 것은 **입력 분포 $p(x)$** — 어떤 심볼을 어떤 빈도로 쓸 것인가.

최적의 입력 분포는 채널의 특성을 최대한 활용하는 분포. BSC 에서는 균등(1/2, 1/2), AWGN 에서는 가우시안.

### 정보를 "한 번에 몇 비트"?

$C$ 는 "채널 1회 사용당 전달 가능한 최대 비트 수"이다.
- BSC(p=0.1): $C = 1 - H(0.1) \approx 0.53$ bit/use — 1비트 보내도 0.47 bit 은 잡음에 먹힘.
- AWGN(SNR=10): $C = \frac{1}{2} \log_2(11) \approx 1.73$ bit/use.
- 이상적 전송: 블록 길이 $n$ 사용 → $nC$ bits 전달.

## ✏️ 엄밀한 정의

### 정의 5.1.1 (이산 메모리리스 채널, DMC)

**이산 메모리리스 채널** (Discrete Memoryless Channel) 은 세 가지로 구성:

$$
(\mathcal{X},\ \mathcal{Y},\ p(y|x))
$$

- $\mathcal{X}$: 입력 알파벳 (유한 집합)
- $\mathcal{Y}$: 출력 알파벳
- $p(y|x)$: 전이 확률 — 입력 $x$ 일 때 출력이 $y$ 일 확률

**Memoryless**: $n$ 번 사용 시 $p(y^n | x^n) = \prod_{i=1}^n p(y_i | x_i)$. 이전 입력/출력과 독립.

### 정의 5.1.2 (채널 용량)

$$
\boxed{C = \max_{p(x)} I(X; Y)}
$$

여기서 $I(X;Y) = \sum_{x,y} p(x) p(y|x) \log \frac{p(y|x)}{\sum_{x'} p(x') p(y|x')}$.

최대는 입력 분포 $p(x)$ 에 대해 (simplex $\{p(x) \geq 0, \sum p(x) = 1\}$ 위에서). 단위: bits/channel use (log$_2$) 또는 nats (log$_e$).

### Lemma 5.1.3 (용량의 존재성)

$I(X;Y)$ 는:
- $p(x)$ 에 대해 **continuous** (compact simplex 위에서)
- $p(x)$ 에 대해 **concave** — Chapter 3.1 에서 증명함

⟹ simplex 위에서 최대값이 반드시 존재하며, 유일한 maximizer 가 있다 (strict concavity — 조건 필요).

## 🔬 주요 채널의 용량 — 증명 포함

### 정리 5.1.4 (이진 대칭 채널, BSC)

BSC$(p)$: $\mathcal{X} = \mathcal{Y} = \{0, 1\}$, $p(1|0) = p(0|1) = p$ (0 ≤ $p$ ≤ 1/2).

$$
\boxed{C_\text{BSC} = 1 - H(p)}
$$

(단위: bits; $H(p) = -p\log_2 p - (1-p)\log_2(1-p)$)

**증명.**

$I(X;Y) = H(Y) - H(Y|X)$.

**Step 1**: $H(Y|X)$ 계산.
주어진 $X = x$ 일 때 $Y$ 는 "확률 $p$ 로 flip" 되므로 $Y|X=x \sim \text{Bernoulli}(p)$. 따라서
$$
H(Y|X=x) = H(p), \quad \forall x.
$$
$$
H(Y|X) = \sum_x p(x) H(Y|X=x) = H(p).
$$
(입력 분포와 무관!)

**Step 2**: $H(Y)$ 최대화.
$Y$ 는 $\{0,1\}$ 값을 가지므로 $H(Y) \leq 1$ bit, 등호는 $Y \sim \text{Uniform}$ 일 때.
대칭성에 의해 $X \sim \text{Uniform}(0, 1/2)$ 이면 $Y$ 도 균등:
$$
p(Y=0) = \tfrac{1}{2}(1-p) + \tfrac{1}{2} p = \tfrac{1}{2}.
$$
따라서 $\max_{p(x)} H(Y) = 1$, 달성은 $p(x) = (1/2, 1/2)$.

**Step 3**: 
$$
C = \max_{p(x)} I(X;Y) = 1 - H(p). \qquad \blacksquare
$$

**수치**: $p=0$ 이면 $C=1$ (완벽 채널), $p=1/2$ 이면 $C=0$ (완전 무작위), $p=0.1$ 이면 $C \approx 0.531$.

### 정리 5.1.5 (이진 소거 채널, BEC)

BEC$(\varepsilon)$: $\mathcal{X} = \{0,1\}$, $\mathcal{Y} = \{0, 1, e\}$ (e = erasure). 확률 $1-\varepsilon$ 로 정확히 전달, $\varepsilon$ 로 $e$ 출력.

$$
\boxed{C_\text{BEC} = 1 - \varepsilon}
$$

**증명 스케치.**

$Y$ 에서 $e$ 가 관측되면 $X$ 에 대한 정보 0. $Y \in \{0,1\}$ 이면 $X$ 완전 복원 (오류 없음).

$I(X;Y) = H(X) - H(X|Y)$, 여기서
$$
H(X|Y=e) = H(X), \quad H(X|Y=0) = H(X|Y=1) = 0.
$$
따라서 $H(X|Y) = \varepsilon H(X)$, $I(X;Y) = (1-\varepsilon) H(X)$.
최대 $H(X) = 1$ (균등 입력) → $C = 1 - \varepsilon$. $\blacksquare$

### 정리 5.1.6 (가산 백색 가우시안 잡음 채널, AWGN — Shannon–Hartley)

연속 채널: $Y = X + N$, 여기서 $N \sim \mathcal{N}(0, \sigma^2)$, 입력 파워 제약 $\mathbb{E}[X^2] \leq P$.

$$
\boxed{C_\text{AWGN} = \frac{1}{2} \log_2\!\left(1 + \frac{P}{\sigma^2}\right)}
$$

**증명.**

$I(X;Y) = h(Y) - h(Y|X) = h(Y) - h(N)$ (잡음이 입력 독립이므로).

$h(N) = \frac{1}{2}\log_2(2\pi e \sigma^2)$ (정규분포의 미분 엔트로피, Ch1-05).

$Y = X + N$ 의 분산 $\leq P + \sigma^2$ (독립 가정). **주어진 분산 조건 하에서 $h(Y)$ 최대는 $Y$ 가 정규분포일 때** (Ch1-06 MaxEnt 정리).

$Y$ 가 정규가 되려면 $X \sim \mathcal{N}(0, P)$ (가우시안 입력). 이때 $Y \sim \mathcal{N}(0, P+\sigma^2)$, 
$$
h(Y) = \frac{1}{2}\log_2(2\pi e (P+\sigma^2)).
$$
$$
C = \frac{1}{2}\log_2\!\left(\frac{2\pi e(P+\sigma^2)}{2\pi e \sigma^2}\right) = \frac{1}{2}\log_2\!\left(1 + \frac{P}{\sigma^2}\right). \qquad \blacksquare
$$

SNR (signal-to-noise ratio) = $P/\sigma^2$.

대역폭 $W$ Hz 의 연속 채널: $C = W \log_2(1 + \text{SNR})$ bits/sec — **Shannon–Hartley 공식**, 5G/Wi-Fi 스펙의 이론 한계.

### 정리 5.1.7 (대칭 채널 일반화)

**대칭 채널** (symmetric channel): 전이행렬의 모든 행이 같은 값의 permutation, 모든 열도 같은 값의 permutation.

⟹ $C = \log |\mathcal{Y}| - H(\text{row})$, maximizer 는 **균등 입력**.

(BSC 는 $|\mathcal{Y}|=2$, row = $(1-p, p)$ 의 특수 케이스.)

## 💻 NumPy 로 BSC·BEC·AWGN 용량 검증

```python
import numpy as np

# 5.1.A BSC capacity
def h_binary(p):
    if p == 0 or p == 1:
        return 0.0
    return -p*np.log2(p) - (1-p)*np.log2(1-p)

def C_BSC(p):
    return 1 - h_binary(p)

print(f"BSC(0.0)  C = {C_BSC(0.0):.4f}")   # 1.0000
print(f"BSC(0.1)  C = {C_BSC(0.1):.4f}")   # 0.5310
print(f"BSC(0.5)  C = {C_BSC(0.5):.4f}")   # 0.0000

# 5.1.B BSC MI 를 입력분포별로 grid-search (C = max 확인)
def I_BSC(p_flip, p_x):
    # p_x = P(X=1), X: Bernoulli
    P = np.array([[1-p_flip, p_flip],
                  [p_flip, 1-p_flip]])   # rows: X=0, X=1
    px = np.array([1-p_x, p_x])
    py = px @ P                          # output marginal
    pxy = P * px[:, None]                # joint
    mi = 0.0
    for i in range(2):
        for j in range(2):
            if pxy[i,j] > 0:
                mi += pxy[i,j] * np.log2(pxy[i,j] / (px[i] * py[j]))
    return mi

pxs = np.linspace(0, 1, 101)
I_vals = [I_BSC(0.1, px) for px in pxs]
best = np.argmax(I_vals)
print(f"\nBSC(0.1) best input p_X(1) = {pxs[best]:.2f}, "
      f"I = {I_vals[best]:.4f}")  # 0.50, 0.5310

# 5.1.C BEC capacity
def C_BEC(eps):
    return 1 - eps

for eps in [0.0, 0.1, 0.3, 0.5, 0.9]:
    print(f"BEC({eps})  C = {C_BEC(eps):.4f}")

# 5.1.D AWGN capacity
def C_AWGN(P, sigma2):
    return 0.5 * np.log2(1 + P/sigma2)

for snr_db in [-10, 0, 10, 20, 30]:
    snr = 10**(snr_db/10)
    print(f"SNR={snr_db:+3d}dB  C = {C_AWGN(snr, 1):.3f} bits/use")
```

출력 예:
```
BSC(0.0)  C = 1.0000
BSC(0.1)  C = 0.5310
BSC(0.5)  C = 0.0000

BSC(0.1) best input p_X(1) = 0.50, I = 0.5310

BEC(0.0)  C = 1.0000
BEC(0.1)  C = 0.9000
BEC(0.3)  C = 0.7000
BEC(0.5)  C = 0.5000
BEC(0.9)  C = 0.1000

SNR=-10dB  C = 0.069 bits/use
SNR= +0dB  C = 0.500 bits/use
SNR=+10dB  C = 1.730 bits/use
SNR=+20dB  C = 3.329 bits/use
SNR=+30dB  C = 4.983 bits/use
```

→ grid-search 로 구한 BSC 최적 $p_X(1) = 0.5$ 와 $I = 0.5310$ 이 이론값과 일치.

### Blahut–Arimoto 알고리즘 (일반 채널의 수치적 capacity)

```python
def blahut_arimoto(P, n_iter=200, tol=1e-10):
    """
    P[i,j] = p(y=j | x=i).
    반복적으로 p(x) 를 갱신하여 capacity 수렴.
    """
    nx, ny = P.shape
    px = np.ones(nx) / nx     # 초기: 균등
    for _ in range(n_iter):
        py = px @ P
        # q(x|y) = p(x)p(y|x) / p(y)
        log_q = np.log(P + 1e-300) - np.log(py + 1e-300)   # log q(y|x) / p(y)
        # p_new(x) ∝ exp( sum_y p(y|x) log q(y|x)/p(y) )
        exponent = (P * log_q).sum(axis=1)
        px_new = np.exp(exponent)
        px_new /= px_new.sum()
        if np.linalg.norm(px_new - px) < tol:
            break
        px = px_new
    C = I_from_joint(P, px)
    return px, C

def I_from_joint(P, px):
    py = px @ P
    mi = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i,j] > 0 and px[i] > 0:
                mi += px[i]*P[i,j] * np.log2(P[i,j] / (py[j] + 1e-300))
    return mi

# 비대칭 채널 예: Z-channel
P_Z = np.array([[1.0, 0.0],
                [0.3, 0.7]])   # 1 -> 0 with prob 0.3
px_opt, C_opt = blahut_arimoto(P_Z)
print(f"\nZ-channel: p_X(0)={px_opt[0]:.3f}, p_X(1)={px_opt[1]:.3f}, C={C_opt:.4f}")
# → 비대칭: 균등 아님
```

## 🔗 AI/ML 연결

### Information Bottleneck 과 "유효 용량"

Tishby (2000) 의 IB: 신경망 layer $Z$ 는 noisy channel.
- 입력 → 표현 $Z$: $I(X; Z)$ 가 작을수록 compression (정규화)
- 표현 → 레이블 $Y$: $I(Z; Y)$ 가 클수록 정보 보존

**각 layer 의 bottleneck capacity** = $\max I(X; Z)$ under constraint. 딥러닝의 "generalization gap" 을 채널 용량 관점에서 해석 (Ch6-04 에서 상세).

### Attention 의 channel 해석

Transformer attention: $\text{Attn}(Q,K,V) = \text{softmax}(QK^\top/\sqrt{d}) V$.
각 head 는 $Q \to V$ 채널, 용량은 $d$ (embedding dim) 과 attention temperature 로 조절됨.
**Multi-head** = 병렬 채널 → 총 capacity 증가.

### CLIP 의 symmetric InfoNCE = two-way channel

CLIP 의 loss $-\log \frac{\exp(s(x,y))}{\sum \exp(s(x, y'))}$ 는 $I(X; Y)$ 의 하한 (Ch3-05).
학습은 $I_\theta(\text{image}; \text{text})$ 을 용량에 근접시키는 것.

### RLHF reward 채널

Human preference → reward model: noisy, limited samples.
Bradley-Terry likelihood 의 MLE 는 reward channel 의 capacity 수준 정보 전달.
DPO 의 KL-constrained update 는 $R > C$ 에서 발산하는 수학적 이유와 연결.

### 분산 학습의 gradient 압축

All-reduce 시 각 노드 → 서버: bandwidth-limited channel.
Quantization (1-bit SGD, TopK) = lossy source coding → 채널 이후 재구성.
이론적 최소 오류: $H(\text{gradient}) - C$ 조건에서 결정.

## ⚖️ 가정과 한계

1. **Memoryless 가정**: 실제 채널은 correlated noise (burst error). → Markov channel 등으로 확장 필요.
2. **피드백 없음**: 피드백 채널 이용 시 capacity 증가 안 함 (Shannon 1956) — 하지만 delay/complexity 는 감소.
3. **입력 제약**: BSC/BEC 는 discrete 무제약, AWGN 은 power constraint 있음. 제약마다 maximizer 다름.
4. **구성적 코드 vs 존재 증명**: $C$ 는 **존재성** 만 말함 — 실제 구현은 5.4 의 Turbo/LDPC/Polar 가 필요.
5. **유한 블록길이**: $n \to \infty$ 극한. 현실은 $n$ 유한 → finite blocklength capacity (Polyanskiy 2010) 가 진짜 한계.

## 📌 핵심 정리

$$
\boxed{
\begin{aligned}
C &= \max_{p(x)} I(X; Y) \\
C_\text{BSC}(p) &= 1 - H(p) \\
C_\text{BEC}(\varepsilon) &= 1 - \varepsilon \\
C_\text{AWGN} &= \frac{1}{2}\log_2(1 + P/\sigma^2)
\end{aligned}}
$$

- $C$ 는 "채널 1회 사용당 신뢰 전송 가능한 최대 비트 수"
- $p(x)$ 에 대해 concave → 유일한 capacity
- 대칭 채널: 균등 입력이 최적
- AWGN: 가우시안 입력이 최적, MaxEnt 와 연결
- Blahut–Arimoto: 일반 채널의 수치적 계산 알고리즘

## 🤔 생각해볼 문제

### 문제 1. BSC(0.25) 의 용량
$C_\text{BSC}(0.25) = $?

<details>
<summary>해설</summary>

$H(0.25) = -0.25\log_2 0.25 - 0.75\log_2 0.75 = 0.5 + 0.75 \cdot 0.4150 = 0.8113$.
$C = 1 - 0.8113 = 0.1887$ bits/use.
</details>

### 문제 2. BSC 와 BEC 비교
BSC$(p)$ 와 BEC$(p)$ 중 어느 것이 capacity 가 더 큰가?

<details>
<summary>해설</summary>

$C_\text{BSC}(p) = 1 - H(p)$, $C_\text{BEC}(p) = 1 - p$.  
$p \in (0, 1/2)$: $H(p) > p$ (Jensen 부등식) → $C_\text{BSC} < C_\text{BEC}$.  

BEC 가 "모른다"는 것을 알려주므로 오히려 정보량이 많다. "wrong" 은 치명적, "erased" 는 회복 가능.
</details>

### 문제 3. Capacity 가 concave 함수?
$I(X;Y)$ 는 $p(x)$ 에 대해 concave 이나 transition matrix $p(y|x)$ 에 대해서는? 

<details>
<summary>해설</summary>

$p(y|x)$ 에 대해서는 **convex** 이다 (Cover & Thomas Thm 2.7.4). 
즉 혼합 채널의 capacity 는 개별 capacity 의 평균 이하. 
→ "bad channel + good channel 의 평균 이 bad channel 보다 덜 나쁘다" 는 의미 있음.
</details>

### 문제 4. AWGN 의 SNR = 1 (0 dB)
$C = ?$, 이론상 1회 사용당 몇 비트 전달?

<details>
<summary>해설</summary>

$C = \tfrac{1}{2}\log_2(2) = 0.5$ bit/use. 2번 사용 → 1 bit 전송 가능.
</details>

### 문제 5. 신경망 layer 와 용량
Attention head 의 $d = 64$ (embedding 차원), softmax temperature $\tau$ 가 작아지면 용량은 어떻게 변하나?

<details>
<summary>해설</summary>

$\tau \to 0$: attention 이 one-hot 에 가까워짐 → 출력 = 하나의 value 복사 = deterministic channel.
$\tau \to \infty$: 모든 value 균등 가중 → 모든 입력에 같은 출력 → $I(Q; \text{out}) \to 0$.

용량은 $\tau$ 가 작을수록 커지지만 학습 불안정. 실무에서 $\tau = 1$ (또는 $1/\sqrt{d}$) 은 안정성-용량 trade-off.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [4.5 Arithmetic Coding](../ch4-source-coding/05-arithmetic-coding.md) | [5.2 Channel Coding Theorem — Achievability](./02-channel-coding-achievability.md) |

[🏠 Home](../README.md)

</div>
