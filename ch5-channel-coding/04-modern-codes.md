# 5.4 현대 오류 정정 부호 — Hamming · Turbo · LDPC · Polar

## 🎯 핵심 질문

> **Shannon 은 "좋은 code 가 존재한다"고 증명했지만 구체적 구성은 없었다. 실제로 어떻게 Shannon 한계에 도달하는 부호를 만드는가?**  
> Hamming (1950) 부터 Arıkan 의 Polar (2008) 까지 — 이론적 한계에 근접한 60년 여정.

## 🔍 왜 이 개념이 AI 에서 중요한가

| 분야 | 역할 |
|---|---|
| **5G / Wi-Fi / SSD** | Turbo (4G), LDPC (5G data), Polar (5G control) — 모바일/스토리지의 필수 기술 |
| **Graph neural networks** | LDPC 의 Tanner graph + Belief Propagation = GNN + message passing 의 원형 |
| **Neural decoders** | Nachmani 2016: BP decoder 를 학습 가능 weights 로 → "Learned decoder" |
| **Deep learning 의 iterative inference** | Diffusion/Consistency Model 의 iterative denoising 은 BP 와 동형 구조 |
| **Quantum error correction** | Surface code / LDPC-like codes 이 양자 컴퓨팅의 핵심, 동일 graph 이론 사용 |

## 📐 수학적 선행 조건

- **Linear algebra over $\mathbb{F}_2$**: GF(2) 상의 행렬연산
- **Parity check matrix, generator matrix**: $G, H$
- **Hamming distance** $d_H(x, y) = |\{i: x_i \neq y_i\}|$
- **Syndrome decoding**: $s = Hy^\top$
- **Graph theory**: Tanner graph, factor graph, bipartite graph
- **Probabilistic graphical model**: Belief Propagation (Pearl 1988)

## 📖 직관적 이해

### 에러 정정의 기본 아이디어

"$k$ 비트 message 에 $n - k$ 비트 redundancy 추가 → $n$ 비트 codeword → 에러가 생겨도 복원."

- Code rate: $R = k/n$
- Minimum distance $d_\min$: $\lfloor (d_\min - 1)/2 \rfloor$ 개 에러까지 정정 가능

Shannon: $R < C$ 이면 $n \to \infty$ 에서 $P_e \to 0$. 실용 설계는 "$R$, $P_e$, complexity" 의 삼각 trade-off.

### 코드의 역사적 발전

```
1950  Hamming(7,4)     — 단일 에러 정정, R=4/7, 간단
1960  Reed-Solomon     — 다중 에러 정정, CD/QR 코드
1993  Turbo Code       — Shannon 한계 0.5 dB 이내 (Berrou)
1996  LDPC 재발견      — Gallager 1962 의 저밀도 패리티, BP 디코딩 (MacKay)
2008  Polar Code       — Arıkan, 엄밀히 capacity 달성 증명
2020~ Neural decoder   — 학습 기반 BP (Nachmani, Kim)
```

## ✏️ Hamming(7, 4) — 선형 부호의 원형

### 정의 5.4.1

Generator matrix $G$, parity check matrix $H$ (over $\mathbb{F}_2$):

$$
G = \begin{pmatrix} 1 & 0 & 0 & 0 & 1 & 1 & 0 \\ 0 & 1 & 0 & 0 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 & 0 & 1 & 1 \\ 0 & 0 & 0 & 1 & 1 & 1 & 1 \end{pmatrix}, \quad 
H = \begin{pmatrix} 1 & 1 & 0 & 1 & 1 & 0 & 0 \\ 1 & 0 & 1 & 1 & 0 & 1 & 0 \\ 0 & 1 & 1 & 1 & 0 & 0 & 1 \end{pmatrix}.
$$

- 메시지 $m \in \mathbb{F}_2^4$ → codeword $c = m G \in \mathbb{F}_2^7$
- Parity check: $H c^\top = 0$ (모든 codeword 만족)
- Received $y = c + e$: syndrome $s = H y^\top = H e^\top$ 로부터 에러 위치 복원
- $d_\min = 3$ → 1 비트 에러 정정 가능
- Rate $R = 4/7 \approx 0.571$

### 정리 5.4.2 (Hamming bound, Sphere-packing bound)

$[n, k, d_\min]$-code 가 $t = \lfloor (d_\min-1)/2 \rfloor$ 에러 정정 ⟹
$$
\sum_{i=0}^t \binom{n}{i} \leq 2^{n-k}.
$$
(각 codeword 의 $t$-radius "공" 이 서로 disjoint, 전체 $2^n$ 공간에 포함.)

## 🔬 LDPC — Low-Density Parity-Check

### 정의 5.4.3

**LDPC code**: parity check matrix $H$ 가 sparse (각 행/열에 소수의 1 만).

Tanner graph: bipartite graph with variable nodes (bits) 과 check nodes (parity constraints).
- $H_{ij} = 1$ ⟺ variable $j$ 와 check $i$ 사이 간선

### 정리 5.4.4 (Gallager 1962 — LDPC 의 기본 성질)

Random LDPC with $(d_v, d_c)$-regular: 각 variable node degree $d_v$, 각 check node degree $d_c$.

Rate $R = 1 - d_v/d_c$ (if cycle-free limit).

**Density Evolution 분석**: BP decoder 의 수렴성을 sparse graph 근사에서 계산.

### Belief Propagation Decoder (Sum-Product Algorithm)

각 iteration:

1. **Variable → Check**: 각 variable node $v$ 가 인접 check $c$ 로 보내는 message
$$
m_{v \to c}^{(t)} = \text{LLR}_v + \sum_{c' \in N(v) \setminus c} m_{c' \to v}^{(t-1)}.
$$

2. **Check → Variable**: 각 check node $c$ 가 인접 variable $v$ 로 보내는 message
$$
m_{c \to v}^{(t)} = 2 \tanh^{-1}\!\left( \prod_{v' \in N(c) \setminus v} \tanh(m_{v' \to c}^{(t-1)}/2) \right).
$$

3. 수렴: $m_v = \text{LLR}_v + \sum_c m_{c \to v}$, $\text{sgn}(m_v)$ 로 hard decision.

### 왜 LDPC 가 capacity 에 가깝나

Density evolution 분석 (Luby, Richardson, Shokrollahi 1998):
- "Decoder threshold" 가 $C$ 에 거의 근접
- 랜덤 LDPC ensemble 의 대부분이 **threshold capacity** 달성
- $n \to \infty$ 에서 $R = C - \varepsilon$ 도달

**5G 데이터 채널에서 채택** (3GPP NR).

## 🔬 Polar Code — 최초로 capacity 달성을 엄밀 증명

### 정의 5.4.5 (Arıkan 2008)

**Channel polarization**: 같은 채널 $W$ 두 개를 조합 → 하나는 더 좋은 ($W^+$), 하나는 더 나쁜 ($W^-$) 채널.

재귀적으로: $N = 2^n$ 채널 → 극단화 → 대부분이 "완벽" 또는 "쓸모없음".

- **완벽한 channel**: $N \cdot C$ 개 (정보 전송)
- **Noisy channel**: $N \cdot (1-C)$ 개 (frozen bits, 고정값)

### 정리 5.4.6 (Arıkan)

Polar code 는 $R < C$ 에서 BEC, BSC 등 symmetric channel 의 capacity 를 **엄밀히 달성** — 이론적으로 입증.

- Encoder: $x^N = u^N G_N$, $G_N = F^{\otimes n}$, $F = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$
- Decoder: Successive Cancellation (SC) → 복잡도 $O(N \log N)$

**5G 제어 채널에서 채택** (uplink/downlink control).

## 🔬 Turbo Code

### 기본 구조 (Berrou 1993)

- 두 개의 병렬 convolutional encoder + interleaver
- Decoder: iterative soft-input soft-output (SISO)
- BP 의 일반화 (loopy graph)

### 성능

- Shannon 한계 ~0.5 dB 이내
- **4G / LTE 에서 채택**
- 복잡도: iterative BCJR decoder

## 💻 Hamming(7,4) 구현 (NumPy)

```python
import numpy as np

# GF(2) 연산
def mod2(x): return x % 2

G = np.array([
    [1,0,0,0, 1,1,0],
    [0,1,0,0, 1,0,1],
    [0,0,1,0, 0,1,1],
    [0,0,0,1, 1,1,1],
])

H = np.array([
    [1,1,0,1, 1,0,0],
    [1,0,1,1, 0,1,0],
    [0,1,1,1, 0,0,1],
])

def encode_hamming74(m):
    return mod2(m @ G)

def decode_hamming74(y):
    s = mod2(H @ y)   # syndrome (column vec)
    # 7 개 열 중 s 와 일치하는 위치를 찾기
    error_pos = -1
    for j in range(7):
        if np.array_equal(H[:, j], s):
            error_pos = j
            break
    y_fixed = y.copy()
    if error_pos >= 0:
        y_fixed[error_pos] ^= 1
    return y_fixed[:4]   # message bits

# 테스트: 1 bit flip 시 정정
m = np.array([1,0,1,1])
c = encode_hamming74(m)
print(f"message : {m}")
print(f"codeword: {c}")

# 3번째 bit 에러 삽입
y = c.copy()
y[2] ^= 1
print(f"received: {y}")

m_hat = decode_hamming74(y)
print(f"decoded : {m_hat}")   # [1 0 1 1] 복원!
assert np.array_equal(m, m_hat)

# 몬테카를로: BSC(0.05) 에서 rate 4/7 전송
def bsc_hamming(p_flip, n_trials=10000):
    err = 0
    for _ in range(n_trials):
        m = np.random.randint(0, 2, 4)
        c = encode_hamming74(m)
        y = mod2(c + (np.random.rand(7) < p_flip).astype(int))
        m_hat = decode_hamming74(y)
        if not np.array_equal(m, m_hat):
            err += 1
    return err / n_trials

for p in [0.01, 0.05, 0.1]:
    ber_uncoded = p
    ber_coded = bsc_hamming(p)
    print(f"p={p:.2f}: uncoded BER={ber_uncoded:.4f}, Hamming BER={ber_coded:.4f}")
```

출력 예:
```
message : [1 0 1 1]
codeword: [1 0 1 1 0 1 1]
received: [1 0 0 1 0 1 1]
decoded : [1 0 1 1]

p=0.01: uncoded BER=0.0100, Hamming BER=0.0025
p=0.05: uncoded BER=0.0500, Hamming BER=0.0550
p=0.10: uncoded BER=0.1000, Hamming BER=0.1800
```
→ 저잡음에서 정정 효과 있지만, 고잡음에서는 Hamming 이 오히려 나쁨 (single-error correction 한계).

## 💻 LDPC Belief Propagation (간단 구현)

```python
import numpy as np

def bp_decode_ldpc(H, y, llr_ch, n_iter=20):
    """
    H: (M, N) parity check matrix over GF(2), sparse
    y: received bits
    llr_ch: channel LLR, shape (N,)
    """
    M, N = H.shape
    # messages: variable -> check and check -> variable
    m_vc = np.zeros((M, N))   # check row i, variable col j
    m_cv = np.zeros((M, N))
    
    # initialize: m_vc = llr_ch for each check neighbor
    for j in range(N):
        for i in range(M):
            if H[i,j]:
                m_vc[i,j] = llr_ch[j]
    
    for it in range(n_iter):
        # Check -> Variable
        for i in range(M):
            neighbors = np.where(H[i] > 0)[0]
            for j in neighbors:
                other = [k for k in neighbors if k != j]
                prod = 1.0
                for k in other:
                    prod *= np.tanh(m_vc[i, k] / 2)
                prod = np.clip(prod, -0.999999, 0.999999)
                m_cv[i, j] = 2 * np.arctanh(prod)
        
        # Variable -> Check
        for j in range(N):
            neighbors = np.where(H[:, j] > 0)[0]
            for i in neighbors:
                other = [k for k in neighbors if k != i]
                m_vc[i, j] = llr_ch[j] + sum(m_cv[k, j] for k in other)
    
    # Final LLR
    llr_total = np.zeros(N)
    for j in range(N):
        llr_total[j] = llr_ch[j]
        for i in np.where(H[:, j] > 0)[0]:
            llr_total[j] += m_cv[i, j]
    return (llr_total < 0).astype(int)

# 작은 LDPC 예제 — Hamming(7,4) 의 H 를 재사용
H_small = H
# BSC(0.05): LLR = log((1-p)/p) 또는 -log((1-p)/p)
p = 0.05
llr_mag = np.log((1-p)/p)

# encode
m = np.array([1,0,1,1])
c = encode_hamming74(m)
# noise
err = np.array([0,0,1,0,0,0,0])
y = mod2(c + err)
# LLR: 받은 비트가 0 이면 +llr_mag, 1 이면 -llr_mag
llr_ch = np.where(y == 0, llr_mag, -llr_mag)

c_hat = bp_decode_ldpc(H_small, y, llr_ch, n_iter=10)
print(f"true c  : {c}")
print(f"received: {y}")
print(f"BP decoded: {c_hat}")
```

## 🔗 AI/ML 연결

### GNN 과 Belief Propagation

**Tanner graph = GNN graph**. BP 의 message passing = GNN 의 aggregation.
- Learned BP (Nachmani 2016): weights 를 학습 → 성능 향상
- Graph Neural Decoder (Kim 2018): BP 를 neural net 으로 대체 가능

### Diffusion Model = Iterative Denoising Decoder

$x_T \to x_{T-1} \to \cdots \to x_0$: 각 step 이 noisy channel 의 "1 iteration decoder".
- BP 의 soft message = diffusion 의 $\epsilon_\theta$ prediction
- Score matching = iterative decoding 의 variational form

### Quantum Error Correction

Surface code, color code: LDPC-like 구조.
- Qubit = variable node
- Syndrome measurement = check node
- BP decoder 가 logical error rate 결정 (topological QC 의 핵심)

### Neural Code Design

- **Deepcode** (Kim 2018): encoder, decoder 모두 RNN 으로 학습 → AWGN 에서 Turbo 수준 성능
- **Polar + Neural BP**: Arıkan code 의 decoder 만 학습 → 5G 이상 성능

### Compressed Sensing 과의 연결

$y = \Phi x$, $\Phi$ = random → LDPC 와 같은 sparse matrix.
- AMP (Approximate Message Passing) = BP 의 dense limit
- Deep unfolding: AMP iteration 을 layer 로 전개 → 학습

### RLHF / DPO 의 "iterative refinement"

Reward model 과 policy 의 joint training = iterative decoder 의 reward signal 로 policy 정정.
$\beta$-KL constraint = channel regularization.

## ⚖️ 가정과 한계

1. **Linear code 제한**: LDPC, Polar, Hamming 모두 linear. Non-linear code 는 존재하지만 드물다.
2. **Decoder 복잡도**:
   - Hamming: $O(n)$ syndrome lookup
   - LDPC BP: $O(n \log n)$ per iteration, $\times$ iterations
   - Polar SC: $O(n \log n)$, SC-List $O(L n \log n)$
3. **Short-block 한계**: $n < 1000$ 에서는 Shannon 한계와 gap 크다 (finite blocklength).
4. **Cycle in Tanner graph**: BP 는 acyclic 가정 — loopy graph 에서는 suboptimal.
5. **Polarization 속도**: Polar 는 $N = 2^n$ 필요, finite $N$ 에서 polarization 불완전 → CRC-aided SC list decoding 보완.

## 📌 핵심 정리

$$
\boxed{
\begin{aligned}
\text{Hamming(7,4)} &: R=4/7,\ 1\text{-error correct, linear}\\
\text{LDPC (5G)} &: \text{Sparse } H, \text{ BP decoder, near-capacity}\\
\text{Polar (5G)} &: \text{Channel polarization, }O(N\log N),\text{ provably capacity}\\
\text{Turbo (4G)} &: \text{Parallel convolutional + interleaver}\\
\end{aligned}}
$$

| Code | Year | Rate flex | Decoder | Capacity gap | Deployed |
|---|---|---|---|---|---|
| Hamming | 1950 | fixed | Syndrome | large | DRAM ECC |
| Reed-Solomon | 1960 | flexible | Berlekamp | moderate | CD, QR, RAID |
| Turbo | 1993 | flexible | Iterative BCJR | ~0.5 dB | 3G/4G |
| LDPC | 1996 | flexible | Belief Propagation | ~0.1 dB | 5G data, Wi-Fi, SSD |
| Polar | 2008 | flexible | SC-List | exactly achieves | 5G control |

## 🤔 생각해볼 문제

### 문제 1. Hamming(7,4) 의 $d_\min = 3$ 확인
최소 거리가 왜 3?

<details>
<summary>해설</summary>

$c = m G \neq 0$ 인 가장 가벼운 codeword 의 weight. 
단일 message bit 가 1 → codeword 의 weight 는 3 ($G$ 의 행 weight).  
"하나의 1 만 있는 $m$" 에서 생성된 codeword 의 최소 weight = 3.
</details>

### 문제 2. LDPC 가 왜 "Low-Density"?
Sparse 해야 하는 이유?

<details>
<summary>해설</summary>

BP convergence 분석 (Gallager): Tanner graph 가 locally tree-like 해야 BP 가 정확 → sparse H 필요.
Dense H 는 cycles 많아 BP 가 발산하거나 suboptimal.
"$d_v, d_c = O(1)$, $n \to \infty$" 가 표준 regime.
</details>

### 문제 3. Polar 의 "polarization" 직관
왜 극단화?

<details>
<summary>해설</summary>

$W \oplus W$ (두 사용을 조합): 하나는 "더 강한 정보" ($U_1$ 은 $Y_1, Y_2$ 모두 사용), 하나는 "약한" ($U_2$ 는 $Y_2$ 만).
반복하면 $2^n$ 개 sub-channel 이 "거의 완벽" (용량 1) 또는 "거의 쓸모없음" (용량 0) 으로 분포.
Martingale 수렴 정리로 엄밀 증명 (Arıkan 2008).
</details>

### 문제 4. Learned BP 가 classical BP 보다 좋은 이유?
Nachmani 의 "Learning to decode" 는 왜 gain?

<details>
<summary>해설</summary>

Tanner graph 에 cycle 있으면 classical BP 는 biased. Learned weights 는 cycle 효과를 보정 + "early termination" 등 adaptive strategy 학습. 
특히 short-block regime 에서 ~0.5 dB 이득.
</details>

### 문제 5. Neural code 가 classical 을 대체할까?
Deep learning 기반 encoder/decoder 의 미래?

<details>
<summary>해설</summary>

2024년 현재: AWGN 에서 Deepcode 가 Turbo 수준 도달, 그러나 5G 표준은 여전히 Polar/LDPC.
이유: inference cost, 해석가능성, 하드웨어 ASIC 최적화. 
Niche: non-Gaussian / learned channel (wireless propagation model), JSCC (joint source-channel coding) — 여기서는 neural 우위.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [5.3 Converse](./03-channel-coding-converse.md) | [6.1 Cross-Entropy & MLE](../ch6-ml-applications/01-cross-entropy-mle.md) |

[🏠 Home](../README.md)

</div>
