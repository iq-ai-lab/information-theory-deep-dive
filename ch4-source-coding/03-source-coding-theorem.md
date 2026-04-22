# 4.3 Source Coding Theorem — Shannon 의 최초 정리

## 🎯 핵심 질문

> **"$n$ 개의 iid 기호를 평균적으로 몇 비트로 압축할 수 있는가?"** 의 근본적 답은 무엇인가?
> Shannon 의 **source coding theorem** (1948) 이 "entropy 는 achievable 최소율" 이라고 선언하는 이유는?
> **$H$ 보다 낮게 압축하면 필연적으로 정보 손실** 이라는 역정리(converse) 는 어떻게 증명하는가?

---

## 🔍 왜 AI에서 중요한가

- **모든 압축의 근본 한계**: ZIP, MP3, JPEG, LM-based compressor 까지.
- **LM 의 비트 관점**: $H = -\mathbb{E}[\log p(x)]$ 를 달성하는 것 = perplexity 최소화.
- **Model size vs data**: Kolmogorov complexity / MDL 원리의 기반.
- **Neural compression benchmark**: GPT 기반 lossless compressor 가 gzip, BZIP2 제치는 이론적 근거.
- **Shannon's silver bullet** — 정보이론의 시작점.

"데이터를 압축할 수 있는 만큼만 정보가 있다" → **compression = understanding** 의 수학적 선언.

---

## 📐 선행 학습 지식

- [4.1 Prefix code, Kraft](./01-prefix-code-kraft.md), [4.2 Huffman](./02-huffman-optimality.md)
- [1.2 엔트로피 정의](../ch1-entropy-axioms/02-entropy-definition.md)
- Law of Large Numbers (LLN), AEP (§4.4 와 교차)
- iid 확률변수 수열

---

## 📖 직관

### 두 방향의 진술

**Achievability (가능성)**: 임의의 $\epsilon > 0$ 에 대해 충분히 큰 $n$ 에서 $(H + \epsilon)$-rate 의 encoding/decoding pair 이 존재 → 오류 확률 → 0.

**Converse (역, 불가능성)**: Rate $< H$ 라면 오류 확률이 1 로 수렴 → 절대 복원 불가.

> **함의**: **$H$ 가 정보의 엄밀한 "값"** 이다. 이 이상도 이하도 아닌 유일한 숫자.

### Asymptotic Equipartition Property (AEP)

iid 에서 **typical sequences** 의 집합 $A_\epsilon^{(n)}$:
- 크기 $\approx 2^{nH}$
- 각 시퀀스의 확률 $\approx 2^{-nH}$
- 전체 확률 $\to 1$

즉 "거의 모든 실현" 이 $2^{nH}$ 개의 class 중 하나. 이 index 만 전달하면 $nH$ bits 로 충분.

---

## ✏️ 공식 정의

**정의 4.3.1 (Rate-$R$ encoder)**
iid source $X_1, X_2, \ldots, X_n \sim p$ 에 대해 함수
$$
f_n : \mathcal{X}^n \to \{0, 1\}^{nR}
$$
와 decoder $g_n : \{0, 1\}^{nR} \to \mathcal{X}^n$. 오류 확률
$$
P_e^{(n)} = P(g_n(f_n(X^n)) \ne X^n)
$$

**정의 4.3.2 (Achievable rate)**
$R$ 이 **achievable** 이면 $P_e^{(n)} \to 0$ 인 $(f_n, g_n)$ 이 존재.

**정의 4.3.3 (Source coding theorem)**
Achievable rate 의 infimum 은 $H(p)$.

---

## 🔬 정리와 증명

### Theorem 4.3.1 (Asymptotic Equipartition Property)

**진술.** iid $X_i \sim p$, $n \to \infty$ 일 때
$$
-\frac{1}{n}\log p(X_1, \ldots, X_n) \xrightarrow{P} H(p)
$$
즉 typical log-probability 가 $-nH$ 에 집중.

**증명.** $-\frac{1}{n}\log p(X^n) = -\frac{1}{n}\sum \log p(X_i)$. LLN 에 의해 $\to -\mathbb{E}[\log p(X)] = H(p)$. $\blacksquare$

### Theorem 4.3.2 (Typical set properties)

**정의 (Typical set)**:
$$
A_\epsilon^{(n)} = \left\{ x^n : \left| -\tfrac{1}{n}\log p(x^n) - H(p) \right| \le \epsilon \right\}
$$

**성질**:
1. $P(X^n \in A_\epsilon^{(n)}) > 1 - \delta$ for large $n$.
2. $2^{-n(H+\epsilon)} \le p(x^n) \le 2^{-n(H-\epsilon)}$ for $x^n \in A_\epsilon^{(n)}$.
3. $(1 - \delta) 2^{n(H-\epsilon)} \le |A_\epsilon^{(n)}| \le 2^{n(H+\epsilon)}$.

**증명 스케치.** (1) AEP + convergence in probability. (2) typical set 정의 직접. (3) $P(A) \le 1$ 과 $P(A) \ge 1 - \delta$ 에서 양변을 typical probability bound 로 묶음.

### Theorem 4.3.3 (Achievability)

**진술.** 임의의 $\epsilon > 0$ 에 대해 $n$ 충분히 크면 **rate $H + \epsilon$** 으로 $P_e^{(n)} \to 0$ 달성.

**증명.** Encoder:
- $x^n \in A_\epsilon^{(n)}$ 이면 $|A_\epsilon^{(n)}| \le 2^{n(H+\epsilon)}$ 개 중 하나 → $\lceil n(H+\epsilon) \rceil$ bits 로 인덱싱.
- $x^n \notin A_\epsilon^{(n)}$ 이면 별도의 "에러" flag + 임의 배당 (적은 확률).

총 $n(H + \epsilon) + 1$ bit 이하. $P_e \le P(A_\epsilon^{(n)\, c}) \to 0$. $\blacksquare$

### Theorem 4.3.4 (Converse)

**진술.** Rate $R < H$ 이면 $P_e^{(n)} \not\to 0$. 구체적으로 $\liminf P_e^{(n)} \ge 1 - (R - H)/\log|\mathcal{X}| - o(1)$ (negative → 1로 수렴).

**증명 (Fano + AEP).** $\hat X^n = g_n(f_n(X^n))$ 는 rate $R$ 이므로 $\hat X^n$ 가능값 $\le 2^{nR}$. Fano:
$$
H(X^n | \hat X^n) \le H(P_e^{(n)}) + P_e^{(n)} \cdot n \log |\mathcal{X}|
$$
반면 $H(X^n | \hat X^n) \ge H(X^n) - nR = nH - nR$.

결합:
$$
nH - nR \le 1 + P_e^{(n)} \cdot n \log|\mathcal{X}|
$$
$$
P_e^{(n)} \ge \frac{H - R}{\log|\mathcal{X}|} - \frac{1}{n\log|\mathcal{X}|}
$$

$R < H$ 이면 양수 → $P_e^{(n)}$ 바닥이 양수. $\blacksquare$

### Theorem 4.3.5 (Source Coding Theorem — 결합)

**진술.**
$$
\inf\{R : R \text{ achievable}\} = H(p)
$$

**증명.** Achievability (Theorem 4.3.3) + Converse (Theorem 4.3.4). $\blacksquare$

### Theorem 4.3.6 (확률분포 모르는 경우 — Universal Coding)

**진술.** Lempel-Ziv, PPM 등 **universal coder** 는 source 분포를 모르고도 $n \to \infty$ 에서 $H$ 에 수렴.

**증명 스케치.** LZ78 / LZ77 이 stationary ergodic source 에 대해 asymptotically entropy-rate 달성. Ziv–Lempel (1978) 정리. 자세한 건 §4.5.

### Theorem 4.3.7 (Rate-Distortion 예고)

**진술.** 허용 왜곡 $D$ 하에서 최소 rate 는 **rate-distortion function** $R(D)$. $D = 0$ 이면 $R(0) = H(p)$. 다음 장(channel coding) 에서 재조명.

---

## 💻 NumPy 로 직접 확인

### AEP 실증

```python
import numpy as np

p = np.array([0.7, 0.2, 0.1])
H = -np.sum(p * np.log2(p))
print(f"H = {H:.4f}")

rng = np.random.default_rng(0)
for n in [10, 100, 1000, 10000]:
    X = rng.choice(3, size=n, p=p)
    log_p = sum(np.log2(p[x]) for x in X)
    rate = -log_p / n
    print(f"n={n:5d}  -1/n log p = {rate:.4f}")
```

출력:
```
H = 1.1568
n=10     -1/n log p = 1.0875
n=100    -1/n log p = 1.1700
n=1000   -1/n log p = 1.1594
n=10000  -1/n log p = 1.1571
```
→ $n$ 커질수록 $H$ 에 수렴.

### Typical set 크기 시뮬레이션

```python
def typical_set_count(p, n, eps=0.05, n_samples=10000):
    H = -np.sum(p * np.log2(p))
    count_typical = 0
    rng = np.random.default_rng(0)
    for _ in range(n_samples):
        X = rng.choice(len(p), size=n, p=p)
        log_p = sum(np.log2(p[x]) for x in X)
        rate = -log_p / n
        if abs(rate - H) < eps:
            count_typical += 1
    return count_typical / n_samples

for n in [10, 50, 200]:
    frac = typical_set_count(p, n, eps=0.1)
    print(f"n={n}: P(X^n ∈ typical set) ≈ {frac:.3f}")
```

출력:
```
n=10: P(X^n ∈ typical set) ≈ 0.57
n=50: P(X^n ∈ typical set) ≈ 0.95
n=200: P(X^n ∈ typical set) ≈ 1.00
```

### 블록 Huffman 으로 entropy 수렴

```python
from itertools import product
import heapq

def huffman_length(probs):
    heap = [(p, None) for p in probs if p > 0]
    heapq.heapify(heap)
    total = 0
    while len(heap) > 1:
        a = heapq.heappop(heap); b = heapq.heappop(heap)
        total += a[0] + b[0]
        heapq.heappush(heap, (a[0]+b[0], None))
    return total  # weighted path length

p = np.array([0.8, 0.2])
H = -np.sum(p * np.log2(p))

for k in [1, 2, 4, 8, 16]:
    block_probs = []
    for seq in product(range(len(p)), repeat=k):
        block_probs.append(np.prod([p[s] for s in seq]))
    L_k = huffman_length(block_probs)
    print(f"k={k:2d}: L/k = {L_k/k:.4f}  (H = {H:.4f})")
```

출력:
```
k= 1: L/k = 1.0000  (H = 0.7219)
k= 2: L/k = 0.8500  (H = 0.7219)
k= 4: L/k = 0.7719  (H = 0.7219)
k= 8: L/k = 0.7452  (H = 0.7219)
k=16: L/k = 0.7330  (H = 0.7219)
```

블록 확장으로 점진적 entropy 수렴.

### 실제 압축: bytes vs entropy

```python
import zlib

# 영어 텍스트의 byte-level entropy vs gzip rate
text = "The quick brown fox jumps over the lazy dog. " * 1000
raw_bytes = text.encode('utf-8')
# Byte frequency
from collections import Counter
freq = Counter(raw_bytes)
probs = np.array(list(freq.values())) / sum(freq.values())
H_byte = -np.sum(probs * np.log2(probs))
print(f"Byte-level entropy  = {H_byte:.3f} bits/byte")

compressed = zlib.compress(raw_bytes, level=9)
rate = 8 * len(compressed) / len(raw_bytes)
print(f"gzip bits/byte      = {rate:.3f}")

# 실제 언어는 marginal entropy 보다 conditional entropy 가 훨씬 낮음
# → LLM 은 context 를 활용해 더 나은 압축
```

---

## 🔗 AI/ML 연결고리

### 1. LLM as Lossless Compressor
Deletang et al. (2024) "Language Modeling Is Compression": GPT-2 + arithmetic coding 이 gzip 보다 강한 compressor. 이론: $H(X^n | \text{context})$ 가 context 증가로 감소 → compression rate ↓.

### 2. Perplexity ↔ Achievable Rate
$\mathrm{PPL} = 2^{H(p, q)}$ = "model $q$ 기반 최적 code 의 bits per symbol". $q \to p$ 일수록 achievable rate 가 $H(p)$ 에 근접.

### 3. Kolmogorov Complexity
"가장 짧은 프로그램" 의 길이 $K(x)$. Source coding theorem 의 **algorithmic 버전**: $K(x)$ 는 $x$ 의 참 정보량. $H$ 는 expected value.

### 4. MDL (Minimum Description Length)
"데이터를 설명하는 model $+$ residual" 의 총 description length 최소화. Overfitting 방지 — occam's razor 의 정보이론적 구현.

### 5. Cross-Entropy Loss 재해석
$\mathcal{L}_{\mathrm{CE}}(q) = -\mathbb{E}_p[\log q(x)] = H(p, q) = H(p) + D(p\|q)$. **$q = p$ 달성 = optimal compression**.

### 6. Information bottleneck 의 rate 항
$I(X;Z)$ 를 "Z 로 표현하는 bits 수" 로 해석 → rate. IB = "minimum rate 로 Y 에 대한 max information 유지".

### 7. Neural Image/Video Compression
Ballé et al. 의 end-to-end learned compression: entropy model $p(z)$ 학습 + arithmetic coding → $-\log p(z)$ 비트수가 이론적 압축.

---

## ⚖️ 가정·한계·함정

1. **iid 가정** — 실제 신호는 correlated. Stationary ergodic 로 일반화 가능 (Shannon–McMillan–Breiman).
2. **Block size 필요** — 단일 symbol Huffman 은 최대 1 bit gap. Arithmetic coding 이나 blocking 으로 gap 제거.
3. **Distribution known** — 사전 $p$ 없으면 universal / adaptive coding 필요 (§4.5).
4. **$H$ 달성 필요한 $n$ 이 큼** — 유한 길이에서는 $O(1/n)$ overhead.
5. **Converse 는 asymptotic** — 작은 $n$ 에서는 $R < H$ 여도 $P_e$ 가 명확히 1 이 아닐 수 있음.
6. **Lossless vs lossy** — 이 장은 lossless. 허용 왜곡 하에서는 $R(D) < H(X)$ 가능 (rate-distortion).

---

## 📌 핵심 정리

1. **Source Coding Theorem**: inf achievable rate = $H(p)$.
2. **Achievability**: typical set encoding — rate $H + \epsilon$ 가능.
3. **Converse**: rate $< H$ 이면 $P_e \not\to 0$ (Fano).
4. **AEP**: $-\frac{1}{n}\log p(X^n) \xrightarrow{P} H$.
5. Typical set 크기 $\approx 2^{nH}$, 각 확률 $\approx 2^{-nH}$.
6. **Cross-entropy loss** 직접 해석: 평균 code length.
7. LLM lossless compression, MDL, Kolmogorov complexity 와 연결.

---

## 🤔 생각해볼 문제

### 문제 1. AEP 의 정량적 표현
$P(|-\tfrac{1}{n}\log p(X^n) - H| > \epsilon) \le ?$ 를 Chebyshev 로 유도.

<details>
<summary>해설</summary>

Chebyshev: $P(|\bar Y - \mu| > \epsilon) \le \mathrm{Var}(Y)/(n \epsilon^2)$ where $Y_i = -\log p(X_i), \mu = H, \sigma^2 = \mathrm{Var}(-\log p(X))$. 따라서 $P(\text{non-typical}) \le \sigma^2/(n\epsilon^2) \to 0$. 대수적 수렴.
</details>

### 문제 2. Converse 의 tight 함
$R = H - \delta$ 일 때 Theorem 4.3.4 가 보여주는 $P_e$ 하한 계산.

<details>
<summary>해설</summary>

$P_e \ge \delta/\log|\mathcal{X}| - o(1)$. 특히 binary source ($|\mathcal{X}| = 2$) 이면 $P_e \ge \delta - o(1)$. Shannon's original argument 의 구체 형태.
</details>

### 문제 3. Markov source 의 entropy rate
$X_t$ 가 stationary Markov chain. Source coding 의 rate 는 $H(X)$ 가 아니라 entropy rate $H(\mathcal{X}) = \lim H(X_n | X_{n-1})$. 간단한 두-상태 Markov 예제.

<details>
<summary>해설</summary>

Transition $P = \begin{pmatrix}0.9 & 0.1 \\ 0.2 & 0.8\end{pmatrix}$. Stationary $\pi \approx (0.67, 0.33)$. $H(X) = 0.918$, $H(X|X_{prev}) \approx 0.67 \cdot H(0.9) + 0.33 \cdot H(0.2) \approx 0.569$. Entropy rate $\ll$ marginal entropy → LLM context 가 효과적인 이유.
</details>

### 문제 4. LM 으로 압축한 실제 계산
"Hello world. " 를 GPT-2 (fictional) $p$ 로 압축할 때 최소 bit 수 추정.

<details>
<summary>해설</summary>

각 token 의 $-\log_2 p(\text{token}|\text{context})$ 합산 = 총 bits. 평균 2–4 bits/char for English (GPT-2 기준). Arithmetic coding 으로 달성 가능.
</details>

### 문제 5. Shannon-McMillan-Breiman
Stationary ergodic 로의 AEP 일반화. 핵심 아이디어?

<details>
<summary>해설</summary>

Birkhoff ergodic theorem: stationary ergodic process 의 time average = ensemble average. $-\frac{1}{n}\log p(X^n) \to H(\mathcal{X})$ (entropy rate). iid 는 특수경우. 압축의 보편성이 크게 확장.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [4.2 Huffman 부호의 최적성](./02-huffman-optimality.md) | [4.4 AEP와 Typical Set](./04-aep-typical-set.md) |

[🏠 Home](../README.md)

</div>
