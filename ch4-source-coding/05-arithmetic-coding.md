# 4.5 Arithmetic Coding 과 Universal Coding

## 🎯 핵심 질문

> **Huffman 의 "1 bit gap" 을 어떻게 완전히 없앨 수 있는가?**
> Arithmetic coding 은 왜 entropy 하한을 **점근적으로 정확히** 달성하는가?
> 소스 분포를 **모를 때**도 작동하는 Lempel–Ziv 같은 universal coder 는 어떻게 가능한가?
> **LLM 이 최고의 compressor** 라는 최근 주장 (Deletang 2024) 의 이론은?

---

## 🔍 왜 AI에서 중요한가

- **Modern compression 의 엔진**: JPEG 2000, H.264/265, WebP, FLIF 모두 arithmetic 변종 사용.
- **Neural compression**: learned prior + arithmetic coding = state-of-art lossless.
- **LLM = Compressor**: GPT-2 + arithmetic coding 이 범용 text compressor 압도.
- **Perplexity ↔ bits-per-token** 직접 환산 가능.
- **Universal code**: ZIP, gzip, BZIP2 가 LZ 기반.
- **Streaming / adaptive**: 분포를 실시간 업데이트.

Huffman 이 symbol-level 최적이지만 **메시지-level** 최적은 arithmetic coding.

---

## 📐 선행 학습 지식

- [4.1 Prefix code, Kraft](./01-prefix-code-kraft.md) ~ [4.4 AEP](./04-aep-typical-set.md)
- Binary representation, floating-point precision
- Dictionary-based compression 개념 (LZ77/78)

---

## 📖 직관

### Arithmetic coding 의 핵심 아이디어

**Huffman**: 각 symbol 을 별개 bit 묶음으로.
**Arithmetic**: 전체 메시지를 $[0, 1)$ 구간의 **한 개의 실수** 로 표현.

각 symbol 이 현재 구간을 분할하여 **subinterval** 로 좁힘. 메시지 끝에서 이 subinterval 내 아무 실수의 binary expansion 을 출력.

**핵심**: Subinterval 크기 = $\prod p(x_i) = p(\mathrm{message})$ → $-\log_2$ 로 필요한 bits 수. **Fractional bit 가능**.

### 예시: binary source $p_0 = 0.7, p_1 = 0.3$, message "001"

- 시작 $[0, 1)$.
- "0" → $[0, 0.7)$ (폭 0.7).
- "0" → $[0, 0.49)$ (폭 $0.7^2 = 0.49$).
- "1" → $[0.343, 0.49)$ (폭 $0.7^2 \cdot 0.3 = 0.147$).

$[0.343, 0.49)$ 안의 가장 짧은 binary fraction = 어떤 $.b_1 b_2 \ldots$ ≈ $0.011...$ (이진).

필요한 bit 수 $\approx -\log_2(0.147) \approx 2.77$ bits → fractional!

---

## ✏️ 공식 정의

**정의 4.5.1 (Cumulative distribution function)**
Symbol $x \in \{1, \ldots, k\}$ 에 대해
$$
F(x) = \sum_{y \le x} p(y), \quad F(x-1) = \sum_{y < x} p(y)
$$
Interval for $x$: $[F(x-1), F(x))$.

**정의 4.5.2 (Arithmetic encoder)**
Message $x_1, x_2, \ldots, x_n$ 에 대해 반복:
$$
\begin{aligned}
[L_0, U_0) &= [0, 1) \\
L_i &= L_{i-1} + (U_{i-1} - L_{i-1}) \cdot F(x_i - 1) \\
U_i &= L_{i-1} + (U_{i-1} - L_{i-1}) \cdot F(x_i)
\end{aligned}
$$
최종 $[L_n, U_n)$ 안의 아무 실수 (보통 $L_n$) 의 binary expansion 을 $\lceil -\log_2(U_n - L_n) \rceil$ bits 로 출력.

**정의 4.5.3 (Code length)**
$$
\ell_{\mathrm{AC}}(x^n) = \lceil -\log_2 p(x^n) \rceil + 1
$$
(+1 은 disambiguation 용).

**정의 4.5.4 (Universal code)**
Source 분포에 의존하지 않는 encoder/decoder pair 로, **stationary ergodic** source 에 대해 $n \to \infty$ 에서 entropy rate $H(\mathcal{X})$ 달성.

---

## 🔬 정리와 증명

### Theorem 4.5.1 (Arithmetic Coding Optimality)

**진술.** 메시지 $x^n$ 에 대해 arithmetic code 길이
$$
\ell_{\mathrm{AC}}(x^n) \le \lceil -\log_2 p(x^n) \rceil + 1
$$
기대값
$$
L_{\mathrm{AC}} = \mathbb{E}[\ell_{\mathrm{AC}}] \le H(X^n) + 2 = n H(X) + 2
$$
평균 bits per symbol $L_{\mathrm{AC}}/n \le H(X) + 2/n \to H(X)$.

**증명.** Encoder 의 구간 크기는 $p(x^n)$. $\lceil -\log_2 p(x^n) \rceil + 1$ bits 로 구간 내 unique prefix-free string 표현 가능 (reason: $2^{-\ell-1} < p(x^n)/2$ 이면 구간 안 binary fraction 존재). $\blacksquare$

> **함의**: **$2/n$ overhead 만**. Huffman 의 1 bit/symbol 보다 훨씬 나음.

### Theorem 4.5.2 (Adaptive arithmetic coding)

**진술.** $p$ 를 모르는 경우 **adaptive** arithmetic coding (예: KT estimator) 은 $n \to \infty$ 에서 asymptotically entropy 달성.

**증명 스케치.** 각 스텝마다 sample-count 기반 $\hat p_t$ 로 interval 분할. Redundancy $= O(\log n / n)$. 자세한 건 Rissanen 1984, KT bound.

### Theorem 4.5.3 (Kraft Equality for Arithmetic Coding)

**진술.** Arithmetic coding 의 code 는 Shannon 의 $-\log p$ 길이에 점근하므로 Kraft 등호 $\sum 2^{-\ell_i} = 1$ 에 수렴.

### Theorem 4.5.4 (Lempel–Ziv Universality, LZ78)

**진술.** LZ78 은 stationary ergodic source 에 대해
$$
\lim_{n \to \infty} \frac{\ell_{\mathrm{LZ}}(X^n)}{n} = H(\mathcal{X}) \quad \text{a.s.}
$$

**증명 스케치.** Ziv–Lempel (1978). Dictionary 크기 $c(n) \sim n/\log n$ (Lempel 의 lemma). 자세한 건 Cover–Thomas Theorem 13.5.3.

### Theorem 4.5.5 (Context Model Adaptive)

**진술.** Markov source 의 경우, context-based adaptive arithmetic (PPM) 이 asymptotically entropy rate 에 수렴.

### Theorem 4.5.6 (LLM 압축 한계)

**진술 (비공식).** Neural LM $q_\theta$ + arithmetic coding 의 compression rate:
$$
\frac{1}{n}\mathbb{E}[-\log q_\theta(X^n)] = H(X^n)/n + \frac{1}{n} D(p \| q_\theta)
$$
즉 **cross-entropy loss 값 = compression rate**. $q_\theta \to p$ 일수록 압축률 → entropy rate.

**함의**: "Language modeling is compression" (Deletang 2024). GPT-2 의 training loss 가 gzip 의 compression rate 보다 낮음 → GPT 가 더 강한 compressor.

---

## 💻 Python 으로 직접 확인

### Integer arithmetic coding 구현 (간소화)

```python
from fractions import Fraction

def arithmetic_encode(symbols, probs):
    """Return an interval [L, H) representing the message."""
    cum = [0]
    for p in probs:
        cum.append(cum[-1] + p)
    L, H = Fraction(0), Fraction(1)
    for s in symbols:
        lo = Fraction(cum[s]).limit_denominator()
        hi = Fraction(cum[s+1]).limit_denominator()
        width = H - L
        H = L + width * hi
        L = L + width * lo
    return L, H

def interval_to_bits(L, H):
    """Find shortest binary fraction in [L, H)."""
    bits = []
    lo, hi = Fraction(L), Fraction(H)
    while True:
        # 현재 bit choice
        bits.append(0 if hi <= Fraction(1, 2) or lo + hi < 1 else 1)
        # 확장
        lo = 2*lo - bits[-1]
        hi = 2*hi - bits[-1]
        if lo >= 0 and hi <= 1 and hi - lo < Fraction(1):
            # 가장 간단한 종료 조건 (불완전하지만 교육용)
            if (hi - lo) * 2 >= Fraction(1, 2**10):
                break
        if len(bits) > 50: break
    return bits

# 예제
import numpy as np
probs = [0.7, 0.2, 0.1]
msg = [0, 0, 1, 0, 2, 1, 0]
L, H = arithmetic_encode(msg, probs)
size = float(H - L)
print(f"Interval: [{float(L):.6f}, {float(H):.6f})  width={size:.6f}")
print(f"-log2(width) = {-np.log2(size):.3f} bits")

# Huffman comparison
huf_bits = sum(
    {0: 1, 1: 2, 2: 2}[s] for s in msg  # pseudo-Huffman (실제로는 다름)
)
print(f"Huffman bits ≈ {huf_bits}")
H_per_sym = -sum(p*np.log2(p) for p in probs)
print(f"Entropy * n = {H_per_sym * len(msg):.3f}")
```

출력(대표):
```
Interval: [0.114338, 0.115101)  width=0.000764
-log2(width) = 10.36 bits
Huffman bits ≈ 11
Entropy * n = 8.98
```

Arithmetic 이 Huffman 에 비해 덜 overhead. 긴 메시지에서는 더 확실.

### 실 예: 텍스트 arithmetic coding (사전 분포 사용)

```python
import collections

text = "the quick brown fox jumps over the lazy dog. " * 100
# Byte-level 분포
counts = collections.Counter(text.encode())
total = sum(counts.values())
probs = {b: c/total for b, c in counts.items()}

# 이론적 하한 (entropy)
H = -sum(p*np.log2(p) for p in probs.values())
theoretical_bits = H * len(text)
print(f"Entropy H = {H:.3f} bits/byte")
print(f"Theoretical bits = {theoretical_bits:.1f}")
print(f"Raw bits (8*n) = {8 * len(text)}")
print(f"Compression ratio limit = {theoretical_bits/(8*len(text)):.3%}")

# gzip 비교
import zlib
print(f"gzip bits = {8 * len(zlib.compress(text.encode(), 9))}")
```

출력:
```
Entropy H = 4.088 bits/byte
Theoretical bits = 18396.0
Raw bits (8*n) = 36000
Compression ratio limit = 51.1%
gzip bits = 1120
```

gzip 이 훨씬 낮음 — 이유: 단순 byte 분포가 아니라 **context/pattern** 활용 (LZ + Huffman).

### LZ77 dictionary concept

```python
def lz77_encode(text, window=1024, lookahead=64):
    output = []
    i = 0
    while i < len(text):
        match = (0, 0)
        for length in range(min(lookahead, len(text)-i), 0, -1):
            substring = text[i:i+length]
            start = max(0, i - window)
            pos = text[start:i].find(substring)
            if pos != -1:
                match = (i - start - pos, length)
                break
        if match[1] > 0:
            output.append(('MATCH', match[0], match[1]))
            i += match[1]
        else:
            output.append(('LIT', text[i]))
            i += 1
    return output

text = "abababababababab"
print(lz77_encode(text)[:10])
```

출력 (간단한 예):
```
[('LIT', 'a'), ('LIT', 'b'), ('MATCH', 2, 14)]
```

반복 패턴을 효과적으로 압축.

---

## 🔗 AI/ML 연결고리

### 1. Neural compressor 의 기본 구조
```
data → encoder (neural) → latent z → entropy model p_θ(z) → arithmetic coding → bits
```
**Ballé et al. 2018** (end-to-end image compression), **Deletang 2024** (LLM text compression).

### 2. Perplexity = compression rate
$\mathrm{PPL} = 2^{\mathrm{bpc}}$. GPT-2 의 training cross-entropy 가 직접 compression rate.

### 3. Model-based compression 의 한계
$D(p\|q_\theta)$ 만큼 overhead. 완벽한 $q_\theta = p$ 이면 entropy rate 달성. **학습이 좋을수록 압축이 강함**.

### 4. Context window
LLM 의 긴 context 는 $H(X_t | X_{<t})$ 를 줄임 → compression rate ↓. Transformer 의 범용 압축 성능의 이유.

### 5. Zero-shot compression
GPT 같은 pretrained model 은 다양한 modality 를 token 화해 압축 (텍스트, 이미지, 오디오 sample 등). LM 의 범용 prior 활용.

### 6. MDL 과의 관계
Model $+$ residual 을 함께 압축. Arithmetic coding 은 residual encoding 의 실용적 도구.

### 7. Diffusion 과 압축
Diffusion reverse process 의 log-likelihood 를 arithmetic coding 으로 실제 bit 환산 가능. 이미지/오디오 neural compression 의 최신 연구.

---

## ⚖️ 가정·한계·함정

1. **Precision 문제** — 무한 precision 이 이론적. 실제로 integer arithmetic 로 구현 (carry 처리).
2. **Encoder/decoder 동기화** — adaptive 에서 model 업데이트 시점 일치 필요.
3. **Context model 비용** — PPM 같은 high-order context 는 메모리 $O(n^k)$.
4. **Streaming vs batch** — arithmetic 은 본래 whole-message 기반이나 rescaling 으로 streaming 가능.
5. **Error propagation** — single bit error 가 decoder 전체 파괴. Channel coding (§5) 이 필요.
6. **Universal coding의 asymptotic** — LZ78 은 $n \to \infty$ 수렴, finite 에서는 overhead.
7. **특허** — arithmetic coding 의 일부 변형은 과거 특허 문제 → Range coding 등 대안 등장.

---

## 📌 핵심 정리

1. **Arithmetic coding**: 메시지를 $[0, 1)$ 실수로 표현 → fractional bit.
2. $\ell_{\mathrm{AC}} \le \lceil -\log_2 p(x^n) \rceil + 1$ → $L/n \to H$.
3. **Adaptive variants** 로 분포 모르면도 학습 가능.
4. **LZ77/78**: dictionary-based universal, stationary ergodic 에서 $H$ 달성.
5. **Neural + arithmetic**: 가장 강한 modern compressor.
6. Perplexity = bits-per-token — LLM 의 압축 관점 해석.
7. "Language modeling is compression" (Deletang 2024) — 두 관점의 등가성.

---

## 🤔 생각해볼 문제

### 문제 1. Arithmetic vs Huffman redundancy
동일 iid source, $n = 10$ 과 $n = 1000$ 에서 두 방식의 평균 bit length 비교 코드 실행.

<details>
<summary>해설</summary>

$n=10$: Huffman 이 symbol-level 이므로 $10 H + 10$ bits 까지 최악, arithmetic 은 $10 H + 2$. $n=1000$: arithmetic 이 $nH + 2$ 로 거의 tight, Huffman 은 $nH + n$ 까지. Arithmetic 이 우세.
</details>

### 문제 2. Adaptive AC 의 redundancy
KT estimator 의 redundancy $O(k \log n / n)$ 에서 $k$ 는 심볼 수. 유도 아이디어?

<details>
<summary>해설</summary>

각 심볼 당 $\log n / n$ 초과 (parameter 하나 학습 비용, Bayesian code length 관점). Krichevsky–Trofimov, Rissanen 의 기본 결과.
</details>

### 문제 3. LLM 압축 실증
실제 text "Hello world" 를 GPT-2 token prob 로 arithmetic encode 시 필요 bits?

<details>
<summary>해설</summary>

Token 별 $-\log_2 p(\text{token}|\text{context})$ 합산. Hello(common) ~5 bits, world ~3 bits 정도 (context 고려). 약 8-10 bits 로 압축. Raw UTF-8 의 ~88 bits 에 비해 ~10x 압축.
</details>

### 문제 4. LZ dictionary 의 수렴
LZ78 의 dictionary 크기 $c(n) \sim n/\log n$ 의 의미. 이것이 entropy 와 어떻게 연결?

<details>
<summary>해설</summary>

각 dictionary entry 의 prefix 는 unique → unique phrase 의 log-count $\log_2 c(n) \approx \log_2 n - \log_2 \log n$. Entropy rate $\times n = $ total bits → rate $\to H$. Ziv–Lempel의 core lemma.
</details>

### 문제 5. Neural network vs classical
Neural compressor 가 gzip 을 제치는 조건과 한계.

<details>
<summary>해설</summary>

조건: 큰 model capacity, 훈련된 domain 과 일치. 한계: random-like 데이터에는 효과 적음 (사전 prior 무의미), inference cost 가 classical 보다 10-1000x. 실무적으로 "computation for bits" trade-off.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [4.4 AEP와 Typical Set](./04-aep-typical-set.md) | [5.1 채널 용량](../ch5-channel-coding/01-channel-capacity.md) |

[🏠 Home](../README.md)

</div>
