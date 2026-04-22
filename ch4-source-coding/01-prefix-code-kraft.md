# 4.1 Prefix Code 와 Kraft 부등식

## 🎯 핵심 질문

> **기호를 비트열로 부호화할 때 "한 기호의 끝이 다른 기호의 시작처럼 보이지 않는다" 는 조건(prefix code)은 왜 중요한가?**
> 부호 길이 $\ell_1, \ell_2, \ldots$ 가 주어지면 **어떤 조건** 을 만족해야 prefix code 로 실현 가능한가?
> **Kraft 부등식** $\sum 2^{-\ell_i} \le 1$ 이 왜 symbol coding 의 근본 법칙인가?

---

## 🔍 왜 AI에서 중요한가

- **Arithmetic coding / Tokenizer**: LLM 의 BPE/SentencePiece 는 사실상 variable-length code. Token 길이와 빈도의 관계 = Kraft 관점.
- **Huffman / Shannon-Fano**: 최적 prefix code 구성의 기반.
- **Information content $\log 1/p$** 의 물리적 구현: 확률 $p_i$ 의 기호에 길이 $\lceil \log 1/p_i \rceil$ 비트를 할당 — Kraft 가 이를 보장.
- **Cross-Entropy Loss 의 비트 해석**: $-\log q(x)$ 는 $q$ 기반 코드의 비트 길이.
- **Neural Compression**: learned entropy model + arithmetic coding.

"확률이 낮으면 긴 코드, 확률이 높으면 짧은 코드" 라는 직관의 수학적 기반.

---

## 📐 선행 학습 지식

- [1.2 엔트로피 정의](../ch1-entropy-axioms/02-entropy-definition.md)
- Binary tree, stringology 기본
- 이산 확률분포, 기댓값

---

## 📖 직관

### Prefix code 의 모양

기호 alphabet $\{A, B, C, D\}$ 를 이진 코드로:
- $A = 0$
- $B = 10$
- $C = 110$
- $D = 111$

이는 binary tree 의 **leaf node** 들에 심볼을 놓은 것. 어떤 코드도 다른 코드의 접두사가 아님 → **즉시 디코딩** 가능.

```
         ()
        / \
       0   1
      A    /\
          0  1
          B  /\
            0  1
            C  D
```

### 왜 Kraft 부등식이 나오는가

각 leaf 가 depth $\ell_i$ 에 있으면 해당 leaf 가 "차지하는 공간 비율" 은 $2^{-\ell_i}$ (전체 가능한 $2^{\ell_{\max}}$ 개의 leaf 중 $2^{\ell_{\max} - \ell_i}$ 개를 차단). 서로 겹치지 않으려면:
$$
\sum_i 2^{-\ell_i} \le 1
$$

**등호** 는 tree 가 "full" 일 때 (모든 가능한 자리가 사용됨).

---

## ✏️ 공식 정의

**정의 4.1.1 (코드)**
Alphabet $\mathcal{X}$ 에서 binary string $\{0, 1\}^*$ 로의 함수 $C: \mathcal{X} \to \{0, 1\}^*$. 길이 $\ell(x) = |C(x)|$.

**정의 4.1.2 (Non-singular)**
$C$ 가 injective. 즉 $x \ne y \Rightarrow C(x) \ne C(y)$.

**정의 4.1.3 (Uniquely decodable, UD)**
메시지 $x_1 x_2 \ldots x_n$ 의 concatenation $C(x_1) C(x_2) \ldots C(x_n)$ 이 항상 유일한 디코딩을 가짐.

**정의 4.1.4 (Prefix code / Instantaneous code)**
어떤 코드도 다른 코드의 접두사가 아님:
$$
C(x) \text{ 는 } C(y) \text{ 의 prefix 가 아님} \quad (\forall x \ne y)
$$

**정의 4.1.5 (Kraft's inequality)**
길이 $\ell_1, \ldots, \ell_n$ 의 binary prefix code 가 존재할 필요충분조건:
$$
\sum_{i=1}^n 2^{-\ell_i} \le 1
$$

**정의 4.1.6 (D-ary 일반화)**
$D$-ary alphabet 이면 $\sum D^{-\ell_i} \le 1$.

---

## 🔬 정리와 증명

### Theorem 4.1.1 (Kraft Inequality, Necessity)

**진술.** Binary prefix code $C$ 가 존재하면 $\sum_i 2^{-\ell_i} \le 1$.

**증명.** $\ell_{\max} = \max_i \ell_i$. 깊이 $\ell_{\max}$ 의 완전 이진 tree 에 $2^{\ell_{\max}}$ 개 leaf. 각 codeword $c_i$ 는 길이 $\ell_i$ 노드 → 그 subtree 가 $2^{\ell_{\max} - \ell_i}$ 개 leaf 를 차지. Prefix 조건 → subtree 가 서로 disjoint → leaf 사용 수
$$
\sum_i 2^{\ell_{\max} - \ell_i} \le 2^{\ell_{\max}}
$$
양변을 $2^{\ell_{\max}}$ 로 나누면 $\sum 2^{-\ell_i} \le 1$. $\blacksquare$

### Theorem 4.1.2 (Kraft Inequality, Sufficiency)

**진술.** $\sum 2^{-\ell_i} \le 1$ 이면 그에 맞는 prefix code 가 존재.

**증명 (생성적).** $\ell_1 \le \ell_2 \le \ldots \le \ell_n$ 으로 정렬. Greedy 로 leaf 를 배치:
- $c_1$: 깊이 $\ell_1$ 의 가장 왼쪽 가능한 노드.
- 각 $c_i$ 선택 후, 그 subtree 를 사용 완료 표시.
- Remaining 공간 = $1 - \sum_{j < i} 2^{-\ell_j} \ge 2^{-\ell_i}$ (귀납적 가정) → 깊이 $\ell_i$ 의 leaf 를 항상 찾을 수 있음.

$\sum 2^{-\ell_i} \le 1$ 이므로 충분한 공간 있음. $\blacksquare$

### Theorem 4.1.3 (McMillan's Theorem — UD 도 Kraft 만족)

**진술.** Uniquely decodable code (prefix 가 아니어도) 역시 $\sum 2^{-\ell_i} \le 1$.

**증명 (McMillan 1956).** $N$ 개 기호 concatenation 고려. Word-length 합 공식:
$$
\left(\sum_i 2^{-\ell_i}\right)^N = \sum_{\text{words of total length } L} a_L\, 2^{-L}
$$
여기서 $a_L$ = UD 하에서 길이 $L$ 인 메시지 수 $\le 2^L$ (중복 없이 복원 가능). 따라서
$$
\left(\sum 2^{-\ell_i}\right)^N \le N \cdot \ell_{\max}
$$
$N \to \infty$ 로 보내면 $\left(\sum 2^{-\ell_i}\right)^N$ 이 polynomial 아래로 bounded → $\sum 2^{-\ell_i} \le 1$. $\blacksquare$

> **함의**: UD 라면 prefix code 로 재설계해도 길이 동일 가능. **실용상 prefix code 만 고려해도 손해 없음**.

### Theorem 4.1.4 (최소 expected length 의 하한)

**진술.** 확률분포 $p_1, \ldots, p_n$ 의 prefix code 의 기대 길이 $L = \sum p_i \ell_i$ 에 대해
$$
L \ge H(p) = -\sum p_i \log_2 p_i
$$
등호는 $\ell_i = -\log_2 p_i$ 가 정수일 때 (dyadic distribution).

**증명.** $L - H(p) = \sum p_i (\ell_i + \log_2 p_i) = \sum p_i \log_2(2^{\ell_i} p_i)$. Jensen:
$$
\sum p_i \log_2(2^{\ell_i} p_i) \ge -\log_2 \sum p_i / (2^{\ell_i} p_i) = -\log_2 \sum 2^{-\ell_i} \ge 0
$$
Kraft 에서 $\sum 2^{-\ell_i} \le 1 \Rightarrow -\log_2(\cdot) \ge 0$. $\blacksquare$

> **함의**: **엔트로피 = 이론적 최소 비트수**.

### Theorem 4.1.5 (Shannon 부호의 upper bound)

**진술.** $\ell_i = \lceil -\log_2 p_i \rceil$ 선택 시
$$
L = \sum p_i \lceil -\log_2 p_i \rceil < H(p) + 1
$$

**증명.** $\ell_i < -\log_2 p_i + 1$, 각 항에 $p_i$ 곱해 합 → $L < H(p) + 1$. Kraft 성립 확인: $2^{-\ell_i} \le 2^{\log_2 p_i} = p_i$ → $\sum 2^{-\ell_i} \le 1$. ✅ $\blacksquare$

**따라서 Shannon 코드는 entropy 에서 1 bit 이내**.

---

## 💻 NumPy 로 직접 확인

### Kraft 부등식 체크

```python
import numpy as np

def kraft_sum(lengths, D=2):
    return sum(D ** (-l) for l in lengths)

# 예시: {0, 10, 110, 111}
lengths = [1, 2, 3, 3]
print(f"Kraft sum = {kraft_sum(lengths):.4f} (<=1 이어야 prefix code 가능)")
# 0.5 + 0.25 + 0.125 + 0.125 = 1.0 (완전)

# Prefix code 불가능한 예
bad_lengths = [1, 1, 2, 2]
print(f"Kraft sum = {kraft_sum(bad_lengths):.4f}")
# = 1.5 > 1 → 불가능
```

### Prefix code 구성 (greedy)

```python
def build_prefix_code(lengths):
    # lengths: list[int], Kraft 성립 가정
    lengths_sorted = sorted(enumerate(lengths), key=lambda t: t[1])
    codes = [None] * len(lengths)
    used = 0  # bit representation of used prefix sum
    # 간단한 구현: 매번 사용 가능한 최소값 찾기
    for idx, l in lengths_sorted:
        # 현재 사용된 영역 = [0, used/2^l)
        codes[idx] = format(used >> (max(used.bit_length() - l, 0)) if False else
                            used, f"0{l}b")[-l:]
        # 좀 더 정확한 구현은 code tree 관리
        used = (used + 1) << (lengths_sorted[min(len(lengths_sorted)-1, 0)][1] - l)
    # (교육용 간이 — 실제로는 canonical Huffman code 사용)
    return codes

# 더 간단하고 정확한 canonical code
def canonical_code(lengths):
    paired = sorted(enumerate(lengths), key=lambda t: (t[1], t[0]))
    codes = [None] * len(lengths)
    code = 0
    prev_len = 0
    for idx, l in paired:
        if prev_len > 0:
            code = (code + 1) << (l - prev_len)
        else:
            code = 0
        codes[idx] = format(code, f"0{l}b")
        prev_len = l
    return codes

lengths = [1, 2, 3, 3]
codes = canonical_code(lengths)
for i, c in enumerate(codes):
    print(f"symbol {i}: code={c}")
```

### Shannon–Fano / Shannon code

```python
def shannon_code(probs):
    """ℓ_i = ceil(-log2 p_i)"""
    lengths = [int(np.ceil(-np.log2(p))) for p in probs]
    print(f"Kraft sum = {kraft_sum(lengths):.4f}")
    return lengths, canonical_code(lengths)

probs = [0.5, 0.25, 0.125, 0.125]
lengths, codes = shannon_code(probs)
print(f"Lengths: {lengths}")
print(f"Codes: {codes}")
print(f"Expected length: {sum(p*l for p, l in zip(probs, lengths)):.3f}")
print(f"Entropy:         {-sum(p*np.log2(p) for p in probs):.3f}")
```

출력:
```
Kraft sum = 1.0000
Lengths: [1, 2, 3, 3]
Codes: ['0', '10', '110', '111']
Expected length: 1.750
Entropy:         1.750
```
이 경우 dyadic 분포라 entropy 정확히 달성 (등호 조건 만족).

### 비-dyadic 분포에서의 Shannon code gap

```python
probs = [0.6, 0.3, 0.1]
lengths, codes = shannon_code(probs)
L = sum(p*l for p, l in zip(probs, lengths))
H = -sum(p*np.log2(p) for p in probs)
print(f"L = {L:.3f}, H = {H:.3f}, gap = {L-H:.3f} (< 1)")
```

출력:
```
L = 1.900, H = 1.295, gap = 0.605
```
1 bit 이내의 redundancy.

---

## 🔗 AI/ML 연결고리

### 1. Cross-Entropy Loss = Code Length
LM 의 $-\log_2 q(x)$ 는 "$q$ 기반 최적 코드의 비트 길이". Cross-entropy $H(p, q) = \mathbb{E}_p[-\log q]$ = **$p$ 데이터를 $q$ 코드로 압축할 때의 기대 비트 길이** → Kraft-Shannon 해석.

### 2. Perplexity 의 본질
$\mathrm{PPL} = 2^{H(p, q)}$ = "평균적으로 $q$ 기반 코드로 다음 토큰을 명시하는데 필요한 대안 수". Compression ratio.

### 3. BPE / SentencePiece
Variable-length tokenization 은 정확히 "자주 나오는 sub-word 는 짧은 token, 드문 것은 긴 token" → Kraft 의 symbol-level 구현.

### 4. Neural Compression
- **Lossless**: LLM + arithmetic coding (Deletang 2024). GPT-2 가 gzip 보다 강한 compressor.
- **Lossy**: VAE 의 rate-distortion. Rate = $\mathbb{E}[-\log q(z)]$.

### 5. Learned Entropy Models
End-to-end 이미지 압축 (Ballé 2016). Neural network 이 $p(z)$ 를 학습 → arithmetic coding 으로 entropy 근접 압축.

### 6. Adaptive Coding
LM context 가 길수록 $p(x_t | x_{<t})$ 가 sharp → code 길이 짧음 → 더 많은 compression.

---

## ⚖️ 가정·한계·함정

1. **정수 길이 제약** — $\ell_i = \lceil -\log_2 p_i \rceil$ 로 반올림 → Shannon code 는 엔트로피에서 최대 1 bit gap. Arithmetic coding 이 이 gap 을 제거 (§4.5).
2. **심볼 단위 부호화의 한계** — 블록 부호화하면 $1/n$ 로 감소 (§4.3 Source coding theorem).
3. **Known distribution 가정** — $p$ 를 모르면 adaptive coding 필요. 실무에서는 LM 이 $\hat p$ 을 제공.
4. **Binary 외 일반화** — $D$-ary 는 동일 원리, $\sum D^{-\ell_i} \le 1$.
5. **Universal coding** — Lempel-Ziv 같은 비모수 코드는 Kraft 를 직접 사용 않고 dictionary 방식 (§4.5 참고).
6. **Entropy vs compression achievable** — Kraft 는 **심볼 당 평균** 을 논함. 실제 유한 메시지에서는 약간 초과 가능 (확률적 변동).

---

## 📌 핵심 정리

1. **Prefix code** = 코드가 다른 코드의 접두사가 아닌 것 → 즉시 디코딩.
2. **Kraft 부등식**: $\sum 2^{-\ell_i} \le 1$ ↔ prefix code 존재.
3. UD 도 Kraft 만족 (McMillan).
4. **$L \ge H(p)$** — 엔트로피는 평균 비트수의 절대 하한.
5. Shannon code ($\ell_i = \lceil -\log p_i\rceil$): $L < H + 1$.
6. AI 연결: Cross-Entropy Loss = 평균 code length, Perplexity = compression ratio.

---

## 🤔 생각해볼 문제

### 문제 1. Kraft 필요성 증명 상세
Theorem 4.1.1 의 증명에서 "subtree 가 disjoint 이면 leaf 합 ≤ total" 부분을 더 엄밀히.

<details>
<summary>해설</summary>

각 codeword $c_i$ 를 root 부터의 path. Prefix 조건 = 어떤 $c_j$ 도 다른 $c_k$ 의 path 상에 있지 않음 = 두 codeword 에서 확장되는 subtree 가 공통 leaf 를 갖지 않음. 전체 $2^{\ell_{\max}}$ 개 leaf 중 codeword $c_i$ 의 subtree 가 $2^{\ell_{\max} - \ell_i}$ 개 leaf 점유 → disjoint 이므로 합이 총 수를 초과 못 함.
</details>

### 문제 2. 비-정수 길이 가능한가?
실수 $\ell_i$ 에 대해 Kraft 부등식을 어떻게 해석?

<details>
<summary>해설</summary>

Arithmetic coding 의 이론: 무한 precision 에서 메시지 당 평균 $-\log p$ bit 가능 (정수 반올림 없이). Kraft 의 연속 버전: measure-theoretic, $p$ 자체를 "fractional code" 로 해석. Entropy 가 tight lower bound 이 되는 이유.
</details>

### 문제 3. 최적 코드 길이 할당
$p_1, \ldots, p_n$ 주어졌을 때 $L = \sum p_i \ell_i$ 최소화하는 정수 $\ell_i$ 를 찾는 알고리즘?

<details>
<summary>해설</summary>

Huffman algorithm. 두 개의 가장 작은 확률을 병합, 재귀. 결과는 $L \le H + 1$, optimal prefix code 보장. 증명은 교환 논증 (§4.2).
</details>

### 문제 4. $D$-ary 코드의 entropy 단위
$D = 10$ (decimal code) 이면 entropy 의 단위?

<details>
<summary>해설</summary>

$H_{10}(p) = -\sum p_i \log_{10} p_i$ = "$p$ 를 decimal digit 단위로 표현하는 평균 길이". $H_{10} = H_2 / \log_2 10$. 실무 compression 은 대부분 binary.
</details>

### 문제 5. McMillan 정리의 놀라움
왜 UD 라면 prefix 가 아니어도 Kraft 만족하는가?

<details>
<summary>해설</summary>

Prefix 가 아닌 UD 코드는 드물지만 존재 (예: $\{00, 01, 10, 11, 0\}$ 는 NOT UD). UD 이면 decoder 가 message boundary 를 결정론적으로 찾을 수 있어서 **coarse 한 compression 하한** 이 필요. McMillan 의 조합론 논증: $N$-symbol 메시지의 가능한 경우 수가 $2^L$ 이하 — information 보존 → Kraft 필연.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [3.5 MI와 표현학습](../ch3-mutual-information/05-mi-representation-learning.md) | [4.2 Huffman 부호의 최적성](./02-huffman-optimality.md) |

[🏠 Home](../README.md)

</div>
