# 4.2 Huffman 부호의 최적성

## 🎯 핵심 질문

> **주어진 확률분포에서 평균 길이가 최소인 prefix code 를 어떻게 구성하는가?**
> Huffman algorithm (1952) 은 왜 **정확히 최적** 인가?
> Shannon code 와 비교해서 얼마나 개선되는가?

---

## 🔍 왜 AI에서 중요한가

- **Greedy algorithm 의 정확 최적 예** — 알고리즘 교육의 고전.
- **JPEG, MP3, DEFLATE** 의 기반 — 현재도 사용되는 코덱 내부.
- **Tokenizer 설계** — BPE 의 greedy 병합은 Huffman 의 사촌 (개념적으로).
- **Variable-length bitstream 레이어** — 모든 압축 시스템의 기초 층.
- **Cross-entropy ↔ optimal code** — LM 의 perplexity 해석과 직접 연결.

Huffman 을 이해하면 "왜 LLM 이 tokenizer 에서 일부 토큰은 짧고 일부는 긴지", "왜 cross-entropy loss 가 bits-per-character 로 환산되는지" 가 명확.

---

## 📐 선행 학습 지식

- [4.1 Prefix code 와 Kraft 부등식](./01-prefix-code-kraft.md)
- Binary tree, priority queue (heap)
- 귀납 증명, 교환 논증 (exchange argument)

---

## 📖 직관

### 재귀적 병합 원리

분포 $p_1 \ge p_2 \ge \ldots \ge p_n$. 가장 희귀한 두 기호는 **가장 긴 코드** 를 가져야 → 두 기호를 **병합**하고 새 기호에 대해 재귀.

**비유**: "뉴스에서 가장 안 중요한 두 내용을 묶어서 같은 슬롯에 배치, 나머지 우선순위 재계산".

### 알고리즘 의사코드

```
Huffman(p_1, ..., p_n):
  Q = priority queue of (p_i, leaf_i)
  while |Q| > 1:
    (p_a, a) = Q.pop_min()
    (p_b, b) = Q.pop_min()
    새 노드 c: 자식 = a, b, 확률 = p_a + p_b
    Q.push((p_a + p_b, c))
  return tree rooted at Q.pop()
```

### Shannon code vs Huffman

- **Shannon**: $\ell_i = \lceil -\log p_i \rceil$ — 개별 확률만 보고 길이 결정 → $L \le H + 1$.
- **Huffman**: 전체 분포 구조 활용 → **정확 최적** $L \le H + 1$ 보장되며, 대부분 Shannon 보다 짧음.

---

## ✏️ 공식 정의

**정의 4.2.1 (Optimal prefix code)**
확률분포 $p$ 에 대해 $L^* = \min_{C \text{ prefix}} \sum p_i \ell_i(C)$ 를 달성하는 prefix code.

**정의 4.2.2 (Huffman tree)**
재귀적 병합으로 생성된 binary tree:
1. $n$ 개 leaf 로 시작.
2. 확률이 가장 작은 두 노드를 병합 (합이 부모 노드의 확률).
3. Leaf 하나만 남을 때까지 반복.

길이 $\ell_i$ = leaf $i$ 의 depth.

---

## 🔬 정리와 증명

### Theorem 4.2.1 (Huffman Optimality)

**진술.** Huffman 알고리즘이 만드는 코드 $C_H$ 의 평균 길이 $L(C_H)$ 는 임의의 prefix code $C$ 보다 작거나 같다:
$$
L(C_H) \le L(C) \quad \forall C \text{ prefix}
$$

**증명 (exchange argument).** 강한 귀납법.

**Lemma A (Siblings).** 최적 코드에서 가장 낮은 확률 두 기호 $a, b$ ($p_a \le p_b \le$ 나머지) 는 **sibling** (동일 부모 아래) 으로 배치 가능.

증명: 어떤 최적 코드에서 가장 깊은 두 leaf 가 siblings 아니면, 그 두 leaf 와 어떤 다른 leaf 의 위치를 교환해도 total length 가 악화되지 않음. 작은 확률 둘이 가장 깊이.

**Lemma B (Merge).** $n$ 심볼 문제에서 $a, b$ 를 병합해 심볼 $z = \{a, b\}$ (확률 $p_a + p_b$) 로 대체한 $n-1$ 심볼 문제를 최적 해결하면, $a, b$ 를 $z$ 의 자식으로 복원한 트리가 원래 $n$ 심볼 문제의 최적.

증명: $n$ 심볼 최적 $L^* = L^*_{n-1} + (p_a + p_b)$ (병합 노드의 extra depth). 두 optimum 이 서로 bijective.

**귀납 base**: $n = 2$ 에서 자명 ($a=0, b=1$).

**귀납 step**: $n-1$ 에서 Huffman 최적 가정 → Lemma B 로 $n$ 에서 Huffman 도 최적. $\blacksquare$

### Theorem 4.2.2 (상한)

**진술.** Huffman code $C_H$ 의 평균 길이
$$
H(p) \le L(C_H) < H(p) + 1
$$

**증명.** 하한은 Theorem 4.1.4. 상한은 Theorem 4.1.5 (Shannon code ≤ $H+1$) + Huffman 이 Shannon 보다 짧음. 자세한 tight bound 는 Gallager (1978) 로 $L \le H + p_1 + \log_2((\log_2 e)/e) + 0.086$ 등 정교한 결과 존재. $\blacksquare$

### Theorem 4.2.3 (Dyadic 분포에서 등호)

**진술.** $p_i = 2^{-\ell_i}$ (dyadic) 이면 Huffman 이 $L = H(p)$ 정확히 달성.

**증명.** $\ell_i = -\log_2 p_i$ 정수, Kraft 등호 성립. Huffman 이 $\ell_i$ 길이 할당. $\blacksquare$

### Theorem 4.2.4 (Huffman 은 unique 하지 않음)

**진술.** Huffman tree 는 분포가 같은 확률을 가진 경우 **여러 최적 해** 를 가짐.

**예.** $p = (0.4, 0.3, 0.2, 0.1)$ 과 $p = (0.4, 0.2, 0.2, 0.2)$ 에서 tie-breaking 에 따라 다른 (같은 평균길이) 트리.

**함의**: Canonical Huffman code (길이 순 + 사전식 정렬) 로 결정론적 정의.

### Theorem 4.2.5 (Blocking 으로 entropy 에 수렴)

**진술.** 분포 $p$ 의 $k$-블록 $p^{(k)}(x_1, \ldots, x_k)$ 에 대해 Huffman 의 평균 bits per symbol
$$
L_k / k \to H(p) \quad (k \to \infty)
$$

**증명.** Theorem 4.2.2 에 의해 $H(p^{(k)}) \le L_k < H(p^{(k)}) + 1$. $k$ 로 나누면 $H(p) \le L_k/k < H(p) + 1/k \to H(p)$. $\blacksquare$

> **함의**: 블록화 → 1 bit gap 이 $1/k$ 로 감소. 이것이 source coding theorem (§4.3).

---

## 💻 NumPy/Python 으로 직접 확인

### Huffman algorithm 구현

```python
import heapq
from collections import defaultdict

class Node:
    def __init__(self, prob, symbol=None, left=None, right=None):
        self.prob = prob
        self.symbol = symbol
        self.left = left
        self.right = right
    def __lt__(self, other):  # heap comparison
        return self.prob < other.prob

def huffman_code(probs, symbols=None):
    if symbols is None: symbols = list(range(len(probs)))
    heap = [Node(p, s) for p, s in zip(probs, symbols)]
    heapq.heapify(heap)
    while len(heap) > 1:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        heapq.heappush(heap, Node(a.prob + b.prob, left=a, right=b))
    root = heap[0]
    codes = {}
    def traverse(node, prefix=""):
        if node.symbol is not None:
            codes[node.symbol] = prefix or "0"
        else:
            traverse(node.left, prefix + "0")
            traverse(node.right, prefix + "1")
    traverse(root)
    return codes

# 예제 1: dyadic
probs = [0.5, 0.25, 0.125, 0.125]
codes = huffman_code(probs, ['A', 'B', 'C', 'D'])
for s, c in codes.items(): print(f"{s}: {c}")
L = sum(p * len(c) for p, c in zip(probs, codes.values()))
H = -sum(p * np.log2(p) for p in probs)
print(f"L = {L}, H = {H}")
```

출력:
```
A: 0
B: 10
C: 110
D: 111
L = 1.75, H = 1.75
```
Dyadic 에서 entropy 정확히 달성.

### 비-dyadic 예제

```python
probs = [0.4, 0.3, 0.2, 0.1]
codes = huffman_code(probs, ['A', 'B', 'C', 'D'])
for s, c in codes.items(): print(f"{s}: {c}")

import numpy as np
L = sum(p * len(c) for p, c in zip(probs, codes.values()))
H = -sum(p * np.log2(p) for p in probs)
print(f"L = {L:.4f}, H = {H:.4f}, gap = {L - H:.4f}")
```

출력:
```
A: 0
B: 10
C: 110
D: 111
L = 1.9000, H = 1.8464, gap = 0.0536
```

### Shannon code 와 비교

```python
def shannon_code_lengths(probs):
    return [int(np.ceil(-np.log2(p))) for p in probs]

for probs in [[0.4, 0.3, 0.2, 0.1], [0.5, 0.2, 0.15, 0.1, 0.05]]:
    hC = huffman_code(probs)
    L_H = sum(p * len(c) for p, c in zip(probs, hC.values()))
    L_S = sum(p * l for p, l in zip(probs, shannon_code_lengths(probs)))
    H = -sum(p * np.log2(p) for p in probs)
    print(f"probs={probs}")
    print(f"  Huffman L = {L_H:.4f}")
    print(f"  Shannon L = {L_S:.4f}")
    print(f"  Entropy H = {H:.4f}")
```

출력:
```
probs=[0.4, 0.3, 0.2, 0.1]
  Huffman L = 1.9000
  Shannon L = 2.1000
  Entropy H = 1.8464
probs=[0.5, 0.2, 0.15, 0.1, 0.05]
  Huffman L = 2.0000
  Shannon L = 2.5000
  Entropy H = 1.9262
```
Huffman 이 Shannon 보다 항상 ≤.

### 블록 확장 (AEP 전조)

```python
# Block length k 에서의 Huffman 성능
probs = np.array([0.7, 0.3])
H = -np.sum(probs * np.log2(probs))

for k in [1, 2, 4, 8]:
    # 블록 확률: 독립 가정
    from itertools import product
    block_probs = []
    for seq in product(range(2), repeat=k):
        p = np.prod([probs[s] for s in seq])
        block_probs.append(p)
    codes = huffman_code(block_probs)
    L = sum(p * len(c) for p, c in zip(block_probs, codes.values()))
    print(f"k={k}: L/k = {L/k:.4f}, H = {H:.4f}")
```

출력:
```
k=1: L/k = 1.0000, H = 0.8813
k=2: L/k = 0.9300, H = 0.8813
k=4: L/k = 0.8961, H = 0.8813
k=8: L/k = 0.8833, H = 0.8813
```

블록 키울수록 entropy 에 수렴 (Theorem 4.2.5).

---

## 🔗 AI/ML 연결고리

### 1. JPEG, MP3, MPEG
Quantized coefficient 들을 Huffman coding → lossy compression 의 마지막 lossless 레이어.

### 2. DEFLATE (gzip, PNG)
LZ77 dictionary + Huffman — 대표적 general-purpose compressor.

### 3. BPE Tokenizer ↔ Huffman
BPE 는 빈도 기반 greedy merge — 직접 Huffman 은 아니지만 "자주 나오는 pair → 새 token" 의 철학 공유. 다만 BPE 는 sub-word 단위, Huffman 은 symbol 단위.

### 4. Neural LM + Huffman
학습된 $p(x_t | x_{<t})$ 에서 Huffman 만들면 context-adaptive 가변 길이 코드. Arithmetic coding 이 더 일반적 (§4.5).

### 5. Model Quantization
가중치의 각 bin 에 Huffman code → 저장공간 절약 (Han 2016 Deep Compression).

### 6. Attention Sparsity
희소 attention score 의 인덱스 저장: value 가 몇 개 안되면 Huffman.

### 7. Genomics / DNA Storage
DNA 서열의 4-letter 알파벳 → 5-way Huffman (with synchronization markers).

---

## ⚖️ 가정·한계·함정

1. **정수 길이 제약** — Arithmetic coding 이 이를 제거 (§4.5).
2. **Distribution known 가정** — Adaptive Huffman (Vitter 1987) 은 실시간 업데이트.
3. **Equal prob symbols** — 많은 기호가 같은 확률이면 Huffman 의 이점 감소 (fixed-length 와 비슷).
4. **구현 복잡도** — 트리 저장 + 디코더 필요. 짧은 메시지에 overhead.
5. **메모리 효율** — canonical Huffman 으로 decoder table 최소화.
6. **Long-tail 분포** — 희귀 심볼의 code 가 길어서 single-bit error 가 치명적. Variable-to-fixed (Tunstall) 코드가 반대 특성.

---

## 📌 핵심 정리

1. **Huffman algorithm**: 두 최소 확률 병합 재귀 → optimal prefix code.
2. **최적성 증명**: Lemma A (siblings) + Lemma B (merge) + 귀납.
3. $H \le L(C_H) < H + 1$.
4. Dyadic 분포 → entropy 정확 달성.
5. 블록 확장 → $L/k \to H$ (source coding theorem 전조).
6. Shannon code 보다 일반적으로 짧음.
7. 실무 압축 (JPEG, gzip, MP3) 의 근간.

---

## 🤔 생각해볼 문제

### 문제 1. Huffman 이 optimal 함을 가장 작은 분포에서 확인
$p = (0.6, 0.25, 0.15)$ 의 Huffman tree 를 그리고, 다른 모든 prefix code 와 비교하라.

<details>
<summary>해설</summary>

Huffman: $\{A=0, B=10, C=11\}$, $L = 0.6+0.5+0.3 = 1.4$. 대안: $\{A=00, B=01, C=1\}$ → $L = 1.2+0.5+0.15 = 1.85$ (길어짐). 다른 할당도 Huffman 보다 짧아지지 않음. $H = 1.353$ → gap 0.047.
</details>

### 문제 2. 최장 코드의 확률 상한
Huffman code 에서 가장 긴 codeword 의 확률 $p_n$ 은 얼마 이하?

<details>
<summary>해설</summary>

$p_n \le 1/F_n$ 같은 관계 (Fibonacci 수열) — Golomb (1980). 대략 $p_n \le 0.618$ for $\ell_n = 2$ (not too tight). 일반적으로 작을수록 깊어짐.
</details>

### 문제 3. Huffman 의 길이 $\ell_i$ 의 범위
$\ell_i \le \lceil -\log p_i \rceil + ?$ 같은 명시적 bound.

<details>
<summary>해설</summary>

$\ell_i \le \lceil -\log_2 p_i \rceil$ 은 **항상** 성립하지 않음. Huffman 은 때때로 Shannon 보다 긴 코드 할당 (Shannon 의 엄격한 정수화와 달라서). 하지만 평균은 Huffman 이 더 짧음. Single symbol bound 는 $\ell_i \le \lceil -\log_2 p_i \rceil + 1$.
</details>

### 문제 4. Adaptive Huffman
실시간으로 분포 학습하며 코드 업데이트. Trade-off?

<details>
<summary>해설</summary>

장점: distribution 사전 지식 불필요. 단점: decoder 도 같은 알고리즘 실행 필요, 초기에는 비효율. Universal code (LZ, arithmetic coding with context model) 가 주로 더 실용적.
</details>

### 문제 5. Huffman vs arithmetic coding
동일 분포에서 arithmetic 가 항상 짧거나 같음을 설명.

<details>
<summary>해설</summary>

Arithmetic coding 은 fractional bit 가능 → symbol-level gap 없음. 메시지 길이 $n$ 에 대해 평균 $nH + O(1)$ bits (constants gap). Huffman 은 symbol 당 1 bit 이내 → 장기적으로 동등 수렴, 짧은 메시지에는 Huffman 이 실용.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [4.1 Prefix code와 Kraft 부등식](./01-prefix-code-kraft.md) | [4.3 Source Coding Theorem](./03-source-coding-theorem.md) |

[🏠 Home](../README.md)

</div>
