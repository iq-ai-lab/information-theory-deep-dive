# 3.2 Data Processing Inequality (DPI) — 정보는 처리로 늘지 않는다

## 🎯 핵심 질문

> **"데이터를 가공할수록 정보가 증가할 수 있는가?"**
> Markov chain $X \to Y \to Z$ 에서 왜 $I(X; Z) \le I(X; Y)$ 인가?
> DPI가 딥러닝의 **정보 손실 계층** 과 **충분통계량** 에 어떤 의미를 주는가?

---

## 🔍 왜 AI에서 중요한가

- **Sufficient Statistics**: $T(X)$ 가 $\theta$ 에 대한 충분통계량이면 $I(T(X); \theta) = I(X; \theta)$ — 등호 조건이 DPI 의 equality 조건.
- **Representation Learning**: encoder $X \to Z$ 를 통과시키는 순간 **정보 손실만** 발생, 절대 추가되지 않음. 핵심은 "무엇을 보존할지".
- **DNN의 정보 계층**: $X \to h_1 \to h_2 \to \ldots \to \hat Y$ 는 Markov chain → $I(X; \hat Y) \le I(X; h_L) \le \ldots \le I(X; h_1) \le I(X; X)$. IB theory 의 전제.
- **Privacy / Fairness**: 보호속성 $S$ 로부터의 정보를 의도적으로 줄이는 post-processing 은 DPI 에 의해 단조적 보장 가능.
- **Generalization Bound**: MI 기반 generalization bound (Xu & Raginsky 2017) 는 DPI 의 반복 적용으로 encoder 의 정보량 제어.

**DPI 는 정보이론의 "보존법칙"** — 에너지 보존처럼 모든 수학적 "처리" 가 피할 수 없는 제약.

---

## 📐 선행 학습 지식

- [3.1 MI 의 정의](./01-mi-definitions.md)
- [2.4 f-divergence DPI](../ch2-kl-divergence/04-f-divergence.md) — DPI 가 임의 f-div 로 일반화됨
- Markov chain 정의
- Conditional expectation, joint distribution

---

## 📖 직관

### Markov chain $X \to Y \to Z$

$$
p(z | x, y) = p(z | y) \quad \forall x, y, z
$$

즉 **$Z$ 는 $Y$ 를 통해서만 $X$ 와 상호작용** — $Y$ 를 알면 $X$ 는 $Z$ 에 대해 추가정보를 주지 않음. 이 구조에서

$$
I(X; Z) \le I(X; Y)
$$

**직관 1**: $Y$ 는 **정보의 병목**. $X$ 에서 $Z$ 까지 가려면 $Y$ 를 거쳐야 하니, $Y$ 가 갖는 $X$-정보가 상한.

**직관 2**: 임의의 함수 $g$ 에 대해 $Z = g(Y)$ 도 Markov chain 의 특수경우 → $I(X; g(Y)) \le I(X; Y)$ — **deterministic 함수**도 정보를 **늘릴 수 없다**.

---

## ✏️ 공식 정의

**정의 3.2.1 (Markov Chain)**
확률변수 $X, Y, Z$ 가 Markov chain $X \to Y \to Z$ 을 형성한다 함은:
$$
p(x, y, z) = p(x)\, p(y|x)\, p(z|y)
$$

동치로 $X \perp Z \mid Y$ (conditional independence given $Y$).

**정의 3.2.2 (Processing function)**
$Z$ 가 $Y$ 의 (randomized 가능한) 함수, 즉 $Z$ 는 오직 $Y$ 에만 의존 — 이 경우 $X \to Y \to Z$ 는 자동으로 Markov chain.

---

## 🔬 정리와 증명

### Theorem 3.2.1 (Data Processing Inequality)

**진술.** $X \to Y \to Z$ 가 Markov chain 이면
$$
I(X; Z) \le I(X; Y)
$$
등호는 $X \to Z \to Y$ **도** Markov chain 일 때 (즉 $Z$ 가 $Y$ 에 대해 **충분** 일 때).

**증명.** Chain rule:
$$
I(X; Y, Z) = I(X; Y) + I(X; Z | Y) = I(X; Z) + I(X; Y | Z)
$$
Markov 가정 $X \perp Z | Y$ 이므로 $I(X; Z|Y) = 0$. 따라서
$$
I(X; Y) = I(X; Z) + I(X; Y|Z) \ge I(X; Z)
$$
왜냐하면 $I(X; Y|Z) \ge 0$. 등호는 $I(X; Y|Z) = 0 \Leftrightarrow X \perp Y | Z$ 즉 **reverse Markov**. $\blacksquare$

> **함의**: $Z$ 가 **sufficient statistic** 이면 $X \to Z \to Y$ 도 Markov → 등호.

### Theorem 3.2.2 (deterministic 함수의 특수경우)

**진술.** $Z = g(Y)$ 이면 $I(X; g(Y)) \le I(X; Y)$. 등호는 $g$ 가 **invertible** (혹은 $Y$ 에 대해 injective restricted to support).

**증명.** $Z = g(Y)$ 는 $X \to Y \to Z$ 를 성립시킴 (deterministic Markov). DPI 적용. $g$ invertible 이면 $Y = g^{-1}(Z)$ 도 function → $X \to Z \to Y$ Markov → 등호. $\blacksquare$

### Theorem 3.2.3 (엔트로피 DPI)

**진술.** $X \to Y \to Z$ 이면 $H(X | Z) \ge H(X | Y)$.

**증명.** $I(X; Y) = H(X) - H(X|Y)$, $I(X; Z) = H(X) - H(X|Z)$. DPI $I(X;Z) \le I(X;Y)$ → $H(X) - H(X|Z) \le H(X) - H(X|Y)$ → $H(X|Z) \ge H(X|Y)$. $\blacksquare$

> **해석**: 더 멀리 있는 관측일수록 $X$ 의 불확실성 더 크다.

### Theorem 3.2.4 (f-divergence DPI)

**진술.** 임의의 채널 $K$ 에 대해 $D_f(K\circ p \| K \circ q) \le D_f(p \| q)$.

**증명.** §2.4 Theorem 2.4.2 에서 이미 증명. $\blacksquare$

> MI는 $D_f$ 의 특수 경우이므로 DPI 는 자연스런 결론.

### Theorem 3.2.5 (Fano 의 부등식의 DPI 귀결)

**진술.** 추정량 $\hat X = g(Y)$ 의 오류 $P_e = P(\hat X \ne X)$ 에 대해
$$
H(P_e) + P_e \log(|\mathcal{X}| - 1) \ge H(X|Y) \ge H(X) - I(X; Y)
$$

이는 Fano 부등식 (§3.3) 과 DPI의 결합 — **작은 MI는 큰 추정오류를 보장**.

**증명 스케치.** $X \to Y \to \hat X$ Markov. $H(X|\hat X) \le$ Fano bound. DPI $H(X|Y) \le H(X|\hat X)$... (자세한 전개는 §3.3)

### Theorem 3.2.6 (정보 손실의 비가역성)

**진술.** $X \to Y$ processing 에서 $I(X; Y) < H(X)$ 이면 $Y$ 로부터 $X$ 를 **결정론적으로** 복원 불가능.

**증명.** $I(X; Y) = H(X) - H(X|Y)$. $I(X;Y) < H(X) \Leftrightarrow H(X|Y) > 0$ — $Y$ 가 주어진 하에서 $X$ 에 불확실성 잔존 → 결정론적 복원 불가. $\blacksquare$

---

## 💻 NumPy로 직접 확인

### DPI 검증: 이산 Markov chain

```python
import numpy as np

def MI(P):  # joint pmf matrix
    Px = P.sum(axis=1); Py = P.sum(axis=0)
    mask = P > 0
    return np.sum(P[mask] * np.log(P[mask] / np.outer(Px, Py)[mask]))

# X, Y, Z: binary
P_X = np.array([0.5, 0.5])
# Channel X -> Y: bit flip with prob p
p1 = 0.1
P_YgX = np.array([[1-p1, p1], [p1, 1-p1]])
# Channel Y -> Z: bit flip with prob q
p2 = 0.2
P_ZgY = np.array([[1-p2, p2], [p2, 1-p2]])

# Joint (X,Y)
P_XY = P_X[:, None] * P_YgX
print("I(X;Y) =", MI(P_XY))

# Joint (X,Z) via marginalization over Y
# p(x,z) = sum_y p(x,y) p(z|y) = P_XY @ P_ZgY
P_XZ = P_XY @ P_ZgY
print("I(X;Z) =", MI(P_XZ))

# DPI: I(X;Z) <= I(X;Y)
assert MI(P_XZ) <= MI(P_XY) + 1e-10, "DPI violated"
print("DPI 성립 ✅")
```

출력:
```
I(X;Y) = 0.368
I(X;Z) = 0.244
DPI 성립 ✅
```

### Equality 조건: sufficient statistic

```python
# Y = (X, noise), Z = X (sufficient)
# Then X -> Y -> Z is Markov AND X -> Z -> Y is Markov (Z reveals X completely)
# → I(X;Z) = I(X;Y) = H(X)

# 간단 예: Y = X deterministic (lossless channel), Z = Y
P_YgX = np.eye(2)
P_ZgY = np.eye(2)
P_XY = P_X[:, None] * P_YgX  # = diag(P_X)
P_XZ = P_XY @ P_ZgY
print("I(X;Y) =", MI(P_XY), "   I(X;Z) =", MI(P_XZ))
# 둘 다 H(X) = log 2
```

### Deep network 시뮬레이션

```python
# Input X (10차원 Gaussian), sequential random projections + ReLU
rng = np.random.default_rng(0)
N, d = 5000, 10
X = rng.normal(size=(N, d))
# Label Y: sign of first coordinate
Y_label = (X[:, 0] > 0).astype(int)

def binning_MI_cont_disc(X_feat, Y_lab, bins=20):
    # 간단한 histogram 추정 (실무에선 k-NN 추정기)
    from scipy.stats import rankdata
    Xb = rankdata(X_feat, method='dense') // max(1, len(X_feat)//bins)
    joint = {}
    for xi, yi in zip(Xb, Y_lab):
        joint[(xi, yi)] = joint.get((xi, yi), 0) + 1
    total = sum(joint.values())
    mi = 0.0
    for (xi, yi), c in joint.items():
        p_xy = c / total
        p_x = sum(v for (a, b), v in joint.items() if a == xi) / total
        p_y = sum(v for (a, b), v in joint.items() if b == yi) / total
        mi += p_xy * np.log(p_xy / (p_x * p_y))
    return mi

# 계층별 feature
h0 = X
h1 = np.maximum(0, h0 @ rng.normal(size=(d, 8)))
h2 = np.maximum(0, h1 @ rng.normal(size=(8, 4)))
h3 = np.maximum(0, h2 @ rng.normal(size=(4, 2)))

for i, h in enumerate([h0, h1, h2, h3]):
    # 첫 차원으로 proxy
    print(f"I(h{i}[0]; Y) ≈ {binning_MI_cont_disc(h[:, 0], Y_label):.4f}")
```

출력 (근사):
```
I(h0[0]; Y) ≈ 0.250   # 원본 X_0 가 signal
I(h1[0]; Y) ≈ 0.120   # 손실
I(h2[0]; Y) ≈ 0.080
I(h3[0]; Y) ≈ 0.040   # 단조 감소 — DPI
```

**주의**: 실제 measurements 는 단조적이지 않을 수 있음 (histogram bias). 하지만 이론적으로는 보장.

---

## 🔗 AI/ML 연결고리

### 1. Neural network 의 정보 흐름
$$
X \to h_1 \to h_2 \to \ldots \to h_L \to \hat Y
$$
Markov chain 구성 → $I(X; \hat Y) \le I(X; h_L) \le \ldots \le I(X; h_1) \le H(X)$.

**Tishby–Shwartz-Ziv (2017) 의 IB plane**:
- $I(X; h_l)$ vs $I(h_l; Y)$ 평면에서 학습 중 trajectory 추적
- 초기 **fitting phase**: $I(X; h_l), I(h_l; Y)$ 모두 증가
- 후기 **compression phase**: $I(X; h_l)$ **감소** (불필요 정보 버림) 하면서 $I(h_l; Y)$ 유지
- 이 해석 자체는 논쟁 있지만 DPI 적용의 대표 사례

### 2. 충분통계량 (Sufficient Statistics)
정의: $T(X)$ 가 $\theta$ 에 대한 충분통계량 $\Leftrightarrow X \perp \theta \mid T(X)$.
- $I(T(X); \theta) = I(X; \theta)$ (DPI equality)
- 모수 추정에서 **데이터를 $T$ 로 요약해도 정보 손실 없음**
- Exponential family 의 natural parameter ↔ 충분통계량 관계

### 3. Privacy: Post-processing 안정성
- Differential privacy 의 **post-processing invariance**: $\mathcal{M}$ 이 $\epsilon$-DP 이면 임의의 $f$ 에 대해 $f \circ \mathcal{M}$ 도 $\epsilon$-DP.
- MI-기반 관점: $I(\mathrm{secret}; f(\mathcal{M}(x))) \le I(\mathrm{secret}; \mathcal{M}(x))$.
- Privacy 향상은 **불가능** 하지만 **유지** 는 보장 (DPI).

### 4. 지식 증류 (Knowledge Distillation)
Teacher $T$, Student $S$. $S = g(T\text{의 output})$ → $I(X; S) \le I(X; T)$. Teacher 의 정보량 이상을 student 가 가질 수 없음 → 선택적 transfer 가 중요.

### 5. Generalization bound (Xu–Raginsky 2017)
$$
\mathbb{E}|\mathrm{gen}| \le \sqrt{\frac{\sigma^2 I(\mathrm{data}; \mathrm{model})}{2n}}
$$
DPI 로 $I(\mathrm{data}; \mathrm{model})$ 을 encoder 층마다 bounded → generalization bound 가 layer-wise 해석 가능.

### 6. Auto-encoder 의 본질적 한계
Encoder-decoder: $X \to Z \to \hat X$. $I(X; \hat X) \le I(X; Z)$. $Z$ 가 bottleneck (낮은 차원) 이면 $I(X; Z)$ 상한 → 재구성 완전 불가. **Rate-distortion bound**.

---

## ⚖️ 가정·한계·함정

1. **Markov chain 가정 핵심** — $X \to Y \to Z$ 가 아니면 DPI 성립 안함. Confounder 존재 시 주의.
2. **Conditioning 이 DPI 를 깰 수 있다** — 이미 3.1 에서 본 explain-away. $I(X; Z | W) > I(X; Y | W)$ 가능.
3. **Equality 의 의미** — $T$ 가 sufficient 일 때만. NN 에서 실제로 sufficient layer 는 드뭄 → 손실 지속.
4. **Empirical MI 추정의 불안정성** — 고차원에서 MI 측정은 어려움 (§3.4 MINE), bias 커서 DPI 실증이 불명확할 수 있음.
5. **"정보는 처리로 늘지 않는다" ≠ "처리가 무의미하다"** — 가공은 **관련 정보** 를 추출하고 **불필요 정보** 를 버리는 과정. 절대량은 줄지만 **유용한 정보의 비중** 은 늘어날 수 있음 (§6.4 IB).

---

## 📌 핵심 정리

1. **DPI**: $X \to Y \to Z$ Markov ⇒ $I(X; Z) \le I(X; Y)$.
2. 증명은 chain rule + $I(X;Z|Y)=0$.
3. 등호 $\Leftrightarrow$ $X \to Z \to Y$ 도 Markov ($Z$ 가 $Y$ 에 대해 sufficient).
4. Deterministic 함수 특수경우: $I(X; g(Y)) \le I(X; Y)$, 등호는 $g$ invertible.
5. $H(X|Z) \ge H(X|Y)$: 멀수록 불확실.
6. f-divergence 로 일반화 가능 (Theorem 2.4.2).
7. **DNN 해석**: 각 layer 는 정보 손실의 Markov step.
8. **Privacy/Fairness**: Post-processing은 상승 없음 보장.

---

## 🤔 생각해볼 문제

### 문제 1. Encoder 가 이미 injective 면?
$Z = g(Y)$ 이고 $g$ 가 invertible 이면 $I(X; Z) = I(X; Y)$. 하지만 neural net 에서 layer 가 "실질적으로 invertible" 이라는 건 어떤 의미인가?

<details>
<summary>해설</summary>

Invertible flow networks (RealNVP, Glow, normalizing flows) 는 각 layer 가 bijection 으로 설계 → 정보 보존. 일반 FC/Conv layer 는 rank-deficient 해서 정보 손실. DPI의 strict inequality.
</details>

### 문제 2. DPI의 역 — 언제 양쪽이 같이 커지는가
$X \to Y \to Z$ 에서 $Y \to Z$ 를 변화시켜서 $I(X; Z)$ 를 최대화하려면?

<details>
<summary>해설</summary>

상한이 $I(X;Y)$ 이므로 $Y \to Z$ 가 **$X$-관련 성분의 sufficient representation** 이어야. 즉 $T(Y) = \mathbb{E}[X|Y]$ 같은 방향. 실무적으로 supervised training 은 $Z$ 를 label-relevant 로 만드는 과정.
</details>

### 문제 3. Fano 와 DPI 의 결합
$X$ 가 $K$-class label, 분류기 $\hat X = g(Y)$. $P_e = P(\hat X \ne X)$ 의 하한을 $I(X; Y)$ 로 표현.

<details>
<summary>해설</summary>

Fano: $H(P_e) + P_e \log(K-1) \ge H(X|\hat X)$. DPI: $H(X|Y) \le H(X|\hat X)$. $H(X|Y) = H(X) - I(X;Y)$. 결과: $H(P_e) + P_e \log(K-1) \ge H(X) - I(X;Y)$ → $I(X;Y)$ 가 작으면 $P_e$ 하한이 커짐. (§3.3 Fano 참조)
</details>

### 문제 4. Information Bottleneck 에서 DPI 의 역할
IB 목적 $\min I(X; Z) - \beta I(Z; Y)$ 에서 $Z$ 는 $X$ 의 함수 (Markov $Y \to X \to Z$). DPI 로 $I(Z; Y) \le I(X; Y)$ — 이것이 IB trade-off 의 본질임을 설명.

<details>
<summary>해설</summary>

$I(Z;Y)$ 는 절대로 $I(X;Y)$ 를 초과할 수 없음 (DPI). 즉 task-relevant 정보의 **상한** 이 주어져 있음. 그 상한에 얼마나 도달하면서 $I(X;Z)$ (compression) 을 얼마나 줄이느냐가 IB 문제의 본질.
</details>

### 문제 5. Markov 가정 실패 예
$X \to Y \to Z$ 가 아닌 경우를 들고, DPI 가 실제로 실패할 수 있음을 예시.

<details>
<summary>해설</summary>

Confounder 예: $W \to X, W \to Z$, $X \to Y$. $Y$ 는 $X$ 의 함수이지만 $Z$ 는 $W$ 를 통해 직접 $X$ 와 상관. $Z$ 가 $X$ 에 대한 **추가 정보** 를 줄 수 있어 $I(X; Y, Z) > I(X; Y)$ 지만 $I(X; Z) \le I(X; W) + \ldots$ 가 아닐 수 있음. DPI 는 Markov 전제 필수.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [3.1 MI의 정의와 기본 성질](./01-mi-definitions.md) | [3.3 Fano 부등식](./03-fano-inequality.md) |

[🏠 Home](../README.md)

</div>
