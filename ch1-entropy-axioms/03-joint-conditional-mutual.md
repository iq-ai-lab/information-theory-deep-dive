# 03. 결합·조건부·상호정보량

## 🎯 핵심 질문

- 두 확률변수 $X, Y$를 함께 보면 엔트로피는 어떻게 정의되는가?
- **조건부 엔트로피** $H(X \mid Y)$는 "$Y$를 알고 나서 남은 $X$의 불확실성"을 어떻게 수식화하는가?
- **상호정보량** $I(X; Y)$는 왜 "$X$와 $Y$ 사이의 정보의 양"으로 해석되는가?
- 세 가지 표현 $I(X; Y) = H(X) - H(X\mid Y) = H(Y) - H(Y\mid X) = H(X) + H(Y) - H(X, Y)$가 어떻게 모두 같은가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **Representation Learning**: 좋은 표현 $Z$는 입력 $X$에 대한 정보를 잘 보존 → $I(X; Z)$를 크게 유지 (InfoMax 원리)
- **Information Bottleneck**: $\min I(X; Z) - \beta I(Z; Y)$ — 압축과 충실도의 정보이론적 트레이드오프
- **Conditional Entropy = 예측 불확실성**: 분류기 입력 $Y$에서 $H(X \mid Y)$는 "어떤 $Y$를 봐도 예측할 수 없는 남은 불확실성"
- **Feature Selection**: $I(X_i; Y)$가 큰 feature를 선택 — 타깃과 **정보 공유량** 이 큰 특성
- **CLIP · SimCLR**: 두 뷰 $X, Y$ 사이의 MI를 최대화 → 자기지도학습의 수학적 기반

$H$만으로 **하나의 변수** 를 분석할 수 있지만, $H(X \mid Y), I(X; Y)$는 **관계의 정보** 를 본다. ML에서 진짜 중요한 건 "관계"다.

---

## 📐 수학적 선행 조건

- [문서 02](./02-entropy-definition.md)의 엔트로피 $H(X)$, Jensen 부등식
- 결합·주변·조건부 확률: $p(x, y), p(x), p(y \mid x) = p(x, y)/p(x)$
- **곱셈 법칙**: $p(x, y) = p(y) p(x \mid y) = p(x) p(y \mid x)$
- 독립성: $X \perp Y \iff p(x, y) = p(x) p(y)$

> 확률의 곱셈 법칙은 [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-statistics-deep-dive)를 전제합니다.

---

## 📖 직관적 이해

### 벤다이어그램

두 변수의 정보 구조는 집합의 합집합·교집합으로 직관적으로 표현된다:

```
   ┌────────── H(X, Y) ──────────┐
   │                             │
   │   H(X|Y)    I(X;Y)    H(Y|X)│
   │   ┌──────┐ ┌──────┐ ┌──────┐│
   │   │      │ │      │ │      ││
   │   │  X만 │ │ 공유 │ │  Y만 ││
   │   │      │ │ 정보 │ │      ││
   │   └──────┘ └──────┘ └──────┘│
   │    H(X)          H(Y)       │
   │    (좌 두 박스)  (우 두 박스)│
   └─────────────────────────────┘
```

- $H(X) = H(X \mid Y) + I(X; Y)$ (X의 정보 = Y 모를 때 남은 것 + Y와 공유)
- $H(X, Y) = H(X \mid Y) + I(X; Y) + H(Y \mid X) = H(X) + H(Y) - I(X; Y)$
- **독립**: $I(X; Y) = 0 \Rightarrow H(X, Y) = H(X) + H(Y)$
- **완전 결정**: $Y = f(X) \Rightarrow H(Y \mid X) = 0, I(X; Y) = H(Y)$

### 일상 예시

- $X$ = 내일 비가 올지, $Y$ = 오늘 기압 → $I(X; Y) > 0$ (기압은 비에 대한 정보)
- $X$ = 주사위 숫자, $Y$ = 독립적인 동전 결과 → $I(X; Y) = 0$
- $X$ = 사람, $Y$ = 그 사람의 DNA → $I(X; Y) = H(X)$ (DNA는 개인을 완전히 결정)

---

## ✏️ 엄밀한 정의

### 정의 3.1 — 결합 엔트로피 (Joint Entropy)

확률변수 쌍 $(X, Y)$의 결합 분포 $p(x, y)$에 대해
$$H(X, Y) := -\sum_{x, y} p(x, y) \log p(x, y) = \mathbb{E}_{(X,Y) \sim p}\!\big[-\log p(X, Y)\big].$$

단순히 $(X, Y)$를 하나의 확률변수로 본 엔트로피.

### 정의 3.2 — 조건부 엔트로피 (Conditional Entropy)

$$H(X \mid Y) := \sum_{y} p(y) \, H(X \mid Y = y) = -\sum_{x, y} p(x, y) \log p(x \mid y) = \mathbb{E}_{(X,Y)}\!\big[-\log p(X \mid Y)\big].$$

**주의**: $H(X \mid Y)$는 특정 $y$가 아니라 $Y$ 전체에 대한 **평균** 이다. 특정 $y$에서의 엔트로피 $H(X \mid Y = y)$와 혼동하지 말 것.

### 정의 3.3 — 상호정보량 (Mutual Information)

$$I(X; Y) := \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)} = D\big(p_{XY} \,\|\, p_X p_Y\big).$$

$p_X p_Y$는 독립 분포. MI는 결합 분포가 독립 분포로부터 **얼마나 벗어났는지** 를 KL로 측정. 독립일 때 $p(x, y) = p(x) p(y) \Rightarrow I = 0$.

---

## 🔬 정리와 증명

### 정리 3.1 — Chain Rule (결합 엔트로피의 분해)

**명제**:
$$H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y).$$

**증명**: 곱셈 법칙 $p(x, y) = p(x) p(y \mid x)$를 로그에 적용:
$$-\log p(x, y) = -\log p(x) - \log p(y \mid x).$$
기댓값 $\mathbb{E}_{(X, Y)}$을 취하면
$$H(X, Y) = \mathbb{E}[-\log p(X)] + \mathbb{E}[-\log p(Y \mid X)] = H(X) + H(Y \mid X).$$
대칭적으로 $H(X, Y) = H(Y) + H(X \mid Y)$. $\square$

**귀납 확장** (문서 04에서 심화): $H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i \mid X_{<i})$.

---

### 정리 3.2 — 상호정보량의 세 가지 동치 표현

**명제**: 다음 세 식은 모두 같다:
$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y).$$

**증명**:

**(I) $I(X; Y) = H(X) - H(X \mid Y)$**:
$$I(X; Y) = \sum_{x, y} p(x, y) \log \frac{p(x, y)}{p(x) p(y)} = \sum_{x, y} p(x, y) \log \frac{p(x \mid y)}{p(x)}.$$
$$= -\sum_{x, y} p(x, y) \log p(x) + \sum_{x, y} p(x, y) \log p(x \mid y) = H(X) - H(X \mid Y).$$

(첫 항: 마진화 $\sum_y p(x, y) = p(x)$로 $H(X)$. 두 번째 항: $\mathbb{E}[-\log p(X \mid Y)] = -\sum p(x, y)\log p(x\mid y)$는 $-H(X\mid Y)$의 부호 반전이므로 $-H(X\mid Y)$.)

**(II) $I(X; Y) = H(Y) - H(Y \mid X)$**: $X, Y$ 대칭성에서 자동.

**(III) $I(X; Y) = H(X) + H(Y) - H(X, Y)$**: 정리 3.1에서 $H(X \mid Y) = H(X, Y) - H(Y)$를 (I)에 대입:
$$I(X; Y) = H(X) - [H(X, Y) - H(Y)] = H(X) + H(Y) - H(X, Y). \quad \square$$

**함의**: MI는 "집합의 교집합 크기"로 볼 수 있는 벤다이어그램 해석을 수식적으로 보증.

---

### 정리 3.3 — 상호정보량의 비음수성 및 독립성

**명제**: 모든 결합 분포에 대해 $I(X; Y) \geq 0$, 등호는 $X \perp Y$일 때 정확히 성립.

**증명**: 정의에 의해 $I(X; Y) = D(p_{XY} \| p_X p_Y)$. KL의 비음수성 (Gibbs 부등식, 2장에서 완전 증명)을 적용:
$$D(p \| q) \geq 0, \quad \text{with equality iff } p = q \text{ a.s.}$$
따라서 $I(X; Y) \geq 0$이고, 등호는 $p(x, y) = p(x) p(y)$ 즉 $X \perp Y$일 때. $\square$

> Gibbs 부등식의 완전 증명은 [Ch2-01](../ch2-kl-divergence/01-kl-definition-nonnegativity.md)에서 Jensen으로 수행한다. 여기서는 결과를 사용.

---

### 정리 3.4 — 조건은 엔트로피를 감소시킨다 (Conditioning Reduces Entropy)

**명제**: $H(X \mid Y) \leq H(X)$, 등호는 $X \perp Y$일 때 정확히 성립.

**증명**: 정리 3.2(I): $H(X) - H(X \mid Y) = I(X; Y) \geq 0$ (정리 3.3)
$$\Rightarrow H(X \mid Y) \leq H(X),$$
등호는 $I(X; Y) = 0 \iff X \perp Y$. $\square$

> **중요**: 이는 "평균적으로" 감소한다는 뜻이다. 특정 $y$에서 $H(X \mid Y = y) > H(X)$도 가능. 예: $X$ = 공정 동전, $Y$ = 가끔씩 편향된 채널로 관찰. 어떤 $y$에서는 결과가 더 혼란스러울 수 있지만 **$Y$에 대한 평균**은 항상 감소.

---

### 정리 3.5 — 결합 엔트로피의 상한

**명제**:
$$\max(H(X), H(Y)) \leq H(X, Y) \leq H(X) + H(Y),$$
우측 등호는 $X \perp Y$일 때만.

**증명**:
- **하한**: $H(X, Y) = H(X) + H(Y \mid X) \geq H(X)$ ($H(Y \mid X) \geq 0$). 대칭적으로 $\geq H(Y)$.
- **상한**: $H(X, Y) = H(X) + H(Y \mid X) \leq H(X) + H(Y)$ (정리 3.4). 등호는 $X \perp Y$. $\square$

---

### 정리 3.6 — 함수 처리는 엔트로피를 줄인다

**명제**: 임의의 결정적 함수 $f$에 대해 $H(f(X)) \leq H(X)$, 등호는 $f$가 단사(1-1)일 때만.

**증명**: $Y = f(X)$로 놓으면 $p(y \mid x) = \delta_{y, f(x)}$, 즉 $H(Y \mid X) = 0$.
$$H(Y) = H(X, Y) - H(X \mid Y) = H(X) + \underbrace{H(Y \mid X)}_{=0} - H(X \mid Y) = H(X) - H(X \mid Y).$$
$H(X \mid Y) \geq 0$이므로 $H(Y) \leq H(X)$. 등호는 $H(X \mid Y) = 0$ 즉 "$Y$를 알면 $X$가 완전히 결정됨" — $f$가 단사. $\square$

**함의**: 신경망의 층은 $f(X)$로 볼 수 있으므로 "결정적 층이 정보를 늘리지 못한다" — 이는 **Data Processing Inequality** (Ch3-02)의 출발점.

---

## 💻 NumPy 구현/시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# 1. 결합 분포에서 H, H(X|Y), I(X;Y) 계산
# ─────────────────────────────────────────────

def entropy(p, base=2):
    p = np.asarray(p, dtype=np.float64).ravel()
    p = p[p > 0]
    return -np.sum(p * np.log(p)) / np.log(base)

def joint_entropy(pxy, base=2):
    return entropy(pxy, base=base)

def conditional_entropy_x_given_y(pxy, base=2):
    """H(X|Y) = H(X,Y) - H(Y)"""
    py = pxy.sum(axis=0)                     # p(y) = Σ_x p(x,y)
    return joint_entropy(pxy, base) - entropy(py, base)

def mutual_information(pxy, base=2):
    """I(X;Y) = Σ p(x,y) log[p(x,y)/(p(x)p(y))]"""
    px = pxy.sum(axis=1, keepdims=True)
    py = pxy.sum(axis=0, keepdims=True)
    pxpy = px * py
    mask = (pxy > 0) & (pxpy > 0)
    return np.sum(pxy[mask] * np.log(pxy[mask] / pxpy[mask])) / np.log(base)

# ─────────────────────────────────────────────
# 2. 예시 1: 완전 독립
# ─────────────────────────────────────────────

px = np.array([0.5, 0.5])
py = np.array([0.5, 0.5])
pxy_indep = np.outer(px, py)                  # 독립 결합

print("=" * 60)
print("Case 1: X ⊥ Y  (p(x,y) = p(x) p(y))")
print("=" * 60)
print(f"  H(X)   = {entropy(px):.4f} bits")
print(f"  H(Y)   = {entropy(py):.4f} bits")
print(f"  H(X,Y) = {joint_entropy(pxy_indep):.4f} bits   (기댓값 H(X)+H(Y) = 2.0)")
print(f"  H(X|Y) = {conditional_entropy_x_given_y(pxy_indep):.4f} bits   (기댓값 H(X) = 1.0)")
print(f"  I(X;Y) = {mutual_information(pxy_indep):.4f} bits   (기댓값 0.0)")

# ─────────────────────────────────────────────
# 3. 예시 2: 완전 결정 (Y = X)
# ─────────────────────────────────────────────

pxy_det = np.array([[0.5, 0.0],
                    [0.0, 0.5]])              # Y = X

print("\n" + "=" * 60)
print("Case 2: Y = X  (완전 결정)")
print("=" * 60)
print(f"  H(X)   = {entropy(pxy_det.sum(axis=1)):.4f}")
print(f"  H(X,Y) = {joint_entropy(pxy_det):.4f}  (기댓값 H(X) = 1.0)")
print(f"  H(X|Y) = {conditional_entropy_x_given_y(pxy_det):.4f}  (기댓값 0.0)")
print(f"  I(X;Y) = {mutual_information(pxy_det):.4f}  (기댓값 H(X) = 1.0)")

# ─────────────────────────────────────────────
# 4. 예시 3: 상관 있는 쌍 (비 독립)
# ─────────────────────────────────────────────

pxy_corr = np.array([[0.4, 0.1],
                     [0.1, 0.4]])

print("\n" + "=" * 60)
print("Case 3: 약한 상관  p(x,y) = [[0.4, 0.1], [0.1, 0.4]]")
print("=" * 60)
print(f"  H(X)   = {entropy(pxy_corr.sum(axis=1)):.4f}")
print(f"  H(Y)   = {entropy(pxy_corr.sum(axis=0)):.4f}")
print(f"  H(X,Y) = {joint_entropy(pxy_corr):.4f}")
print(f"  H(X|Y) = {conditional_entropy_x_given_y(pxy_corr):.4f}")
print(f"  I(X;Y) = {mutual_information(pxy_corr):.4f}")

# 동치 표현 검증
H_X = entropy(pxy_corr.sum(axis=1))
H_Y = entropy(pxy_corr.sum(axis=0))
H_XY = joint_entropy(pxy_corr)
H_X_given_Y = conditional_entropy_x_given_y(pxy_corr)
I_XY = mutual_information(pxy_corr)

print(f"\n  세 가지 동치 표현 검증:")
print(f"    H(X) - H(X|Y)        = {H_X - H_X_given_Y:.6f}")
print(f"    H(X) + H(Y) - H(X,Y) = {H_X + H_Y - H_XY:.6f}")
print(f"    직접 계산 I(X;Y)     = {I_XY:.6f}")

# ─────────────────────────────────────────────
# 5. 독립/상관을 parameter로 부드럽게 연결
# ─────────────────────────────────────────────

alphas = np.linspace(0, 0.5, 30)              # 0: 독립, 0.5: 완전 결정
I_values = []
H_cond = []
for a in alphas:
    # p(x,y) = (1-2a) * indep + 2a * perfect
    pxy = (1 - 2 * a) * pxy_indep + 2 * a * pxy_det
    I_values.append(mutual_information(pxy))
    H_cond.append(conditional_entropy_x_given_y(pxy))

plt.figure(figsize=(9, 5))
plt.plot(alphas, I_values, label=r'$I(X;Y)$', linewidth=2)
plt.plot(alphas, H_cond, label=r'$H(X|Y)$', linewidth=2)
plt.axhline(entropy([0.5, 0.5]), color='k', linestyle=':', label=r'$H(X) = 1$ bit')
plt.xlabel(r'Mixing parameter $\alpha$  (0: 독립, 0.5: $Y = X$)')
plt.ylabel('bits')
plt.title(r'$I(X;Y) + H(X|Y) = H(X)$ — 에너지 보존')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('03-mi-vs-conditional.png', dpi=150, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
Case 1: X ⊥ Y
  H(X)   = 1.0000
  H(X,Y) = 2.0000
  I(X;Y) = 0.0000   ← 독립

Case 2: Y = X
  H(X,Y) = 1.0000   ← 같은 정보 → 합치면 1 bit
  H(X|Y) = 0.0000   ← Y 알면 X 결정
  I(X;Y) = 1.0000   ← 공유 정보 = H(X)

Case 3: 약한 상관
  H(X) = 1.0000, H(X|Y) = 0.7219
  I(X;Y) = 0.2781

세 가지 동치 표현 검증:
  H(X) - H(X|Y)        = 0.278072
  H(X) + H(Y) - H(X,Y) = 0.278072
  직접 계산 I(X;Y)     = 0.278072
```

세 표현이 소수점 이하까지 같음을 확인 — 정리 3.2 수치 검증.

---

## 🔗 AI/ML 연결

### 분류 문제의 조건부 엔트로피

이미지 $Y$에서 레이블 $X$를 예측하는 분류기의 **비가약 오류 하한**은 $H(X \mid Y)$에 의해 결정된다. 같은 이미지 $Y$에 대해 서로 다른 레이블 $X$가 자연스럽게 존재하는 경우 (label noise), $H(X \mid Y) > 0$이고 어떤 완벽한 분류기도 0% 오류를 달성할 수 없다.

### Mutual Information Maximization (InfoMax)

표현 학습의 원칙:
$$\max_{\phi} I(X; Z), \quad Z = \phi(X).$$
"입력에 대한 정보를 최대한 보존하는 표현을 학습하라." SimCLR, CLIP, Deep InfoMax의 수학적 기반. 구체 구현은 [Ch3-05](../ch3-mutual-information/05-mi-representation-learning.md) (InfoNCE).

### Information Bottleneck

Tishby의 원리:
$$\min_\phi I(X; Z) - \beta \, I(Z; Y).$$
**두 MI가 트레이드오프**: 표현 $Z$가 입력 $X$와 너무 많은 정보를 공유하면 overfit, 타깃 $Y$와 공유 정보가 너무 적으면 underfit. [Ch6-04](../ch6-ml-applications/04-information-bottleneck.md).

### Feature Selection의 정보이론

각 feature $X_i$와 타깃 $Y$의 $I(X_i; Y)$가 큰 순서로 선택하면 **MI-based feature ranking**. Filter-based FS 방법의 대표.

### GAN의 Condition 정보

InfoGAN의 보조 손실:
$$\mathcal{L}_\text{info} = -I(c; G(z, c)),$$
생성자의 코드 $c$와 생성 이미지 사이의 MI 최대화 → $c$가 해석 가능한 latent factor가 되도록 강제.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 이산 결합 분포 | 연속 쌍은 미분 MI (문서 05, Ch3-04)로 확장, 일반적으로 MI는 연속·이산 혼합에서도 정의되나 분포 추정이 어려움 |
| 주변 분포를 알고 있음 | 실전에서는 결합 분포 $p(x, y)$를 관측 샘플로 추정 — 고차원에서 정확 추정 어려움 ("MI 추정의 저주", Ch3-04) |
| 결정적 함수 $f$에만 정리 3.6 적용 | 확률적 처리(노이즈 추가 등)도 일반적으로 정보를 늘리지 못함 — DPI (Ch3-02)에서 확장 |
| 평균의 의미 | $H(X\mid Y) \leq H(X)$는 **평균적**, 개별 $y$에서는 깨질 수 있음 |

**수치적 주의**: $p(x, y)/(p(x)p(y)) = 0$인 항은 MI 합에서 제외해야 log(0) 회피. `np.where(pxy > 0, ...)` 패턴 필수.

---

## 📌 핵심 정리

$$\boxed{I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y) \geq 0}$$

| 양 | 의미 |
|---|------|
| $H(X, Y)$ | 쌍 $(X, Y)$를 관찰하는 총 불확실성 |
| $H(X \mid Y)$ | $Y$를 알고 난 뒤 남은 $X$의 평균 불확실성 |
| $I(X; Y)$ | $X$와 $Y$가 공유하는 정보량 (= KL $(p_{XY} \\| p_X p_Y)$) |

**핵심 항등식들**:
- Chain rule: $H(X, Y) = H(X) + H(Y \mid X)$
- Conditioning reduces entropy: $H(X \mid Y) \leq H(X)$
- Function non-increase: $H(f(X)) \leq H(X)$
- MI ≥ 0, 등호 $\iff$ 독립

---

## 🤔 생각해볼 문제

**문제 1** (기초): 결합 분포
$$p(x, y) = \begin{pmatrix} 1/4 & 1/8 \\ 1/8 & 1/2 \end{pmatrix}$$
에 대해 $H(X), H(Y), H(X \mid Y), H(Y \mid X), I(X; Y)$를 모두 계산하라 (bits).

<details>
<summary>힌트 및 해설</summary>

주변 분포: $p_X = (3/8, 5/8)$, $p_Y = (3/8, 5/8)$.  
$H(X) = H(Y) = -0.375 \log_2 0.375 - 0.625 \log_2 0.625 \approx 0.954$ bits.  
$H(X, Y) = -(0.25\log_2 0.25 + 2 \cdot 0.125 \log_2 0.125 + 0.5 \log_2 0.5) = 0.5 + 0.75 + 0.5 = 1.75$ bits.  
$H(X\mid Y) = H(X, Y) - H(Y) \approx 1.75 - 0.954 = 0.796$ bits.  
대칭: $H(Y\mid X) \approx 0.796$ bits.  
$I(X; Y) = H(X) - H(X\mid Y) \approx 0.954 - 0.796 = 0.158$ bits.

</details>

---

**문제 2** (심화): $I(X; X)$는 무엇인가? 답이 $H(X)$임을 증명하고, "자기 자신과의 MI가 엔트로피다"라는 해석의 의미를 설명하라.

<details>
<summary>힌트 및 해설</summary>

$Y = X$일 때 $p(x, y) = p(x) \delta_{x, y}$.  
$I(X; X) = H(X) - H(X \mid X) = H(X) - 0 = H(X)$.  
해석: 자기 자신을 관찰하면 자기에 대한 모든 정보를 얻는다 — $X$가 **자기 자신과 공유하는 정보** 가 바로 $X$의 엔트로피. 이는 "$X$의 총 정보량 = $H(X)$"라는 소박한 직관을 MI의 언어로 재확인.

</details>

---

**문제 3** (심화): $H(X \mid Y) = 0$과 $Y$가 $X$를 **결정한다**는 명제가 동치임을 증명하라 (즉 $X = g(Y)$인 결정적 함수 $g$가 존재).

<details>
<summary>힌트 및 해설</summary>

$H(X \mid Y) = 0 \iff \sum_y p(y) H(X \mid Y = y) = 0$.  
각 항 $p(y) H(X \mid Y = y) \geq 0$이므로, $p(y) > 0$인 모든 $y$에서 $H(X \mid Y = y) = 0$.  
이는 각 $y$에서 조건부 분포 $p(x \mid y)$가 결정적, 즉 어떤 $x_y$에 대해 $p(x_y \mid y) = 1$. $g(y) := x_y$로 정의하면 $X = g(Y)$ a.s. $\square$

</details>

---

**문제 4** (AI 연결): 분류기 accuracy가 $100\%$가 되려면 어떤 정보이론적 조건이 필요한가? Fano 부등식(Ch3-03)을 찾아보기 전에, 직관적으로 답해보자.

<details>
<summary>힌트 및 해설</summary>

레이블 $X$를 입력 $Y$로부터 완벽 예측 가능 $\iff$ $X$가 $Y$의 결정적 함수 $\iff$ $H(X \mid Y) = 0 \iff I(X; Y) = H(X)$.  
즉 **입력이 레이블을 완전히 결정할 만큼 정보를 포함** 해야 함. 레이블 노이즈가 있거나 입력이 모호하면 $H(X \mid Y) > 0$이고 100% 정확도는 불가능. Fano (Ch3-03)에서 오류 확률의 정확한 하한을 $H(X \mid Y)$로 표현한다.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 02. 엔트로피 $H(X)$의 정의와 성질](./02-entropy-definition.md) | [04. Chain Rule과 정보의 계층 구조 ▶](./04-chain-rule-hierarchy.md) |

</div>
