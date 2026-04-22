# 02. 엔트로피 $H(X)$의 정의와 성질

## 🎯 핵심 질문

- 엔트로피 $H(X) = -\sum p(x) \log p(x)$는 왜 항상 $\geq 0$인가?
- 최대 엔트로피가 **균등분포** 에서 달성되는 이유는?
- 결정적 분포($p(x^*) = 1$)에서만 $H = 0$인가?
- Jensen 부등식은 엔트로피의 성질을 왜 자동으로 결정하는가?

---

## 🔍 왜 이 개념이 AI에서 중요한가

- **Softmax Temperature**: $\tau$를 키우면 분포가 균등해지고 엔트로피가 증가 → 탐색 강화
- **Entropy Regularization**: 강화학습(A3C, SAC)이 정책 엔트로피를 손실에 더하는 이유 — "다양성 유지"
- **레이블 스무딩(Label Smoothing)**: One-hot → $\varepsilon$-soft 분포로 바꾸면 엔트로피가 증가 → 과적합 완화
- **LLM의 Top-k/Top-p Sampling**: 샘플링 분포의 엔트로피가 곧 **생성의 창의성 척도**

엔트로피의 **하한(0)과 상한($\log|\mathcal{X}|$)** 을 알고 있으면 "이 모델이 얼마나 확신/혼란스러운지"를 절대적 스케일로 이야기할 수 있다.

---

## 📐 수학적 선행 조건

- **Jensen 부등식**: $\varphi$가 볼록이면 $\varphi(\mathbb{E}[X]) \leq \mathbb{E}[\varphi(X)]$ (오목이면 부등호 반대)
- **$-\log$의 볼록성**: $-\log x$는 $x > 0$에서 엄격하게 볼록 (2차 도함수 $1/x^2 > 0$)
- 확률 분포의 기본: $p(x) \geq 0$, $\sum_x p(x) = 1$

> Jensen 부등식은 [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-statistics-deep-dive)의 핵심 도구입니다.

---

## 📖 직관적 이해

### "평균 최적 부호 길이"

이산 분포 $p$의 엔트로피 $H(p) = -\sum p(x) \log p(x)$는 다음 해석을 가진다:

> **각 심볼 $x$를 최적으로 부호화했을 때의 평균 부호 길이**(단위 bits).

Shannon의 source coding theorem(4장)은 이 해석을 엄밀하게 만든다. 지금은 직관으로만:

- 심볼 $x$의 최적 부호 길이 ≈ $-\log_2 p(x)$ bits
- 기댓값 $\mathbb{E}[-\log p(X)] = H(X)$ = 평균 부호 길이

### 동전과 주사위

| 분포 | $H$ (bits) | 의미 |
|------|-----------|------|
| 공정한 동전 ($p = 0.5$) | $1$ | 매번 1 bit 필요 |
| 편향 동전 ($p = 0.99$) | $\approx 0.08$ | 거의 확정 → 압축 가능 |
| 공정한 주사위 (6면) | $\log_2 6 \approx 2.585$ | 평균 2.58 bits |
| 결정적 (1이 항상) | $0$ | 정보 없음 |
| 균등한 26글자 | $\log_2 26 \approx 4.7$ | 평균 4.7 bits |

**최대 엔트로피 = 균등분포** 는 "가장 예측 불가능한 상태"라는 직관과 일치한다.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Shannon 엔트로피 (이산)

이산 확률변수 $X$가 알파벳 $\mathcal{X}$ 위에서 분포 $p(x) = \Pr(X = x)$를 가진다고 하자. $X$의 **Shannon 엔트로피** 는
$$H(X) := -\sum_{x \in \mathcal{X}} p(x) \log p(x) = \mathbb{E}_{X \sim p}\!\big[-\log p(X)\big],$$
단 $p(x) = 0$인 항은 $0 \cdot \log 0 = 0$으로 규약한다 (극한 $\lim_{p \to 0^+} p \log p = 0$에서 정당화).

### 표기 주의

- 엔트로피는 분포 $p$에만 의존하므로 $H(X) = H(p)$로도 표기
- 로그 밑: 특별 언급이 없으면 **자연로그 ($\ln$)**, 단위는 nats
- bits 단위는 $\log_2$를 사용, $H_2(X) = H(X) / \ln 2$

---

## 🔬 정리와 증명

### 정리 2.1 — 비음수성

**명제**: 모든 이산 분포 $p$에 대해 $H(p) \geq 0$. 등호 $H(p) = 0$은 $p$가 결정적 분포($\exists x^* \text{ s.t. } p(x^*) = 1$)일 때 **정확히** 성립한다.

**증명**:

각 항 $-p(x) \log p(x)$를 분석한다.
- $p(x) = 0$이면 규약에 의해 항 = 0.
- $0 < p(x) \leq 1$이면 $\log p(x) \leq 0$이므로 $-p(x) \log p(x) \geq 0$.

따라서 모든 항이 $\geq 0$이고 $H(p) = \sum_x [-p(x) \log p(x)] \geq 0$.

**등호 조건**: $H(p) = 0 \iff$ 모든 $x$에서 $-p(x) \log p(x) = 0$.  
- $p(x) = 0$인 경우는 항이 자동으로 0.
- $p(x) > 0$인 경우는 $\log p(x) = 0 \iff p(x) = 1$.

즉 각 $x$에서 $p(x) \in \{0, 1\}$. 확률 합이 1이므로 유일한 $x^*$에서 $p(x^*) = 1$, 나머지에서 $p(x) = 0$. $\square$

---

### 보조정리 2.1 — $-\log$의 볼록성

**명제**: 함수 $\varphi(t) = -\log t$는 $(0, \infty)$에서 **엄격 볼록** 이다.

**증명**: $\varphi''(t) = 1/t^2 > 0$ for $t > 0$. 2차 도함수가 양수이므로 엄격 볼록. $\square$

> 이 볼록성은 **Jensen 부등식**을 통해 엔트로피·KL·MI 등 정보 측도의 비음수성·단조성 증명의 핵심 도구가 된다. 기억해두자.

---

### 정리 2.2 — 엔트로피의 상한 (균등분포가 최대)

**명제**: $|\mathcal{X}| = n$인 유한 알파벳 위에서
$$H(p) \leq \log n,$$
등호는 $p$가 균등분포($p(x) = 1/n$)일 때 **정확히** 성립한다.

**증명 (Jensen 이용)**:

Jensen 부등식을 볼록함수 $\varphi(t) = -\log t$, 확률변수 $T = 1/p(X)$ ($X \sim p$)에 적용한다.

오목함수 $\log$에 대한 Jensen (또는 볼록 $-\log$에 대한 Jensen의 부호 반전):
$$\mathbb{E}[\log T] \leq \log \mathbb{E}[T].$$

**좌변 계산**:
$$\mathbb{E}[\log T] = \mathbb{E}\!\big[\log \tfrac{1}{p(X)}\big] = -\mathbb{E}[\log p(X)] = H(p).$$

**우변 계산** (합의 순서 주의 — $p(x) = 0$인 $x$는 support에서 제외):
$$\mathbb{E}[T] = \mathbb{E}\!\big[\tfrac{1}{p(X)}\big] = \sum_{x:\, p(x) > 0} p(x) \cdot \frac{1}{p(x)} = |\{x : p(x) > 0\}| \leq n.$$

따라서
$$H(p) = \mathbb{E}[\log T] \leq \log \mathbb{E}[T] \leq \log n.$$

**등호 조건**: Jensen의 등호는 $T$가 상수인 경우, 즉 $1/p(X)$가 $X$에 무관한 값 $c$여야 한다. 이는 $p(x) = 1/c$ (상수)를 의미하고, 정규화 조건과 결합하면 $c = n$, $p(x) = 1/n$ (균등분포). 추가로 두 번째 부등식 $|\text{support}| \leq n$의 등호는 모든 $x$에 대해 $p(x) > 0$. 두 조건이 결합되면 **완전 균등분포**가 유일한 등호 도달 분포. $\square$

---

### 정리 2.3 — $-\log$ 볼록성을 이용한 대안 증명

**명제**: (정리 2.2와 동일)

**증명 (직접 계산)**:

$$\log n - H(p) = \log n + \sum_x p(x) \log p(x) = \sum_x p(x) [\log n + \log p(x)] = \sum_x p(x) \log(n p(x)).$$

이제 $q(x) := n p(x) / n = p(x)$이 아니라, 균등 분포 $u(x) = 1/n$에 대한 **KL divergence** 형태임에 주목:

$$\log n - H(p) = \sum_x p(x) \log \frac{p(x)}{1/n} = D(p \| u) \geq 0,$$

여기서 마지막 부등식은 **Gibbs 부등식**(KL $\geq 0$, 다음 챕터에서 증명). 따라서 $H(p) \leq \log n$이고, 등호는 $D(p\|u) = 0 \iff p = u$ (균등). $\square$

> **관찰**: $\log n - H(p) = D(p \| u)$는 아름다운 항등식이다. "균등분포에서 얼마나 멀리 떨어졌는가"가 곧 "엔트로피가 상한에서 얼마나 줄어들었는가"와 같다.

---

### 정리 2.4 — 엔트로피의 오목성 (Concavity)

**명제**: $H(p)$는 $p$에 대해 **오목**이다. 즉 두 분포 $p, q$와 $\lambda \in [0, 1]$에 대해
$$H(\lambda p + (1-\lambda) q) \geq \lambda H(p) + (1-\lambda) H(q).$$

**증명**:

$\varphi(t) = -t \log t$는 $(0, \infty)$에서 오목 ($\varphi''(t) = -1/t < 0$). 분포 $p, q$의 convex combination $r = \lambda p + (1-\lambda) q$에 대해 각 원소별로
$$\varphi(r(x)) = \varphi(\lambda p(x) + (1-\lambda) q(x)) \geq \lambda \varphi(p(x)) + (1-\lambda) \varphi(q(x)).$$

$x$에 대해 합하면
$$H(r) = \sum_x \varphi(r(x)) \geq \lambda H(p) + (1-\lambda) H(q). \quad \square$$

**의미**: 두 분포를 섞으면 엔트로피가 **적어도 평균만큼**은 된다 — "섞음은 불확실성을 증가시킨다"는 직관을 수식화.

---

## 💻 NumPy 구현/시뮬레이션

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import entr  # entr(x) = -x log(x)

# ─────────────────────────────────────────────
# 1. 엔트로피 계산 유틸 (nats + bits)
# ─────────────────────────────────────────────

def entropy(p, base=np.e):
    """Shannon 엔트로피. p가 0인 항은 0으로 처리."""
    p = np.asarray(p, dtype=np.float64)
    p = p[p > 0]                         # support만
    return -np.sum(p * np.log(p)) / np.log(base)

# 검증: 공정한 동전 = 1 bit
print(f"공정 동전 H = {entropy([0.5, 0.5], base=2):.4f} bits  (기댓값 1.0000)")
print(f"결정적 분포 H = {entropy([1.0, 0.0], base=2):.4f} bits  (기댓값 0.0000)")
print(f"균등 6면 주사위 H = {entropy([1/6]*6, base=2):.4f} bits  (기댓값 {np.log2(6):.4f})")

# ─────────────────────────────────────────────
# 2. 이진 엔트로피 H(p) vs p 시각화
# ─────────────────────────────────────────────

p_vals = np.linspace(1e-6, 1 - 1e-6, 500)
H_vals = -p_vals * np.log2(p_vals) - (1 - p_vals) * np.log2(1 - p_vals)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(p_vals, H_vals, linewidth=2, color='steelblue')
ax.fill_between(p_vals, 0, H_vals, alpha=0.2, color='steelblue')
ax.scatter([0.5], [1.0], color='red', s=80, zorder=5, label=r'최대: $p=0.5$, $H=1$ bit')
ax.axhline(1.0, color='r', linestyle='--', alpha=0.5)
ax.set_xlabel(r'$p$')
ax.set_ylabel(r'$H(p)$ (bits)')
ax.set_title(r'이진 엔트로피 $H(p) = -p \log_2 p - (1-p)\log_2(1-p)$')
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig('02-binary-entropy.png', dpi=150, bbox_inches='tight')
plt.show()

# ─────────────────────────────────────────────
# 3. 균등이 최대 엔트로피임을 랜덤 분포로 확인
# ─────────────────────────────────────────────

n = 10
rng = np.random.default_rng(42)
H_uniform = np.log2(n)

print(f"\n{n}차원 랜덤 분포 1000개 vs 균등분포 엔트로피")
print(f"  균등분포 H = {H_uniform:.4f} bits (이론적 상한)")
H_samples = []
for _ in range(1000):
    p = rng.dirichlet(np.ones(n))        # Dirichlet(1,...,1) = 단순체 위 균등
    H_samples.append(entropy(p, base=2))

print(f"  샘플 최대 H = {max(H_samples):.4f} (≤ {H_uniform:.4f}?)  →  {'OK' if max(H_samples) <= H_uniform + 1e-9 else 'FAIL'}")

plt.figure(figsize=(8, 4))
plt.hist(H_samples, bins=40, alpha=0.7, color='steelblue', edgecolor='white')
plt.axvline(H_uniform, color='r', linestyle='--', linewidth=2, label=f'균등분포 상한 {H_uniform:.3f}')
plt.xlabel('H (bits)')
plt.ylabel('빈도')
plt.title(f'{n}차원 랜덤 분포 1000개의 엔트로피 분포 — 상한 확인')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────
# 4. 엔트로피의 오목성 시각화 — 두 분포의 mixture
# ─────────────────────────────────────────────

p = np.array([0.9, 0.1])
q = np.array([0.1, 0.9])
lambdas = np.linspace(0, 1, 100)
H_mix = []
H_line = []
for lam in lambdas:
    r = lam * p + (1 - lam) * q
    H_mix.append(entropy(r, base=2))
    H_line.append(lam * entropy(p, base=2) + (1 - lam) * entropy(q, base=2))

plt.figure(figsize=(8, 5))
plt.plot(lambdas, H_mix, linewidth=2, label=r'$H(\lambda p + (1-\lambda)q)$')
plt.plot(lambdas, H_line, linewidth=2, linestyle='--', label=r'$\lambda H(p) + (1-\lambda) H(q)$')
plt.fill_between(lambdas, H_line, H_mix, alpha=0.2, color='green', label='오목성 gap ≥ 0')
plt.xlabel(r'$\lambda$')
plt.ylabel('Entropy (bits)')
plt.title('엔트로피의 오목성 (정리 2.4)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

**출력 예시**:
```
공정 동전 H = 1.0000 bits  (기댓값 1.0000)
결정적 분포 H = 0.0000 bits  (기댓값 0.0000)
균등 6면 주사위 H = 2.5850 bits  (기댓값 2.5850)

10차원 랜덤 분포 1000개 vs 균등분포 엔트로피
  균등분포 H = 3.3219 bits (이론적 상한)
  샘플 최대 H = 3.2892 (≤ 3.3219?)  →  OK
```

---

## 🔗 AI/ML 연결

### Softmax Temperature와 엔트로피 제어

Softmax 출력 $p_i = \exp(z_i / \tau) / \sum_j \exp(z_j / \tau)$의 엔트로피는 $\tau$에 **단조 증가**한다.
- $\tau \to 0^+$: 원-핫 → $H \to 0$
- $\tau \to \infty$: 균등 → $H \to \log n$

LLM 디코딩에서 temperature는 엔트로피 레벨을 선택하는 노브다.

### Entropy Regularization (A3C, SAC)

강화학습 손실에 정책 엔트로피 항을 추가:
$$\mathcal{L}_\pi = \mathcal{L}_\text{policy} - \alpha H(\pi(\cdot \mid s)).$$

정리 2.2에 의해 $H$의 최댓값은 $\log |\mathcal{A}|$. $-H$를 최소화하는 것은 $\pi$를 균등에 가깝게 유지하려는 압력 → **탐색 유지**. $\alpha$가 탐색·활용 균형을 통제.

### Label Smoothing

One-hot 타깃 $y$를 $\tilde{y} = (1-\varepsilon) y + \varepsilon \cdot u$ (균등분포 섞기)로 변경하면:
$$H(\tilde{y}) = H((1-\varepsilon) y + \varepsilon u) \underset{\text{정리 2.4}}{\geq} (1-\varepsilon) \cdot 0 + \varepsilon \log n = \varepsilon \log n > 0.$$

타깃 분포가 더 "혼란스러워지고" → 모델의 과도한 확신을 억제.

### Maximum Entropy RL

Soft Actor-Critic의 목적 함수:
$$J(\pi) = \mathbb{E}\!\left[\sum_t r_t + \alpha H(\pi(\cdot \mid s_t))\right].$$

**보상 + 엔트로피 보너스** → 다양한 궤적 샘플링 → robust 정책.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| **유한 알파벳** $|\mathcal{X}| < \infty$ | 가산 무한은 $\log n$ 상한이 $\infty$로 발산 — 엔트로피가 유한한지 별도 확인 필요 |
| **이산 분포** | 연속 분포는 **미분 엔트로피** (문서 05)로 이행, 음수가 될 수 있음 |
| 로그 밑 선택은 단위 | bits / nats / digits 혼용 시 수치 비교에서 오류 발생 가능 |
| $p(x) = 0$ 규약 | 이는 극한으로 정당화되지만 KL에서는 $p > 0, q = 0$에 대해 $D = +\infty$가 되므로 주의 |

**수치적 주의**: `-p * log(p)`를 $p = 0$에서 계산하면 NaN이 나오므로 `scipy.special.entr` 또는 `np.where(p > 0, -p * np.log(p), 0)` 패턴을 사용해야 한다.

---

## 📌 핵심 정리

$$\boxed{0 \leq H(p) \leq \log |\mathcal{X}|}$$

| 경계 | 도달 분포 | 의미 |
|------|-----------|------|
| $H = 0$ | 결정적 분포 (한 점에 확률 1) | "정보 없음" — 결과 예측 가능 |
| $H = \log n$ | 균등분포 | "최대 혼란" — 가장 예측 불가능 |

| 성질 | 수식 | 도구 |
|------|------|------|
| 비음수성 | $H(p) \geq 0$ | $-\log p \geq 0$ for $p \leq 1$ |
| 상한 | $H(p) \leq \log n$ | Jensen (또는 $H = \log n - D(p\|u)$) |
| 오목성 | $H(\lambda p + (1-\lambda) q) \geq \lambda H(p) + (1-\lambda) H(q)$ | $-t \log t$의 오목성 |

---

## 🤔 생각해볼 문제

**문제 1** (기초): 3개 심볼 분포 $p = (0.5, 0.3, 0.2)$의 엔트로피를 bits 단위로 계산하라. 또한 균등 3면 분포 대비 얼마나 낮은지 계산하라.

<details>
<summary>힌트 및 해설</summary>

$H(p) = -(0.5 \log_2 0.5 + 0.3 \log_2 0.3 + 0.2 \log_2 0.2)$  
$= -(0.5 \cdot (-1) + 0.3 \cdot (-1.737) + 0.2 \cdot (-2.322))$  
$= 0.5 + 0.521 + 0.464 = 1.485$ bits.  
균등 3면 분포: $\log_2 3 \approx 1.585$ bits.  
차이 $= D(p \| u) \approx 0.100$ bits.

</details>

---

**문제 2** (심화): $H(p) = \log n$이 성립할 조건이 **"$p$가 균등이다"** 와 동치임을 Jensen의 등호 조건을 사용하지 않고 직접 Lagrangian으로 보여라.

<details>
<summary>힌트 및 해설</summary>

$\max_p H(p)$ subject to $\sum_x p(x) = 1$. 라그랑지안:
$$\mathcal{L} = -\sum_x p(x) \log p(x) - \lambda\Big(\sum_x p(x) - 1\Big).$$
$\partial \mathcal{L} / \partial p(x) = -\log p(x) - 1 - \lambda = 0 \Rightarrow p(x) = e^{-1-\lambda}$ (모든 $x$에서 동일한 상수).  
정규화 $\sum p(x) = 1$ 에서 $n \cdot e^{-1-\lambda} = 1 \Rightarrow p(x) = 1/n$. 따라서 임계점은 균등분포.  
이차 조건: $\partial^2 \mathcal{L}/\partial p(x)^2 = -1/p(x) < 0 \Rightarrow$ 최대점. $H = \log n$ 달성.  
(이 방법은 문서 06 최대 엔트로피 분포의 일반 틀을 예고한다.)

</details>

---

**문제 3** (AI 연결): Softmax 출력 $z = (2, 1, 0)$에 대해 $\tau = 0.1, 1, 10$ 각각의 경우 엔트로피를 계산하고 상한 $\log_2 3$과 비교하라.

<details>
<summary>힌트 및 해설</summary>

- $\tau = 0.1$: $p \approx (0.9999, 0.00004, ...)$ → $H \approx 0.001$ bits (거의 결정적)
- $\tau = 1$: $p \approx (0.665, 0.245, 0.090)$ → $H \approx 1.10$ bits
- $\tau = 10$: $p \approx (0.37, 0.34, 0.30)$ → $H \approx 1.58$ bits (거의 균등 = $\log_2 3 \approx 1.585$)

$\tau$를 키울수록 상한 $\log_2 3$에 접근 — Softmax는 $\tau \to \infty$에서 균등으로 수렴.

</details>

---

**문제 4** (증명): $H$가 오목이면 엔트로피의 **maximum principle**: 유한 개 분포 $p_1, \ldots, p_K$와 비음수 가중치 $\alpha_i$ ($\sum \alpha_i = 1$)에 대해 $H(\sum \alpha_i p_i) \geq \sum \alpha_i H(p_i)$를 귀납으로 증명하라.

<details>
<summary>힌트 및 해설</summary>

$K = 2$: 정리 2.4에서 성립.  
$K = n$: 가중 평균 $\sum_{i=1}^{n} \alpha_i p_i = \alpha_n p_n + (1 - \alpha_n) \sum_{i=1}^{n-1} \frac{\alpha_i}{1 - \alpha_n} p_i$로 쪼개고 정리 2.4 + 귀납 가정 사용.  
이는 $H$가 "여러 분포의 mixture"에서도 항상 $\geq$ 평균 — "섞음은 정보를 증가시킨다"의 일반화.

</details>

---

<div align="center">

| | |
|---|---|
| [◀ 01. 정보의 공리적 유도](./01-axiomatic-derivation.md) | [03. 결합·조건부·상호정보량 ▶](./03-joint-conditional-mutual.md) |

</div>
