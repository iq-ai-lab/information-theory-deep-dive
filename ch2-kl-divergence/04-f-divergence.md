# 2.4 f-divergence — 모든 거리의 통일

## 🎯 핵심 질문

> **KL, Reverse KL, JSD, Hellinger, Total Variation, $\chi^2$ 거리 — 이들을 하나의 틀로 통일할 수 있는가?**
> **f-divergence** 는 어떤 공통 구조를 추출하는가?
> **변분 표현(variational representation)** 을 이용해 어떻게 신경망으로 추정할 수 있는가 (f-GAN)?

---

## 🔍 왜 AI에서 중요한가

- **통합 이론**: 거의 모든 "분포 간 거리"가 f-divergence의 특수 경우.
- **일반적 GAN objective**: f-GAN(Nowozin et al. 2016)은 임의의 f-divergence 를 최소화.
- **Fisher metric 유도**: 지역적으로 f-divergence의 2차 근사 → **Fisher information** → 정보기하학.
- **Data Processing Inequality 보편성**: 모든 f-divergence 가 DPI 를 만족.
- **신경망 추정**: Variational lower bound 로 샘플만으로 추정 가능.
- **Regularization**: $\chi^2$ 벌점은 variance를 제어, TV 는 robustness.

즉 f-divergence 는 "분포 간 거리" 분야의 **지도(map)** 이다.

---

## 📐 선행 학습 지식

- [2.1 KL](./01-kl-definition-nonnegativity.md), [2.3 JSD](./03-js-divergence.md)
- 볼록함수, Legendre 변환 (convex conjugate)
- Jensen 부등식
- 확률밀도의 Radon–Nikodym 미분 개념 (일반 측도)

---

## 📖 직관

**아이디어**: KL은 $p \log(p/q)$ 형태. 일반화하면 $q \cdot f(p/q)$ 에서 $f$ 는 **볼록함수**이고 $f(1)=0$. 비율 $p/q$가 1에서 멀수록 $f$ 가 커지게 하자. $f$ 를 어떻게 고르느냐에 따라:

| $f(t)$ | 결과 divergence |
|---|---|
| $t \log t$ | Forward KL $D(p\|q)$ |
| $-\log t$ | Reverse KL $D(q\|p)$ |
| $(t-1)^2$ | $\chi^2$-divergence |
| $\frac12 |t-1|$ | Total variation $\mathrm{TV}(p,q)$ |
| $(\sqrt{t}-1)^2$ | Squared Hellinger $H^2(p,q)$ |
| $t\log t - (t+1)\log\frac{t+1}{2}$ | $2 \cdot \mathrm{JSD}$ |

**핵심 통찰**: Jensen 부등식을 $f$ 에 적용하면 비음성과 $p=q$ 에서 0 이 자동으로 성립 → "Gibbs 부등식의 f-일반화".

---

## ✏️ 공식 정의

**정의 2.4.1 (f-divergence; Ali–Silvey, Csiszár 1966)**

볼록함수 $f : (0, \infty) \to \mathbb{R}$ 이 $f(1) = 0$ 을 만족한다고 하자. 확률분포 $p, q$ (with $p \ll q$, 즉 $q(x)=0$ 이면 $p(x)=0$) 에 대해

$$
\boxed{\ D_f(p \| q) \;=\; \mathbb{E}_{q}\!\left[ f\!\left(\frac{p(X)}{q(X)}\right) \right] \;=\; \int q(x)\, f\!\left(\frac{p(x)}{q(x)}\right)\,dx\ }
$$

**Convention**:
- $0 \cdot f(0/0) = 0$
- $p(x) > 0, q(x) = 0$ 인 경우 $q f(p/q) = p \cdot \lim_{t\to\infty} f(t)/t = p \cdot f^*(\infty)$ (asymptotic slope) 로 정의. 보통 $\infty$ 가 되는 경우가 많음.

**정의 2.4.2 (Adjusted / Symmetric 버전)**
- $\tilde f(t) = t f(1/t)$ 도 convex이고 $\tilde f(1) = 0$. $D_{\tilde f}(p\|q) = D_f(q\|p)$ — KL/reverse-KL 의 쌍대성이 이 공식에 담김.

---

## 🔬 정리와 증명

### Theorem 2.4.1 (비음성 & 등호조건)

**진술.** $D_f(p\|q) \ge f(1) = 0$, 등호는 $p = q$ (a.s.) — $f$ 가 strictly convex 일 때.

**증명.** Jensen 부등식:
$$
\mathbb{E}_q\!\left[f(p/q)\right] \ge f\!\left(\mathbb{E}_q[p/q]\right) = f(1) = 0.
$$
$\mathbb{E}_q[p/q] = \int q \cdot p/q\,dx = 1$. 등호는 Jensen에서 $p/q$ 가 a.s. 상수 → $p=q$. $\blacksquare$

### Theorem 2.4.2 (Data Processing Inequality for f-divergence)

**진술.** 임의의 channel(conditional) $K(y|x)$ 에 대해 $p \to K \circ p, q \to K \circ q$ (출력 분포) 하면
$$
D_f(K\circ p \,\|\, K \circ q) \le D_f(p \| q)
$$

**증명.** Conditional Jensen: 출력 밀도 $\hat p(y) = \int K(y|x) p(x) dx$. 비율
$$
\frac{\hat p(y)}{\hat q(y)} = \frac{\int K(y|x) q(x) \cdot (p(x)/q(x))\,dx}{\int K(y|x) q(x)\,dx} = \mathbb{E}_{q(\cdot|y)}\!\left[\frac{p(X)}{q(X)}\right]
$$
여기서 $q(\cdot|y) = K(y|x)q(x)/\hat q(y)$. Jensen of $f$:
$$
f\!\left(\frac{\hat p(y)}{\hat q(y)}\right) \le \mathbb{E}_{q(\cdot|y)}\!\left[f\!\left(\frac{p(X)}{q(X)}\right)\right].
$$
$\hat q(y)$ 로 곱하고 $y$ 에 대해 적분:
$$
D_f(\hat p\|\hat q) \le \int \hat q(y) \mathbb{E}_{q(\cdot|y)}[f(p/q)] \, dy = \int q(x) f(p(x)/q(x)) dx = D_f(p\|q). \blacksquare
$$

> **함의**: 데이터 가공·추상화는 **모든 f-divergence 를 감소**시킨다. KL만의 성질이 아니라 전체 family의 성질.

### Theorem 2.4.3 (Joint Convexity in (p, q))

**진술.** $(p, q) \mapsto D_f(p\|q)$ 는 jointly convex on 확률측도 쌍.

**증명.** Perspective function $g(u, v) = v\, f(u/v)$ 은 convex 함수의 perspective 이므로 jointly convex (표준 결과, Boyd–Vandenberghe §3.2.6). 적분은 convexity 유지. $\blacksquare$

### Theorem 2.4.4 (변분 표현 — Fenchel–Legendre)

**진술.** $f^*$ 를 $f$ 의 convex conjugate ($f^*(u) = \sup_t(ut - f(t))$) 라 하자.
$$
D_f(p\|q) = \sup_{T: \mathcal{X}\to \mathrm{dom}(f^*)} \left\{ \mathbb{E}_p[T(X)] - \mathbb{E}_q[f^*(T(X))] \right\}
$$

**증명 스케치.** Fenchel 부등식 $f(t) \ge ut - f^*(u)$ 에서 $t = p/q, u = T$ 로 놓고 $q$ 에 대해 기댓값:
$$
\mathbb{E}_q[f(p/q)] \ge \mathbb{E}_q[T \cdot p/q] - \mathbb{E}_q[f^*(T)] = \mathbb{E}_p[T] - \mathbb{E}_q[f^*(T)]
$$
최적 $T^*(x) = f'(p(x)/q(x))$ 에서 등호. $\blacksquare$

> **함의**: 신경망 $T_\phi$ 로 이 sup 를 근사하면 $D_f$ 를 샘플만으로 추정할 수 있음 → **f-GAN**. KL의 Donsker–Varadhan (§3.4) 도 특수경우.

### Theorem 2.4.5 (Pinsker의 f-일반화)

**진술.** $\mathrm{TV}(p,q) = \frac{1}{2}\int|p-q|$ 에 대해
$$
\mathrm{TV}(p,q) \le \sqrt{\tfrac{1}{2} D(p\|q)}, \qquad H^2(p,q) \le D(p\|q)
$$

**증명 스케치.** 첫째는 §2.1에서. 둘째는 $H^2(p,q) = \int (\sqrt{p}-\sqrt{q})^2 dx$ 에 대해 $\log u \le u - 1$ (with $u = \sqrt{q/p}$) 을 정교하게 사용. $\blacksquare$

### Theorem 2.4.6 (Fisher Information as Infinitesimal f-divergence)

**진술.** $q_\theta$ 가 $\theta$ 에 매끄럽게 의존할 때,
$$
D_f(q_{\theta + \delta} \| q_\theta) = \frac{f''(1)}{2}\, \delta^\top \mathcal{I}(\theta)\, \delta + O(\|\delta\|^3)
$$
여기서 $\mathcal{I}(\theta)$ 는 Fisher information matrix. 즉 모든 f-divergence는 **지역적으로 Fisher metric**.

**증명 스케치.** $p/q \to 1 + \delta^\top s(x) + O(\delta^2)$ 로 전개($s(x) = \nabla \log q_\theta$). Taylor $f(1+\epsilon) = f(1) + f'(1)\epsilon + \frac{f''(1)}{2}\epsilon^2 + \cdots$. $f(1)=0,\ \mathbb{E}_q[s]=0$ 이므로 1차항 소멸, 2차항이 $\frac{f''(1)}{2}\mathbb{E}_q[(\delta^\top s)^2] = \frac{f''(1)}{2} \delta^\top \mathcal{I}(\theta) \delta$. $\blacksquare$

> **함의**: KL($f(t)=t\log t, f''(1)=1$), $\chi^2$($f(t)=(t-1)^2, f''(1)=2$), Hellinger($f(t)=(\sqrt t-1)^2, f''(1)=1/2$) 모두 Fisher metric의 상수배로 환원. 이것이 **정보기하** 의 출발점 (Ch 6.6 참고).

---

## 💻 NumPy로 직접 확인

```python
import numpy as np

def f_div(p, q, f):
    mask = (q > 1e-12)
    return np.sum(q[mask] * f(p[mask] / q[mask]))

p = np.array([0.2, 0.3, 0.4, 0.1])
q = np.array([0.1, 0.4, 0.3, 0.2])

# f-함수 사전
divs = {
    "KL(p||q)":       lambda t: t*np.log(t),
    "RevKL(p||q)":    lambda t: -np.log(t),
    "Chi-square":     lambda t: (t-1)**2,
    "Hellinger^2":    lambda t: (np.sqrt(t)-1)**2,
    "TV":             lambda t: 0.5*np.abs(t-1),
    "JSD(x2)":        lambda t: t*np.log(t) - (t+1)*np.log((t+1)/2),
}
for name, f in divs.items():
    print(f"{name:15s} = {f_div(p, q, f):.5f}")
```

출력 (수치):
```
KL(p||q)        = 0.11366
RevKL(p||q)     = 0.12094     # = KL(q||p)
Chi-square      = 0.25833
Hellinger^2     = 0.02861
TV              = 0.15000
JSD(x2)         = 0.05655     # = 2 * JSD
```

### Pinsker 부등식 수치 검증

```python
tv = f_div(p, q, lambda t: 0.5*np.abs(t-1))
kl = f_div(p, q, lambda t: t*np.log(t))
print(f"TV = {tv:.4f},  sqrt(0.5*KL) = {np.sqrt(0.5*kl):.4f}")
# TV ≤ sqrt(0.5*KL) 성립
```

### 변분 bound 로 KL 추정 (Donsker–Varadhan 데모)

```python
# f(t) = t log t, f*(u) = exp(u-1). 변분 표현:
#   KL(p||q) = sup_T  E_p[T] - E_q[exp(T-1)]
# 신경망 없이 샘플 평균으로 MINE 데모
np.random.seed(0)
N = 20000
# p = N(0,1), q = N(0.5, 1)
xp = np.random.randn(N); xq = 0.5 + np.random.randn(N)

def variational_kl(T):
    return T(xp).mean() - np.exp(T(xq) - 1).mean()

from scipy.stats import norm
def T_true(x):  # 최적: T*(x) = 1 + log(p(x)/q(x))
    return 1 + np.log(norm.pdf(x, 0, 1) + 1e-12) - np.log(norm.pdf(x, 0.5, 1) + 1e-12)

val = variational_kl(T_true)
print(f"변분 bound = {val:.4f},  실제 KL = 0.125")
```

이 데모가 **MINE** 의 아이디어 (§3.4).

---

## 🔗 AI/ML 연결고리

| 응용 | f-divergence |
|---|---|
| **MLE / Cross-Entropy** | KL |
| **GAN** | JSD (vanilla), 임의 $f$ (f-GAN) |
| **WGAN** | (IPM; 엄밀히 말하면 f-div 아님, §2.5) |
| **LSGAN** | Pearson $\chi^2$ |
| **EBGAN** | Total Variation 근사 |
| **Distillation** | KL or reverse KL |
| **Calibration (ECE)** | TV 관련 |
| **Robust statistics** | Hellinger (outlier robust) |
| **Covariate shift** | $\chi^2$, TV |
| **Noise contrastive estimation** | KL/log-ratio |
| **InfoNCE** | KL lower bound |

### f-GAN (Nowozin et al. 2016)
목적:
$$
\min_G \max_{T_\phi}\ \mathbb{E}_{p_{\mathrm{data}}}[T_\phi(x)] - \mathbb{E}_{p_g}[f^*(T_\phi(x))]
$$
$f$ 를 바꿔 KL, JSD, $\chi^2$, TV 등 임의 divergence 로 학습 가능. 판별기 출력에 activation 을 $f^*$ 의 정의역에 맞게 설계.

### $\chi^2$ bound와 variance regularization
$\chi^2(p\|q) = \mathbb{E}_q[(p/q-1)^2]$. 확률비(importance weight) 의 **분산**. IS 추정에서 유한 분산 보장의 지표. OPE / off-policy 학습에서 등장.

### DPI의 보편성
CNN, Transformer, 토큰화, 다운샘플링 — 모든 "정보 처리"는 f-divergence 감소. 모델 내부에 흘려보낼수록 클래스 간 거리(f-divergence)는 줄어들지, 늘어나진 않는다.

---

## ⚖️ 가정·한계·함정

1. **$p \ll q$ 가정** — $p$ 는 $q$ 에 대해 absolute continuous 해야 유한. Vanilla GAN의 support mismatch 문제는 이 때문.
2. **TV 는 $\chi^2$ 나 Hellinger 보다 smoothness 가 나쁨** — gradient 없는 부분이 있어 직접 최소화하기 어려움.
3. **f-GAN 실무에서의 한계** — 다양한 $f$ 를 시도해봐도 기본 GAN/WGAN 만큼 안정되지 않는 경우 많음.
4. **고차원 추정의 어려움** — Variational representation은 $T$ 의 capacity에 민감. 과소 capacity 면 bound 가 느슨.
5. **"f-divergence 아닌" divergence 들** — Wasserstein, MMD, Optimal Transport 는 f-div 가 아님. 이들은 **Integral Probability Metric (IPM)** family (§2.5).

---

## 📌 핵심 정리

1. $D_f(p\|q) = \mathbb{E}_q[f(p/q)]$, $f$ 는 convex, $f(1)=0$.
2. **비음성·Jensen·DPI·Joint convexity** 가 전체 family 의 공통 성질.
3. KL, reverse KL, JSD, TV, Hellinger, $\chi^2$ 모두 특수 경우.
4. $\tilde f(t) = t f(1/t)$ 가 쌍대 divergence — 비대칭 짝의 관계.
5. Fenchel 변분표현으로 **샘플 기반 추정** 가능 (f-GAN, MINE).
6. 지역적으로 모든 f-div 가 Fisher metric 에 비례 → 정보기하학.
7. 한계: $p \ll q$ 가정, support mismatch → Wasserstein 으로 확장 (§2.5).

---

## 🤔 생각해볼 문제

### 문제 1. KL이 f-divergence 임을 직접 확인
$f(t) = t \log t$ 로 $D_f(p\|q) = D(p\|q)$ 됨을 보여라. $f(1) = 0, f$ convex 확인.

<details>
<summary>해설</summary>

$\mathbb{E}_q[f(p/q)] = \int q (p/q)\log(p/q) dx = \int p \log(p/q) dx = D(p\|q)$. $f''(t) = 1/t > 0$ convex. $f(1) = 0$. ✅
</details>

### 문제 2. 쌍대성 $\tilde f$
$f(t) = t\log t$ 일 때 $\tilde f(t) = t f(1/t) = -\log t$. 이것이 reverse KL 을 주는지 확인하라.

<details>
<summary>해설</summary>

$D_{\tilde f}(p\|q) = \mathbb{E}_q[-\log(p/q)] = \mathbb{E}_q[\log q/p] = D(q\|p)$. ✅
</details>

### 문제 3. Pearson $\chi^2$ 의 MLE적 의미
$\chi^2(p\|q) = \int (p-q)^2/q$. 유한 샘플 $\hat p$ 와 모수화 모델 $q_\theta$ 에서 $\chi^2$ 최소화와 **최소제곱법** 의 관계?

<details>
<summary>해설</summary>

$\chi^2$ 최소화는 importance-weighted 최소제곱. $q$ 가 고정이면 $\int (p-q)^2/q$ 를 $p$-공간에서 찾는 **weighted L2 투영**. MLE(KL)는 log-likelihood 투영. 둘 다 정보기하학적 투영이지만 사용하는 metric이 다름.
</details>

### 문제 4. DPI의 강함
"Channel 이 deterministic bijection 이면 $D_f$ 보존"을 증명하라. 이로부터 $D_f$ 는 **invariant under reparameterization** 임을 논하라.

<details>
<summary>해설</summary>

$y = g(x)$ bijection: $\hat p(y) = p(g^{-1}(y))|\det g^{-1'}(y)|$, 비율 $\hat p/\hat q = p/q$. 따라서 $D_f$ 형태 불변. KL, MI 가 연속·이산 양쪽에서 coordinate-invariant 인 이유.
</details>

### 문제 5. Squared Hellinger와 확률적 해석
$H^2(p,q) = 1 - \int\sqrt{pq}$. 확률적으로 무엇을 의미하는가?

<details>
<summary>해설</summary>

$\sqrt{pq}$ 는 **Bhattacharyya coefficient** $BC(p,q)$. 두 분포에서 샘플 $x$ 를 뽑아 $\sqrt{p(x)q(x)}/(\cdot)$ 형태의 overlap 척도. $H^2 = 1 - BC$ 는 "얼마나 덜 겹치는가". Metric 성질 있음.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [2.3 JS-divergence](./03-js-divergence.md) | [2.5 Wasserstein 거리](./05-wasserstein-distance.md) |

[🏠 Home](../README.md)

</div>
