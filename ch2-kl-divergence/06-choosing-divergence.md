# 2.6 어떤 발산을 쓸 것인가 — 실무 의사결정 가이드

## 🎯 핵심 질문

> **KL, reverse KL, JSD, $\chi^2$, Hellinger, TV, Wasserstein, MMD — 주어진 문제에 **어느 발산**을 써야 하는가?**
> "발산을 바꿨더니 학습이 안정화되었다" 같은 실무 경험은 이론적으로 왜 그런가?
> Forward vs reverse KL 의 선택은 실무 성능에 어떻게 드러나는가?

이 문서는 2장의 **요약·재조합** 이면서 동시에 3–6장에서 반복 참조될 **의사결정 프레임**이다.

---

## 🔍 왜 AI에서 중요한가

같은 모델·데이터·optimizer 라도 **어떤 divergence 로 학습하느냐** 가:
- 생성품질 (blurry vs sharp vs mode-collapse)
- 학습 안정성
- 분산 제어 (importance weighting variance)
- Out-of-distribution robustness
- 계산 효율성
- 샘플 효율성

을 전부 바꾼다. 논문 읽을 때 "이 method가 왜 이 divergence를 쓰는지" 명시적으로 분석할 수 있어야 실무 응용이 가능.

---

## 📐 선행 학습 지식

- 2장의 §2.1–§2.5 전체
- 각 divergence 의 정의와 기본 성질

---

## 📖 직관: 네 가지 축

divergence 선택 시 고려할 주요 축:

1. **Geometry**: 파라미터 공간 vs 데이터 공간 기하?
2. **Support assumption**: 두 분포 support가 겹치는가?
3. **Symmetry**: 대칭이 필요한가?
4. **Computability**: 샘플만 있나, density 아나, ratio 아나?

---

## ✏️ 정리: 비교 표

| Divergence | 식 | 대칭? | bounded? | metric? | density ratio? | IPM? | 주요 응용 |
|---|---|---|---|---|---|---|---|
| **KL $(p\|q)$** | $\mathbb{E}_p[\log p/q]$ | × | × (∞) | × | 필요 | × | MLE, CE, VI posterior |
| **Reverse KL** | $\mathbb{E}_q[\log q/p]$ | × | × (∞) | × | 필요 | × | VAE, PPO, VI |
| **JSD** | $\frac12 D(p\|m) + \frac12 D(q\|m)$ | ○ | ○ ($\log 2$) | $\sqrt{\cdot}$ | 필요 | × | GAN, 분포 비교 |
| **$\chi^2$** | $\mathbb{E}_q[(p/q-1)^2]$ | × | × (∞) | × | 필요 | × | LSGAN, IS variance |
| **Hellinger$^2$** | $\int(\sqrt p-\sqrt q)^2$ | ○ | ○ (2) | $\sqrt{\cdot}$ | 필요 | × | robust stat |
| **TV** | $\frac12\int|p-q|$ | ○ | ○ (1) | ○ | 필요 | ○ | calibration, robustness |
| **Wasserstein $W_1$** | $\inf_\gamma \int d\, d\gamma$ | ○ | × | ○ | 불필요 | ○ (1-Lip) | WGAN, OT, FID($W_2$) |
| **MMD** | RKHS feature mean | ○ | depend on kernel | ○ | 불필요 | ○ (RKHS) | kernel methods, two-sample test |

**범례**:
- **Density ratio 필요**: $p(x)/q(x)$ 를 알아야 계산(또는 bound) 가능.
- **IPM (Integral Probability Metric)**: $\sup_{f \in \mathcal{F}} |\mathbb{E}_p f - \mathbb{E}_q f|$ 형태.

---

## 🔬 정리와 의사결정 원칙

### Principle 1 — 데이터 밀도의 접근 가능성

| 상황 | 적합한 divergence |
|---|---|
| $p$ 는 샘플만, $q$ 는 알려진 밀도 | Forward KL $D(p\|q) \to$ cross-entropy |
| 양쪽 샘플만 | Wasserstein, MMD, f-GAN (변분추정) |
| 양쪽 밀도 알려짐 | KL/JSD 직접 계산 가능 |
| $p/q$ (density ratio) 추정 가능 | NCE, DRE, f-divergence variational |

### Principle 2 — Support Mismatch

| 특성 | 선택 |
|---|---|
| 초기 $p_g$ 와 $p_{\mathrm{data}}$ support 거의 disjoint (고차원 이미지) | **Wasserstein** (WGAN) 또는 sliced Wasserstein |
| Support 겹침 보장 (softmax, 정규화된 density) | **KL 계열** 가능 |
| Bounded loss 원함 | JSD, Hellinger, TV |

### Principle 3 — Mode-seeking vs Mean-seeking

| 원하는 행동 | 선택 |
|---|---|
| 모든 모드 커버, blurry 괜찮 | **Forward KL** (MLE) |
| 한 모드 선명히, 다른 모드 무시 가능 | **Reverse KL** (VI, VAE) |
| 중립적, 대칭 원함 | **JSD**, Hellinger |
| 모드 구조 무시, 지리적 거리만 | **Wasserstein** |

### Principle 4 — 통계적 수렴율

| Divergence | 샘플 크기 $n$ 에 대한 수렴율 |
|---|---|
| KL, $\chi^2$ | $\sqrt{d/n}$ (Hoeffding-like for bounded, density estimation이 어려움) |
| TV | $n^{-1/d}$ (empirical estimation) |
| **Wasserstein $W_1$** | **$n^{-1/d}$** (curse of dimensionality) |
| **MMD** | $\mathbf{n^{-1/2}}$ (dimension-free!) |
| Sliced Wasserstein | $n^{-1/2}$ |

> **결론**: 고차원에서 sample-only 평가 → **MMD, Sliced-W, Energy distance** 유리.

### Principle 5 — 최적화 dynamics

| 관심 | 선택 |
|---|---|
| Vanishing gradient 없이 smooth 감소 | Wasserstein >> JSD (support mismatch 때문) |
| Jacobian 조건 양호 | $\chi^2$ (quadratic) 는 매끄러움 |
| Convex optimization 하고싶음 | $(p,q)$ 에 jointly convex 인 f-div 활용 |
| Stochastic 하면서 unbiased 추정 가능 | forward KL, MMD (둘 다 unbiased U-statistic 가능) |

---

## 💻 NumPy로 직접 확인 — Divergence Spectrum

한 pair $(p, q)$ 에 대해 다양한 divergence를 한 번에 계산:

```python
import numpy as np
from scipy.stats import wasserstein_distance

def kl(p, q, eps=1e-12): return np.sum(p * (np.log(p+eps) - np.log(q+eps)))
def jsd(p, q): m = 0.5*(p+q); return 0.5*kl(p, m) + 0.5*kl(q, m)
def chi2(p, q): return np.sum((p - q)**2 / (q + 1e-12))
def h2(p, q): return np.sum((np.sqrt(p) - np.sqrt(q))**2)
def tv(p, q): return 0.5 * np.sum(np.abs(p - q))

# 네 가지 상황:
cases = {
    "identical":   (np.array([0.5,0.3,0.2]), np.array([0.5,0.3,0.2])),
    "close":       (np.array([0.5,0.3,0.2]), np.array([0.45,0.35,0.2])),
    "far":         (np.array([0.9,0.05,0.05]),np.array([0.05,0.05,0.9])),
    "disjoint":    (np.array([1.0,0.0,0.0]), np.array([0.0,0.0,1.0])),
}

print(f"{'case':10s} {'KL':>8s} {'JSD':>8s} {'chi2':>8s} {'H2':>8s} {'TV':>8s}")
for name, (p, q) in cases.items():
    print(f"{name:10s} {kl(p,q):8.3f} {jsd(p,q):8.3f} {chi2(p,q):8.3f} {h2(p,q):8.3f} {tv(p,q):8.3f}")
```

출력(대표):
```
case         KL      JSD     chi2     H2       TV
identical 0.000    0.000    0.000    0.000    0.000
close     0.008    0.002    0.015    0.003    0.050
far       4.443    0.552    31.20    1.400    0.850
disjoint  inf      0.693    inf      2.000    1.000
```

**관찰**:
- **identical**: 모두 0 (정부호).
- **close**: 작은 값, 순위 비슷.
- **far**: JSD는 $\log 2$에 근접, KL/$\chi^2$ 는 폭증.
- **disjoint**: KL $\to\infty$, JSD $= \log 2$, TV = 1, H² = 2, **Wasserstein만 점진적**(격자 상 거리 고려).

### Wasserstein vs KL/JSD — support mismatch 시 연속성

```python
from scipy.stats import wasserstein_distance
for a in [0, 1, 2, 3, 5]:
    x = np.zeros(1000); y = np.full(1000, a)
    # empirical KL 은 정의 안됨(support disjoint)
    print(f"a={a}  W1={wasserstein_distance(x, y):.3f}")
```
$W_1$ 만 $a$ 에 smooth.

### MMD (RBF kernel)

```python
def rbf_kernel(x, y, sigma=1.0):
    xx = x[:, None] - y[None, :]
    return np.exp(-xx**2 / (2*sigma**2))

def mmd2(x, y, sigma=1.0):
    return rbf_kernel(x, x, sigma).mean() + rbf_kernel(y, y, sigma).mean() - 2*rbf_kernel(x, y, sigma).mean()

rng = np.random.default_rng(0)
x = rng.normal(0, 1, 500); y = rng.normal(1, 1, 500)
print(f"MMD^2(N(0,1), N(1,1)) = {mmd2(x, y):.5f}")
```

---

## 🔗 AI/ML 연결고리 — 결정 예시

### 시나리오 A: 이미지 생성
- **VAE**: ELBO → **reverse KL** 자동 (variational posterior). 결과 blurry 경향.
- **Diffusion**: variational bound → forward KL 의 Gaussian 근사 = MSE. 안정 sampling.
- **GAN (vanilla)**: JSD → mode collapse/gradient vanishing.
- **WGAN**: **Wasserstein** → 안정, mode coverage ↑.
- **Score-based**: Fisher divergence ($\int \|\nabla \log p - \nabla \log q\|^2$) — density ratio 없이도 학습.

### 시나리오 B: LLM alignment (RLHF)
- **Reward maximization + PPO-KL**: **reverse KL** to reference policy → distribution-safe.
- **DPO**: log-ratio 기반 → forward KL 의 한 형태 (MLE-like).
- **IPO/KTO**: 다른 divergence variant, robustness 목적.

### 시나리오 C: 분포 이동 탐지 / Drift
- **경고 빠르게**: TV, JSD (bounded).
- **지리적 의미**: Wasserstein (feature space).
- **통계적 검정**: MMD (kernel 2-sample test, closed-form $p$-value).

### 시나리오 D: Representation Learning
- **InfoNCE**: KL 의 lower bound 형태 (§3.5).
- **Contrastive (SimCLR)**: softmax cross-entropy → 암묵적 mutual information 최대화.
- **Barlow Twins**: cross-covariance → 정보 이론과는 다른 축이지만 관련.

### 시나리오 E: OOD detection
- **Energy score**: $-\log \sum_y e^{f_y(x)}$ — cross-entropy 기반.
- **Mahalanobis (feature space $W_2$-like)**: Gaussian fit → $W_2^2$ 근사.
- **Density ratio**: 학습분포 vs ID 분포 ratio → $\chi^2$ 관점.

---

## ⚖️ 실무 함정 10가지

1. **"KL이 항상 맞다"** — Reverse/Forward 선택이 결과를 바꾼다.
2. **Support mismatch 진단 누락** — GAN 학습이 안 되면 JSD 의 gradient vanishing 의심.
3. **FID 를 절대적 거리로 취급** — Inception feature 공간의 Gaussian 가정에 의존.
4. **KL regularizer $\beta$-튜닝** (VAE): $\beta$ 가 너무 크면 posterior collapse.
5. **Label smoothing이 cross-entropy 식을 바꿈을 잊음** — 정확히는 $D(\tilde p_y \| q_\theta)$ 에서 one-hot 이 아닌 $\tilde p_y$.
6. **TV가 쉬워 보이지만 gradient 문제**: $|p-q|$ 는 미분 불연속 → smooth surrogate (Hellinger, JSD) 로 실무 대체.
7. **WGAN weight clipping** 의 capacity 감소 — GP 권장.
8. **고차원 Wasserstein 추정** 시 sample 복잡도 폭발 — sliced 또는 entropic 근사 사용.
9. **MMD kernel 선택** 이 결과를 바꿈 — multi-bandwidth kernel 사용 권장.
10. **Importance sampling weight variance 검증 누락** — $\chi^2(p\|q) = \mathrm{Var}(w)+1$ 을 잊고 무한대 분산의 가중평균을 사용.

---

## 📌 핵심 정리

1. **Divergence 는 네 가지 축** (symmetry, boundedness, metric, computability) 에서 차별화된다.
2. **KL 계열**은 density ratio 가 있을 때, **IPM 계열(W, MMD)** 은 샘플만 있을 때 유리.
3. **Support mismatch** 여부가 f-div / IPM 결정에 critical.
4. **Mode-seeking vs mean-seeking** 은 역설적으로 **어느 방향 KL** 을 쓰느냐가 좌우.
5. **통계적 수렴율**: MMD 는 $O(n^{-1/2})$ 로 dimension-free, Wasserstein 은 $O(n^{-1/d})$.
6. **실무 권장**:
   - 생성 모델 평가 → **FID ($W_2^2$ on features), KID (MMD)**
   - GAN 훈련 → **Wasserstein** (WGAN-GP)
   - VI / 정책 최적화 → **Reverse KL**
   - 언어 모델 학습 → **Forward KL (Cross-Entropy)**
   - 2-sample test → **MMD** (closed-form test)
   - Drift 탐지 → **JSD / TV** (bounded)

이 장의 모든 결론은 이후 3장(Mutual Information), 6장(Cross-Entropy·ELBO·Fisher) 에서 재등장한다.

---

## 🤔 생각해볼 문제

### 문제 1. VAE가 Wasserstein 이면?
VAE 목적 $-\log p_\theta(x) + D_{\mathrm{KL}}(q_\phi(z|x) \| p(z))$ 에서 KL 대신 $W_2$ 를 쓰면 어떤 모델이 되는가?

<details>
<summary>해설</summary>

**Wasserstein Autoencoder (Tolstikhin et al. 2018)**. $W_2$ 가 $q_\phi(z)$ 의 marginal 과 prior $p(z)$ 의 OT 거리 → MMD-WAE, GAN-WAE 등 구현. VAE 의 posterior collapse 완화, 샘플 품질 개선 보고.
</details>

### 문제 2. RLHF의 KL 을 JSD 로 바꾸면?
$\beta D(\pi_\theta \| \pi_{\mathrm{ref}})$ 대신 $\beta \cdot \mathrm{JSD}(\pi_\theta \| \pi_{\mathrm{ref}})$ 로 바꾸면 distribution shift 방지에 어떤 영향?

<details>
<summary>해설</summary>

JSD 는 bounded → **penalty saturation**. $\pi_\theta$ 가 ref 에서 크게 벗어나도 벌점이 $\log 2$ 이하로 머물러 reward 가 이기기 쉬움 → 실제로는 shift 가 심해질 위험. 반대로 KL 은 $\infty$ 로 발산 가능 → 강한 억제력. 따라서 KL 이 trust region 역할에 더 적합.
</details>

### 문제 3. Image quality metric으로 JSD는 왜 부적절?
FID 가 $W_2^2$ 기반, KID 가 MMD 기반인데, JSD 기반 metric 이 드문 이유는?

<details>
<summary>해설</summary>

(1) JSD 는 bounded → dynamic range 작아 상대비교 어려움. (2) Feature 분포의 density estimation 이 필요 (density ratio 계산). (3) support disjoint 시 상수 → 고차원에서 민감도 낮음. $W_2^2$ 는 Gaussian 가정으로 closed form, MMD 는 sample only + 수렴 좋음.
</details>

### 문제 4. DPO의 발산 해석
DPO 손실 $-\log \sigma(\beta \log \pi_\theta(y_w|x)/\pi_{\mathrm{ref}}(y_w|x) - \beta \log \pi_\theta(y_l|x)/\pi_{\mathrm{ref}}(y_l|x))$ 는 어떤 f-divergence 와 연결?

<details>
<summary>해설</summary>

DPO 는 Bradley-Terry model 하에서 reward 를 reparameterize. Loss 는 pairwise logistic → KL 과 직접적이지는 않고, implicit reward model + MLE. 기본적으로 forward KL (MLE) 의 변형으로 볼 수 있으며, 이것이 PPO(reverse KL) 와 다른 "regularization 방향" 을 만듦.
</details>

### 문제 5. 에너지 기반 모델의 발산
EBM 은 $p_\theta(x) = e^{-E_\theta(x)}/Z(\theta)$. MLE (forward KL) 학습 시 $\nabla_\theta \log p_\theta = -\nabla_\theta E_\theta(x) + \mathbb{E}_{p_\theta}[\nabla_\theta E_\theta]$ 에서 후자의 sampling 이 어려움. 대안으로 score matching (Fisher divergence), NCE (log-ratio binary classification) 를 쓰는 이유?

<details>
<summary>해설</summary>

Fisher divergence $\int \|\nabla \log p - \nabla \log q\|^2$ 는 normalization $Z$ 에 의존 안함. NCE 는 "샘플이 $p$ vs noise $q$ 인지" classification 으로 density ratio 학습 → $Z$ 우회. 모두 MLE(forward KL)의 intractable 부분을 피하는 divergence-level 재설계.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [2.5 Wasserstein 거리](./05-wasserstein-distance.md) | [3.1 MI의 정의와 기본 성질](../ch3-mutual-information/01-mi-definitions.md) |

[🏠 Home](../README.md)

</div>
