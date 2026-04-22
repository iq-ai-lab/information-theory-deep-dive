# 3.4 연속 MI 와 MINE — 신경망으로 정보량 추정하기

## 🎯 핵심 질문

> **연속 확률변수 $X, Y$ 의 mutual information 을 샘플만으로 어떻게 추정하는가?**
> **MINE**(Belghazi 2018) 은 왜 Donsker–Varadhan bound 로 MI 를 신경망 학습으로 바꾸는가?
> 추정기의 **bias-variance trade-off** 와 **보편적 근사성** 은 어떤 관계인가?

---

## 🔍 왜 AI에서 중요한가

- **고차원 MI 측정의 근본 난제**: density estimation 없이는 MI 직접 계산 불가.
- **Representation learning 의 메트릭**: $I(X; Z)$ 직접 측정 → auto-encoder, SSL 평가.
- **Information Bottleneck 실측 가능성**: IB 목적 $I(X;Z) - \beta I(Z;Y)$ 를 그라디언트 추정.
- **Generative 모델 분석**: VAE 의 posterior MI, GAN mode coverage.
- **신경과학**: fMRI 신호 간 MI, spike train 분석.
- **Feature selection**: 비선형 의존성까지 잡아냄 (correlation 이 못잡는 것).
- **Causality**: Granger causality 의 MI 일반화.

MINE, InfoNCE, CLUB, SMILE 등 대규모 modern ML 에서 MI 추정은 필수 도구가 되었다.

---

## 📐 선행 학습 지식

- [2.4 f-divergence 와 변분표현](../ch2-kl-divergence/04-f-divergence.md)
- [3.1 MI 정의](./01-mi-definitions.md), [3.2 DPI](./02-data-processing-inequality.md)
- Fenchel–Legendre duality, conjugate function
- Neural network 기본 (backprop, SGD)

---

## 📖 직관

### 왜 연속 MI 는 어려운가?

$I(X; Y) = \mathbb{E}_{p_{XY}}[\log p(X,Y)/(p(X)p(Y))]$.

샘플만 있고 밀도 미지:
- Histogram: 차원 저주.
- Kernel density estimation (KDE): 대역폭 선택 민감, 고차원 실패.
- k-NN (Kozachenko–Leonenko, Kraskov): 100차원까지는 가능하지만 이상 복잡한 의존성은 놓침.

**새로운 아이디어**: $I$ 를 **변분 bound** 로 써서 **신경망이 최적화** 하도록 만든다 → MINE.

### Donsker–Varadhan Representation

KL 의 변분 표현:
$$
D(p \| q) = \sup_{T} \mathbb{E}_p[T] - \log \mathbb{E}_q[e^T]
$$

$T$ 는 $\mathcal{X} \to \mathbb{R}$ 인 아무 함수. 최적 $T^*(x) = \log p/q + C$.

MI 에 적용:
$$
I(X; Y) = D(p_{XY} \| p_X p_Y) = \sup_T \mathbb{E}_{p_{XY}}[T(X, Y)] - \log \mathbb{E}_{p_X p_Y}[e^{T(X, Y)}]
$$

$T$ 를 신경망 $T_\theta$ 로 잡고 이 bound 를 maximize → $I(X;Y)$ 하한의 tightest estimator.

---

## ✏️ 공식 정의

**정의 3.4.1 (Donsker–Varadhan DV bound)**
$$
\boxed{\ I_{\mathrm{DV}}(X;Y) = \sup_T\ \mathbb{E}_{p_{XY}}[T(X,Y)] - \log \mathbb{E}_{p_X p_Y}[e^{T(X,Y)}]\ }
$$

**정의 3.4.2 (MINE)**
$T_\theta$ 를 신경망으로 파라미터화. 미니배치 $B$ 에 대해
$$
\hat I_{\mathrm{MINE}} = \frac{1}{|B|}\sum_{(x,y)\in B} T_\theta(x,y) - \log\!\left(\frac{1}{|B'|}\sum_{(x,y')\in B'} e^{T_\theta(x, y')}\right)
$$
- $B$: joint 샘플 (진짜 pair)
- $B'$: marginal 샘플 ($y'$ 를 배치 내 permutation)

**정의 3.4.3 (f-divergence variational bound)**
Nowozin-style: $\sup_T \mathbb{E}_{p_{XY}}[T] - \mathbb{E}_{p_Xp_Y}[f^*(T)]$ ($f(t) = t \log t - (t+1)\log\frac{t+1}{2}$ 가 JSD 대응, 등).

**정의 3.4.4 (InfoNCE lower bound)**
$$
I(X; Y) \ge \log K - \mathcal{L}_{\mathrm{InfoNCE}}
$$
여기서
$$
\mathcal{L}_{\mathrm{InfoNCE}} = -\mathbb{E}\left[\log \frac{f(x_0, y_0)}{\frac{1}{K}\sum_{j=0}^{K-1} f(x_0, y_j)}\right]
$$
$f(x, y) = e^{T(x, y)}$ 형태, $y_0$ 는 실제 pair, $y_1, \ldots, y_{K-1}$ 은 marginal 에서 샘플.

**정의 3.4.5 (CLUB: contrastive log-ratio upper bound)**
$$
I_{\mathrm{CLUB}}(X; Y) = \mathbb{E}_{p_{XY}}[\log q_\theta(y|x)] - \mathbb{E}_{p_X}\mathbb{E}_{p_Y}[\log q_\theta(y|x)]
$$
MI 의 **상한** 추정기 (MINE 은 하한). 상한 필요시 사용 — IB 에서 $I(X; Z)$ 를 minimize 할 때.

---

## 🔬 정리와 증명

### Theorem 3.4.1 (Donsker–Varadhan)

**진술.** 임의의 측정 가능 함수 $T$ 에 대해
$$
D(p \| q) \ge \mathbb{E}_p[T] - \log \mathbb{E}_q[e^T]
$$
등호는 $T^* = \log(p/q)$ (scaled).

**증명.** Fenchel-type: $\log \mathbb{E}_q[e^T] = \log \int q e^T = \log \int p e^T \cdot (q/p)$. Jensen with $\log$:
$$
\log \mathbb{E}_q[e^T] = \log \mathbb{E}_p[e^T (q/p)] \ge \mathbb{E}_p[\log(e^T q/p)] = \mathbb{E}_p[T] - D(p\|q)
$$
정리하면 $D(p\|q) \ge \mathbb{E}_p[T] - \log \mathbb{E}_q[e^T]$. $\blacksquare$

### Theorem 3.4.2 (MINE consistency)

**진술.** $T_\theta$ 의 capacity 가 충분히 크고 training converges 하면 MINE 추정기는 진짜 MI 로 수렴:
$$
\hat I_{\mathrm{MINE}} \xrightarrow{N \to \infty} I(X; Y)
$$

**증명 스케치.** Universal approximation: $T_\theta$ 는 $\log p_{XY}/(p_Xp_Y)$ 에 임의 근사 가능. DV bound 의 gap 이 수렴. (Belghazi et al. 2018, Theorem 2)

### Theorem 3.4.3 (InfoNCE lower bound)

**진술.**
$$
I(X; Y) \ge \log K - \mathcal{L}_{\mathrm{InfoNCE}}
$$
$K$ 가 커질수록 bound 가 타이트.

**증명 스케치.** Oord et al. (2018). 분모의 log-sum-exp 를 DV 와 유사한 방식으로 bound. 자세한 유도는 §3.5 에서.

### Theorem 3.4.4 (CLUB upper bound)

**진술.**
$$
I(X; Y) \le \mathbb{E}_{p_{XY}}[\log q(y|x)] - \mathbb{E}_{p_Xp_Y}[\log q(y|x)]
$$
임의의 $q(y|x)$ 에 대해, 등호는 $q = p(y|x)$.

**증명 스케치.** $I(X;Y) = \mathbb{E}_{p_{XY}}[\log p(y|x)/p(y)]$. $q$ 를 plug in 하고 Gibbs 부등식으로 bound. (Cheng et al. 2020)

### Theorem 3.4.5 (Bias 의 논리적 한계, McAllester-Stratos 2020)

**진술.** $I(X;Y) \ge B$ 인 MI 를 **$2^B$ 이하** 의 샘플로는 **어떤 추정기** 도 안정적으로 얻을 수 없다.

**결과**: 큰 MI 값 (≥ 수십 bits) 은 구조적으로 추정 불가능. MINE, InfoNCE 가 고차원에서 MI 를 underestimate 하는 근본 이유.

**증명 스케치.** 하한 증명은 Fano + sample complexity. 자세한 건 원논문.

### Theorem 3.4.6 (Variance of MINE gradient)

**진술.** DV bound 에서 gradient 는 exponential moving estimator 를 이용한 biased 추정으로 구현 (원 MINE 논문). 그렇지 않으면 $e^T$ 의 분산이 폭증해 학습 불안정.

**대안**: Nguyen–Wainwright–Jordan (NWJ) bound
$$
I \ge \mathbb{E}_{p_{XY}}[T] - \mathbb{E}_{p_X p_Y}[e^{T-1}]
$$
(biased 이나 분산 낮음). SMILE (Song & Ermon 2020) 은 DV 와 NWJ 를 결합한 balanced estimator.

---

## 💻 NumPy / PyTorch 로 직접 확인

### 간단한 MINE 구현 (PyTorch)

```python
import torch
import torch.nn as nn
import numpy as np

# Gaussian: (X, Y) ~ N(0, [[1, rho], [rho, 1]])
# True MI = -0.5 * log(1 - rho^2)

rho = 0.8
true_mi = -0.5 * np.log(1 - rho**2)
print(f"True MI = {true_mi:.4f}")

N = 10000
rng = np.random.default_rng(0)
Z = rng.normal(size=(N, 2))
L = np.array([[1, 0], [rho, np.sqrt(1 - rho**2)]])
XY = Z @ L.T
X, Y = XY[:, 0:1], XY[:, 1:2]

X = torch.tensor(X, dtype=torch.float32)
Y = torch.tensor(Y, dtype=torch.float32)

class T_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1))
    def forward(self, x, y):
        return self.fc(torch.cat([x, y], dim=1))

T = T_net()
opt = torch.optim.Adam(T.parameters(), lr=1e-3)
batch_size = 1024

for step in range(3000):
    idx = torch.randint(0, N, (batch_size,))
    idx_shuffle = torch.randperm(batch_size)
    xb, yb = X[idx], Y[idx]
    yb_shuffle = yb[idx_shuffle]

    t_joint = T(xb, yb)
    t_marg  = T(xb, yb_shuffle)
    # DV bound
    mi_lower = t_joint.mean() - torch.logsumexp(t_marg, dim=0) + np.log(batch_size)

    loss = -mi_lower
    opt.zero_grad(); loss.backward(); opt.step()

    if step % 500 == 0:
        print(f"step {step:5d}  MINE estimate = {mi_lower.item():.4f}  (true {true_mi:.4f})")
```

출력(대표):
```
step     0  MINE estimate = 0.0102  (true 0.5108)
step   500  MINE estimate = 0.3214  (true 0.5108)
step  1000  MINE estimate = 0.4520  (true 0.5108)
step  2000  MINE estimate = 0.4987  (true 0.5108)
step  2999  MINE estimate = 0.5030  (true 0.5108)
```

### k-NN estimator (Kraskov–Stögbauer–Grassberger)

```python
from sklearn.neighbors import NearestNeighbors
from scipy.special import digamma

def KSG_MI(X, Y, k=3):
    # Kraskov 2004 estimator
    XY = np.hstack([X, Y])
    N, dX = X.shape; dY = Y.shape[1]
    # Find k-NN distance in joint space
    nn_xy = NearestNeighbors(n_neighbors=k+1, metric='chebyshev').fit(XY)
    d_xy, _ = nn_xy.kneighbors(XY)
    eps_i = d_xy[:, k]
    # Count neighbors in X and Y marginal balls
    nn_x = NearestNeighbors(metric='chebyshev').fit(X)
    nn_y = NearestNeighbors(metric='chebyshev').fit(Y)
    nx = np.array([len(nn_x.radius_neighbors([x], radius=e, return_distance=False)[0]) - 1 for x, e in zip(X, eps_i)])
    ny = np.array([len(nn_y.radius_neighbors([y], radius=e, return_distance=False)[0]) - 1 for y, e in zip(Y, eps_i)])
    return digamma(k) - np.mean(digamma(nx + 1) + digamma(ny + 1)) + digamma(N)

# 실행 (소규모 샘플)
X_np = XY[:2000, 0:1]; Y_np = XY[:2000, 1:2]
print(f"KSG MI estimate = {KSG_MI(X_np, Y_np, k=3):.4f} (true {true_mi:.4f})")
```

### 다양한 추정기 비교

| 방법 | 장점 | 단점 |
|---|---|---|
| Histogram | 단순 | 차원 저주 |
| KDE | smooth | 대역폭 선택, 고차원 실패 |
| k-NN (KSG) | non-parametric, 중간 차원 강함 | 복잡한 구조 놓침, $O(N^2)$ |
| MINE (DV) | 고차원, neural flexibility | bias, 분산, training cost |
| InfoNCE | 안정, contrastive 자연 | $\log K$ 제한 상한 |
| CLUB | upper bound (IB 용) | $q(y\|x)$ 모델 필요 |
| SMILE | DV/NWJ 균형 | 하이퍼파라미터 추가 |

---

## 🔗 AI/ML 연결고리

### 1. Deep InfoMax (DIM, Hjelm 2019)
Representation $Z = f(X)$ 의 $I(X; Z)$ 를 MINE 으로 최대화 → 고품질 self-supervised 표현.

### 2. Contrastive Predictive Coding (CPC, Oord 2018)
InfoNCE objective 자체가 MI lower bound. 시계열 / 이미지 patch 의 $I(\mathrm{context}; \mathrm{future})$ 최대화.

### 3. Variational Information Bottleneck (VIB, Alemi 2017)
$\min I(X; Z) - \beta I(Z; Y)$. $I(X;Z)$ 는 CLUB upper bound 로 minimize, $I(Z;Y)$ 는 MINE/InfoNCE lower bound 로 maximize.

### 4. Neural Compressor
Rate-distortion: rate = $I(X; Z)$ (encoded bits). 신경망이 정보 손실 거의 없이 rate 제어.

### 5. Disentanglement 측정
VAE의 latent 각 차원 $z_i$ 간 $I(z_i; z_j)$ 를 MINE 으로 최소화 → 축별 독립성 강제.

### 6. GAN Mode Collapse 탐지
$I(\mathrm{noise}; \mathrm{output})$ 를 MINE 으로 추정 → 낮으면 mode collapse.

### 7. Continual Learning
Task 간 $I$ 추정으로 forgetting 이 정보량 손실임을 정량화.

---

## ⚖️ 가정·한계·함정

1. **Bias 에 의한 overestimation** — DV bound 의 log-sum-exp 가 sample-average 로 근사될 때 유한 batch bias. 큰 $I$ 값에서 심각 (Theorem 3.4.5).
2. **분산 폭주** — $e^{T}$ 가 heavy-tail → gradient 분산 큼. Clipping, temperature, NWJ variant 로 완화.
3. **Capacity vs Overfitting** — $T_\theta$ 가 너무 유연하면 data 에 overfit → bound 가 불신할 만큼 tight. Regularization 필수.
4. **$\log K$ cap for InfoNCE** — InfoNCE bound 의 최대값 $\log K$ (batch size). 진짜 MI 가 $\log K$ 보다 크면 무조건 underestimate.
5. **Lower vs Upper bound 분별** — MINE 은 하한, CLUB 은 상한. 부등식 방향 혼동 금지.
6. **High-dim discrete** — "binning 후 MI" 는 bin size 에 매우 민감. 카테고리 수 많은 언어데이터 조심.
7. **Stationarity** — 샘플이 iid 아니면 추정 편향.

---

## 📌 핵심 정리

1. **DV bound**: $I(X;Y) \ge \mathbb{E}_{p_{XY}}[T] - \log \mathbb{E}_{p_Xp_Y}[e^T]$.
2. **MINE**: $T_\theta$ 를 신경망, DV bound 를 maximize.
3. **InfoNCE**: $I \ge \log K - \mathcal{L}_{\mathrm{NCE}}$, batch 기반 실용적 하한.
4. **CLUB**: upper bound (IB minimize 용).
5. **Bias vs Variance**: DV 는 biased (log-sum-exp), NWJ 는 unbiased 이나 variance 큼.
6. **고차원 MI 큰 값은 구조적으로 추정 불가** (Theorem 3.4.5).
7. 적용: DIM, CPC, VIB, fairness adversarial, disentanglement.

---

## 🤔 생각해볼 문제

### 문제 1. DV bound 증명 재유도
$\log \mathbb{E}_q[e^T] \ge \mathbb{E}_p[T] - D(p\|q)$ 를 Fenchel 형태로 증명.

<details>
<summary>해설</summary>

$f(u) = u \log u - u + 1$ 의 conjugate $f^*(v) = e^v - 1$. 따라서 $D(p\|q) = \sup_v \mathbb{E}_p[v] - \mathbb{E}_q[e^v - 1] = \sup_v \mathbb{E}_p[v] - \mathbb{E}_q[e^v] + 1$. $v' = v - \log\mathbb{E}_q[e^v]$ 로 normalization → $D \ge \mathbb{E}_p[T] - \log \mathbb{E}_q[e^T]$.
</details>

### 문제 2. InfoNCE의 $\log K$ cap
InfoNCE bound 가 왜 $\log K$ 이상 올릴 수 없는가?

<details>
<summary>해설</summary>

$\mathcal{L}_{\mathrm{NCE}} \ge 0$ 이므로 $I \ge \log K - 0 = \log K$ 가 최대. 실제로 $K-1$ 개의 negative sample 이 marginal 에서 오기에 batch 크기가 대안 수를 제한. $K=65536$ (MoCo) 같은 큰 batch 필요성.
</details>

### 문제 3. Gaussian 에서 MINE vs KSG 비교
코드로 $\rho = 0.3, 0.6, 0.9$ 에서 두 추정기 비교. 어느 쪽이 더 정확한가?

<details>
<summary>해설</summary>

저차원 Gaussian 에서는 KSG 가 매우 정확. MINE 은 neural net training overhead 에 비해 덜 정확. 하지만 고차원/복잡 의존성에서는 MINE 이 압승. 저차원 → KSG, 고차원 → MINE 이 rule of thumb.
</details>

### 문제 4. Bias 제어
MINE 의 EMA (exponential moving average) 기법 의 필요성.

<details>
<summary>해설</summary>

원 MINE 은 gradient 의 분모 $\mathbb{E}_q[e^T]$ 를 EMA 로 추정. 이유: batch-wise estimate 가 biased ($\log \hat E \ne \log E$), gradient of $\log(\text{sample mean})$ 이 noisy. EMA 로 smoothing. 그러나 theoretical bias 존재.
</details>

### 문제 5. Fairness 적용
예측 $\hat Y$ 와 민감속성 $S$ 간 $I(\hat Y; S) = 0$ 을 강제. MINE 을 어떻게 adversarial training 에 활용?

<details>
<summary>해설</summary>

Generator $f$: $X \to \hat Y$, Critic $T$: $(\hat Y, S) \to \mathbb{R}$. $T$ 는 $I(\hat Y; S)$ 를 estimate (MINE loss 최대화), $f$ 는 accuracy 유지 + $T$ 의 estimate 를 최소화 (adversarial). Equilibrium 에서 $I(\hat Y; S) \approx 0$ 달성.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [3.3 Fano 부등식](./03-fano-inequality.md) | [3.5 MI와 표현학습](./05-mi-representation-learning.md) |

[🏠 Home](../README.md)

</div>
