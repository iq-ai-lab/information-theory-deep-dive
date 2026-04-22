# 6.4 Information Bottleneck — $\min I(X;Z) - \beta I(Z;Y)$

## 🎯 핵심 질문

> **"최적 표현(representation)" 을 수식으로 정의할 수 있는가?**  
> Tishby 의 답: $Z$ 가 $X$ 의 정보를 "충분히" 담되 "필요 이상" 담지 않는 점 — 즉 **$\min I(X;Z)$ subject to $I(Z;Y) \geq $ 목표**.

## 🔍 왜 이 개념이 AI 에서 중요한가

| 분야 | IB 의 역할 |
|---|---|
| **Representation learning** | "무엇이 좋은 표현인가" 의 정보이론적 정의 |
| **Generalization** | Tishby 의 "compression phase" 가설 — 학습 동역학 해석 |
| **VIB (Variational IB)** | VAE 와 거의 동일 구조, label 을 이용한 조건 |
| **β-VAE** | $\beta$ IB 와 동등한 수학적 프레임 |
| **Invariant features** | Nuisance invariance = $I(X; Z|Y) \to 0$ |
| **Sufficient statistics** | Classical IB 의 limit: minimal sufficient statistic |
| **Disentanglement** | $I(Z_i; Z_j) \to 0$ between latent dimensions |

## 📐 수학적 선행 조건

- **Mutual Information** (Ch3)
- **Data Processing Inequality**: $X \to Z \to Y$, $I(Z;Y) \leq I(X;Y)$
- **Variational bounds on MI**: MINE, InfoNCE (Ch3-04, 05)
- **Lagrangian optimization**
- **Markov chain** $Y \leftarrow X \to Z$ (IB 의 가정 구조)

## 📖 직관적 이해

### "압축" 과 "예측" 의 Trade-off

$X$ = 입력 (고차원, noisy), $Y$ = 레이블, $Z$ = 표현.

- $I(X; Z)$ 작을수록: $Z$ 는 $X$ 의 **압축된** 표현 (simpler, noise-invariant)
- $I(Z; Y)$ 클수록: $Z$ 는 $Y$ 예측에 **충분**

IB 목적:
$$
\min_{p(z|x)} \big[ I(X;Z) - \beta \, I(Z;Y) \big].
$$

$\beta$: compression vs. prediction 조정.
- $\beta \to 0$: $Z$ 는 상수 (압축 극대, 예측력 0)
- $\beta \to \infty$: $Z = X$ (예측력 최대, 압축 0)
- Intermediate: **sufficient yet minimal** 표현

### "정보 평면" (Information Plane)

Tishby 의 도식: x축 $I(X; Z)$, y축 $I(Z; Y)$.
- 가능한 모든 $Z$ 의 점은 $I(Z;Y) \leq I(X; Y)$ 하의 compact region
- IB curve: 같은 compression 수준에서 최대 prediction
- Neural network 학습 과정이 이 평면 위에서 어떻게 움직이는가?

## ✏️ 엄밀한 정의

### 정의 6.4.1 (Information Bottleneck Problem)

분포 $p(X, Y)$ 주어짐. 확률적 매핑 $p(z|x)$ 를 찾아 다음을 최소화:

$$
\mathcal{L}_\text{IB}[p(z|x)] = I(X; Z) - \beta I(Z; Y).
$$

제약: Markov chain $Y \leftarrow X \to Z$ (즉 $Z$ 는 $X$ 만의 함수, $Y$ 와 직접 연결 없음).

### 정리 6.4.2 (IB Self-Consistent Equations)

IB 의 stationary point 는 다음 세 방정식을 만족 (Tishby, Pereira, Bialek 1999):

$$
p(z|x) = \frac{p(z)}{Z(x, \beta)} \exp\left( -\beta \, D(p(y|x) \| p(y|z)) \right)
$$

$$
p(y|z) = \sum_x p(y|x) p(x|z) = \frac{1}{p(z)} \sum_x p(y|x) p(x) p(z|x)
$$

$$
p(z) = \sum_x p(x) p(z|x).
$$

(자기무결적 fixed-point iteration, Blahut-Arimoto 유사.)

### 정리 6.4.3 (IB 와 minimal sufficient statistic)

$\beta \to \infty$ 이면 IB 해는 **$Y$ 에 대한 minimal sufficient statistic** 으로 수렴.
$\beta \to 0$ 이면 $I(X;Z) \to 0$ (trivial representation).

## 🔬 Variational Information Bottleneck (VIB)

### 문제점
$I(X; Z), I(Z; Y)$ 모두 **intractable** (연속 분포에서 MI 추정).

### Alemi 2016 의 해결: Variational bounds

**Upper bound on $I(X; Z)$**:
$$
I(X;Z) = \mathbb{E}_{p(x,z)}\left[\log \frac{p(z|x)}{p(z)}\right] \leq \mathbb{E}_{p(x)}[D(p(z|x) \| r(z))],
$$
여기서 $r(z)$ 는 "variational prior" (e.g., $\mathcal{N}(0, I)$).

**Lower bound on $I(Z; Y)$**:
$$
I(Z;Y) \geq \mathbb{E}_{p(y,z)}[\log q(y|z)] + H(Y),
$$
$q(y|z)$ = variational decoder.

### VIB Objective

$$
\mathcal{L}_\text{VIB} = \underbrace{-\mathbb{E}_{p(y|x)}\mathbb{E}_{p(z|x)}[\log q(y|z)]}_{\text{classification loss}} + \beta \cdot \underbrace{\mathbb{E}_{p(x)}[D(p(z|x) \| r(z))]}_{\text{KL regularization}}
$$

**VAE 와 비교**:
- VAE: reconstruction ($x$ 재구성) + KL(posterior || prior)
- VIB: classification ($y$ 예측) + KL(encoder || marginal prior)
- 구조 동일, target 만 다름 (self-supervised vs supervised)

## 🔬 Tishby 의 "Two-Phase" 학습 가설

### 정리 6.4.4 (Tishby 2017, 논쟁적)

학습 과정이 정보 평면에서 두 단계:
1. **Fitting phase**: $I(Z; Y)$ 증가, $I(X; Z)$ 도 증가
2. **Compression phase**: $I(Z; Y)$ 유지, $I(X; Z)$ 감소

즉 처음에는 memorize, 이후 compress → generalization.

### 논쟁

- Saxe 2018 "On the information bottleneck theory of deep learning" 은 이 현상이 saturating nonlinearity (tanh) 에서만 일어남을 지적
- ReLU network 에서는 $I(X; Z) = \infty$ (deterministic mapping) 로 관찰 자체가 불가능
- 여전히 활발한 연구 주제 (MINE 기반 추정)

## 💻 VIB 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VIB(nn.Module):
    def __init__(self, d_in=784, d_hidden=256, d_latent=32, n_classes=10):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(d_hidden, d_latent)
        self.log_sigma = nn.Linear(d_hidden, d_latent)
        self.classifier = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, n_classes),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        return self.mu(h), self.log_sigma(h)
    
    def forward(self, x, n_samples=1):
        mu, log_sigma = self.encode(x)
        sigma = torch.exp(log_sigma)
        # Sample z ~ N(mu, sigma²)
        z_samples = []
        for _ in range(n_samples):
            eps = torch.randn_like(mu)
            z = mu + sigma * eps
            z_samples.append(z)
        z = torch.stack(z_samples).mean(0)   # average across samples
        logits = self.classifier(z)
        return logits, mu, log_sigma

def vib_loss(logits, y, mu, log_sigma, beta=1e-3):
    # Classification loss: -E_q log q(y|z)
    ce = F.cross_entropy(logits, y)
    # KL(q(z|x) || N(0, I))
    kl = 0.5 * torch.mean(torch.sum(mu**2 + torch.exp(2*log_sigma) - 1 - 2*log_sigma, dim=-1))
    return ce + beta * kl, ce, kl

# 학습 개념 루프 (MNIST 가정)
model = VIB()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# for x, y in dataloader:
#     logits, mu, log_sigma = model(x)
#     loss, ce, kl = vib_loss(logits, y, mu, log_sigma, beta=1e-3)
#     opt.zero_grad(); loss.backward(); opt.step()

# 6.4.A β 효과 관찰
# β가 클수록 bottleneck 강해짐 → test accuracy 감소, robustness 증가
# 작은 β: almost MLE, no compression

# 6.4.B 정보 평면 추정 (개념)
# I(X; Z) ≈ E_x [D(q(z|x) || q(z))] (aggregate)
# I(Z; Y) ≈ H(Y) - H(Y|Z) (classifier 의 log-likelihood 로 하한)
```

## 🔗 AI/ML 연결

### β-VAE = Unsupervised IB

$Y = X$ 로 두면 ($X$ 자신을 예측):
$$
\min I(X; Z) - \beta I(Z; X) = \min (1 - \beta) I(X;Z) \cdot ?
$$
Higgins 2017 의 β-VAE 는 약간 다른 재구성이지만 philosophical 동일 — compression vs reconstruction.

### Sufficient Statistics

Classical: $T(X)$ 가 $Y$ 에 대한 sufficient statistic ⟺ $I(T; Y) = I(X; Y)$.

Minimal sufficient: 추가로 $I(X; T)$ 가 최소. 
IB 의 $\beta \to \infty$ 는 이것.

### Invariant Representations

Achille 2018 "Emergence of invariance and disentanglement":
- Nuisance variable $N$ 에 대한 invariance: $I(Z; N) \to 0$
- IB 로 $Z$ 의 정보를 제한하면 자연스럽게 nuisance 감소

### Contrastive Learning ↔ IB

InfoNCE:
$$
\mathcal{L}_\text{NCE} \geq I(X; Y) - \log N.
$$
InfoMax objective 는 IB 의 second term 만. 
하지만 augmentation invariance → implicit compression (Chen 2020 SimCLR).

### Adversarial Robustness

Bottleneck 이 robust features 를 찾음: $\beta > 0$ 이 adversarial 공격에 덜 민감 (Ilyas 2019 "Adversarial Examples Are Features").

### Mutual Information Neural Estimation (MINE)

IB 의 $I(X; Z), I(Z; Y)$ 를 MINE 으로 추정 → 정보 평면 측정.
Belghazi 2018, Saxe 2018 등이 활용.

### InfoGAN

GAN 에 latent code $c$ 와 $I(c; G(z, c))$ 최대화 항 추가. IB 의 역방향 — prediction 강화.

## ⚖️ 가정과 한계

1. **MI 추정의 어려움**: 고차원에서 정확한 MI 는 계산 불가. 추정기(MINE) 도 variance 큼.
2. **Deterministic DNN 의 $I(X; Z)$**: 이론상 $\infty$ (단일값 매핑). 실무는 discretization/noise 추가로 측정.
3. **β 선택**: task 마다 최적 다름, grid search 필요.
4. **Markov 가정**: $Y \leftarrow X \to Z$ 는 "supervised" 가정. semi-supervised 확장 있음.
5. **Non-convex optimization**: IB self-consistent equations 는 local optima 에 빠질 수 있음.
6. **Two-phase 가설 의심**: Saxe 2018 이후 "information plane" 해석은 논쟁 중.

## 📌 핵심 정리

$$
\boxed{
\begin{aligned}
\min_{p(z|x)} &\big[ I(X;Z) - \beta I(Z;Y) \big]\\
\text{VIB: } &-\mathbb{E}_q \log q(y|z) + \beta \cdot D(q(z|x) \| r(z))
\end{aligned}}
$$

- **Compression-prediction trade-off** 의 정량화
- β: small ↔ accuracy priority, large ↔ compression priority
- VIB 는 VAE 의 supervised version
- IB plane = 학습 동역학의 시각화 도구 (논쟁적)

## 🤔 생각해볼 문제

### 문제 1. $\beta = 1$ 일 때 IB
$\min I(X;Z) - I(Z;Y)$ 의 의미?

<details>
<summary>해설</summary>

$I(X;Z) - I(Z;Y) = I(X;Z|Y)$ (DPI, $Y$ 와 독립인 $X$ 정보) + $[I(X;Y) - I(Z;Y)] \geq 0$ 항들.
즉 "레이블과 무관한 정보" 최소화 + "레이블 예측력 손실" 페널티. β=1 은 neutral 설정.
</details>

### 문제 2. IB ↔ Minimum Sufficient Statistic
왜 $\beta \to \infty$ 에서?

<details>
<summary>해설</summary>

$\beta \to \infty$: $I(Z;Y)$ 을 $I(X;Y)$ 와 같게 (sufficient) 유지하면서 $I(X;Z)$ 최소화 → **minimal**. 정확히 고전적 정의와 일치.
</details>

### 문제 3. Tishby two-phase 논쟁
Saxe 의 비판이 옳다면 학습은 정말 무엇을 하나?

<details>
<summary>해설</summary>

ReLU NN 에서 $I(X;Z)$ 측정이 잘못됨 (discretization sensitive). 
그러나 "compression" 이라는 개념 자체는 유효 — flat minima, Rademacher complexity 등으로 재측정.  
정확한 IB 관찰은 noisy (stochastic) NN 에서.
</details>

### 문제 4. VIB 와 β-VAE 차이
둘 다 KL regularization + reconstruction / classification. 핵심 차이?

<details>
<summary>해설</summary>

VIB: supervised. $Y$ 를 예측. Labeled data 필요.  
β-VAE: unsupervised. $X$ 재구성. Labels 없이도 disentanglement 시도.  
수식 구조는 동일.
</details>

### 문제 5. IB 로 LLM 이해
GPT 의 intermediate layer 를 IB 로 해석하면?

<details>
<summary>해설</summary>

각 layer $\ell$ 의 activation $Z_\ell$:
- Early layers: $I(X; Z_\ell)$ 크고 $I(Z_\ell; Y)$ 작음 (syntax)
- Late layers: $I(X; Z_\ell)$ 작아지고 $I(Z_\ell; Y)$ 커짐 (semantics)
- 최후 layer: prediction 에 집중된 minimal representation

Probing 실험 (Belinkov 2017) 이 부분적으로 이를 지지.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [6.3 MDL 원리](./03-mdl-principle.md) | [6.5 Diffusion ELBO](./05-diffusion-elbo.md) |

[🏠 Home](../README.md)

</div>
