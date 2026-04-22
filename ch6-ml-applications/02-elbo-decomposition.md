# 6.2 ELBO 의 정보이론적 분해

## 🎯 핵심 질문

> **$\log p(x) = \text{ELBO}(q) + D(q(z|x) \| p(z|x))$ — 왜 이 항등식이 VAE 의 모든 것을 설명하는가?**  
> ELBO 가 왜 정확히 **reconstruction − KL regularizer** 로 나뉘며, 두 항이 각각 어떤 정보이론적 의미를 가지는가?

## 🔍 왜 이 개념이 AI 에서 중요한가

| 모델 | ELBO 의 역할 |
|---|---|
| **VAE** (Kingma 2013) | 손실 자체가 ELBO. "Reconstruction + KL" 로 학습 |
| **β-VAE / Disentanglement** | KL 계수 $\beta$ 조정으로 latent 정보량 제어 |
| **VQ-VAE** | Discrete latent + commitment loss, ELBO 프레임에 적합 |
| **Diffusion (DDPM)** | 학습 목적이 $T$ step ELBO 의 합 (Ch6-05) |
| **IWAE / Multi-sample ELBO** | Tighter bound via importance weighting |
| **Amortized inference** | Encoder 공유 → 변분추정의 computational 효율 |

**깊은 통찰**: ELBO 는 "evidence $\log p(x)$ 를 하한"하는 도구가 아니라 **정확한 분해** — lower-bound gap 이 항상 $D(q \| p_\text{posterior})$.

## 📐 수학적 선행 조건

- **KL divergence 비음수성** (Ch2-01)
- **Bayes 정리**: $p(z|x) = p(x|z)p(z)/p(x)$
- **Chain rule**: $\log p(x, z) = \log p(x|z) + \log p(z)$
- **Jensen 부등식**
- **Expectation under variational distribution**

## 📖 직관적 이해

### 왜 "variational"?

$p(z|x)$ (true posterior) 은 대부분의 경우 **분석적으로 계산 불가**:
$$
p(z|x) = \frac{p(x|z) p(z)}{p(x)} = \frac{p(x|z) p(z)}{\int p(x|z) p(z) dz}.
$$
분모 $p(x)$ 는 intractable.

**Variational Inference** (VI): $p(z|x)$ 를 다루기 쉬운 $q_\phi(z|x)$ 로 근사. 최적화 대상을 만들 필요 → ELBO.

### ELBO 분해의 의미

$$
\text{ELBO}(q) = \underbrace{\mathbb{E}_q[\log p(x|z)]}_{\text{reconstruction}} - \underbrace{D(q(z|x) \| p(z))}_{\text{regularization to prior}}.
$$

- Reconstruction: "$z$ 로부터 $x$ 를 얼마나 잘 복원하나"
- KL: "Posterior $q(z|x)$ 가 prior $p(z)$ 로부터 얼마나 떨어져 있나"

두 힘의 균형:
- KL 항이 없으면 encoder 는 $x$ 에 대한 정보를 모두 $z$ 에 압축 (useless latent)
- Reconstruction 항이 없으면 $q \to p(z)$ (useless encoder)

## ✏️ 엄밀한 정의와 분해

### 정의 6.2.1 (ELBO)

관측 변수 $x$, 잠재 변수 $z$, 생성 모델 $p_\theta(x, z) = p_\theta(x|z) p(z)$, 변분분포 $q_\phi(z|x)$.

$$
\text{ELBO}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x, z)] - \mathbb{E}_{q_\phi(z|x)}[\log q_\phi(z|x)].
$$

동등하게:
$$
\text{ELBO} = \mathbb{E}_q[\log p(x|z)] - D(q(z|x) \| p(z)).
$$

### 정리 6.2.2 (핵심 항등식)

$$
\boxed{\log p(x) = \text{ELBO}(x; \theta, \phi) + D(q_\phi(z|x) \| p_\theta(z|x))}.
$$

**증명.**

$D(q(z|x) \| p(z|x))$ 전개:
$$
D(q \| p(z|x)) = \mathbb{E}_q[\log q(z|x)] - \mathbb{E}_q[\log p(z|x)].
$$

$\log p(z|x) = \log p(x, z) - \log p(x)$ 대입 (Bayes):
$$
= \mathbb{E}_q[\log q(z|x)] - \mathbb{E}_q[\log p(x, z)] + \log p(x)
$$
(마지막 항은 $z$ 에 무관, $\mathbb{E}_q$ 빠져나옴)

$$
= -\text{ELBO} + \log p(x).
$$

재배치:
$$
\log p(x) = \text{ELBO} + D(q \| p(z|x)). \qquad \blacksquare
$$

**따름**:
- $D \geq 0$ → $\text{ELBO} \leq \log p(x)$ (lower bound)
- Gap = $D(q \| p(z|x))$ (true posterior 와의 distance)
- $q = p(z|x)$ 이면 gap = 0, ELBO = $\log p(x)$ (정확히 달성)

### 정리 6.2.3 (ELBO 의 두 가지 분해)

**분해 1 (VAE 관점)**:
$$
\text{ELBO} = \underbrace{\mathbb{E}_q[\log p(x|z)]}_{\text{reconstruction log-likelihood}} - \underbrace{D(q(z|x) \| p(z))}_{\text{KL to prior}}.
$$

**분해 2 (EM / expected complete likelihood)**:
$$
\text{ELBO} = \mathbb{E}_q[\log p(x, z)] + H(q) = \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q].
$$

**증명**:
$\text{ELBO} = \mathbb{E}_q[\log p(x|z) + \log p(z) - \log q(z|x)]$.  
$= \mathbb{E}_q[\log p(x|z)] + \mathbb{E}_q[\log p(z) - \log q(z|x)]$  
$= \mathbb{E}_q[\log p(x|z)] - D(q \| p(z))$. $\blacksquare$

### 정리 6.2.4 (ELBO 와 Mutual Information — β-VAE 해석)

ELBO 를 dataset 평균으로 쓰면:
$$
\mathbb{E}_{p_D(x)}[\text{ELBO}] = \mathbb{E}_{p_D(x)}[\mathbb{E}_{q(z|x)}[\log p(x|z)]] - \mathbb{E}_{p_D(x)}[D(q(z|x) \| p(z))].
$$

두 번째 항 (aggregate KL) 은 다음과 같이 재구성:
$$
\mathbb{E}_{p_D}[D(q(z|x) \| p(z))] = I_q(X; Z) + D(q(z) \| p(z)),
$$
여기서 $q(z) = \mathbb{E}_{p_D(x)}[q(z|x)]$ 는 aggregate posterior.

**해석**: KL 페널티는 두 요소로 분해 —
- $I_q(X;Z)$: "encoder 가 $x$ 의 정보를 $z$ 에 얼마나 담는가" (압축과 관련)
- $D(q(z) \| p(z))$: "aggregate posterior 가 prior 에 얼마나 맞나"

β-VAE: $\beta > 1$ → KL 에 가중 → $I(X;Z)$ 강하게 제한 → disentanglement 유도.

## 🔬 Reparameterization Trick

### 문제
$\nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f(z)]$ 를 어떻게 추정?

"직접" 은 고분산 (score function estimator, REINFORCE). → reparameterization.

### 풀이 (Gaussian case)
$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$:
$$
z = \mu_\phi(x) + \sigma_\phi(x) \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
$$

$$
\nabla_\phi \mathbb{E}_{\epsilon}[f(\mu_\phi + \sigma_\phi \epsilon)] = \mathbb{E}_\epsilon[\nabla_\phi f(\mu_\phi + \sigma_\phi \epsilon)].
$$

→ pathwise gradient, 저분산.

## 💻 VAE 완전 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, d_in=784, d_hidden=256, d_latent=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(d_hidden, d_latent)
        self.log_sigma = nn.Linear(d_hidden, d_latent)
        self.dec = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_in), nn.Sigmoid()   # Bernoulli output
        )
    
    def encode(self, x):
        h = self.enc(x)
        return self.mu(h), self.log_sigma(h)
    
    def reparameterize(self, mu, log_sigma):
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        return mu + sigma * eps
    
    def forward(self, x):
        mu, log_sigma = self.encode(x)
        z = self.reparameterize(mu, log_sigma)
        x_recon = self.dec(z)
        return x_recon, mu, log_sigma

def vae_loss(x, x_recon, mu, log_sigma, beta=1.0):
    """
    -ELBO = -recon + β·KL
    KL for N(μ, σ²) vs N(0, I):
      (1/2) Σ (μ² + σ² - 1 - log σ²)
    """
    # Reconstruction (Bernoulli CE)
    recon = F.binary_cross_entropy(x_recon, x, reduction='sum')
    # KL closed form
    sigma_sq = torch.exp(2 * log_sigma)
    kl = 0.5 * torch.sum(mu**2 + sigma_sq - 1 - 2*log_sigma)
    return recon + beta * kl, recon, kl

# 학습 루프 (개념 — MNIST 가정)
model = VAE()
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# for x in dataloader:
#     x_recon, mu, log_sigma = model(x)
#     loss, recon, kl = vae_loss(x, x_recon, mu, log_sigma)
#     opt.zero_grad(); loss.backward(); opt.step()

# 6.2.A KL 항등식 검증: closed-form vs 샘플링
torch.manual_seed(0)
mu = torch.tensor([0.5, -0.3])
log_sigma = torch.tensor([0.1, -0.2])
sigma = torch.exp(log_sigma)

# Closed form
kl_closed = 0.5 * torch.sum(mu**2 + sigma**2 - 1 - 2*log_sigma)

# Monte Carlo
n_samples = 100000
eps = torch.randn(n_samples, 2)
z = mu + sigma * eps
# log q(z) - log p(z)
log_q = -0.5*((z - mu)/sigma)**2 - torch.log(sigma) - 0.5*torch.log(torch.tensor(2*torch.pi))
log_p = -0.5 * z**2 - 0.5*torch.log(torch.tensor(2*torch.pi))
kl_mc = (log_q.sum(-1) - log_p.sum(-1)).mean()

print(f"KL closed-form: {kl_closed.item():.4f}")
print(f"KL Monte Carlo: {kl_mc.item():.4f}")

# 6.2.B ELBO 항등식 검증: log p(x) = ELBO + D(q || p(z|x))
# 간단 예: p(x, z) = N(z, 1), p(z) = N(0, 1), q(z|x) = N(μ_q, σ_q²)
# 이 경우 true posterior p(z|x) = N(x/2, 1/2)

x = torch.tensor(2.0)
mu_q, sigma_q = torch.tensor([1.0]), torch.tensor([1.0])
# ELBO = E_q[log p(x|z)] - D(q || p(z))
# log p(x|z) = -0.5(x-z)² - 0.5 log(2π)
# Sample ELBO
n = 100000
eps = torch.randn(n)
z_samples = mu_q + sigma_q * eps
log_lik = -0.5*(x - z_samples)**2 - 0.5*torch.log(torch.tensor(2*torch.pi))
# KL(q || N(0, 1))
kl_prior = 0.5*(mu_q**2 + sigma_q**2 - 1 - 2*torch.log(sigma_q))
elbo = log_lik.mean() - kl_prior

# True log p(x): marginalize z → N(0, 2)
log_px = -0.5*x**2/2 - 0.5*torch.log(torch.tensor(4*torch.pi))

# D(q || p(z|x)) — KL between two Gaussians
mu_p_zx, sigma_p_zx = x/2, (0.5)**0.5
kl_to_posterior = torch.log(torch.tensor(sigma_p_zx)/sigma_q) + \
    (sigma_q**2 + (mu_q - mu_p_zx)**2)/(2 * sigma_p_zx**2) - 0.5

print(f"\nlog p(x)          = {log_px.item():.4f}")
print(f"ELBO              = {elbo.item():.4f}")
print(f"D(q||p(z|x))      = {kl_to_posterior.item():.4f}")
print(f"ELBO + KL         = {(elbo + kl_to_posterior).item():.4f}  (should equal log p(x))")
```

## 🔗 AI/ML 연결

### VAE = Generative Autoencoder

Standard autoencoder 는 deterministic. VAE 의 차별점:
- Encoder 는 분포 $q(z|x)$ 출력 (μ, σ)
- Latent 에서 **샘플링** → generate 가능
- KL 항이 latent space 를 "continuous and meaningful" 하게 만듦 → interpolation, generation

### β-VAE 와 Disentanglement

$\beta = 1$: 표준 VAE (ELBO 그대로).  
$\beta > 1$: KL 강조 → $I(X; Z)$ 제한 → factor 가 disentangled.

$$
\mathcal{L}_\beta = \mathbb{E}_q[\log p(x|z)] - \beta \cdot D(q(z|x) \| p(z)).
$$

Higgins 2017: $\beta = 4 \sim 16$ 에서 latent 축이 "회전", "크기", "색상" 등으로 분리.

### IWAE (Importance Weighted Autoencoder)

Tighter bound: $K$ 샘플의 log-mean-exp:
$$
\mathcal{L}_K = \mathbb{E}_{z_{1:K} \sim q}\left[ \log \frac{1}{K}\sum_k \frac{p(x, z_k)}{q(z_k|x)} \right].
$$

$K = 1$: ELBO. $K \to \infty$: $\log p(x)$. 
→ IWAE 는 ELBO 와 true likelihood 사이의 interpolation.

### Diffusion 의 ELBO

DDPM: $p_\theta(x_0) \geq$ ELBO = $\sum_{t=1}^T L_t$.
각 $L_t = D(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$ — KL form.  
Ch6-05 에서 상세.

### Rate-Distortion 과 VAE

Alemi 2018 "Fixing a broken ELBO": β-VAE 의 KL 제약은 **rate**, reconstruction 은 **distortion** → rate-distortion curve 상의 점.

### Amortized Inference

Encoder $q_\phi(z|x)$ 가 모든 $x$ 에 대해 공유 — "amortization". 
전통적 VI 는 각 $x$ 마다 별도 $\phi$ 최적화 → $N$ 배 비용. VAE 의 핵심 기여.

## ⚖️ 가정과 한계

1. **ELBO 는 lower bound 일 뿐**: gap $D(q \| p(z|x))$ 만큼 suboptimal.  
2. **Posterior collapse**: KL → 0 (encoder 가 $x$ 무시) 하는 degenerate 해, powerful decoder 에서 빈번.
3. **Gaussian assumption**: Encoder/prior 가 대부분 $\mathcal{N}$ → multi-modal posterior 표현 불가.
   - 해결: Normalizing flow, hierarchical VAE
4. **Blurry reconstructions**: pixel CE 는 L2 loss 와 유사 → 고차원 이미지에서 blur.
   - 해결: VQ-VAE (discrete latent), VAE-GAN hybrid, Diffusion
5. **β 선택**: β-VAE 의 disentanglement 는 annealing schedule, architectural bias 등에 민감.

## 📌 핵심 정리

$$
\boxed{
\begin{aligned}
\log p(x) &= \text{ELBO}(q) + D(q(z|x) \| p(z|x)) \\
\text{ELBO} &= \mathbb{E}_{q(z|x)}[\log p(x|z)] - D(q(z|x) \| p(z)) \\
\mathbb{E}_{p_D}[\text{ELBO}] &= \text{recon} - I_q(X;Z) - D(q(z) \| p(z))
\end{aligned}}
$$

- Lower-bound gap = posterior approximation error
- Reconstruction ↔ KL 균형
- β 조정 → mutual information 제어 → disentanglement
- Reparameterization trick → low-variance gradient

## 🤔 생각해볼 문제

### 문제 1. ELBO 의 tightness
$q = p(z|x)$ 이면 ELBO = $\log p(x)$. 왜?

<details>
<summary>해설</summary>

$D(q \| p(z|x)) = D(p(z|x) \| p(z|x)) = 0$. → gap 0.  
단 $p(z|x)$ 는 intractable, 실용은 근사만 가능.
</details>

### 문제 2. Posterior collapse 의 원인
VAE 에서 KL = 0 으로 수렴하는 현상의 수식적 해석?

<details>
<summary>해설</summary>

Decoder $p(x|z)$ 가 매우 표현력이 크면 (e.g., autoregressive) $z$ 없이도 $x$ 를 복원 가능 → KL 페널티 줄여서 $q(z|x) \to p(z)$ 가 local minimum.  
해결: KL annealing (초기에 $\beta$ 작게), free bits, architectural constraint.
</details>

### 문제 3. ELBO 최대화와 log-likelihood 최대화
두 objective 가 같은가?

<details>
<summary>해설</summary>

No. $\log p(x) = \text{ELBO} + D$. 
$\theta$ 가 둘 다 영향을 주면 ELBO maximization 이 $\log p(x)$ maximization 과 diverge 가능. 
단 $q = p(z|x; \theta)$ 이면 $D = 0$ → 동일 (EM 알고리즘).
</details>

### 문제 4. β = 0 vs β → ∞
극단의 해석?

<details>
<summary>해설</summary>

$\beta = 0$: KL 무시 → encoder 자유롭게 정보 저장 → overfit deterministic AE.  
$\beta \to \infty$: KL dominant → $q \to p(z)$ → uninformative latent, generation 망가짐.  
Sweet spot 은 task 마다 다름.
</details>

### 문제 5. ELBO 가 아닌 upper bound?
$\log p(x)$ 의 upper bound 를 구성할 수 있나?

<details>
<summary>해설</summary>

Yes: CUBO (Dieng 2017), $\chi$-divergence 사용. 이론적으로 가능하지만 unbiased gradient 추정이 어려워 실무적으로 덜 쓰임.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [6.1 Cross-Entropy & MLE](./01-cross-entropy-mle.md) | [6.3 MDL 원리](./03-mdl-principle.md) |

[🏠 Home](../README.md)

</div>
