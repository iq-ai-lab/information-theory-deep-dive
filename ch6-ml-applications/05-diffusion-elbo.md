# 6.5 Diffusion Model 의 변분 한계 — ELBO 의 KL 합 분해

## 🎯 핵심 질문

> **Diffusion model 의 학습 목적이 왜 정확히 $T$ 개 KL divergence 의 합으로 분해되는가?**  
> DDPM 의 단순한 MSE loss $\|\epsilon - \epsilon_\theta\|^2$ 가 어떻게 엄밀한 ELBO 에서 유도되는가?

## 🔍 왜 이 개념이 AI 에서 중요한가

| 분야 | Diffusion ELBO 의 역할 |
|---|---|
| **DDPM / DDIM** | 학습 목표가 ELBO 각 step 의 KL |
| **Score-based generative models** | ELBO ↔ score matching 동등성 (DSM) |
| **Classifier-Free Guidance** | Conditional diffusion 의 ELBO 유도 |
| **Consistency Models** | ELBO 기반 distillation |
| **Video / 3D diffusion** | 동일 ELBO 구조 재사용 |
| **Flow Matching / Rectified Flow** | ELBO 를 ODE 기반으로 reformulate |

## 📐 수학적 선행 조건

- **ELBO** (Ch6-02): $\log p(x) \geq \mathbb{E}_q[\log p(x,z) - \log q(z|x)]$
- **Markov chain**: $x_0 \to x_1 \to \cdots \to x_T$
- **KL between Gaussians**: closed form
- **Jensen 부등식**
- **Reparameterization trick**

## 📖 직관적 이해

### Forward vs Reverse Process

**Forward** (데이터 → 잡음, 고정):
$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I).
$$
$T$ step 후: $x_T \approx \mathcal{N}(0, I)$.

**Reverse** (잡음 → 데이터, 학습):
$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t)).
$$

학습: $q(x_{t-1} | x_t, x_0)$ (true posterior) 을 $p_\theta$ 가 닮도록.

### VAE ↔ Diffusion

VAE 의 latent $z$ 가 **하나**.
Diffusion 의 "latent" 는 **sequence $x_{1:T}$** — 각 step 이 하나의 잠재변수.

$$
\text{ELBO}_\text{diff} = \mathbb{E}_q\left[ \log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)} \right].
$$

## ✏️ 엄밀한 유도

### 정의 6.5.1 (Forward Process, DDPM)

$$
q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}\, x_{t-1},\ \beta_t I), \quad t = 1, \ldots, T.
$$

$\bar\alpha_t = \prod_{s=1}^t (1 - \beta_s)$ 로 표기.

**Closed form** (marginal):
$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar\alpha_t}\, x_0,\ (1 - \bar\alpha_t) I).
$$

즉 $x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon$, $\epsilon \sim \mathcal{N}(0, I)$.

### 정의 6.5.2 (Reverse Process)

$$
p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t),\ \Sigma_\theta(x_t, t)).
$$

$p(x_T) = \mathcal{N}(0, I)$ (starting point).

### 정리 6.5.3 (핵심 ELBO 분해)

$$
\boxed{
-\log p_\theta(x_0) \leq \mathbb{E}_q \left[ \underbrace{D(q(x_T | x_0) \| p(x_T))}_{L_T} + \sum_{t=2}^T \underbrace{D(q(x_{t-1} | x_t, x_0) \| p_\theta(x_{t-1} | x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0 | x_1)}_{L_0} \right]
}.
$$

**증명.**

$\log p_\theta(x_0) \geq \text{ELBO} = \mathbb{E}_q[\log p_\theta(x_{0:T})] - \mathbb{E}_q[\log q(x_{1:T}|x_0)]$.

$p_\theta(x_{0:T}) = p(x_T) \prod_{t=1}^T p_\theta(x_{t-1}|x_t)$,  
$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$.

Bayes 재작성: $q(x_t|x_{t-1}) = \frac{q(x_{t-1}|x_t, x_0) q(x_t|x_0)}{q(x_{t-1}|x_0)}$ (Markov forward).

대입 후 telescoping:

$$
-\text{ELBO} = \mathbb{E}_q[D(q(x_T|x_0) \| p(x_T))] + \sum_{t \geq 2} \mathbb{E}_q[D(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))] - \mathbb{E}_q[\log p_\theta(x_0|x_1)].
$$

$\blacksquare$

### 정리 6.5.4 (True Posterior $q(x_{t-1} | x_t, x_0)$ — Gaussian)

Bayes + Gaussian identities:
$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde\mu_t(x_t, x_0), \tilde\beta_t I),
$$

$$
\tilde\mu_t(x_t, x_0) = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t} x_0 + \frac{\sqrt{1-\beta_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t} x_t,
$$

$$
\tilde\beta_t = \frac{1 - \bar\alpha_{t-1}}{1 - \bar\alpha_t} \beta_t.
$$

### 정리 6.5.5 (DDPM 의 단순 MSE 유도)

$\Sigma_\theta = \sigma_t^2 I$ (고정) 으로 두고, $p_\theta$ 의 평균 $\mu_\theta(x_t, t)$ 로 파라미터화.

Reparameterize: $x_0 = \frac{1}{\sqrt{\bar\alpha_t}}(x_t - \sqrt{1-\bar\alpha_t} \epsilon)$.

$$
\tilde\mu_t(x_t, x_0) = \frac{1}{\sqrt{1-\beta_t}}\left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon \right).
$$

모델: 
$$
\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}}\left( x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}} \epsilon_\theta(x_t, t) \right).
$$

두 Gaussian 의 KL:
$$
D(q \| p_\theta) = \frac{1}{2\sigma_t^2}\|\tilde\mu_t - \mu_\theta\|^2 = \frac{\beta_t^2}{2\sigma_t^2 (1-\beta_t)(1-\bar\alpha_t)} \|\epsilon - \epsilon_\theta(x_t, t)\|^2.
$$

**단순화** (Ho 2020): 계수를 무시한 weighted MSE:
$$
\boxed{\mathcal{L}_\text{simple}(\theta) = \mathbb{E}_{t, x_0, \epsilon}\left[ \|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t} \epsilon, t)\|^2 \right]}.
$$

이 목표가 full ELBO 의 **reweighted** 버전임이 유명.

### 정리 6.5.6 (Diffusion = Score Matching, Song 2021)

$s_\theta(x_t, t) = \nabla_{x_t} \log p_t(x_t)$ (score) 로 파라미터화:
$$
\epsilon_\theta(x_t, t) = -\sqrt{1 - \bar\alpha_t} \cdot s_\theta(x_t, t).
$$

Denoising Score Matching (Vincent 2011):
$$
\mathcal{L}_\text{DSM} = \mathbb{E}\left[ \|s_\theta(x_t) - \nabla_{x_t} \log q(x_t|x_0)\|^2 \right].
$$

$\nabla_{x_t} \log q(x_t|x_0) = -\frac{x_t - \sqrt{\bar\alpha_t} x_0}{1 - \bar\alpha_t} = -\frac{\epsilon}{\sqrt{1-\bar\alpha_t}}$.

→ DSM = ELBO (상수 factor 차).

## 💻 DDPM 완전 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPM:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        self.T = T
        self.beta = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)
    
    def q_sample(self, x0, t, noise=None):
        """Forward: x_t | x_0"""
        if noise is None:
            noise = torch.randn_like(x0)
        a = self.alpha_bar[t].view(-1, 1, 1, 1)  # shape broadcasts
        return torch.sqrt(a) * x0 + torch.sqrt(1 - a) * noise, noise
    
    def q_posterior_mean(self, x0, xt, t):
        """True posterior μ_t(x_t, x_0)"""
        bt = self.beta[t]
        a_bar_t = self.alpha_bar[t]
        a_bar_tm1 = self.alpha_bar[t-1] if t > 0 else torch.tensor(1.0)
        at = self.alpha[t]
        
        c1 = (torch.sqrt(a_bar_tm1) * bt) / (1 - a_bar_t)
        c2 = (torch.sqrt(at) * (1 - a_bar_tm1)) / (1 - a_bar_t)
        return c1 * x0 + c2 * xt

class SimpleUNet(nn.Module):
    """Minimal UNet-like ε_θ"""
    def __init__(self, c=64):
        super().__init__()
        self.down = nn.Conv2d(3, c, 3, stride=2, padding=1)
        self.mid = nn.Conv2d(c, c, 3, padding=1)
        self.up = nn.ConvTranspose2d(c, 3, 4, stride=2, padding=1)
        self.t_embed = nn.Linear(1, c)
    
    def forward(self, x, t):
        t_feat = self.t_embed(t.float().unsqueeze(-1) / 1000)
        t_feat = t_feat.view(-1, t_feat.shape[-1], 1, 1)
        h = F.silu(self.down(x)) + t_feat
        h = F.silu(self.mid(h)) + t_feat
        return self.up(h)

# 학습 loop (개념)
def train_step(ddpm, model, x0, opt):
    """x0: batch of images"""
    B = x0.shape[0]
    t = torch.randint(0, ddpm.T, (B,))
    xt, noise = ddpm.q_sample(x0, t)
    
    pred_noise = model(xt, t)
    loss = F.mse_loss(pred_noise, noise)    # L_simple
    
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()

# 샘플링 (generate)
@torch.no_grad()
def sample(ddpm, model, shape):
    x = torch.randn(shape)   # x_T ~ N(0, I)
    for t in reversed(range(ddpm.T)):
        ts = torch.full((shape[0],), t)
        eps = model(x, ts)
        alpha_t = ddpm.alpha[t]
        alpha_bar_t = ddpm.alpha_bar[t]
        # DDPM 평균 공식
        mean = (x - ((1-alpha_t)/torch.sqrt(1-alpha_bar_t)) * eps) / torch.sqrt(alpha_t)
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt(ddpm.beta[t])
            x = mean + sigma * noise
        else:
            x = mean
    return x

# Example:
# ddpm = DDPM(T=1000)
# model = SimpleUNet()
# opt = torch.optim.Adam(model.parameters(), lr=1e-4)
# for x0 in dataloader:
#     loss = train_step(ddpm, model, x0, opt)
```

## 🔗 AI/ML 연결

### DDPM vs DDIM

- **DDPM** (Ho 2020): stochastic reverse, $T$ step 필요 (1000 typical)
- **DDIM** (Song 2021): deterministic, ODE 형태, 10-50 step 로 샘플링

둘 다 같은 $\epsilon_\theta$ 모델 학습, inference 절차만 다름.

### Classifier-Free Guidance (CFG)

조건부 diffusion: $\epsilon_\theta(x_t, t, c)$ (text embedding 등).
- Unconditional: $\epsilon_\theta(x_t, t, \emptyset)$
- Guidance: $\hat\epsilon = (1+w)\epsilon_\theta(x_t, t, c) - w \epsilon_\theta(x_t, t, \emptyset)$
- $w$ = guidance scale (보통 3~10)
- ELBO 프레임에서 conditional + unconditional jointly learned

### Consistency Models (Song 2023)

Diffusion step 을 하나로 합침: $f_\theta(x_t, t) \to x_0$ (any $t$).
- ODE trajectory 에 대한 self-consistency
- ELBO-like loss 로 distillation

### Latent Diffusion Model (LDM, Stable Diffusion)

$x$ 를 latent $z = \mathcal{E}(x)$ 로 압축 후 $z$ 공간에서 diffusion.
- 64×64×4 latent (512² RGB 에서 ×1/64 compute)
- ELBO 는 latent 에서 계산

### Video / 3D Diffusion

같은 ELBO 구조, $x$ 차원만 확장:
- Imagen Video: $\mathbb{R}^{T \times H \times W \times 3}$
- NeRF diffusion: 3D density + color

### Flow Matching (Lipman 2022)

Probability path 의 ODE 관점에서 loss 재구성:
$$
\mathcal{L}_\text{FM} = \mathbb{E}_{t, x_1}[\|v_\theta(x_t, t) - u_t(x_t | x_1)\|^2],
$$
where $u_t$ = optimal vector field. ELBO 는 특수 case.

### Score-based SDE (Song 2021)

Continuous time: $dx = f(x, t) dt + g(t) dW$.
Reverse: $dx = [f(x, t) - g(t)^2 \nabla \log p_t(x)] dt + g(t) d\bar W$.
ELBO → 각 $t$ 에서 $\|s_\theta - \nabla \log p_t\|^2$ 적분.

## ⚖️ 가정과 한계

1. **Gaussian 가정**: Forward/reverse kernel 이 모두 Gaussian → non-Gaussian data (categorical, etc.) 에는 확장 필요 (discrete diffusion).
2. **$T$ 큼 필요**: ELBO 의 tightness 는 $T$ 클수록 좋지만 compute cost 선형 증가.
3. **Simplified loss 는 reweighted ELBO**: 엄밀한 density 는 full ELBO 로만.
4. **Intractable $\log p(x)$**: 직접 likelihood 계산 불가, ELBO 또는 ODE 기반 likelihood evaluation (DDIM).
5. **Mode coverage**: GAN 보다 좋지만 low-density modes 누락 가능.
6. **Compute cost**: inference 시 수십 step 의 network evaluation — real-time inference 어려움 (distillation 으로 완화).

## 📌 핵심 정리

$$
\boxed{
-\log p_\theta(x_0) \leq \underbrace{L_T}_{\text{prior match}} + \sum_{t=2}^T \underbrace{L_{t-1}}_{\text{step KL}} + \underbrace{L_0}_{\text{final decode}}
}
$$

- $L_T \approx 0$ (forward 가 $\mathcal{N}(0, I)$ 에 도달)
- $L_{t-1} = D(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))$ — 두 Gaussian 간 KL
- 단순화: $\mathcal{L}_\text{simple} = \mathbb{E}\|\epsilon - \epsilon_\theta\|^2$
- Score matching 과 **정확히** 동등 (상수 factor 제외)

## 🤔 생각해볼 문제

### 문제 1. $L_T$ 가 왜 0 에 가까운가?
$q(x_T|x_0)$ vs $p(x_T) = \mathcal{N}(0, I)$?

<details>
<summary>해설</summary>

$\bar\alpha_T \approx 0$ → $q(x_T|x_0) \approx \mathcal{N}(0, I) = p(x_T)$. 
따라서 $L_T \to 0$ as $T \to \infty$. 이 항은 학습 불필요.
</details>

### 문제 2. $L_0$ 의 의미
왜 따로 처리하나?

<details>
<summary>해설</summary>

$L_0 = -\log p_\theta(x_0 | x_1)$. 이산 픽셀 가정 시 discretized Gaussian (Ho 2020). 
마지막 step 의 reconstruction loss 역할, image space 로 복귀.
</details>

### 문제 3. DDPM loss 가 "weighted" ELBO 라는 의미
단순 MSE 와 full ELBO 의 차이?

<details>
<summary>해설</summary>

Ho 2020: $\mathcal{L}_\text{simple}$ 는 각 $t$ 의 KL 의 계수 ($\beta_t^2 / (2\sigma_t^2 \ldots)$) 를 1 로 간주.
실험적으로 이 reweighting 이 더 나은 샘플 품질을 준다 — full ELBO 는 likelihood 에 최적 (Kingma 2021 "Variational Diffusion").
</details>

### 문제 4. Consistency model 의 이론
$f_\theta(x_t, t) \to x_0$ 의 수렴 이유?

<details>
<summary>해설</summary>

PF-ODE (Probability Flow ODE): forward SDE 와 같은 marginal 을 주는 ODE. 
ODE 해 $\phi_t(x_0)$: $f_\theta(x_t, t) = x_0$ 은 정확한 ODE reverse.
Self-consistency: $f_\theta(x_t, t) = f_\theta(x_s, s)$ for $s < t$. → distillation.
</details>

### 문제 5. Diffusion 이 GAN 을 대체했는가?
생성 품질에서 지금 어느 쪽이 우위?

<details>
<summary>해설</summary>

2024 현재: 이미지에서 diffusion 이 dominant (StableDiffusion, FLUX, Imagen). 
GAN 은 빠른 inference 가 필요한 특정 domain (face editing, super-resolution) 에서 여전히 쓰임. 
Consistency/distillation 으로 diffusion 도 1-step generation 접근 → 차이 축소.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [6.4 Information Bottleneck](./04-information-bottleneck.md) | [6.6 Fisher 정보계량](./06-fisher-information-geometry.md) |

[🏠 Home](../README.md)

</div>
