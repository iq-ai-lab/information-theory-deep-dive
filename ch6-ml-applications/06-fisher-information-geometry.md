# 6.6 Fisher Information 과 정보 기하 입문

## 🎯 핵심 질문

> **KL divergence 의 **2차 근사** 가 왜 정확히 Fisher Information 행렬로 나타나는가?**  
> 이 "정보 기하" 관점에서 **Natural Gradient**, **K-FAC**, **Amari-Chentsov** 정리는 어떻게 등장하는가?

Fisher Information 은 통계학, 정보이론, 미분기하학을 잇는 다리다.  
다음 레포 **Information Geometry Deep Dive** 의 진입점.

## 🔍 왜 이 개념이 AI 에서 중요한가

| 분야 | Fisher 의 역할 |
|---|---|
| **Natural Gradient** | Fisher-scaled gradient, Amari 1998 |
| **K-FAC** | Layer-wise Fisher 근사, Martens 2015 |
| **Elastic Weight Consolidation** | Continual learning, Fisher 로 중요도 가중 |
| **Laplace Approximation** | Posterior 의 Gaussian 근사 (curvature) |
| **Meta-learning (MAML)** | Natural gradient 기반 second-order |
| **Model Merging** | Fisher-weighted averaging |

## 📐 수학적 선행 조건

- **KL divergence** (Ch2-01)
- **Taylor 전개**
- **Hessian, gradient**
- **Cramér-Rao bound** (수리통계 기초)
- **Riemannian 기하 기초** (manifold, tangent space — 간단히)

## 📖 직관적 이해

### Fisher = "파라미터 변화에 얼마나 민감한가"

$p_\theta(x)$ 파라미터 모델. $\theta$ 가 조금 바뀌면 likelihood 가 얼마나 바뀌나?

$$
I(\theta) = \mathbb{E}_{p_\theta}\left[ \left(\frac{\partial \log p_\theta(X)}{\partial \theta}\right)^2 \right].
$$

큰 $I$ → $\theta$ 의 작은 변화가 likelihood 에 큰 영향. 즉 "민감한" 파라미터.

### Amari 의 통찰

$\theta \in \mathbb{R}^k$ 공간은 단순 Euclidean 이 아니라 **통계 다양체**.  
각 점 $\theta$ 에서 Fisher $I(\theta)$ 가 **metric tensor** 역할:
$$
ds^2 = d\theta^\top I(\theta) d\theta.
$$

이 metric 이 바로 **KL divergence 의 무한소 근사**.

## ✏️ 엄밀한 정의

### 정의 6.6.1 (Fisher Information Matrix)

Score function: $\ell(\theta; x) = \log p_\theta(x)$.

$$
I(\theta) = \mathbb{E}_{p_\theta}\left[ \nabla_\theta \ell(\theta; X) \, \nabla_\theta \ell(\theta; X)^\top \right] = -\mathbb{E}_{p_\theta}[\nabla^2_\theta \ell(\theta; X)].
$$

($I(\theta)$ 는 positive semi-definite $k \times k$ 행렬.)

두 형식의 동등성 (정리 6.6.2 에서 증명).

### 정리 6.6.2 (두 정의의 동등성)

$$
\mathbb{E}[\nabla \ell \nabla \ell^\top] = -\mathbb{E}[\nabla^2 \ell].
$$

**증명.**
$\nabla \log p_\theta = \frac{\nabla p_\theta}{p_\theta}$. $\mathbb{E}[\nabla \log p_\theta] = \int p_\theta \cdot \frac{\nabla p_\theta}{p_\theta} dx = \nabla \int p_\theta dx = 0$.

$\mathbb{E}[\nabla^2 \log p_\theta] = \mathbb{E}\left[ \frac{\nabla^2 p_\theta}{p_\theta} - \frac{(\nabla p_\theta)(\nabla p_\theta)^\top}{p_\theta^2} \right]$
$= \int \nabla^2 p_\theta dx - \mathbb{E}[\nabla \log p_\theta \nabla \log p_\theta^\top]$
$= 0 - I(\theta)$.

따라서 $I(\theta) = -\mathbb{E}[\nabla^2 \ell]$. $\blacksquare$

### 정리 6.6.3 (KL 의 2차 근사 = Fisher)

$$
\boxed{D(p_\theta \| p_{\theta + d\theta}) = \frac{1}{2} d\theta^\top I(\theta) d\theta + O(\|d\theta\|^3)}.
$$

**증명.**

$D(p_\theta \| p_{\theta + d\theta}) = \mathbb{E}_{p_\theta}[\log p_\theta - \log p_{\theta + d\theta}] = -\mathbb{E}_{p_\theta}[\ell(\theta + d\theta) - \ell(\theta)]$.

Taylor 전개:
$\ell(\theta + d\theta) = \ell(\theta) + \nabla \ell^\top d\theta + \frac{1}{2} d\theta^\top \nabla^2 \ell \, d\theta + O(\|d\theta\|^3)$.

$\mathbb{E}_{p_\theta}[\nabla \ell] = 0$ (score mean zero).

$\mathbb{E}_{p_\theta}[\nabla^2 \ell] = -I(\theta)$.

$$
D = -\mathbb{E}[\nabla \ell^\top d\theta + \tfrac{1}{2} d\theta^\top \nabla^2 \ell d\theta] + O(\|d\theta\|^3) = \tfrac{1}{2} d\theta^\top I(\theta) d\theta + O(\|d\theta\|^3). \qquad \blacksquare
$$

### 정리 6.6.4 (Cramér-Rao Lower Bound)

Unbiased estimator $\hat\theta$ 의 variance:
$$
\text{Var}(\hat\theta) \geq I(\theta)^{-1}.
$$

즉 Fisher information 이 클수록 추정 가능한 precision 이 높음.

**중요**: MLE $\hat\theta_N$ 은 asymptotically $\mathcal{N}(\theta, I(\theta)^{-1}/N)$ → MLE 는 점근적으로 efficient.

### 정의 6.6.5 (Natural Gradient)

$\theta$ 공간이 Riemannian (Fisher metric) 이면, "가장 가파른 방향"은 Euclidean gradient 가 아니라:
$$
\tilde\nabla L(\theta) = I(\theta)^{-1} \nabla L(\theta).
$$

**업데이트 규칙**: $\theta_{t+1} = \theta_t - \eta \, I(\theta_t)^{-1} \nabla L(\theta_t)$.

### 정리 6.6.6 (Natural gradient 의 invariance)

Natural gradient step 은 파라미터 재구성에 **invariant**.

*즉*: $\psi = g(\theta)$ 로 reparameterize 해도 natural gradient step 은 같은 분포 변화를 줌 (infinitesimal).
Euclidean gradient 는 $g$ 의 Jacobian 에 따라 달라짐.

이 invariance 가 natural gradient 의 "natural" 한 이유.

### 정리 6.6.7 (Amari-Chentsov 정리)

통계 다양체 위에서 **Fisher metric 은 유일한 (up to scaling) Markov-invariant Riemannian metric**.

즉 "정보이론적으로 자연스러운" metric 은 Fisher 뿐 (Chentsov 1972).

### Exponential Family 의 Fisher

$p_\theta(x) = h(x) \exp(\theta^\top T(x) - A(\theta))$ (exponential family).
$$
I(\theta) = \nabla^2 A(\theta).
$$

즉 log-partition function 의 Hessian = Fisher. 대표:
- Gaussian $\mathcal{N}(\mu, 1)$: $I(\mu) = 1$
- Bernoulli$(p)$ with $\theta = \log(p/(1-p))$: $I(\theta) = p(1-p)$
- Poisson$(\lambda)$: $I(\lambda) = 1/\lambda$

## 💻 Fisher & Natural Gradient 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 6.6.A Fisher information 수치 계산: Bernoulli
# X ~ Bernoulli(σ(θ)), p = σ(θ)
# I(θ) = Var(∂ℓ/∂θ) = p(1-p) · ... (σ' = σ(1-σ))
# 직접 유도: ℓ = y log σ + (1-y) log(1-σ), dℓ/dθ = y - σ
# I(θ) = Var(Y - σ) = σ(1-σ)

import numpy as np
thetas = np.linspace(-3, 3, 100)
fisher = [1/(1 + np.exp(-t)) * (1 - 1/(1 + np.exp(-t))) for t in thetas]

# 6.6.B KL ≈ (1/2) dθ² · I(θ) 검증
def kl_bernoulli(p, q):
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

theta = 0.5
dtheta = 0.01
p_theta = 1/(1+np.exp(-theta))
p_theta2 = 1/(1+np.exp(-(theta+dtheta)))

kl_exact = kl_bernoulli(p_theta, p_theta2)
I_theta = p_theta * (1 - p_theta)
kl_approx = 0.5 * dtheta**2 * I_theta

print(f"Exact KL:  {kl_exact:.6e}")
print(f"Fisher approx: {kl_approx:.6e}")
# → 매우 가까움 (O(dθ³) 오차)

# 6.6.C Natural gradient for Gaussian mean
# Model: p(x; μ) = N(μ, 1), loss = ||μ - target||²
# Gradient: ∇L = 2(μ - target)
# Fisher for N(μ, 1): I(μ) = 1
# → Natural gradient = gradient (same)

# Model: p(x; θ) = Bernoulli(σ(θ)), loss = -log p(x_i; θ)
# ∇ℓ = y - σ(θ)
# I(θ) = σ(1-σ)
# Natural gradient = (y - σ) / (σ(1-σ))
# Euclidean gradient is σ-scaled, natural removes that

# 6.6.D 신경망의 empirical Fisher
def empirical_fisher(model, data_batch, n_samples=100):
    """
    Empirical Fisher: E[∇log p · ∇log p^T]
    log p = -L (negative loss = log likelihood for Gaussian likelihood)
    """
    params = list(model.parameters())
    n_params = sum(p.numel() for p in params)
    F_diag = torch.zeros(n_params)   # diagonal approx
    
    for x, y in data_batch:
        model.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        grad_vec = torch.cat([p.grad.flatten() for p in params])
        F_diag += grad_vec**2
    
    F_diag /= len(data_batch)
    return F_diag   # diagonal of empirical Fisher

# 6.6.E K-FAC 개념 구현 (layer-wise block-diagonal Fisher)
# Martens 2015: Fisher ≈ block diagonal across layers
#                per-layer block is Kronecker product A ⊗ G
# (실제 구현은 복잡 - scaffold 만)
```

## 🔗 AI/ML 연결 (상세)

### Natural Gradient Descent 의 실무적 어려움

$I(\theta)^{-1}$ 계산: $k \times k$ 행렬 역 (현대 NN 에서 $k = 10^9$). Intractable.

**근사**:
1. **K-FAC** (Martens 2015): block-diagonal Kronecker factorization
2. **Diagonal Fisher**: $I \approx \text{diag}$ — Adam 의 $v_t$ term 이 이것의 running estimate
3. **Shampoo**: per-layer preconditioning (Google)
4. **CG-based**: conjugate gradient 로 $I^{-1} g$ 계산

### Adam ≈ Approximate Natural Gradient

Adam 의 업데이트:
$$
\theta_{t+1} = \theta_t - \eta \cdot \frac{m_t}{\sqrt{v_t} + \varepsilon}.
$$

$v_t \approx \mathbb{E}[g^2]$ (diagonal empirical Fisher).  
→ Adam 은 **diagonal natural gradient** 의 추정.  
(정확히 말하면 "gradient 의 제곱" 의 EMA — empirical Fisher vs. true Fisher 차이 있음 — Kunstner 2019.)

### Elastic Weight Consolidation (EWC)

Kirkpatrick 2017: continual learning. 이전 task 의 Fisher $I_A$ 로:
$$
\mathcal{L} = \mathcal{L}_B(\theta) + \lambda \sum_i I_{A, i} (\theta_i - \theta_{A,i}^*)^2.
$$

"중요한" 파라미터 (큰 Fisher) 는 변화 페널티 큼 → catastrophic forgetting 완화.

### Laplace Approximation

$p(\theta | D) \approx \mathcal{N}(\hat\theta, (-\nabla^2 \log p(D|\theta))^{-1})$.
$-\nabla^2 \log p = $ Fisher 의 empirical version.

Bayesian NN, uncertainty estimation 의 표준 도구.

### MAML 과 Natural Gradient

MAML (Finn 2017):
$$
\theta^* = \arg\min_\theta \sum_\tau \mathcal{L}_\tau(\theta - \alpha \nabla \mathcal{L}_\tau(\theta)).
$$

Inner step 이 Euclidean. Meta-MAML (Grant 2018): natural gradient inner step → task space 의 기하 반영.

### Model Merging via Fisher

Matena & Raffel 2021: 여러 모델 $\theta_1, \ldots, \theta_K$ 를 합칠 때 Fisher-weighted average:
$$
\theta^* = \left(\sum_k I_k\right)^{-1} \sum_k I_k \theta_k.
$$

Uniform averaging 보다 나은 성능.

### Information Geometry 와의 연결 (다음 레포)

Fisher metric → Riemannian geometry on statistical manifolds.
- $\alpha$-divergence (Amari 2000)
- Dual flat structure ($e$-connection, $m$-connection)
- Natural exponential family = flat manifold
- Pythagorean theorem: projection 기하

이 모든 것이 **Information Geometry Deep Dive** 레포에서 다뤄짐.

## ⚖️ 가정과 한계

1. **Fisher 존재 가정**: $p_\theta$ 가 미분 가능, $\log p$ 가 integrable 한 derivative 가정.
2. **True vs Empirical Fisher**:
   - True: $\mathbb{E}_{p_\theta}[\nabla \ell \nabla \ell^\top]$
   - Empirical: $\frac{1}{N}\sum_i \nabla \ell(x_i) \nabla \ell(x_i)^\top$ (일반적으로 다름 — 특히 under model misspecification)
3. **Non-identifiable parameters**: 비식별 모델에서 $I$ singular → natural gradient 정의 안 됨.
4. **NN 규모**: $k \approx 10^9$ → full $I^{-1}$ 저장 불가. 근사 필수.
5. **Fisher vs GGN vs Hessian**: 각각 다른 curvature 정의. GGN (Gauss-Newton) ≈ Fisher under softmax + CE (Martens 2020).
6. **Singular perturbation**: 딥러닝의 local minima 근처 Fisher 가 singular 해지는 현상 (Watanabe 2009 — singular learning theory).

## 📌 핵심 정리

$$
\boxed{
\begin{aligned}
I(\theta) &= \mathbb{E}[\nabla \ell \nabla \ell^\top] = -\mathbb{E}[\nabla^2 \ell] \\
D(p_\theta \| p_{\theta + d\theta}) &\approx \tfrac{1}{2} d\theta^\top I(\theta) d\theta \\
\text{Natural gradient: } & \tilde\nabla L = I^{-1} \nabla L \\
\text{Cramér-Rao: } & \text{Var}(\hat\theta) \geq I^{-1}
\end{aligned}}
$$

| 개념 | Fisher 의 역할 |
|---|---|
| Cramér-Rao bound | 추정 precision 하한 |
| Natural gradient | Riemannian steepest descent |
| K-FAC / Adam | NN 용 근사 |
| Laplace approximation | Bayesian posterior 근사 |
| EWC | Continual learning regularizer |
| Amari-Chentsov | Markov-invariant 유일 metric |

## 🤔 생각해볼 문제

### 문제 1. Gaussian 의 Fisher
$p_\theta = \mathcal{N}(\theta, 1)$ 의 Fisher = ?

<details>
<summary>해설</summary>

$\ell = -\frac{(x-\theta)^2}{2} - \frac{1}{2}\log 2\pi$. $\nabla \ell = x - \theta$. $\mathbb{E}[(\nabla \ell)^2] = 1 = I(\theta)$.
</details>

### 문제 2. 왜 Fisher 는 "information" 이라 부르나?
직관적 의미?

<details>
<summary>해설</summary>

$I$ 클수록 $X$ 로부터 $\theta$ 를 잘 추정 가능. "데이터가 파라미터에 대해 얼마나 정보를 주는가" 의 측도.  
CR bound: $\text{Var} \geq 1/I$ → 큰 $I$ = 정확한 추정 가능.
</details>

### 문제 3. Natural gradient 가 항상 좋은가?
Practical trade-off?

<details>
<summary>해설</summary>

이론적으로 optimal (invariance, convergence). 실무:
- $I^{-1}$ 계산 비용 커서 대규모 NN 에서 부담
- Small batch 에서 Fisher 추정 noisy
- Adam 이 대부분 case 에서 "good enough" + 간단
K-FAC, Shampoo 등이 대안. Meta-learning 에서는 natural gradient 효과 뚜렷.
</details>

### 문제 4. Fisher vs Hessian 차이
두 행렬의 관계?

<details>
<summary>해설</summary>

$\mathbb{E}[\nabla^2 \log p] = -I$. 즉 Fisher = $-$Hessian of expected log-lik. 
경험적으로: empirical Fisher vs empirical Hessian. 양의 definite 여부 다름 — Fisher 는 항상 PSD, Hessian 은 아닐 수 있음. 
GGN (Gauss-Newton) 은 Fisher 에 더 가까운 근사.
</details>

### 문제 5. Deep learning 의 Fisher 는 "singular"?
왜 문제가 되나?

<details>
<summary>해설</summary>

NN 에 symmetry (permutation, scale) → 같은 함수를 주는 여러 $\theta$ 존재 → $I$ 가 그 방향으로 singular.  
Watanabe 의 Singular Learning Theory: 이 singularity 가 NN 의 generalization 과 핵심적으로 연결 (free energy, RLCT).  
Information geometry 의 깊은 영역 (다음 레포에서 다룸).
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [6.5 Diffusion ELBO](./05-diffusion-elbo.md) | [🏠 Home](../README.md) |

[🏠 Home](../README.md)

</div>
