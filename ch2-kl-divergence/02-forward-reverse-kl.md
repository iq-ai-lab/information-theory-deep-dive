# 2.2 Forward KL vs Reverse KL — 비대칭성의 기하학

## 🎯 핵심 질문

> **$D(p \| q)$와 $D(q \| p)$는 왜 그렇게 다른 결과를 낳는가?**
> 어떤 경우에 **mean-seeking(평균 추종)** 이 되고, 어떤 경우에 **mode-seeking(모드 추종)** 이 되는가?
> VAE와 변분추론(VI)은 왜 **reverse KL** 을 고르는가? Maximum Likelihood는 왜 **forward KL** 인가?

---

## 🔍 왜 AI에서 중요한가

| 선택 | 무엇을 최소화? | 결과적 행동 | 대표 적용 |
|---|---|---|---|
| **Forward KL** $D(p \| q_\theta)$ | $p$(데이터)의 평균으로 $q$ 학습 | **mean-seeking / zero-avoiding** | MLE, Cross-Entropy 학습, Diffusion |
| **Reverse KL** $D(q_\theta \| p)$ | $q$ 아래에서 expected log-ratio | **mode-seeking / zero-forcing** | VI, VAE ELBO, RLHF(PPO KL) |
| **Symmetric (JSD)** | 둘의 중간 | 균형적 | GAN 이론, 탐지기 |

**왜 이게 critical 한가?** 같은 모델 구조여도 어느 KL을 쓰느냐에 따라:
- 생성 샘플이 **blurry** 해지거나 (forward KL: VAE의 과도한 평균화 느낌)
- **mode collapse** 가 일어나거나 (reverse KL: 하나의 모드만 잡음)
- 분포의 **support** 가 과도하게 넓어지거나 좁아진다.

특히 **변분추론**, **KL-regularized policy optimization**(PPO), **Diffusion의 variational bound**는 전부 reverse KL 기반이고, **MLE / Cross-Entropy 최소화**는 forward KL이다. 이 비대칭을 이해하지 못하면 손실함수를 오해한다.

---

## 📐 선행 학습 지식

- [2.1 KL의 정의와 비음성](./01-kl-definition-nonnegativity.md)
- [1.2 엔트로피 정의와 성질](../ch1-entropy-axioms/02-entropy-definition.md)
- Jensen 부등식, 볼록성
- 최적화: 기울기, 고정점 방정식

---

## 📖 직관: 왜 비대칭인가

### 정의 복습

$$
D(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}
$$

두 분포에서 기댓값을 **$p$에 대해** 취하느냐, **$q$에 대해** 취하느냐가 근본 차이다.

### 핵심 관찰 — 어느 분포의 "눈"으로 보느냐

- **Forward** $D(p \| q)$ : $p$의 **support** 에서 적분. $p(x) > 0$ 인데 $q(x) \to 0$ 이면 $\log(p/q) \to +\infty$ 가 되어 **큰 벌점**. 따라서 $q$ 는 $p$ 가 있는 곳을 **절대로 비우면 안 됨** → **zero-avoiding** → 넓게 퍼지는 **mean-seeking** 해.
- **Reverse** $D(q \| p)$ : $q$의 **support** 에서 적분. $q(x) > 0$ 인데 $p(x) \to 0$ 이면 발산. 반면 $p(x) > 0$ 이어도 $q(x) = 0$ 이면 해당 항 $0 \cdot \log 0 = 0$. 즉, $q$ 는 $p$ 가 0이 아닌 곳에 있기만 하면 됨 → **zero-forcing** → $q$ 가 한 **모드**에 쑥 박힘 → **mode-seeking**.

### 이미지: 이중모드 $p$ 를 단일 Gaussian $q$ 로 근사

$p$가 두 개의 봉우리 (bimodal) 이고 우리가 단봉 정규분포 $q_\theta$로 근사한다면:

- **Forward KL 최소화**: $q$는 두 모드의 **중간**에 자리잡고 너비를 양쪽을 모두 커버하도록 넓게 → **평균**을 향함.
- **Reverse KL 최소화**: $q$는 두 모드 중 **하나**에만 수렴, 다른 모드는 무시 → **모드 하나**만 잡음.

> 이 이중모드 예제는 §💻 NumPy 섹션에서 수치로 확인한다.

---

## ✏️ 공식 정의

**정의 2.2.1 (Forward / Reverse KL)**

목표 분포 $p$ 와 파라미터 모델 $q_\theta$ 에 대해:

- **Forward KL (inclusive, M-projection)**:
  $$
  \theta^{\mathrm{fwd}} = \arg\min_\theta D(p \| q_\theta) = \arg\min_\theta \mathbb{E}_{x \sim p}[\log p(x) - \log q_\theta(x)]
  $$
- **Reverse KL (exclusive, I-projection)**:
  $$
  \theta^{\mathrm{rev}} = \arg\min_\theta D(q_\theta \| p) = \arg\min_\theta \mathbb{E}_{x \sim q_\theta}[\log q_\theta(x) - \log p(x)]
  $$

**M / I projection** 용어는 정보기하학(Information Geometry)에서 유래:
- **M-projection**: "Moment-matching" — $q$ 가 지수족(exponential family)이면 forward KL 최소화는 **충분통계량의 기댓값**을 맞춘다.
- **I-projection**: "Information projection" — reverse KL 최소화는 $q_\theta$ 의 support 를 $p$의 서포트 안에 끼워 맞춘다.

---

## 🔬 정리와 증명

### Theorem 2.2.1 (Forward KL = MLE)

**진술.** 데이터 $x_1, \ldots, x_N \overset{iid}{\sim} p_{\mathrm{data}}$ 에 대해

$$
\arg\min_\theta D(p_{\mathrm{data}} \| q_\theta) = \arg\max_\theta \mathbb{E}_{x \sim p_{\mathrm{data}}}[\log q_\theta(x)] \approx \arg\max_\theta \frac{1}{N}\sum_{i=1}^N \log q_\theta(x_i)
$$

즉 forward KL 최소화는 **Maximum Likelihood Estimation** 이다.

**증명.**
$$
D(p_{\mathrm{data}} \| q_\theta) = \underbrace{\mathbb{E}_{p_{\mathrm{data}}}[\log p_{\mathrm{data}}]}_{\text{상수 }(= -H)} - \mathbb{E}_{p_{\mathrm{data}}}[\log q_\theta]
$$
첫 항은 $\theta$ 에 의존하지 않으므로 최소화는 $\mathbb{E}_{p_{\mathrm{data}}}[\log q_\theta]$ 의 **최대화** 와 등가. 이것이 기대 log-likelihood. 유한 샘플에서는 대수의 법칙에 의해 $\frac{1}{N}\sum \log q_\theta(x_i)$. $\blacksquare$

> **함의**: Cross-Entropy loss = Forward KL 최소화 = MLE. 이 세 관점이 같은 식이라는 게 2장 6절에서 엄밀히 정리된다.

### Theorem 2.2.2 (Moment Matching for Exponential Families)

**진술.** $q_\theta(x) = h(x) \exp(\theta^\top T(x) - A(\theta))$ 가 지수족일 때 forward KL 최소화의 정류점(stationary point)은

$$
\mathbb{E}_{q_\theta}[T(X)] = \mathbb{E}_{p}[T(X)]
$$

즉 **충분통계량 기댓값이 일치**.

**증명 스케치.** $\nabla_\theta \log q_\theta(x) = T(x) - \nabla A(\theta)$. Forward KL의 기울기:
$$
\nabla_\theta D(p \| q_\theta) = -\mathbb{E}_p[\nabla_\theta \log q_\theta] = \nabla A(\theta) - \mathbb{E}_p[T(X)]
$$
지수족 항등식 $\nabla A(\theta) = \mathbb{E}_{q_\theta}[T(X)]$ 로부터 결과. $\blacksquare$

**예.** $q_\theta = \mathcal{N}(\mu, \sigma^2)$ 이면 $T(X) = (X, X^2)$, 정류점에서 $\mu = \mathbb{E}_p[X], \sigma^2 = \mathrm{Var}_p[X]$. **모멘트를 잡는 Gaussian** 이 forward-KL 최적해.

### Theorem 2.2.3 (Zero-Forcing Property of Reverse KL)

**진술.** $D(q \| p) < \infty$ 이면 $\mathrm{supp}(q) \subseteq \mathrm{supp}(p)$. 즉 $p(x) = 0 \Rightarrow q(x) = 0$ (a.s.).

**증명.** 만약 $q(A) > 0$ 인 집합 $A$ 에서 $p(x) = 0$ 이면
$$
\int_A q(x) \log \frac{q(x)}{p(x)}\,dx = +\infty
$$
이므로 $D(q\|p) = \infty$. 유한이라는 가정에 모순. $\blacksquare$

> **함의**: Reverse KL 을 쓰면 $q$ 의 support 가 $p$ 의 support 를 **넘지 못함**. 그래서 $p$ 가 두 봉우리여도 $q$ 가 둘 중 하나 안쪽에만 자리잡는 것이 **허용**된다.

### Theorem 2.2.4 (Gaussian Variational Approximation — Bimodal 예제)

**진술.** $p(x) = \frac{1}{2}\mathcal{N}(-a, 1) + \frac{1}{2}\mathcal{N}(a, 1)$ 이고 $q_{\mu, \sigma} = \mathcal{N}(\mu, \sigma^2)$ 라 하자. $a$ 가 충분히 크면:

- Forward KL 최적해: $\mu^* = 0, \sigma^{*2} \approx 1 + a^2$ (넓게 퍼짐, **mean-seeking**)
- Reverse KL 최적해: $\mu^* \approx \pm a, \sigma^{*2} \approx 1$ (한 봉우리에 박힘, **mode-seeking**)

**증명 스케치.**
- **Forward**: Theorem 2.2.2 에 의해 평균·분산 matching. $p$ 의 평균은 $0$, 분산은 $1 + a^2$.
- **Reverse**: $D(q\|p) = \mathbb{E}_q[\log q - \log p]$. $a$ 가 크면 $p(x) \approx \frac{1}{2}\mathcal{N}(x; \mu_k, 1)$ 근방에 국소화. $q$ 를 $\mu \approx \pm a$ 로 놓으면 교차항이 잘 상쇄되어 KL이 작아진다. 쌍대칭이어서 해가 두 개. $\blacksquare$

---

## 💻 NumPy로 직접 확인

### 이중 Gaussian에 대한 forward vs reverse 최적해

```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

np.random.seed(0)
a = 3.0  # 봉우리 분리
def p(x):
    return 0.5 * norm.pdf(x, -a, 1) + 0.5 * norm.pdf(x, a, 1)

# 격자 기반 적분
xs = np.linspace(-10, 10, 4001)
dx = xs[1] - xs[0]
p_grid = p(xs)

def q_grid(mu, sigma):
    return norm.pdf(xs, mu, sigma)

def forward_kl(params):   # min_q  D(p||q)
    mu, log_sigma = params
    q = q_grid(mu, np.exp(log_sigma))
    # p log (p/q) — 0 log 0 = 0
    mask = p_grid > 1e-12
    val = np.sum((p_grid[mask] * (np.log(p_grid[mask]) - np.log(q[mask] + 1e-300))) * dx)
    return val

def reverse_kl(params):   # min_q  D(q||p)
    mu, log_sigma = params
    q = q_grid(mu, np.exp(log_sigma))
    mask = q > 1e-12
    val = np.sum((q[mask] * (np.log(q[mask]) - np.log(p_grid[mask] + 1e-300))) * dx)
    return val

r_fwd = minimize(forward_kl, x0=[0.5, 0.0], method='Nelder-Mead').x
r_rev1 = minimize(reverse_kl, x0=[-3.5, 0.0], method='Nelder-Mead').x
r_rev2 = minimize(reverse_kl, x0=[ 3.5, 0.0], method='Nelder-Mead').x

print("Forward KL 해:  mu=%.3f  sigma=%.3f" % (r_fwd[0], np.exp(r_fwd[1])))
print("Reverse KL 해(좌):  mu=%.3f  sigma=%.3f" % (r_rev1[0], np.exp(r_rev1[1])))
print("Reverse KL 해(우):  mu=%.3f  sigma=%.3f" % (r_rev2[0], np.exp(r_rev2[1])))
```

출력(수치):
```
Forward KL 해:  mu=0.000  sigma=3.162      # 넓게, 평균 잡음
Reverse KL 해(좌):  mu=-3.000  sigma=1.000  # 왼쪽 봉우리만
Reverse KL 해(우):  mu= 3.000  sigma=1.000  # 오른쪽 봉우리만
```

$\sigma_{\mathrm{fwd}} \approx \sqrt{1 + a^2} = \sqrt{10} = 3.162$, 이론값과 일치 ✅. Reverse KL은 두 개의 **별개의 local minimum**이 있고 어디서 시작하느냐에 따라 다른 봉우리에 붙는다.

### Mode collapse 시각화

```python
import matplotlib.pyplot as plt
plt.plot(xs, p_grid, 'k', lw=2, label='p (target)')
plt.plot(xs, q_grid(r_fwd[0], np.exp(r_fwd[1])), 'b', label='forward KL q')
plt.plot(xs, q_grid(r_rev1[0], np.exp(r_rev1[1])), 'r--', label='reverse KL q (mode)')
plt.legend(); plt.title('Forward vs Reverse KL'); plt.show()
```

### 기울기 관점: 왜 zero-forcing 인가

Reverse KL은 $\mathbb{E}_q[\log q - \log p]$. $q$ 가 $p \approx 0$ 인 곳에 질량을 두면 $\log p \to -\infty$ 라는 **무한 벌점**. 그래서 기울기가 $q$ 를 $p$가 큰 영역으로 밀어붙이되 **한 영역**만 골라도 된다. 반면 forward KL 은 $p$가 큰 모든 곳을 $q$ 가 커버해야만 $\log q$ 항이 유한.

```python
# zero-forcing 검증: p(x)=0 근방에서 q가 0이어도 reverse는 문제없지만 forward는 폭발
x0 = 0.0  # 두 모드 사이 (p≈0)
print("p(0) =", p(x0))
# q=N(3,1) : q(0)=0.004, p(0)=0.004 — 양쪽 다 작으므로 문제 없음
```

---

## 🔗 AI/ML 연결고리

| 응용 | 어떤 KL? | 의미 |
|---|---|---|
| **MLE / Cross-Entropy** | Forward | $\min D(p_{\mathrm{data}} \| q_\theta)$ |
| **VAE ELBO** | **Reverse** | $\min D(q_\phi(z\|x) \| p_\theta(z\|x))$ |
| **Mean-Field VI** | Reverse | 한 모드 수렴, 분산 과소추정 경향 |
| **PPO (RLHF)** | Reverse | $D(\pi_\theta \| \pi_{\mathrm{ref}})$ — $\pi_\theta$ 가 ref의 support 안에 머무름 |
| **Expectation Propagation** | Forward (local) | moment matching |
| **Knowledge Distillation (teacher→student)** | Forward | teacher를 soft label로 MLE |
| **Reverse KD** | Reverse | mode-seeking, smaller support |

### VAE가 reverse KL인 이유

ELBO:
$$
\log p_\theta(x) = \mathrm{ELBO}(x, \phi, \theta) + D(q_\phi(z|x) \| p_\theta(z|x))
$$

ELBO 최대화는 오른쪽 KL을 **줄인다**. 이건 $q_\phi$ 를 true posterior $p_\theta(z|x)$ 에 맞추는 **reverse KL**. 그래서 VI는 posterior의 하나의 모드를 찾고 분산을 **과소추정**한다(흔히 "VI underestimates variance").

### PPO-KL과 RLHF

$$
\max_\theta \mathbb{E}_{\pi_\theta}[r(s,a)] - \beta\, D(\pi_\theta \| \pi_{\mathrm{ref}})
$$

**Reverse KL** ($\pi_\theta$가 기준). 효과: $\pi_\theta$ 가 $\pi_{\mathrm{ref}}$의 support 안쪽으로 머무름 → **distribution shift 방지**. Forward KL 이었다면 $\pi_{\mathrm{ref}}$ 의 모든 행동을 커버해야 해서 탐색이 지나치게 넓어졌을 것.

### Diffusion — 둘 다 등장

DDPM 손실의 variational bound:
$$
\mathcal{L}_{\mathrm{vlb}} = \sum_t D(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))
$$

$q$(forward diffusion의 posterior) 가 고정이고 $p_\theta$ 가 학습되므로 **$q$ 의 평균에 대해 기댓값** 을 취함 → 실질적으로 **forward KL** 형태이고 이것이 Gaussian이면 MSE로 환원. (자세한 유도는 [6.5 Diffusion ELBO](../ch6-ml-applications/05-diffusion-elbo.md))

---

## ⚖️ 가정·한계·함정

1. **"reverse KL이 항상 mode-collapse 한다"는 오해** — 단봉 $q$ 로 다봉 $p$ 를 근사할 때만 심각. $q$ 가 충분히 유연하면 양쪽 KL 해가 비슷해진다.
2. **유한샘플에서 unbiased 추정기** — Forward KL $D(p\|q_\theta)$ 은 $x \sim p$ 로 쉽게 추정 (cross-entropy의 MC 평균). Reverse는 $x \sim q_\theta$ 가 필요한데, 이때 $\log p$ 가 정규화상수를 요구하면 intractable → **변분 bound / MCMC / Stein** 기법 필요.
3. **Reparameterization** — Reverse KL의 $\nabla_\theta \mathbb{E}_{q_\theta}[\cdot]$ 는 score function estimator(REINFORCE) 가 기본, VAE는 reparameterization trick으로 저분산 gradient 확보.
4. **Infinite value 문제** — 두 support가 어긋나면 KL이 **$\infty$**. 학습 중에 이런 상황이 생기면 gradient도 의미가 없어짐. 이 때문에 실무에서는 종종 **soft KL** (label smoothing, Dirichlet prior) 로 회피.
5. **대칭화의 유혹** — Symmetric KL = $D(p\|q) + D(q\|p)$ 는 수학적으로 자연스러워 보이지만 최적화적으로 deterministic 모델을 끌어당기는 효과가 강해서 불안정. JSD를 쓰는 편이 보통 더 낫다(다음 절).

---

## 📌 핵심 정리

1. $D(p\|q)$ 와 $D(q\|p)$ 는 **어느 분포의 서포트 위에서 기댓값을 취하느냐** 가 결정.
2. **Forward KL**: mean-seeking / zero-avoiding / MLE / moment-matching.
3. **Reverse KL**: mode-seeking / zero-forcing / VI / PPO / distribution-safe.
4. 지수족에서 forward KL 해 = 충분통계량 matching.
5. Reverse KL 해는 여러 모드 중 **하나**에 자리잡을 수 있음(국소 해 다수).
6. VAE/RLHF가 reverse KL 을 쓰는 이유는 "근사 분포가 참조 분포의 support 안에 머무르게" 하려는 설계철학.
7. 실무에서는 **symmetric 필요성** → JSD (§2.3), **더 일반적 family** → f-divergence (§2.4), **support 불일치 robust** → Wasserstein (§2.5) 로 확장된다.

---

## 🤔 생각해볼 문제

### 문제 1. Gaussian 근사의 완전 유도
$p(x) = 0.5 \mathcal{N}(-a,1) + 0.5 \mathcal{N}(a,1)$, $q = \mathcal{N}(\mu, \sigma^2)$ 일 때 forward KL 의 정류점을 **손으로** 유도하라. (힌트: $\mathbb{E}_p[X]=0,\ \mathbb{E}_p[X^2]=1+a^2$)

<details>
<summary>해설</summary>

$D(p\|q)$ 은 $\theta$-항만 보면 $-\mathbb{E}_p[\log q] = \frac{1}{2}\log(2\pi\sigma^2) + \frac{\mathbb{E}_p[(X-\mu)^2]}{2\sigma^2}$. $\mu$ 에 대한 기울기 0 → $\mu = \mathbb{E}_p[X] = 0$. $\sigma^2$ 에 대한 기울기 0 → $\sigma^2 = \mathbb{E}_p[(X-\mu)^2] = 1 + a^2$. 이론과 수치 예제가 일치. ✅
</details>

### 문제 2. Reverse KL의 비볼록성
이중 Gaussian 예제에서 reverse KL은 $\mu = \pm a$ 두 국소 최소를 갖는다. 이 objective 가 $\mu$ 에 대해 **비볼록**임을 기울기 그래프로 보이고, 어떤 식으로 initialization 이 중요한지 논하라.

<details>
<summary>해설</summary>

$\mu=0$ 근방에서 기울기는 0 (대칭성), 하지만 Hessian 이 음수 → 안장점. $\mu = \pm a$ 가 국소 최소. 초기값 $\mu_0 > 0$ 이면 오른쪽 모드로, $\mu_0 < 0$ 이면 왼쪽 모드로 수렴. VI 구현시 **random restart** 가 필요한 이유.
</details>

### 문제 3. PPO KL이 reverse인 실무적 귀결
만약 RLHF의 KL 항을 forward $D(\pi_{\mathrm{ref}} \| \pi_\theta)$ 로 바꾸면 어떤 문제가 생길까?

<details>
<summary>해설</summary>

Forward KL 은 $\pi_\theta$ 가 $\pi_{\mathrm{ref}}$ 의 support 전체를 커버하도록 강요 → 이미 낮은 확률로 샘플링되는 부적절한 응답까지 살려야 함 → 탐색이 노이즈로 번짐. 또한 $\mathbb{E}_{\pi_{\mathrm{ref}}}[\log \pi_\theta]$ 계산이 ref에서 샘플링을 요구 → 샘플 효율 저하. Reverse KL은 $\pi_\theta$ 에서만 샘플링하면 충분.
</details>

### 문제 4. α-divergence로의 일반화
Rényi $\alpha$-divergence $D_\alpha$ 는 $\alpha \to 1$ forward, $\alpha \to 0$ reverse 극한을 갖는다. $\alpha = 0.5$ 는 무엇을 최소화하는가? (힌트: Hellinger)

<details>
<summary>해설</summary>

$\alpha = 0.5$ 은 **Hellinger squared distance** 와 비례: $H^2(p,q) = \frac{1}{2}\int(\sqrt{p}-\sqrt{q})^2 = 1 - \int\sqrt{pq}$. 대칭, metric 성질 있음. Forward 와 reverse의 사이 (중립적). 이 일반화가 §2.4 f-divergence 로 이어진다.
</details>

### 문제 5. Forward KL로 mode collapse가 생기지 않는 이유
MLE / forward KL 학습으로 GAN의 mode collapse 문제가 (이론상) 없는 이유를 설명하라.

<details>
<summary>해설</summary>

Forward KL 은 $p$ 의 모든 support를 $q$ 가 커버하도록 강제하므로 $q$ 가 한 모드에만 수렴하면 그 외 support 에서 $\log q \to -\infty$ 벌점이 터짐. 따라서 MLE 학습의 생성모델(Normalizing flow, Autoregressive LM, VAE의 reconstruction)은 mode coverage가 좋지만 대신 **blurry / over-spread** 해지는 trade-off 를 가진다. GAN은 JS-divergence를 최적화하지만 학습 역학이 mode-seeking 성격이 강함.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [2.1 KL의 정의와 비음성](./01-kl-definition-nonnegativity.md) | [2.3 JS-divergence](./03-js-divergence.md) |

[🏠 Home](../README.md)

</div>
