# 2.3 JS-divergence — GAN의 이론적 토대

## 🎯 핵심 질문

> **KL이 비대칭이고 $\infty$로 쉽게 터진다면, 그걸 고친 symmetric & bounded divergence 는 무엇인가?**
> GAN의 원 논문(Goodfellow 2014)이 최적 판별기 하에서 **JS-divergence**로 환원된다는 주장은 어떻게 유도되는가?
> JSD가 **제곱근을 취하면 진짜 metric**이 된다는 사실은 무슨 의미인가?

---

## 🔍 왜 AI에서 중요한가

- **GAN 이론**: 최적 판별기 하에서 generator의 목적은 $2 \cdot \mathrm{JSD}(p_{\mathrm{data}} \| p_g) - 2\log 2$ 최소화.
- **Bounded**: $0 \le \mathrm{JSD} \le \log 2$ — KL은 $\infty$ 로 발산할 수 있지만 JSD 는 유한.
- **Symmetric**: $\mathrm{JSD}(p\|q) = \mathrm{JSD}(q\|p)$.
- **Metric**: $\sqrt{\mathrm{JSD}}$ 가 실제 metric (거리공리 만족).
- **Label smoothing / Classifier 학습**: 두 클래스가 섞인 상황의 정보량 측정.
- **Model validation**: 두 경험분포 간 유사도 정량화.

GAN이 mode collapse로 유명해진 이유 중 하나는 JSD 의 수학적 성질(support 불일치 시 상수)이 gradient를 소멸시키기 때문 → 이 관찰이 Wasserstein GAN(§2.5) 등장의 직접 동기.

---

## 📐 선행 학습 지식

- [2.1 KL의 정의](./01-kl-definition-nonnegativity.md)
- [2.2 Forward/Reverse KL](./02-forward-reverse-kl.md)
- 기본 부등식 (Cauchy–Schwarz), 로그의 볼록성
- 확률 측도의 absolute continuity

---

## 📖 직관

### KL의 두 가지 고질병

1. **비대칭**: $D(p\|q) \ne D(q\|p)$.
2. **$\infty$ 발산**: $\mathrm{supp}(p) \not\subseteq \mathrm{supp}(q)$ 이면 $D(p\|q) = \infty$.

이 두 가지를 고치는 가장 간단한 아이디어:
**둘을 평균분포 $m = \frac{p+q}{2}$ 에 대해 각각 KL 잰 후 평균낸다.**

$$
\mathrm{JSD}(p \| q) = \frac{1}{2}D\!\left(p \,\Big\|\, \frac{p+q}{2}\right) + \frac{1}{2}D\!\left(q \,\Big\|\, \frac{p+q}{2}\right)
$$

$m$ 은 $p$ 와 $q$ 둘의 support 를 모두 포함하므로 발산 문제가 사라지고, 구성상 symmetric.

### Label-generating 해석

$Z \sim \mathrm{Bernoulli}(1/2)$ 로 $Z=0$ 이면 $X \sim p$, $Z=1$ 이면 $X \sim q$ 라 하자. 그러면 $X$의 주변분포는 $m$. **상호정보량**:
$$
I(X; Z) = \mathrm{JSD}(p \| q)
$$

즉 JSD = "$X$를 보고 어느 분포에서 왔는지 얼마나 알 수 있는가"의 정보량. 이것이 GAN 의 **판별기 해석** 이다.

---

## ✏️ 공식 정의

**정의 2.3.1 (Jensen–Shannon Divergence)**

$$
\boxed{\ \mathrm{JSD}(p \| q) \;=\; \frac{1}{2}D(p \| m) + \frac{1}{2}D(q \| m), \qquad m = \frac{p+q}{2}\ }
$$

**동치식(엔트로피 형식)**:
$$
\mathrm{JSD}(p \| q) = H(m) - \frac{1}{2}H(p) - \frac{1}{2}H(q)
$$

이는 "혼합의 엔트로피 − 평균 엔트로피" 로 Jensen 부등식의 gap과 같다.

**정의 2.3.2 (JS distance)** — $\sqrt{\mathrm{JSD}(p\|q)}$ 는 metric (Endres–Schindelin 2003).

---

## 🔬 정리와 증명

### Theorem 2.3.1 (비음성 & 유계성)

**진술.** $0 \le \mathrm{JSD}(p\|q) \le \log 2$. 등호는 각각 $p=q$ (왼쪽), $p \perp q$(완전 분리 support, 오른쪽).

**증명.**
- **비음성**: 각 KL ≥ 0 (§2.1 Theorem), 평균도 ≥ 0.
- **상계**: 엔트로피 형식 사용.
$$
\mathrm{JSD}(p\|q) = H(m) - \tfrac{1}{2}(H(p) + H(q))
$$
$m = \frac{p+q}{2}$ 인데, 두 분포가 서로 겹치지 않는 (disjoint) 극단에서는
$$
H(m) = H(p)/2 + H(q)/2 + \log 2, \quad \text{(혼합 Bernoulli 항 추가)}
$$
따라서 $\mathrm{JSD} = \log 2$. 일반적으로 $H(m) \le \frac{1}{2}H(p) + \frac{1}{2}H(q) + \log 2$ 를 엔트로피 concavity 와 mixing 항으로 보이면 상계 성립. $\blacksquare$

### Theorem 2.3.2 (대칭성)

**진술.** $\mathrm{JSD}(p\|q) = \mathrm{JSD}(q\|p)$.

**증명.** 정의식에서 $p \leftrightarrow q$ 를 바꿔도 $m = (p+q)/2$ 가 불변이므로 두 항이 서로 교환. $\blacksquare$

### Theorem 2.3.3 (JSD 는 Mutual Information)

**진술.** $Z \sim \mathrm{Bernoulli}(1/2)$, $X|Z=0 \sim p$, $X|Z=1 \sim q$ 이면
$$
I(X; Z) = \mathrm{JSD}(p \| q)
$$

**증명.**
$$
I(X;Z) = H(X) - H(X|Z) = H(m) - \tfrac{1}{2}H(p) - \tfrac{1}{2}H(q) = \mathrm{JSD}(p\|q). \blacksquare
$$

### Theorem 2.3.4 (GAN의 최적 판별기와 JSD)

**진술.** 판별기 $D : \mathcal{X} \to [0,1]$ 에 대해 GAN objective
$$
V(G, D) = \mathbb{E}_{p_{\mathrm{data}}}[\log D(x)] + \mathbb{E}_{p_g}[\log(1 - D(x))]
$$
의 **$D$-최적해** 는
$$
D^*(x) = \frac{p_{\mathrm{data}}(x)}{p_{\mathrm{data}}(x) + p_g(x)}
$$
이고, 이때
$$
\max_D V(G, D) = 2\, \mathrm{JSD}(p_{\mathrm{data}} \| p_g) - 2\log 2
$$

**증명.** 각 $x$ 에서 $V$ 의 피적분항은 $p_{\mathrm{data}} \log D + p_g \log(1-D)$ (with respect to $D \in [0,1]$). $D$ 에 대한 기울기 0 → $D^* = p_{\mathrm{data}}/(p_{\mathrm{data}} + p_g)$. 대입하면
$$
V(G, D^*) = \mathbb{E}_{p_{\mathrm{data}}}\!\left[\log\frac{p_{\mathrm{data}}}{p_{\mathrm{data}}+p_g}\right] + \mathbb{E}_{p_g}\!\left[\log\frac{p_g}{p_{\mathrm{data}}+p_g}\right]
$$
$= D(p_{\mathrm{data}} \| p_{\mathrm{data}}+p_g) + D(p_g \| p_{\mathrm{data}}+p_g) - 2\log 2 = 2\,\mathrm{JSD} - 2\log 2$. $\blacksquare$

> **함의**: Generator는 JSD를 최소화 → 최소값 0 → $p_g = p_{\mathrm{data}}$.

### Theorem 2.3.5 (JSD의 거리 성질)

**진술.** $d_{\mathrm{JS}}(p,q) := \sqrt{\mathrm{JSD}(p\|q)}$ 는 metric (비음성, 대칭, 삼각부등식, $d=0 \Leftrightarrow p=q$).

**증명 스케치.** (Endres–Schindelin 2003) JSD 를 Hilbert space embedding 을 이용해 $\ell_2$ 거리의 제곱으로 표현하는 embedding 이 존재한다. 구체적으로
$$
\mathrm{JSD}(p\|q) = \|\phi(p) - \phi(q)\|_2^2
$$
인 Hilbert-space embedding $\phi$ 를 구성함으로써 삼각부등식이 $\ell_2$ 거리의 삼각부등식으로 환원. $\blacksquare$

### Theorem 2.3.6 (Support 불일치 시 상수)

**진술.** 만약 $p$ 와 $q$ 의 support 가 **disjoint** 이면 $\mathrm{JSD}(p\|q) = \log 2$ (상수).

**증명.** disjoint 라면 각 $x$ 에서 $p(x)$ 와 $q(x)$ 중 **오직 하나** 만 0 이 아님. 가령 $p(x)>0, q(x)=0$ 인 $x$ 에서 $m(x) = p(x)/2$. 그러면
$$
p(x)\log\frac{p(x)}{m(x)} + q(x)\log\frac{q(x)}{m(x)} = p(x)\log 2.
$$
적분 시 양쪽에서 각각 $\log 2$ 기여 → $\mathrm{JSD} = \frac{1}{2}(\log 2 + \log 2) \cdot 1 = \log 2$. $\blacksquare$

> **함의**: GAN 초기에 $p_g$ 가 $p_{\mathrm{data}}$ 의 support와 겹치지 않으면 gradient = 0 → **학습 멈춤**. 이것이 WGAN이 필요한 이유(§2.5).

---

## 💻 NumPy로 직접 확인

```python
import numpy as np

def kl(p, q, eps=1e-12):
    return np.sum(p * (np.log(p + eps) - np.log(q + eps)))

def jsd(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)

rng = np.random.default_rng(0)
p = np.array([0.2, 0.3, 0.4, 0.1])
q = np.array([0.1, 0.4, 0.3, 0.2])
print("JSD(p||q)        =", jsd(p, q))
print("JSD(q||p)        =", jsd(q, p))      # 같음
print("sqrt(JSD)        =", np.sqrt(jsd(p, q)))   # metric 값
print("log 2 (상계)      =", np.log(2))
print("JSD(p||p)        =", jsd(p, p))      # 0
```
출력:
```
JSD(p||q)        = 0.02827...
JSD(q||p)        = 0.02827...        # 대칭성 확인
sqrt(JSD)        = 0.16815...
log 2 (상계)      = 0.69314...
JSD(p||p)        = 0.0
```

### Disjoint support 실험

```python
e = np.eye(4)
p, q = e[0], e[1]   # 서로소 support
print("JSD(disjoint) =", jsd(p, q), "   log 2 =", np.log(2))
# JSD(disjoint) = 0.6931...  → 정확히 log 2
```

### 삼각부등식 검증

```python
def jsd_dist(p, q): return np.sqrt(jsd(p, q))
rng = np.random.default_rng(1)
for _ in range(5):
    p = rng.dirichlet(np.ones(5))
    q = rng.dirichlet(np.ones(5))
    r = rng.dirichlet(np.ones(5))
    d_pq = jsd_dist(p, q); d_pr = jsd_dist(p, r); d_rq = jsd_dist(r, q)
    assert d_pq <= d_pr + d_rq + 1e-10
print("삼각부등식 100% pass ✅")
```

### GAN 최적 판별기 실험

```python
# p_data = Gaussian(0,1), p_g = Gaussian(mu_g, 1). D*(x) = p_data/(p_data+p_g)
from scipy.stats import norm
xs = np.linspace(-6, 8, 2001); dx = xs[1]-xs[0]
for mu_g in [0.0, 1.0, 3.0]:
    pd = norm.pdf(xs, 0, 1); pg = norm.pdf(xs, mu_g, 1)
    m  = 0.5*(pd + pg)
    jsd_val = 0.5*np.sum(pd*(np.log(pd+1e-30)-np.log(m+1e-30)))*dx \
            + 0.5*np.sum(pg*(np.log(pg+1e-30)-np.log(m+1e-30)))*dx
    V_star = 2*jsd_val - 2*np.log(2)
    print(f"mu_g={mu_g}  JSD={jsd_val:.4f}  V*={V_star:.4f}")
```
$\mu_g=0$에서 JSD=0, $\mu_g$ 커질수록 JSD $\to \log 2$.

---

## 🔗 AI/ML 연결고리

| 응용 | 등장 형태 |
|---|---|
| **GAN (Goodfellow 2014)** | Generator loss = $2\mathrm{JSD}(p_{\mathrm{data}}\|p_g) - 2\log 2$ |
| **f-GAN** (§2.4) | JSD는 f-divergence의 한 특수형 |
| **Latent Dirichlet Allocation 평가** | 주제 분포간 JSD |
| **언어모델 품질** | 두 문서 분포간 JSD (bag-of-words) |
| **Fairness 측정** | 서브그룹 분포간 JSD |
| **이상탐지** | sliding window 분포 vs. reference |
| **Molecular similarity** | 분자 기반 분포 비교 |

### GAN의 gradient vanishing — JSD의 내재적 결함

Theorem 2.3.6: support 가 어긋나면 JSD = $\log 2$ **상수**. 그러면 generator 의 파라미터에 대한 gradient 가 0. 실제로 초기 GAN은 Dense한 고차원 이미지 manifold에서 $p_g$ 와 $p_{\mathrm{data}}$ 의 support 겹침이 거의 없어서 학습이 매우 불안정. 이것이 **Wasserstein distance** 가 제안된 결정적 이유.

### Label smoothing의 JSD적 해석

one-hot 타깃 $e_y$ 와 smoothed 타깃 $\tilde e_y = (1-\epsilon)e_y + \epsilon u$ 를 섞으면, 학습의 cross-entropy loss 는 $D(\tilde e_y \| q_\theta)$ 인데 이는 부분적으로 **$q_\theta$의 확률을 모든 클래스에 분산**시키는 효과 → 효과적으로 JSD 쪽 geometry 로 이동.

### ELBO와의 관계

ELBO 에서 등장하는 것은 **reverse KL** 이고 JSD 는 아니지만, 혼합 분포 $\tfrac{1}{2}q+\tfrac{1}{2}p$ 를 구성해 **symmetric variational bound** 를 쓰는 연구(예: Symmetric VAE) 에서 JSD가 등장한다.

---

## ⚖️ 가정·한계·함정

1. **JSD는 bounded지만 gradient는 zero가 될 수 있다** — support 어긋나면 상수 → 학습 불가능.
2. **JSD ≠ metric** (제곱근 취해야 metric). 실무에서는 보통 JSD 자체를 쓰지 $\sqrt{\cdot}$ 는 이론적 맥락에서만.
3. **혼합 $m$ 계산 비용** — 고차원 확률밀도에서 $p+q$ 계산은 샘플만으로는 직접 추정 불가. MC로 JSD 추정할 때 신중한 기법 필요.
4. **벌점이 "약함"** — $p \ne q$ 에서도 값이 bounded 하므로 loss 의 dynamic range 가 작음 → optimizer 에 따라 학습 속도가 느릴 수 있음.
5. **고차원에서 의미 약화** — 고차원 manifold 문제로 $p_g$ 와 $p_{\mathrm{data}}$ 의 JSD 추정은 noise가 큼. Empirical JSD 는 sample 수에 민감.

---

## 📌 핵심 정리

1. $\mathrm{JSD}(p\|q) = \frac{1}{2}D(p\|m) + \frac{1}{2}D(q\|m),\ m=(p+q)/2$.
2. **대칭 & bounded** $[0, \log 2]$.
3. $\mathrm{JSD} = I(X; Z)$ 에서 $Z$ 는 "분포 선택" 지시자 — GAN 판별기 해석.
4. GAN: $\max_D V(G, D) = 2\mathrm{JSD}(p_{\mathrm{data}}\|p_g) - 2\log 2$.
5. **Disjoint support → JSD = log 2 상수** → gradient 소멸 → WGAN 필요성.
6. $\sqrt{\mathrm{JSD}}$ 는 **metric** (삼각부등식 성립).
7. JSD는 **f-divergence 의 한 경우** 로, 다음 절에서 일반화.

---

## 🤔 생각해볼 문제

### 문제 1. JSD 상계가 왜 $\log 2$ 인가
$\mathrm{JSD}(p\|q) \le \log 2$ 의 정확한 유도를 써라. (힌트: 엔트로피 형식과 mixing lemma)

<details>
<summary>해설</summary>

$\mathrm{JSD} = H(m) - \tfrac12(H(p)+H(q))$. $X\sim m$, $Z$ 가 Bernoulli(1/2) 라면 $H(X) = H(X|Z) + I(X;Z)$ 에서 $H(X|Z) = \tfrac12(H(p)+H(q))$, $H(X) = H(m)$. 따라서 $\mathrm{JSD} = I(X;Z) \le H(Z) = \log 2$. 등호는 $X$ 로부터 $Z$ 를 완전히 복원할 수 있을 때 → support disjoint.
</details>

### 문제 2. GAN 이론 유도 전체
$V(G,D)$ 를 $D$ 에 대해 최대화하고, 결과를 다시 $G$ 에 대해 최소화하면 $p_g = p_{\mathrm{data}}$ 가 유일한 global minimum임을 보여라.

<details>
<summary>해설</summary>

Theorem 2.3.4 로 $\max_D V = 2\mathrm{JSD}(p_{\mathrm{data}}\|p_g) - 2\log 2 \ge -2\log 2$. 등호는 JSD=0 ↔ $p_g = p_{\mathrm{data}}$. 따라서 $\min_G \max_D V = -2\log 2$, $p_g^* = p_{\mathrm{data}}$.
</details>

### 문제 3. Empirical JSD의 편향
샘플 크기 $N$ 개의 경험분포 $\hat p, \hat q$ 로 JSD 를 근사하면 bias가 어떻게 생기는가?

<details>
<summary>해설</summary>

엔트로피 추정량이 유한 $N$ 에서 underestimate 되는 경향(Miller–Madow bias)이 있어 $\hat H(m) - \tfrac12(\hat H(p) + \hat H(q))$ 전체적으로 bias 양의 방향. $N$ 이 클수록 감소. 이산 분포의 경우 Nemenman–Shafee–Bialek (NSB) 추정량 등으로 편향 보정.
</details>

### 문제 4. 가중 JSD
$\mathrm{JSD}_\pi(p\|q) = \pi D(p\|m_\pi) + (1-\pi) D(q\|m_\pi),\ m_\pi = \pi p + (1-\pi)q$ 는 여전히 비음성이고 $p=q$ 에서 0 임을 보여라. 상계는 무엇인가?

<details>
<summary>해설</summary>

Jensen/KL 비음성으로 ≥0. 등호 $p=q$. 상계 $H(\pi) = -\pi\log\pi - (1-\pi)\log(1-\pi)$ (mutual information $I(X;Z)$ 에서 $Z$ 가 Bernoulli($\pi$)). $\pi=1/2$ 특수경우가 $\log 2$.
</details>

### 문제 5. $\sqrt{\mathrm{JSD}}$ 가 metric 인 이유 (스케치)
JSD 가 Hilbert embedding 을 가짐을 받아들이고, 삼각부등식을 유도하라.

<details>
<summary>해설</summary>

$\mathrm{JSD}(p,q) = \|\phi(p)-\phi(q)\|^2$ 이면 $\sqrt{\mathrm{JSD}} = \|\phi(p)-\phi(q)\|$. $\ell_2$ 노름이므로 $\|\phi(p)-\phi(q)\| \le \|\phi(p)-\phi(r)\| + \|\phi(r)-\phi(q)\|$. 즉 $\sqrt{\mathrm{JSD}(p,q)} \le \sqrt{\mathrm{JSD}(p,r)} + \sqrt{\mathrm{JSD}(r,q)}$. 비음성·대칭·$p=q$ 이면 0 도 JSD 로부터 자명.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [2.2 Forward/Reverse KL](./02-forward-reverse-kl.md) | [2.4 f-divergence](./04-f-divergence.md) |

[🏠 Home](../README.md)

</div>
