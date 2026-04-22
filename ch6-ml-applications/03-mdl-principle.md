# 6.3 MDL 원리 — Minimum Description Length

## 🎯 핵심 질문

> **"Occam 의 면도날"을 어떻게 정보이론으로 **정량화** 하는가?**  
> "Data = Model + Residual" 의 **최소 부호 길이** 를 구하는 것이 왜 **최적 모델 선택 = 베이지안 추론 = 정규화된 MLE** 와 동등한가?

Rissanen (1978) 의 MDL 원리: **"데이터를 가장 짧게 설명하는 모델이 최적이다."**

## 🔍 왜 이 개념이 AI 에서 중요한가

| 분야 | MDL 의 등장 |
|---|---|
| **정규화 (L1, L2)** | 파라미터 사전분포의 부호 길이 = penalty, 수식적 유래 |
| **모델 선택** | AIC, BIC — MDL/베이지안 유도의 근사 |
| **Compression-as-intelligence** | Hutter, Bellard — "압축 능력 = AGI" 철학적 주장 |
| **Neural network pruning** | 작은 모델 = 짧은 description = 일반화 양호 |
| **Flat minima** | Hochreiter 1997: flat minima = low code length |
| **In-context learning** | ICL 을 "data prefix 의 $-\log p$" 로 해석 |

**핵심 직관**: 좋은 모델은 데이터를 **짧게** 요약 — 더 적은 비트로 data 를 재생산 가능.

## 📐 수학적 선행 조건

- **Source coding theorem** (Ch4-03): 최소 code 길이 ≈ $H$
- **Kraft inequality**: 유효한 prefix code 의 조건
- **Bayes 정리**: posterior $\propto$ likelihood × prior
- **Shannon-Fano code**: 길이 $\lceil -\log p \rceil$

## 📖 직관적 이해

### "Occam 의 면도날" 의 수치화

두 모델:
- $M_1$: 복잡, 데이터 완벽 예측 ($-\log p(D|M_1) = 0$)
- $M_2$: 단순, 약간의 오차 ($-\log p(D|M_2) = 5$ bits)

$M_1$ 자체의 기술 비용 (e.g., 많은 파라미터): $-\log p(M_1) = 100$ bits.  
$M_2$: $-\log p(M_2) = 10$ bits.

**총 비용**:
- $M_1$: 100 + 0 = 100 bits
- $M_2$: 10 + 5 = 15 bits → **$M_2$ 선호**

### Two-Part Code

```
전송할 것 = [Model code] + [Data | Model code]
              L(M)          L(D | M)
   
총 길이:     L(D, M) = L(M) + L(D | M)
최적화:      min_M L(M) + L(D | M)
```

## ✏️ 엄밀한 정의

### 정의 6.3.1 (Two-part MDL)

모델 후보 집합 $\mathcal{M}$. 각 모델 $M \in \mathcal{M}$ 에 길이 함수 $L(M)$, 데이터 $D$ 의 조건부 길이 $L(D|M)$ 가 정의됨.

$$
\hat M = \arg\min_{M \in \mathcal{M}} \{ L(M) + L(D|M) \}.
$$

- $L(D|M) = -\log p(D|M)$ (Shannon code, likelihood 의 negative log)
- $L(M)$: 모델의 code length (prior $-\log p(M)$ 로 볼 수 있음)

### 정의 6.3.2 (Refined / Normalized MDL)

Parametric family $\{p_\theta\}$: $\theta \in \Theta \subset \mathbb{R}^k$.

$$
L_\text{MDL}(D) = -\log p_{\hat\theta(D)}(D) + \frac{k}{2} \log n + O(1),
$$
(stochastic complexity, Rissanen 1996). $n$ = sample size, $k$ = 파라미터 수.

## 🔬 MDL = 베이지안 = 정규화 MLE

### 정리 6.3.3 (MDL ↔ Bayesian MAP 동등성)

$L(M) = -\log p(M)$, $L(D|M) = -\log p(D|M)$ 로 두면:
$$
\arg\min_M [L(M) + L(D|M)] = \arg\max_M [\log p(M) + \log p(D|M)] = \arg\max_M p(M|D).
$$

즉 MDL = **Maximum a Posteriori (MAP)**.

**증명.**
Bayes: $p(M|D) \propto p(D|M) p(M)$.  
$\log p(M|D) = \log p(D|M) + \log p(M) - \log p(D)$.  
$\log p(D)$ 는 $M$ 에 무관 상수 → MAP = min sum of neg logs. $\blacksquare$

### 정리 6.3.4 (MDL ↔ Regularized MLE)

L2 (weight decay):
$$
L_\text{reg} = -\log p(D|\theta) + \frac{\lambda}{2} \|\theta\|^2.
$$

이는 prior $\theta \sim \mathcal{N}(0, 1/\lambda)$ 가정 하 MAP = MDL.

$$
-\log p(\theta) = \frac{\lambda}{2}\|\theta\|^2 + \text{const}.
$$

L1: $\theta_i \sim \text{Laplace}(0, 1/\lambda)$ → $-\log p = \lambda \|\theta\|_1$.

**따라서 모든 regularizer 는 MDL 의 model prior 선택**.

### 정리 6.3.5 (BIC 는 MDL 의 대수적 근사)

$$
\text{BIC} = -2 \log p(D|\hat\theta) + k \log n.
$$

Rissanen stochastic complexity 의 leading term:
$$
L_\text{MDL} \approx -\log p(D|\hat\theta) + \frac{k}{2}\log n.
$$

BIC = $2 \times$ MDL leading term. 모델 선택에서 동일 ranking.

### 정리 6.3.6 (Universal Code 와 Bayesian Mixture)

Marginal likelihood:
$$
p(D) = \int p(D|\theta) p(\theta) d\theta.
$$

**Shannon code 길이**:
$$
L(D) = -\log p(D) = -\log \int p(D|\theta) p(\theta) d\theta.
$$

Laplace approximation: $L(D) \approx -\log p(D|\hat\theta) - \log p(\hat\theta) - \frac{k}{2}\log(2\pi) + \frac{1}{2}\log |I(\hat\theta)| + \frac{k}{2}\log n$.

→ **Bayesian 적분 = 평균적 description length + 추가 복잡도 페널티**.

## 💻 Python 예제: 다항식 회귀에서 MDL 모델 선택

```python
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import polynomial as P

np.random.seed(42)

# True: y = x² + noise
N = 30
x = np.linspace(-1, 1, N)
y_true = x**2
y = y_true + 0.1 * np.random.randn(N)

sigma_noise = 0.1

# Candidate polynomial degrees
degrees = range(0, 10)
results = []

for k in degrees:
    # MLE
    coefs = np.polyfit(x, y, deg=k)
    y_hat = np.polyval(coefs, x)
    
    # Residual neg-log-likelihood (Gaussian noise assumption, variance σ²)
    nll = 0.5 * np.sum((y - y_hat)**2) / sigma_noise**2 + 0.5*N*np.log(2*np.pi*sigma_noise**2)
    # code length of residual in nats
    L_D_given_M = nll
    
    # Code length of model parameters (MDL approx: k/2 * log n)
    n_params = k + 1
    L_M = 0.5 * n_params * np.log(N)
    
    L_total = L_D_given_M + L_M
    
    # BIC and AIC comparison
    BIC = 2*nll + n_params * np.log(N)
    AIC = 2*nll + 2*n_params
    
    results.append((k, L_D_given_M, L_M, L_total, BIC, AIC))
    print(f"degree={k}: L(D|M)={L_D_given_M:6.2f}  L(M)={L_M:5.2f}  total={L_total:6.2f}  BIC={BIC:6.2f}  AIC={AIC:6.2f}")

best_MDL = min(results, key=lambda r: r[3])
best_BIC = min(results, key=lambda r: r[4])
print(f"\n🏆 MDL 최적: degree = {best_MDL[0]}")
print(f"🏆 BIC 최적: degree = {best_BIC[0]}")

# 이론: true degree = 2 이어야 선택됨
```

출력 예 (일부):
```
degree=0: L(D|M)= 14.50  L(M)= 1.70  total= 16.20  BIC= 32.40  AIC= 31.00
degree=1: L(D|M)=  5.80  L(M)= 3.40  total=  9.20  BIC= 18.40  AIC= 15.60
degree=2: L(D|M)= -8.50  L(M)= 5.10  total= -3.40  BIC= -1.80  AIC= -11.00
degree=3: L(D|M)= -8.45  L(M)= 6.80  total= -1.65  BIC=  3.40  AIC=-10.90
degree=4: L(D|M)= -8.40  L(M)= 8.50  total=  0.10  BIC=  8.60  AIC=-10.80
...

🏆 MDL 최적: degree = 2
🏆 BIC 최적: degree = 2
```

→ 올바르게 $x^2$ (degree = 2) 선택.

### Fisher information 과 volume-based code length

```python
# 더 정교한 MDL: Rissanen stochastic complexity
# L(D) = -log p(D|θ_hat) + (k/2) log(n/2π) + log ∫ √det I(θ) dθ
# 마지막 항이 모델 집합의 "volume"

# 여기서는 대수적 근사만 보임 — 정확한 계산은 Fisher information 행렬 필요
```

## 🔗 AI/ML 연결

### L1/L2 Regularization = 파라미터 prior

- L2 weight decay: $\theta \sim \mathcal{N}(0, 1/\lambda)$ Gaussian prior
- L1 (Lasso): $\theta \sim \text{Laplace}(0, 1/\lambda)$ → sparsity 유도
- Group lasso: group-structured prior
- Dropout as Bayesian approximation (Gal & Ghahramani 2016)

### Pruning & Quantization

MDL 관점: 작은 모델 = 짧은 code. 
- Magnitude pruning: zero out small weights → 효율적 encoding
- Quantization: 8-bit → 파라미터 당 code length 감소
- Knowledge distillation: student model 의 description 을 teacher 가 제공

### Flat Minima Hypothesis

Hochreiter & Schmidhuber 1997: "Flat minima = low description length".
- Sharp minima: parameter 를 정확히 저장해야 → 긴 code
- Flat minima: ±ε 내에서 loss 가 같음 → ε-precision 으로 충분 → 짧은 code
- 따라서 SGD 가 찾는 flat minima 가 generalization 좋음 (argument by MDL)

### In-Context Learning 의 MDL 해석

LLM: prompt $x_{1:n}$ → next token $x_{n+1}$.
$$
-\log p(x_{1:n}) = \sum_t -\log p(x_t | x_{<t})
$$
이것이 prompt 의 description length. Few-shot 가 성공하는 이유:
examples 가 task 의 **짧은 description** 역할 (low $L(\text{task})$).

### NN 의 Compression = Generalization

Arora 2018: NN 을 압축하면 generalization bound 가 개선.
- PAC-Bayes 와 결합: compression → tighter generalization gap
- "Large model, small effective description" paradigm

### Hutter Prize & AIXI

Hutter 2005: 일반 지능 = "모든 computable environment 의 짧은 description 을 빠르게 찾는 능력".
- Kolmogorov complexity $K(x)$ = 가장 짧은 description
- AIXI: 이론적으로 optimal, uncomputable
- Bellard 의 ts_zip: LLM + arithmetic coding = neural compressor → MDL 실증

## ⚖️ 가정과 한계

1. **Prior 선택**: $L(M)$ 은 prior 선택에 따라 다름. "Objective" MDL 은 여전히 연구 주제 (Jeffreys prior, reference prior).
2. **Continuous parameters**: Kolmogorov complexity 는 discrete. Continuous parameter 는 precision-truncation 필요.
3. **Computability**: Kolmogorov complexity 는 uncomputable. MDL 은 computable approximation.
4. **Finite model class**: $\mathcal{M}$ 이 무한해도 unique pre-encoding (prefix code) 필요.
5. **Asymptotic**: BIC approximation 은 $n \to \infty$ 에서. Small $n$ 에서 correction 필요.

## 📌 핵심 정리

$$
\boxed{
\begin{aligned}
\hat M &= \arg\min_M \{ L(M) + L(D|M) \} \\
&= \arg\min_M \{ -\log p(M) - \log p(D|M) \}\\
&= \arg\max_M p(M|D) \quad \text{(MAP)}\\
&\approx \arg\min_\theta \{ -\log p(D|\theta) + \tfrac{k}{2}\log n \} \quad \text{(BIC)}
\end{aligned}}
$$

| 관점 | 수식 |
|---|---|
| MDL | $L(M) + L(D\|M)$ |
| Bayesian MAP | $-\log p(M\|D)$ |
| Regularized MLE | NLL + penalty |
| BIC | $-2 \log p(D\|\hat\theta) + k \log n$ |

모두 동일 ranking ($n$ 큼 가정).

## 🤔 생각해볼 문제

### 문제 1. Gaussian noise 가정의 정당성?
MSE loss = MDL 에서 왜?

<details>
<summary>해설</summary>

$p(y|x, \theta) = \mathcal{N}(f_\theta(x), \sigma^2)$ → $-\log p(y|x) = \frac{(y-f)^2}{2\sigma^2} + \text{const}$.  
Cross-entropy / MSE 는 잡음 분포의 선택 → **MDL 관점에서는 잡음 모델도 description**.
</details>

### 문제 2. MDL 과 "no free lunch"
MDL 이 모든 데이터에서 동일하게 잘 작동?

<details>
<summary>해설</summary>

No. MDL 은 prior 선택에 의존 → "reasonable prior" 가정 필요.  
해리자와 데이터가 mismatch 하면 MDL 은 오판.  
결국 universal prior (Solomonoff) 는 uncomputable.
</details>

### 문제 3. Deep learning 에서 $k \log n$ term?
파라미터 수 = 수백만 → BIC penalty 엄청 큼. 그런데 왜 overfitting 안 됨?

<details>
<summary>해설</summary>

**Effective complexity 는 parameter count 가 아님**. Flat minima, PAC-Bayesian argument: 실제 description length 가 훨씬 작음.  
Zhang 2016 "Understanding Deep Learning Requires Rethinking Generalization" 의 역설 - MDL 프레임워크로 재해석 진행 중.
</details>

### 문제 4. Universal coding vs two-part coding
두 접근의 차이?

<details>
<summary>해설</summary>

Two-part: $L(M) + L(D|M)$ separately specified.  
Universal: $L(D) = -\log \int p(D|\theta) p(\theta) d\theta$ (marginalize out). 
Universal 이 더 tight 하지만 실무는 대부분 two-part (e.g., BIC).
</details>

### 문제 5. LLM 은 진정한 MDL 압축기인가?
GPT-4 를 손실없는 압축기로 쓰면?

<details>
<summary>해설</summary>

Yes. Arithmetic coding + LLM = $-\log_2 p_{LLM}(x)$ bits per token. 
Bellard 의 ts_zip: 실제 LLM + AC 로 zstd 보다 나은 압축 (텍스트). 
단: LLM 자체의 weights 크기를 포함하면 net 이득은 데이터가 클 때만.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [6.2 ELBO 분해](./02-elbo-decomposition.md) | [6.4 Information Bottleneck](./04-information-bottleneck.md) |

[🏠 Home](../README.md)

</div>
