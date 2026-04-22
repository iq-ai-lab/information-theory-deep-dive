# 6.1 Cross-Entropy 와 MLE 의 정보이론적 해석

## 🎯 핵심 질문

> **왜 "Cross-Entropy 손실을 최소화" 하는 것이 곧 "$D(p \| q)$ 를 최소화", "MLE", 그리고 "codebook 최적화" 와 모두 동일한가?**

Cross-entropy 는 ML 손실함수의 거의 모든 곳 — 분류, 언어모델링 (next-token prediction), Diffusion (VLB), contrastive learning — 에 등장한다.  
그 수식 $-\sum p \log q$ 가 왜 이토록 보편적인가?

## 🔍 왜 이 개념이 AI 에서 중요한가

| 분야 | Cross-Entropy 의 등장 |
|---|---|
| **분류** | `CrossEntropyLoss` (PyTorch), `sparse_categorical_crossentropy` (TF) — 기본 중의 기본 |
| **언어 모델** | Next-token prediction loss = CE. GPT-4 의 "perplexity" 도 같은 값 |
| **VAE** | Reconstruction term = $-\mathbb{E}_q \log p(x|z)$ = CE (Bernoulli output) |
| **Diffusion** | Variational bound 의 각 term 이 cross-entropy / KL 형태 |
| **Contrastive (InfoNCE)** | Softmax over negatives = categorical CE |
| **RLHF** | Bradley-Terry likelihood 최대화 = CE |

**한 가지 깨달음**: 이 모든 게 **같은 수학적 대상**. "올바른 분포 $p$ 로 샘플을 설명할 때 평균 code 길이 $-\log q$" 를 최소화.

## 📐 수학적 선행 조건

- **엔트로피** $H(p) = -\mathbb{E}_p[\log p]$ — Ch1
- **KL divergence** $D(p\|q) = \mathbb{E}_p[\log(p/q)] \geq 0$ — Ch2
- **Jensen 부등식**
- **MLE** (Maximum Likelihood Estimation) — 통계학 기초
- **Source coding theorem** (Ch4-03): $L^* \geq H$

## 📖 직관적 이해

### "코드 길이" 해석

$H(p) = $ $p$ 에 맞춘 최적 code 의 **평균 길이** (bits/symbol).

$H(p, q) = \mathbb{E}_p[-\log q(X)] = $ $q$ 에 맞춘 code 를 쓰는데 **실제 데이터는 $p$** 에서 올 때의 평균 길이.

$$
H(p, q) = H(p) + D(p \| q).
$$

즉 잘못된 모델 $q$ 를 사용하면 $D(p\|q)$ 만큼 추가 비용 발생 — **"mismatch penalty"**.

### 분류 손실의 의미

샘플 $(x_i, y_i)$, 모델 $q_\theta(y|x)$.
$$
\mathcal{L} = -\frac{1}{N}\sum_{i=1}^N \log q_\theta(y_i | x_i).
$$
이것은 경험분포 $\hat p$ 하에서 $H(\hat p, q_\theta)$:
$$
-\frac{1}{N}\sum \log q_\theta(y_i|x_i) = \mathbb{E}_{\hat p}[-\log q_\theta(Y|X)] = H(\hat p, q_\theta).
$$

### MLE 와의 동등성

MLE: $\max \prod q_\theta(y_i|x_i) \iff \max \sum \log q_\theta \iff \min H(\hat p, q_\theta)$.

즉 **CE minimization = MLE**.

## ✏️ 엄밀한 정의

### 정의 6.1.1 (Cross-Entropy)

분포 $p, q$ 가 같은 sample space 를 공유. Cross-entropy:
$$
H(p, q) = -\sum_x p(x) \log q(x) = \mathbb{E}_p[-\log q(X)].
$$

(연속: $H(p,q) = -\int p(x) \log q(x) dx$, density 가정 하.)

### 정의 6.1.2 (Conditional Cross-Entropy)

$p(y|x), q(y|x)$ 두 조건부 분포에 대해:
$$
H(p, q \mid X) = \mathbb{E}_{X \sim p_X}\left[ H(p(\cdot|X), q(\cdot|X)) \right] = -\sum_x p(x) \sum_y p(y|x) \log q(y|x).
$$

## 🔬 핵심 정리

### 정리 6.1.3 (Cross-Entropy 분해)

$$
\boxed{H(p, q) = H(p) + D(p \| q)}.
$$

**증명.**
$$
H(p, q) = -\sum p(x) \log q(x) = -\sum p(x) \log \left( p(x) \cdot \frac{q(x)}{p(x)} \right)
$$
$$
= -\sum p(x) \log p(x) - \sum p(x) \log \frac{q(x)}{p(x)} = H(p) + D(p\|q). \quad \blacksquare
$$

**따름정리**:
- $H(p, q) \geq H(p)$, 등호는 $p = q$.
- $\arg\min_q H(p, q) = \arg\min_q D(p\|q) = p$.

### 정리 6.1.4 (MLE = Cross-Entropy Minimization)

iid 샘플 $x_1, \ldots, x_N \sim p$. 모델 $q_\theta$. 경험분포 $\hat p_N = \frac{1}{N}\sum \delta_{x_i}$.

**MLE**:
$$
\hat\theta_\text{MLE} = \arg\max_\theta \prod_{i=1}^N q_\theta(x_i) = \arg\max_\theta \frac{1}{N}\sum \log q_\theta(x_i).
$$

**Cross-Entropy minimization**:
$$
\arg\min_\theta H(\hat p_N, q_\theta) = \arg\min_\theta -\frac{1}{N}\sum \log q_\theta(x_i).
$$

두 식이 동일한 $\theta$ 를 준다. $\blacksquare$

### 정리 6.1.5 ($N \to \infty$ 에서 KL 최소화)

Law of large numbers:
$$
-\frac{1}{N}\sum \log q_\theta(x_i) \xrightarrow{a.s.} \mathbb{E}_p[-\log q_\theta(X)] = H(p, q_\theta) = H(p) + D(p\|q_\theta).
$$

$H(p)$ 는 $\theta$ 에 무관한 상수. 따라서:
$$
\hat\theta_\text{MLE} \xrightarrow{N \to \infty} \arg\min_\theta D(p\|q_\theta).
$$

**MLE 는 "forward KL 최소화"** — 이것이 Ch2-02 의 mean-seeking 성질과 일관.

### 정리 6.1.6 (Cross-Entropy 와 Perplexity)

$$
\text{PPL}(q_\theta) = \exp\!\left( H(\hat p, q_\theta) \right) = \left( \prod q_\theta(x_i) \right)^{-1/N}.
$$

언어 모델의 PPL 은 $\hat p$ 평균 당 모델이 "애매한" degree 를 측정. PPL = 10 이면 "매 토큰 10 개 중 1 개를 맞출 만큼 모름".

$$
\log \text{PPL} = H(\hat p, q_\theta) = H(\hat p) + D(\hat p \| q_\theta).
$$

## 💻 NumPy/PyTorch 검증

```python
import numpy as np
import torch
import torch.nn.functional as F

# 6.1.A 경험적 검증: H(p,q) = H(p) + D(p||q)
p = np.array([0.1, 0.2, 0.3, 0.4])
q = np.array([0.25, 0.25, 0.25, 0.25])

H_p = -np.sum(p * np.log2(p))
KL_pq = np.sum(p * np.log2(p / q))
H_pq = -np.sum(p * np.log2(q))

print(f"H(p)      = {H_p:.4f}")
print(f"D(p||q)   = {KL_pq:.4f}")
print(f"H(p,q)    = {H_pq:.4f}")
print(f"H(p)+D(p||q) = {H_p + KL_pq:.4f}  (should equal H(p,q))")
assert np.isclose(H_pq, H_p + KL_pq)

# 6.1.B PyTorch: nn.CrossEntropyLoss = softmax + NLL
logits = torch.tensor([[2.0, 1.0, 0.5]])
target = torch.tensor([0])

# 내장 CrossEntropyLoss
loss_builtin = F.cross_entropy(logits, target, reduction='none')
# 직접: -log softmax
probs = F.softmax(logits, dim=-1)
loss_manual = -torch.log(probs[0, target[0]])
print(f"PyTorch CE: {loss_builtin.item():.4f}")
print(f"Manual CE : {loss_manual.item():.4f}")

# 6.1.C Perplexity 계산
log_probs = torch.log(probs)
token_ids = torch.tensor([0, 1, 2, 0, 1])      # example sequence
nll = -log_probs[0, token_ids].mean()
ppl = torch.exp(nll)
print(f"\nAverage NLL: {nll.item():.4f}, PPL: {ppl.item():.4f}")

# 6.1.D MLE 수렴: Bernoulli 데이터에서 true p 추정
torch.manual_seed(0)
true_p = 0.3
N = 10000
samples = (torch.rand(N) < true_p).float()

# CE 최소화: θ = logit(p) 학습
theta = torch.tensor(0.0, requires_grad=True)
opt = torch.optim.SGD([theta], lr=0.1)
for step in range(200):
    p_theta = torch.sigmoid(theta)
    # CE = -(y log p + (1-y) log(1-p))
    loss = -(samples * torch.log(p_theta) + (1-samples)*torch.log(1-p_theta)).mean()
    opt.zero_grad(); loss.backward(); opt.step()

p_hat = torch.sigmoid(theta).item()
print(f"\nMLE via CE: p̂ = {p_hat:.4f}, true = {true_p}")
# → 매우 근접 (LLN)
```

## 🔗 AI/ML 연결 (상세)

### Softmax + NLL = 정확히 cross-entropy

분류 neural net:
$$
q_\theta(y | x) = \frac{\exp(f_\theta(x)_y)}{\sum_{y'} \exp(f_\theta(x)_{y'})}.
$$
$$
\mathcal{L} = -\log q_\theta(y|x) = -f_\theta(x)_y + \log \sum_{y'} \exp(f_\theta(x)_{y'}).
$$

Gradient:
$$
\nabla_\theta \mathcal{L} = -\nabla f_\theta(x)_y + \sum_{y'} q_\theta(y'|x) \nabla f_\theta(x)_{y'} = \sum_{y'} (q_\theta - \mathbb{1}_y) \nabla f_\theta(x)_{y'}.
$$

**"Error = prediction − target"** 의 정보이론적 의미: $q$ 를 $p$ 에 맞추려는 직접적 force.

### Label Smoothing = regularized KL

Hard target $\mathbb{1}_y$ 대신 $p_\text{LS} = (1-\varepsilon) \mathbb{1}_y + \varepsilon / K$.

CE:
$$
-\sum p_\text{LS}(y') \log q(y') = (1-\varepsilon)[-\log q(y)] + \varepsilon H(\text{unif}, q).
$$

= (원래 CE) + (균등분포로의 attraction) → overconfident 방지, calibration 개선.

### Focal Loss = 재가중된 CE

$$
\mathcal{L}_\text{focal} = -(1 - q_\theta(y|x))^\gamma \log q_\theta(y|x).
$$

KL 은 $q$ 가 $p$ 에 가까워지는 샘플의 loss 를 줄임 → 어려운 샘플에 집중.

### Bradley-Terry (RLHF reward) = sigmoid CE

Preference $(x, y_+, y_-)$: 
$$
P(y_+ \succ y_- | x) = \sigma(r(x, y_+) - r(x, y_-)).
$$
$$
\mathcal{L} = -\log \sigma(r_+ - r_-) = -\log \text{softmax}_+ = \text{CE}.
$$

### Diffusion 의 $L_{t-1}$ term

DDPM loss $\sum_t \mathbb{E}[D(q(x_{t-1}|x_t, x_0) \| p_\theta(x_{t-1}|x_t))]$.
각 KL 은 CE − H 로 분해 → "각 시간 step 의 코드 길이".

### InfoNCE / contrastive = categorical CE

Positive sample $y$, negatives $y_1, \ldots, y_N$:
$$
\mathcal{L}_\text{NCE} = -\log \frac{e^{s(x,y)}}{\sum_{y'} e^{s(x, y')}} = \text{categorical CE}.
$$

## ⚖️ 가정과 한계

1. **Support mismatch**: $p(x) > 0$ 이지만 $q(x) = 0$ 이면 $H(p,q) = \infty$. 
   - 해결: smoothing (add-ε), label smoothing, mixup
2. **Class imbalance**: CE 는 $p(y|x)$ 에 비례해 loss 가 축적 → 희귀 클래스가 학습 안 됨.
   - 해결: weighted CE, focal loss, resampling
3. **Calibration**: CE 최소화가 **true probability** 를 주지는 않음 — overconfident 경향 (온도 스케일링 필요)
4. **Finite-sample MLE bias**: $\hat\theta_\text{MLE}$ 는 점근적으로 unbiased, 유한 $N$ 에서 bias 존재 (e.g., variance estimator: $\frac{1}{N}$ vs. $\frac{1}{N-1}$)
5. **$H(p)$ 가 상수라는 전제**: 모델링 대상이 바뀌지 않을 때만 성립. Curriculum learning 등에서는 상수 아님.

## 📌 핵심 정리

$$
\boxed{
\begin{aligned}
H(p, q) &= H(p) + D(p \| q) \\
\text{MLE } &= \arg\min_\theta H(\hat p, q_\theta) \underset{N\to\infty}{=} \arg\min_\theta D(p \| q_\theta) \\
\text{PPL} &= e^{H(\hat p, q_\theta)} \\
\text{Cross-Entropy minimization} &= D \text{ minimization} = \text{MLE} = \text{최적 source code 학습}
\end{aligned}}
$$

## 🤔 생각해볼 문제

### 문제 1. 왜 $H(p)$ 는 MLE 의 관점에서 "무시되는 상수"인가?
$p$ 는 참 데이터 분포 (fixed). 모델 $\theta$ 에 대한 최적화에서 영향 없음. 단 $H(p)$ 는 "불가능한 lower bound" 를 말해준다 — PPL 이 $e^{H(p)}$ 보다 작아질 수 없음.

### 문제 2. CE loss = 0 의 의미?
$\mathcal{L} = 0 \iff q_\theta(y_i|x_i) = 1$ for all $i$. training 세트에 대한 완벽 암기 (overfit). 반드시 $p = q$ 가 되는 것은 아님 — 다른 $x$ 에 대해서는 다를 수 있음.

### 문제 3. Reverse KL 을 쓰면?
$\min D(q_\theta \| p)$ → mode-seeking (Ch2-02). 
분류에서는 거의 안 쓰이지만 VI (VAE 의 $q(z|x)$) 에서는 reverse KL 이 표준.

### 문제 4. Perplexity 의 비직관적 크기
vocab size $V = 50000$ 의 언어모델에서 PPL = 20 이면?
**해설**: 모델이 다음 토큰을 "20 개 후보 중 하나" 수준으로 압축. $\log_2 20 \approx 4.3$ bits/token. Uniform 이면 $\log_2 50000 \approx 15.6$ → 11 bits 압축 달성.

### 문제 5. Label smoothing 의 optimal $\varepsilon$?
**해설**: $p_\text{LS}$ 의 entropy 가 원본 $H(p) \approx 0$ 인 hard target 보다 커진다 → model 의 "uncertainty budget" 제공. 실무적으로 $\varepsilon = 0.1$ (Inception), ImageNet 에서 top-1 ~0.5% 향상. 이론적 optimal 은 noise level 과 task 에 따름.

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [5.4 현대 오류 정정 부호](../ch5-channel-coding/04-modern-codes.md) | [6.2 ELBO 분해](./02-elbo-decomposition.md) |

[🏠 Home](../README.md)

</div>
