# 3.5 MI와 표현학습 — InfoNCE, Contrastive Learning, CLIP

## 🎯 핵심 질문

> **"좋은 representation 이란 무엇인가?"** 에 대한 정보이론적 답은 MI 최대화다.
> InfoNCE, SimCLR, CLIP, MoCo 는 왜 모두 같은 objective 의 변형인가?
> **Positive/Negative pairs** 의 contrastive loss 는 왜 mutual information lower bound 인가?

---

## 🔍 왜 AI에서 중요한가

- **Self-Supervised Learning (SSL) 의 표준**: SimCLR, MoCo, BYOL, CLIP, DINO — 대부분 contrastive 아이디어 기반.
- **Representation learning 의 통일 프레임**: "InfoMax 원리" + "probabilistic criterion".
- **CLIP 의 문자·이미지 alignment**: 자연어와 시각의 joint embedding.
- **Retrieval, Search, RAG**: dense retrieval 는 contrastive embedding.
- **Foundation Models**: 사전학습의 대표적 손실함수.

"Contrastive loss = MI lower bound" 는 modern SSL 의 수학적 기반. 이 장은 2장 KL + 3장 MI 을 응용한 **가장 실무적인 결정체**.

---

## 📐 선행 학습 지식

- [3.1 MI 정의](./01-mi-definitions.md)
- [3.4 MINE / InfoNCE](./04-continuous-mi-mine.md)
- Softmax cross-entropy, similarity kernel
- Embedding, neural encoder

---

## 📖 직관

### InfoMax principle

"좋은 representation $Z = f(X)$ 는 입력의 정보를 최대한 보존한다":
$$
\max_f\ I(X; Z)
$$

딱 이대로는 **trivial solution**: $f$ 를 identity 로 잡으면 $I(X; X) = H(X)$ 무한대 (continuous) / 최댓값. 실질적 representation learning 이 아님.

### 해결: Contrastive / Predictive 구조

두 가지 **augmented view** 또는 **두 시점**:
- $X$ = anchor (이미지 or 텍스트)
- $Y$ = positive view (같은 이미지의 다른 augmentation, 또는 연관된 텍스트)
- $\{Y^-_k\}$ = negatives (다른 이미지)

$f$ 가 $X, Y$ 의 **공유 정보** 만 추출 → trivial 방지.

$$
\max_f\ I(f(X); f(Y))
$$

### Why contrastive loss 가 MI lower bound 인가

InfoNCE는 $K$-way classification: "positive 가 어느 것인지 맞춰라". 분류 정확도와 MI 간의 직접 연결.

---

## ✏️ 공식 정의

**정의 3.5.1 (Anchor–Positive pair)**
$(X, Y) \sim p(X, Y)$ 가 positive pair. Negatives $Y^-_k \sim p(Y)$ (marginal) 가 독립.

**정의 3.5.2 (InfoNCE loss)**
Scoring function $f(x, y) = e^{\phi(x)^\top \psi(y) / \tau}$ (cosine similarity + temperature):
$$
\mathcal{L}_{\mathrm{NCE}} = -\mathbb{E}\left[\log \frac{f(X, Y)}{f(X, Y) + \sum_{k=1}^{K-1} f(X, Y^-_k)}\right]
$$
여기서 기댓값은 positive $(X, Y)$ + $K-1$ negatives.

**정의 3.5.3 (MI lower bound — Oord et al. 2018)**
$$
\boxed{\ I(X; Y) \ge \log K - \mathcal{L}_{\mathrm{NCE}}\ }
$$

**정의 3.5.4 (SimCLR loss, Chen 2020)**
$\tilde X, \tilde Y$ = 같은 이미지의 두 augmentation. Batch 내 $N$ 샘플에서 $2N$ views. Normalized temperature cross entropy:
$$
\mathcal{L}_{i,j} = -\log \frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k \ne i} \exp(\mathrm{sim}(z_i, z_k)/\tau)}
$$
$\mathrm{sim}(u, v) = u^\top v / (\|u\|\|v\|)$.

**정의 3.5.5 (CLIP loss, Radford 2021)**
Image $X_i$, Text $Y_i$ pair 로 symmetric InfoNCE:
$$
\mathcal{L}_{\mathrm{CLIP}} = \frac{1}{2}(\mathcal{L}_{X \to Y} + \mathcal{L}_{Y \to X})
$$

**정의 3.5.6 (Alignment & Uniformity, Wang & Isola 2020)**
Contrastive loss 의 두 성분 분해:
- **Alignment**: $\mathbb{E}_{(x,y)\sim p_{\mathrm{pos}}}[\|f(x) - f(y)\|^2]$ — positives 가까움.
- **Uniformity**: $\log \mathbb{E}_{x, x' \sim p}[e^{-t\|f(x) - f(x')\|^2}]$ — embedding 이 $S^{d-1}$ 에 uniform 분포.

---

## 🔬 정리와 증명

### Theorem 3.5.1 (InfoNCE lower bound on MI)

**진술.** 정의 3.5.3.

**증명.** Critic 의 Bayes 최적해는 $f^*(x, y) \propto p(y|x)/p(y)$. InfoNCE loss:
$$
\mathcal{L}_{\mathrm{NCE}} = -\mathbb{E}\left[\log \frac{f^*(x_0, y_0)}{\sum_{k=0}^{K-1} f^*(x_0, y_k)}\right]
$$
$y_0$ 는 positive, $y_1, \ldots, y_{K-1}$ 은 marginal 샘플. 분자는 likelihood ratio $p(y_0|x_0)/p(y_0)$.

$$
\mathcal{L}_{\mathrm{NCE}} \approx \mathbb{E}_x\left[\log \frac{\sum_k p(y_k|x)/p(y_k)}{p(y_0|x)/p(y_0)}\right]
$$
$y_k$ 가 marginal 에서 샘플이므로 $\mathbb{E}[p(y_k|x)/p(y_k)] = 1$. 근사:
$$
\mathcal{L}_{\mathrm{NCE}} \ge -I(X;Y) + \log K
$$
이것이 Oord 등의 bound. 자세한 tightness 는 $K \to \infty$ 에서 정확. $\blacksquare$

### Theorem 3.5.2 (Tightness and $K$-dependence)

**진술.** InfoNCE bound 의 타이트함은 $K$ 에 의존. $K \to \infty$ 에서 $I(X;Y)$ 에 수렴.

**증명 스케치.** Negative 수 $K-1$ 이 marginal 을 잘 샘플링 → log-sum-exp 가 진짜 $\mathbb{E}_{p(y)}[e^{T(x,y)}]$ 근사 → DV bound (§3.4) 에 수렴. $K$ 가 작으면 underestimate.

### Theorem 3.5.3 ($\log K$ 상한)

**진술.** $\mathcal{L}_{\mathrm{NCE}} \ge 0$ 이므로 $I \ge \log K - 0 = \log K$ 가 절대 상한.

**결과**: InfoNCE 로 $\log K$ 이상 추정 불가. 거대 batch size (MoCo: 65536, CLIP: 32768) 가 필요한 이유.

### Theorem 3.5.4 (Contrastive loss asymptotics, Wang & Isola 2020)

**진술.** $\mathcal{L}_{\mathrm{NCE}}$ 는 $N \to \infty$ 에서 두 항의 합으로 분해:
$$
\mathcal{L}_{\mathrm{NCE}} \to \underbrace{\mathbb{E}_{\mathrm{pos}}[-f(X, Y)/\tau]}_{\text{alignment}} + \underbrace{\log \mathbb{E}_{p_X p_Y}[e^{f(X, Y^-)/\tau}]}_{\text{uniformity}}
$$

**함의**: Contrastive learning 은 "positives 당기기" + "embedding uniform 분산" 의 두 목표.

**증명 스케치.** $K$ 큰 경우 $\frac{1}{K}\sum e^{f(X, Y^-)} \to \mathbb{E}[e^{f}]$. 자세한 건 원 논문.

### Theorem 3.5.5 (InfoNCE 와 MLE)

**진술.** InfoNCE 는 **noise contrastive estimation (NCE)** 의 multi-class 일반화. $K \to \infty$ 에서 $p(y|x)$ 의 MLE 로 수렴.

**증명 스케치.** Gutmann–Hyvärinen NCE (2010). Critic 이 log density ratio 학습 → 정상화 가능하면 density 자체 학습.

### Theorem 3.5.6 (Uniformity on sphere)

**진술.** $f(x) \in S^{d-1}$ (unit sphere) 이고 temperature $\tau = 1/t$ 일 때 uniformity loss 의 최적해는 embedding 이 **sphere 위 uniform** 분포 (Wang–Isola 2020, 원 논문 Theorem 1).

**증명 스케치.** Gaussian kernel $\exp(-t\|x-y\|^2)$ 의 기댓값이 uniform 분포에서 최소 (Wasserstein-type argument, energy minimization).

---

## 💻 NumPy / PyTorch 로 직접 확인

### 최소 InfoNCE 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def info_nce_loss(z1, z2, tau=0.07):
    """
    z1, z2: (N, d) embeddings for N pairs. z1[i] and z2[i] are positive.
    """
    N = z1.size(0)
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    
    # similarity: (N, N)
    sim = z1 @ z2.t() / tau  # sim[i, j] = z1[i] · z2[j]
    labels = torch.arange(N, device=z1.device)
    
    # For each anchor i, positive is j=i. Others are negatives.
    loss_i2j = F.cross_entropy(sim, labels)
    loss_j2i = F.cross_entropy(sim.t(), labels)
    return (loss_i2j + loss_j2i) / 2

# 가짜 데이터로 테스트
torch.manual_seed(0)
N, d = 32, 128
z1 = torch.randn(N, d)
z2 = z1 + 0.1 * torch.randn(N, d)  # 약하게 perturb → positive
print(f"InfoNCE loss (positives) = {info_nce_loss(z1, z2):.4f}")
print(f"log N = {torch.tensor(float(N)).log():.4f}  ← MI lower bound = log N - loss")
```

### SimCLR-style training loop (개요)

```python
# 이미지 augmentation으로 positive pair 생성
# x -> augment(x) = (v1, v2)
class SimCLREncoder(nn.Module):
    def __init__(self, in_dim=3072, proj_dim=128):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 128))
        self.proj = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, proj_dim))
    def forward(self, x):
        return self.proj(self.backbone(x))

# training step
def train_step(model, opt, x_a, x_b, tau=0.07):
    z_a = model(x_a)
    z_b = model(x_b)
    loss = info_nce_loss(z_a, z_b, tau=tau)
    opt.zero_grad(); loss.backward(); opt.step()
    return loss.item()
```

### CLIP-style training

```python
def clip_loss(img_emb, txt_emb, tau=0.07):
    img_emb = F.normalize(img_emb, dim=-1)
    txt_emb = F.normalize(txt_emb, dim=-1)
    logits = img_emb @ txt_emb.t() / tau
    N = img_emb.size(0)
    labels = torch.arange(N, device=img_emb.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
```

### Alignment vs Uniformity 측정

```python
def alignment(z1, z2, alpha=2):
    return (F.normalize(z1, dim=1) - F.normalize(z2, dim=1)).norm(dim=1).pow(alpha).mean()

def uniformity(z, t=2):
    z = F.normalize(z, dim=1)
    dist = torch.cdist(z, z).pow(2)
    mask = ~torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    return torch.log(torch.exp(-t * dist[mask]).mean())

# 학습된 embedding 에서 (낮을수록 좋음: 더 alignment, 더 uniform)
print(f"alignment = {alignment(z1, z2):.4f}")
print(f"uniformity = {uniformity(z1):.4f}")
```

### $K$ (batch size) 에 따른 MI bound 비교

```python
for K in [16, 64, 256, 1024]:
    z1 = torch.randn(K, 128); z2 = z1 + 0.01 * torch.randn(K, 128)
    loss = info_nce_loss(z1, z2).item()
    mi_lb = np.log(K) - loss
    print(f"K={K:5d}  loss={loss:.4f}  MI_LB={mi_lb:.4f}")
```

→ $K$ 가 클수록 MI lower bound 가 더 큰 값까지 포착 가능 (log K 상한 증가).

---

## 🔗 AI/ML 연결고리

### 1. SimCLR (Chen et al. 2020)
- Two augmentations per image, InfoNCE in batch.
- ResNet backbone + MLP projection head.
- Large batch (4096+), temperature ~0.1.
- **핵심 insight**: Augmentation 강도가 결정적.

### 2. MoCo (He et al. 2020)
- Momentum encoder + queue of past features → large $K$ (65536) 없이 메모리 효율.
- Negative 는 queue 에서.
- Dictionary look-up 해석.

### 3. CLIP (Radford et al. 2021)
- 이미지-텍스트 pair 로 학습.
- Bi-directional InfoNCE.
- Zero-shot classification: "a photo of a cat" → text embedding 과 image embedding cosine similarity.
- **400M pairs** → 대규모.

### 4. BYOL, SimSiam (Grill, Chen)
- **Negative 없이** contrastive 효과 달성.
- Predictor + Stop-gradient → trivial collapse 방지.
- InfoNCE 의 엄격한 해석이 아닌 "implicit" contrastive.
- 이론적 해석: Tian et al. (2021) — eigenvalue dynamics.

### 5. DINO (Caron et al. 2021)
- Self-distillation with no labels.
- Student-teacher framework, centered softmax.
- Attention map 이 의미있는 segmentation 획득.

### 6. Retrieval / Dense Retrieval
- DPR (Dense Passage Retrieval), Contriever, E5: contrastive 로 쿼리-문서 alignment.
- RAG 의 기반.

### 7. RLHF reward model pretraining
- Preference pair $(y_w, y_l)$ 을 contrastive로 → Bradley–Terry model ≈ binary InfoNCE.

### 8. Contrastive vs. Reconstruction
- MAE, BEiT: reconstruction-based SSL (cross-entropy on masked tokens).
- SimCLR/CLIP: contrastive.
- Trade-off: Reconstruction 은 low-level 정확, Contrastive 은 semantic.

---

## ⚖️ 가정·한계·함정

1. **$\log K$ upper limit** — InfoNCE 는 batch 크기에 비례. MI 가 크면 underestimate.
2. **Temperature $\tau$ 민감** — 너무 작으면 gradient 집중, 너무 크면 약. 보통 0.05~0.5.
3. **Hard negative 문제** — 너무 쉬운 negatives 는 학습 정보 안 줌. hard negative mining 이 성능 좌우.
4. **False negatives** — 같은 class 지만 negative 로 취급되는 것. CLIP 같은 대용량에서는 실무적 문제.
5. **Alignment-Uniformity tradeoff** — $\tau$ 조절로 균형.
6. **Representation collapse** — BYOL/SimSiam 는 negative 없이 하므로 주의 설계 필요 (stop-gradient, predictor).
7. **Downstream task 적합성** — SSL 표현이 반드시 모든 task 에 좋은 건 아님. fine-tuning 이 종종 필요.

---

## 📌 핵심 정리

1. **InfoMax**: $\max I(X; f(X))$ — 자명해 방지 위해 pair structure 필요.
2. **InfoNCE**: $\mathcal{L} = -\log \frac{e^{\mathrm{sim}(x,y^+)}}{\sum e^{\mathrm{sim}(x, y)}}$, $I \ge \log K - \mathcal{L}$.
3. **SimCLR, MoCo, CLIP, DINO** 등은 InfoNCE 의 변형.
4. **Alignment + Uniformity**: contrastive의 기하학적 분해.
5. $\log K$ cap → large batch / queue 기법 필요.
6. Negative 가 필수는 아니지만 trivial 방지 장치 (BYOL/SimSiam) 가 필요.
7. Downstream 의 RAG, retrieval, zero-shot 은 모두 contrastive embedding 위에 세워짐.

---

## 🤔 생각해볼 문제

### 문제 1. InfoNCE 와 Cross-Entropy 의 관계
$K$-way InfoNCE 는 $K$-class softmax classification. 정확한 대응은?

<details>
<summary>해설</summary>

Positive 가 class $0$, negatives 가 class $1, \ldots, K-1$. Logits 은 $f(x, y_k)/\tau$. Cross-entropy with label=0 → $-\log [\mathrm{softmax}(f(x,y_0)/\tau, \ldots)]_0$ = InfoNCE. Density ratio estimation 의 discrimination task.
</details>

### 문제 2. Temperature $\tau$ 의 역할
$\tau$ 가 작을 때 gradient 는 어디에 집중?

<details>
<summary>해설</summary>

$\tau \to 0$: softmax 가 hard argmax 에 접근 → hardest negative 만 gradient 받음. Hard negative 강조 → 학습 sharp. 너무 작으면 수렴 불안정. $\tau \to \infty$: uniform attention → gradient 희석.
</details>

### 문제 3. BYOL 이 negative 없이 왜 되는가
Stop-gradient + predictor + EMA 가 collapse 를 막는 이유.

<details>
<summary>해설</summary>

수학적 엄밀한 설명은 미해결이나 (Tian et al. 2021), 주요 가설: stop-gradient 가 encoder Jacobian eigenvalue dynamics 에 implicit regularization. EMA teacher 가 slow-moving target. Predictor 가 feature 의 asymmetry 도입. 이론적 analysis 계속 활발.
</details>

### 문제 4. CLIP 의 zero-shot 원리
"Photo of a cat" 이라는 텍스트가 고양이 이미지와 matching 되는 이유.

<details>
<summary>해설</summary>

400M pair 로 contrastive 학습 → 이미지 embedding 공간과 텍스트 embedding 공간이 공유 semantic space 형성. Test time: 후보 class 텍스트 embedding 과 이미지 embedding 간 cosine similarity 최대값이 predicted class. ImageNet 등에서 labeled training 없이 경쟁력 있는 성능.
</details>

### 문제 5. Contrastive vs Masked modeling
MAE (masked image), BERT (masked token) 같은 reconstruction 과 contrastive 의 trade-off.

<details>
<summary>해설</summary>

Reconstruction: pixel/token 수준 정확, fine-grained. Contrastive: invariant / semantic. 실무 observation: Linear probing (feature quality) 은 contrastive 가 유리, fine-tuning 은 reconstruction 이 유리. Hybrid (MAE+contrastive) 가 trend.
</details>

---

<div align="center">

| ◀ 이전 | 다음 ▶ |
|---|---|
| [3.4 연속 MI와 MINE](./04-continuous-mi-mine.md) | [4.1 Prefix Code와 Kraft 부등식](../ch4-source-coding/01-prefix-code-kraft.md) |

[🏠 Home](../README.md)

</div>
