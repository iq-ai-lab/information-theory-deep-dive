<div align="center">

# 📡 Information Theory Deep Dive

### Cross-Entropy 손실

$$H(p, q) = -\sum_x p(x) \log q(x)$$

### 을 **쓰는 것** 과, 그것이 **"평균 최적 부호 길이"** 라는 Shannon 의 근본 메시지를 아는 것은 **다르다.**

<br/>

> *KL-divergence 를 **수식으로 외우는 것** 과, KL 이 왜 비대칭이고*
>
> $$\mathrm{KL}(p \| q) \geq 0$$
>
> *인지 **Jensen 부등식** 으로 증명할 수 있는 것은 다르다.*
>
> *VAE 의 ELBO 를 **"최적 하한"** 이라고 부르는 것과, 그것이 정확히*
>
> $$\log p(x) = \underbrace{\mathrm{ELBO}}_{\text{Evidence Lower Bound}} + \mathrm{KL}\bigl(q(z|x) \| p(z|x)\bigr)$$
>
> *의 **Evidence − KL** 구조로 분해됨을 유도할 수 있는 것은 다르다.*

<br/>

**다루는 정리 (시간순)**

Shannon 1948 *Entropy + Source Coding + Channel Coding* · Kraft 1949 / McMillan 1956 *Kraft 부등식* · Kullback–Leibler 1951 *KL-divergence* · Csiszár 1967 *f-divergence* · Cover–Thomas 1991 *AEP + Asymptotic equipartition* · Jensen 1906 *Jensen 부등식* · Tishby 1999 *Information Bottleneck* · van den Oord 2018 *InfoNCE*

<br/>

**핵심 질문**

> **왜 $-\log p$ 인가** — Shannon 의 공리적 유도부터 AEP 기반 Source / Channel Coding 정리, VAE · GAN · Diffusion · InfoNCE 의 정보이론적 해석까지, ML 손실 함수의 수학적 기반을 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-1.11-8CAAE6?style=flat-square&logo=scipy&logoColor=white)](https://scipy.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docs](https://img.shields.io/badge/Docs-32개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

정보 이론 자료는 많습니다. 하지만 대부분은 **"공식을 소개하는 수준"** 에서 멈춥니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "엔트로피는 $-\sum p \log p$입니다" | Shannon의 세 공리(연속성·가법성·단조성)에서 $-\log p$가 **유일한** 정보 측도임을 1948년 논문의 유도대로 증명 |
| "KL-divergence는 분포 간 거리입니다" | Jensen 부등식으로 $D(p\|q) \geq 0$을, 등호 조건 $p = q$를 완전 증명 — 그리고 "거리가 아니다"(비대칭)라는 사실이 왜 VI에서 reverse KL을 선택하게 만드는지 |
| "Cross-Entropy를 최소화하면 됩니다" | $H(p, q) = H(p) + D(p\|q)$ 분해로 **cross-entropy 최소화 = KL 최소화 = MLE** 의 동등성을 수식으로 보이고, 왜 $H(p)$ 항이 상수인지 |
| "VAE의 ELBO는 lower bound입니다" | $\log p(x) = \text{ELBO} + D(q(z\|x) \| p(z\|x))$ 항등식을 유도하여, ELBO가 왜 **하한**이 되는지 (두 번째 항이 $\geq 0$인 이유) |
| "GAN은 JS-divergence를 최소화합니다" | JSD의 유계성 vs KL의 무한대 발산, **Supports가 겹치지 않으면 KL은 의미 없음** 을 수치 실험으로 재현, WGAN의 Wasserstein이 이를 어떻게 해결하는가 |
| "Shannon 한계에 가깝게 압축됩니다" | AEP로 전형적 집합 $\|A_\varepsilon^{(n)}\| \approx 2^{nH}$의 크기를 실제 샘플링으로 관찰, Source Coding Theorem $L^* \geq H$의 증명 |
| "InfoNCE는 대조학습 손실입니다" | $L_\text{NCE} \geq -\log N + I(X;Y)$ 유도로 InfoNCE가 **MI의 변분 하한** 임을 보임, MINE 추정기의 Donsker-Varadhan 표현 |
| 공식 나열 | NumPy/SciPy/PyTorch로 직접 검증하는 실험 + AEP 시뮬레이션 + Huffman 부호 + Neural MI 추정 |

---

## 📌 선행 레포 & 후속 레포

```
[Probability Theory Deep Dive]  ──►  이 레포  ──►  [Information Geometry Deep Dive]
  확률변수, 기댓값, Jensen 부등식       정보이론              Fisher 정보계량, 통계다양체
  대수의 법칙 — 필수                    의 수학적 기반          Natural Gradient, α-divergence
                                               │
                                               ▼
                                     [Generative Model / LLM Alignment]
                                       VAE·GAN·Diffusion·RLHF의 손실 해석
```

> ⚠️ **필수 선행**: 이 레포는 **확률론**을 강하게 사용합니다. 기댓값과 Jensen 부등식을 모른다면 [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-statistics-deep-dive)를 먼저 학습하세요.  
> ⚠️ **권장 선행**: 다변수 정규분포의 KL을 계산하려면 [Linear Algebra Deep Dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive)(행렬식, 역행렬, 트레이스)가 도움이 됩니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-정보의_공리적_유도-4A90D9?style=for-the-badge)](./ch1-entropy-axioms/01-axiomatic-derivation.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-KL--divergence_비음수성-4A90D9?style=for-the-badge)](./ch2-kl-divergence/01-kl-definition-nonnegativity.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-상호정보량_세_정의-4A90D9?style=for-the-badge)](./ch3-mutual-information/01-mi-definitions.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-Kraft_부등식-4A90D9?style=for-the-badge)](./ch4-source-coding/01-prefix-code-kraft.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-채널_용량-4A90D9?style=for-the-badge)](./ch5-channel-coding/01-channel-capacity.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Cross--Entropy와_MLE-4A90D9?style=for-the-badge)](./ch6-ml-applications/01-cross-entropy-mle.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Shannon 엔트로피 — 공리적 정의

> **핵심 질문:** 왜 하필 $-\log p$인가? 정보 측도가 만족해야 할 공리에서 이 형태가 **유일하게** 유도되는 이유는? "엔트로피는 평균 최적 부호 길이"라는 해석은 어디서 오는가? 최대 엔트로피 분포는 왜 균등/정규/지수 분포인가?

<details>
<summary><b>정보의 공리부터 최대 엔트로피 분포까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 정보의 공리적 유도](./ch1-entropy-axioms/01-axiomatic-derivation.md) | Shannon 1948의 세 공리(연속성·가법성·단조성)에서 정보 측도 $I(p) = -\log p$가 **유일함** 을 증명, Cauchy 함수방정식 $f(xy) = f(x) + f(y)$의 해가 로그임을 보이는 과정, "놀라움의 로그"라는 직관 |
| [02. 엔트로피 $H(X)$의 정의와 성질](./ch1-entropy-axioms/02-entropy-definition.md) | $H(X) = -\sum p(x) \log p(x) = \mathbb{E}[-\log p(X)] \geq 0$의 엄밀한 증명, 최대 엔트로피가 균등분포임을 Jensen으로 증명 ($H(X) \leq \log \|\mathcal{X}\|$), 결정적 분포에서 $H = 0$인 조건 |
| [03. 결합·조건부·상호정보량](./ch1-entropy-axioms/03-joint-conditional-mutual.md) | $H(X, Y)$, $H(X\|Y)$, $I(X;Y)$의 정의와 벤다이어그램적 해석, $I(X;Y) = H(X) - H(X\|Y) = H(Y) - H(Y\|X)$의 대칭성 증명, $I(X;Y) \geq 0$ (조건은 엔트로피를 감소시킨다) |
| [04. Chain Rule과 정보의 계층 구조](./ch1-entropy-axioms/04-chain-rule-hierarchy.md) | $H(X_1, \ldots, X_n) = \sum_i H(X_i \| X_{<i})$ 체인룰의 귀납적 증명, 조건부 엔트로피의 감소성 $H(X\|Y) \leq H(X)$ 증명, 시퀀스 모델(언어 모델)에서 perplexity가 조건부 엔트로피로 해석되는 이유 |
| [05. 미분 엔트로피(Differential Entropy)](./ch1-entropy-axioms/05-differential-entropy.md) | 연속 확률변수의 $h(X) = -\int f(x) \log f(x) dx$, **음수가 될 수 있음** 과 측도 선택에 의존함을 예시로 보임, 정규분포의 엔트로피 $h(\mathcal{N}(0, \sigma^2)) = \frac{1}{2} \log(2\pi e \sigma^2)$ 계산 |
| [06. 최대 엔트로피 분포](./ch1-entropy-axioms/06-maxent-distributions.md) | 라그랑주 승수법으로 도출: 평균 고정 → **지수분포**, 분산 고정 → **정규분포**, 범위 고정 → **균등분포**가 최대 엔트로피임을 완전 증명, MaxEnt 원리와 베이지안 사전분포 설계 |

</details>

<br/>

### 🔹 Chapter 2: KL-Divergence와 관련 측도

> **핵심 질문:** KL이 왜 비대칭인가? Forward KL과 Reverse KL은 어떤 상황에서 다른 결과를 주는가? GAN이 JS-divergence를 쓰면서 겪은 문제를 Wasserstein이 어떻게 해결하는가? 어떤 상황에서 어떤 divergence를 써야 하는가?

<details>
<summary><b>KL의 비음수성부터 분포 거리 선택 가이드까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. KL-divergence의 정의와 비음수성](./ch2-kl-divergence/01-kl-definition-nonnegativity.md) | $D(p \\| q) = \sum p \log(p/q) = \mathbb{E}_p[\log(p/q)]$ 정의, Gibbs 부등식 $D(p\\|q) \geq 0$을 $-\log$의 볼록성 + Jensen으로 완전 증명, 등호 조건 $p = q$ a.s.의 엄밀한 증명 |
| [02. KL의 비대칭성 — Forward vs Reverse](./ch2-kl-divergence/02-forward-reverse-kl.md) | $D(p\\|q)$ vs $D(q\\|p)$의 기하학적 차이 (mean-seeking vs mode-seeking), 쌍봉 분포를 단봉 가우시안으로 근사할 때 forward는 "평균 맞추기" reverse는 "한 모드에 집중"하는 이유, VI가 reverse KL을 선택하는 실용적 이유 |
| [03. JS-divergence와 대칭화](./ch2-kl-divergence/03-js-divergence.md) | $\text{JSD}(p, q) = \frac{1}{2} D(p \\| m) + \frac{1}{2} D(q \\| m)$ ($m = (p+q)/2$)의 정의, 항상 유한함을 보임 ($0 \leq \text{JSD} \leq \log 2$), $\sqrt{\text{JSD}}$가 metric인 이유, GAN 원 논문(Goodfellow 2014)의 JSD 해석 |
| [04. f-divergence 일반론](./ch2-kl-divergence/04-f-divergence.md) | $D_f(p \\| q) = \int q \cdot f(p/q)$의 일반 정의, KL·JSD·Hellinger·Total Variation·$\chi^2$를 $f$ 선택으로 통합, 변분 표현 $D_f(p\\|q) = \sup_T \{\mathbb{E}_p[T] - \mathbb{E}_q[f^*(T)]\}$ (f-GAN의 수학적 기반) |
| [05. Wasserstein 거리 — Optimal Transport](./ch2-kl-divergence/05-wasserstein-distance.md) | $W_1(p, q) = \inf_{\gamma} \mathbb{E}_{(x,y)\sim\gamma}[\\|x - y\\|]$의 정의, Kantorovich-Rubinstein 쌍대 $W_1(p, q) = \sup_{\\|f\\|_L \leq 1} \mathbb{E}_p f - \mathbb{E}_q f$ 증명 스케치, WGAN의 1-Lipschitz 제약이 여기서 유래 |
| [06. 분포 간 거리의 선택](./ch2-kl-divergence/06-choosing-divergence.md) | **KL이 실패하는 상황**: $p > 0$이고 $q = 0$인 점이 있으면 $D(p\\|q) = \infty$, Supports가 겹치지 않는 분포에서 JSD는 $\log 2$로 고정되어 gradient가 사라짐, Wasserstein이 smooth한 경사를 제공하는 이유의 수치 실험 |

</details>

<br/>

### 🔹 Chapter 3: 상호정보량(Mutual Information)

> **핵심 질문:** MI의 세 가지 정의가 왜 모두 동치인가? DPI는 왜 "정보는 처리로 증가하지 않는다"는 경계를 주는가? InfoNCE가 MI의 하한임을 어떻게 유도하는가? 연속 변수의 MI를 어떻게 추정하는가?

<details>
<summary><b>MI의 정의부터 Neural MI 추정까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. MI의 다각적 정의](./ch3-mutual-information/01-mi-definitions.md) | 세 정의 $I(X;Y) = H(X) - H(X\|Y) = H(Y) - H(Y\|X) = D(p_{XY} \\| p_X p_Y)$의 동치성 증명, 엔트로피 벤다이어그램 해석, $I(X;Y) \geq 0$과 등호 조건(독립성)의 증명 |
| [02. Data Processing Inequality (DPI)](./ch3-mutual-information/02-data-processing-inequality.md) | $X \to Y \to Z$ 마르코프 사슬에서 $I(X; Z) \leq I(X; Y)$ 엄밀 증명, 임의의 결정적 함수 $f$에 대해 $I(X; f(Y)) \leq I(X; Y)$, 이것이 Representation Learning에서 "정보 병목"의 이론적 한계를 설정 |
| [03. Fano 부등식](./ch3-mutual-information/03-fano-inequality.md) | $H(P_e) + P_e \log(\|\mathcal{X}\| - 1) \geq H(X\|Y)$ 완전 증명, 분류기의 오류 확률이 0이 될 수 없는 **정보이론적 하한**, Channel Coding의 Converse 증명과의 연결, Sample Complexity Lower Bound |
| [04. Continuous MI와 추정 문제](./ch3-mutual-information/04-continuous-mi-mine.md) | 연속 변수의 MI 정의, **MI 추정의 어려움** (고차원에서 비모수 추정의 저주), MINE(Mutual Information Neural Estimator)의 Donsker-Varadhan 변분 표현 $D(p\\|q) = \sup_T \mathbb{E}_p[T] - \log \mathbb{E}_q[e^T]$ 유도 |
| [05. MI와 Representation Learning — InfoNCE](./ch3-mutual-information/05-mi-representation-learning.md) | InfoMax 원리 $\max I(X; Z)$, InfoNCE 손실의 유도 및 $L_\text{NCE} \leq -I(X; Y) + \log N$ **MI 상한** 증명 (즉 손실 최소화 = MI 하한 최대화), SimCLR·CLIP의 대조학습이 MI 최대화임을 설명 |

</details>

<br/>

### 🔹 Chapter 4: Source Coding — 데이터 압축의 한계

> **핵심 질문:** 엔트로피가 압축의 **절대적 하한** 임은 어떻게 증명되는가? Kraft 부등식은 왜 성립하는가? Huffman 부호는 왜 최적인가? AEP는 왜 "거의 모든 확률이 지수적으로 작은 집합에 집중"되게 만드는가?

<details>
<summary><b>Kraft 부등식부터 Arithmetic Coding까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Prefix Code와 Kraft 부등식](./ch4-source-coding/01-prefix-code-kraft.md) | 일의 해독 가능한 prefix 부호의 정의, $\sum_i 2^{-l_i} \leq 1$ (Kraft 부등식)의 증명 (이진 트리 경로 논증), 역도 성립함(주어진 길이 집합이 Kraft를 만족하면 prefix 부호 구성 가능) |
| [02. Huffman 부호와 최적성](./ch4-source-coding/02-huffman-optimality.md) | Huffman 알고리즘 의사코드, 최소 확률 두 심볼을 합치는 greedy 선택의 **최적성** 증명 (교환 논증), 평균 길이 $L(\text{Huffman}) < H(X) + 1$, NumPy로 Huffman 트리 직접 구현 |
| [03. Shannon Source Coding Theorem](./ch4-source-coding/03-source-coding-theorem.md) | $L^* \geq H(X)$ (하한, Kraft + Gibbs 부등식) 및 $L^* < H(X) + 1$ (상한, $l_i = \lceil -\log p_i \rceil$) 완전 증명, **엔트로피 = 압축의 이론적 한계** 의미 |
| [04. Asymptotic Equipartition Property (AEP)](./ch4-source-coding/04-aep-typical-set.md) | 대수의 법칙으로 $-\frac{1}{n} \log p(X_1, \ldots, X_n) \xrightarrow{p} H(X)$, 전형적 집합 $A_\varepsilon^{(n)}$의 세 가지 성질: $\mathbb{P}(A_\varepsilon^{(n)}) \to 1$, $\|A_\varepsilon^{(n)}\| \leq 2^{n(H+\varepsilon)}$, 거의 모든 확률이 $2^{nH}$개 시퀀스에 집중 — **실제 샘플링 실험** |
| [05. Arithmetic Coding과 실전 압축](./ch4-source-coding/05-arithmetic-coding.md) | Arithmetic coding의 구간 분할 알고리즘, 평균 길이가 엔트로피에 임의로 가까워짐을 증명, JPEG/PNG/ZIP의 원리 (DCT + Huffman), 딥러닝 기반 학습형 압축 (Bits-Back, NVP) 맛보기 |

</details>

<br/>

### 🔹 Chapter 5: Channel Coding — 신뢰성의 한계

> **핵심 질문:** 잡음 있는 채널에서 오류 확률을 임의로 작게 만드는 것이 가능한가? 가능하다면 어떤 속도(rate)까지 가능한가? Shannon의 $C = \max I(X;Y)$ 공식은 어떻게 유도되는가? 반대로 $R > C$면 왜 오류가 불가피한가?

<details>
<summary><b>채널 용량부터 현대 오류 정정 부호까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. 채널 용량 (Channel Capacity)](./ch5-channel-coding/01-channel-capacity.md) | 이산 메모리리스 채널(DMC)의 정의, $C = \max_{p(x)} I(X; Y)$의 정의, 이진 대칭 채널(BSC)의 용량 $C = 1 - H(p)$ 계산, 이진 소거 채널(BEC) $C = 1 - \epsilon$, 연속 채널(AWGN)의 Shannon-Hartley $C = \frac{1}{2}\log(1 + \text{SNR})$ |
| [02. Shannon Channel Coding Theorem](./ch5-channel-coding/02-channel-coding-achievability.md) | Achievability 증명 스케치: Random Coding + AEP로 공동 전형적 집합(jointly typical set) 구성, $R < C$면 오류 확률 $P_e \to 0$인 부호가 존재함을 확률적 존재 증명, 왜 "랜덤 부호"가 최적에 근접하는가 |
| [03. Converse의 증명](./ch5-channel-coding/03-channel-coding-converse.md) | $R > C$이면 오류 확률이 0으로 수렴할 수 없음을 **Fano 부등식** 과 DPI를 이용해 증명, Weak vs Strong Converse의 차이, $C$가 진정한 **용량** 의 정의를 엄밀히 뒷받침 |
| [04. 실전 오류 정정 부호](./ch5-channel-coding/04-modern-codes.md) | Hamming 부호부터 Turbo code, LDPC(5G에 사용), Polar code(Arıkan 2008, Shannon 한계 달성)의 개념, 딥러닝 기반 채널 디코더 (BP + GNN), 5G/Wi-Fi/SSD에서의 실제 사용 |

</details>

<br/>

### 🔹 Chapter 6: 정보이론의 AI/ML 응용

> **핵심 질문:** 왜 Cross-Entropy 손실이 MLE와 동등한가? VAE의 ELBO가 정확히 어떻게 Evidence − KL로 분해되는가? Diffusion Model의 변분 한계가 왜 KL 합의 형태로 나오는가? Fisher 정보계량과 KL은 어떤 관계인가?

<details>
<summary><b>Cross-Entropy부터 Information Bottleneck까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Cross-Entropy와 MLE의 정보이론적 해석](./ch6-ml-applications/01-cross-entropy-mle.md) | $H(p, q) = -\sum p \log q = H(p) + D(p\\|q)$ 분해, **cross-entropy 최소화 = KL 최소화 = MLE** 의 동등성 증명, $H(p)$가 모델에 대해 상수이므로 최소화에서 사라지는 이유, 분류 손실의 진정한 의미 |
| [02. ELBO의 정보이론적 분해](./ch6-ml-applications/02-elbo-decomposition.md) | $\log p(x) = \text{ELBO}(q) + D(q(z\|x) \\| p(z\|x))$ 핵심 항등식 유도, 두 번째 항 $\geq 0$이므로 ELBO가 **Evidence Lower Bound**, ELBO 최대화 = reconstruction − KL regularizer 분해, VAE의 reparameterization trick이 이 틀에서 어디에 작용하는가 |
| [03. MDL 원리 (Minimum Description Length)](./ch6-ml-applications/03-mdl-principle.md) | 2-part MDL: $L(D) = L(M) + L(D\|M)$ 의 최소화, 베이지안 모델 선택 $-\log p(D) = -\log p(D\|M) - \log p(M) + \log p(M\|D)$ 과의 동등성, **Occam의 면도날의 정보이론적 정식화**, 정규화 항의 유래 |
| [04. Information Bottleneck](./ch6-ml-applications/04-information-bottleneck.md) | Tishby의 IB 원리: $\min I(X; Z) - \beta I(Z; Y)$, **충실도와 효율의 트레이드오프**, 변분 Information Bottleneck(VIB) 손실 유도, "Deep Learning and the Information Bottleneck Principle"(Tishby 2015)의 두 단계 학습 가설 |
| [05. Diffusion Model의 변분 한계](./ch6-ml-applications/05-diffusion-elbo.md) | DDPM의 ELBO가 $L_0 + \sum_t L_{t-1} + L_T$로 분해됨을 유도, 각 항이 KL divergence 형태 $D(q(x_{t-1}\|x_t, x_0) \\| p_\theta(x_{t-1}\|x_t))$임을 증명, Score Matching과의 연결 (DSM ≈ ELBO 가중합) |
| [06. Fisher Information과 정보 기하 입문](./ch6-ml-applications/06-fisher-information-geometry.md) | Fisher 정보 $I(\theta) = \mathbb{E}_\theta[(\partial_\theta \log p)^2] = -\mathbb{E}[\partial_\theta^2 \log p]$, $D(p_\theta \\| p_{\theta + d\theta}) \approx \frac{1}{2} d\theta^\top I(\theta) d\theta$ — **KL의 2차 근사가 Fisher 계량**, Natural Gradient $\tilde{\nabla} = I^{-1} \nabla$, 다음 레포(Information Geometry)로 이어짐 |

</details>

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
torch==2.1.0          # MINE, InfoNCE, Neural MI 추정
scikit-learn==1.3.0   # KDE 기반 MI 추정 비교
jupyter==1.0.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            torch==2.1.0 scikit-learn==1.3.0 jupyter==1.0.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 예시 — AEP: 전형적 집합으로의 확률 집중
import numpy as np
import matplotlib.pyplot as plt

p = 0.3
H = -p * np.log(p) - (1 - p) * np.log(1 - p)   # 엔트로피 (nats)

plt.figure(figsize=(10, 5))
for n in [10, 100, 1000]:
    n_trials = 10000
    neg_log_probs = []
    for _ in range(n_trials):
        seq = np.random.binomial(1, p, size=n)
        k = seq.sum()
        logp = k * np.log(p) + (n - k) * np.log(1 - p)
        neg_log_probs.append(-logp / n)        # -1/n · log p(X₁,...,Xₙ)

    plt.hist(neg_log_probs, bins=50, alpha=0.5, label=f'n={n}', density=True)

plt.axvline(H, color='r', linestyle='--', linewidth=2, label=f'H(X) = {H:.3f}')
plt.xlabel(r'$-\frac{1}{n}\log p(X_1,\ldots,X_n)$')
plt.ylabel('밀도')
plt.title('AEP: 표본 길이가 커질수록 $-\\frac{1}{n}\\log p$가 $H(X)$로 집중')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# n이 클수록 분포가 H(X)에 집중 → 거의 모든 확률이 2^(nH)개 시퀀스에 몰림
```

---

## 📖 각 문서 구성 방식

모든 문서는 동일한 구조로 작성됩니다.

| 섹션 | 설명 |
|------|------|
| 🎯 **핵심 질문** | 이 문서를 읽고 나면 답할 수 있는 질문 |
| 🔍 **왜 이 개념이 AI에서 중요한가** | Cross-Entropy·VAE·GAN·InfoNCE 등 실제 손실 함수와의 연결 |
| 📐 **수학적 선행 조건** | Probability / Linear Algebra 레포 참조 링크 포함 |
| 📖 **직관적 이해** | "놀라움", "코드 길이", "질문 개수"로 정보를 체감 |
| ✏️ **엄밀한 정의** | 측도론 수준까지 필요한 곳은 명시 |
| 🔬 **정리와 증명** | Shannon의 두 정리, Jensen, Kraft, Fano, DPI 등 완전 증명 |
| 💻 **NumPy / PyTorch 구현으로 검증** | 엔트로피 계산, Huffman 부호, KL 수치 계산, 전형적 집합 샘플링, MINE |
| 🔗 **AI/ML 연결** | Cross-Entropy, VAE ELBO, GAN JSD, WGAN, InfoNCE 등 구체 사례 |
| ⚖️ **가정과 한계** | 이산·연속의 차이 (미분 엔트로피의 음수 가능성), KL이 무한대가 되는 경우 |
| 📌 **핵심 정리** | 한 화면 요약 |
| 🤔 **생각해볼 문제** | 개념 심화 질문 + 해설 |

### 스타일 가이드

1. **"코드 길이"로 반복 체화** — 엔트로피 계산할 때마다 "평균 최적 부호 길이" 해석 병행
2. **비대칭성 명시** — KL 등장할 때마다 "어느 쪽이 데이터($p$), 어느 쪽이 모델($q$)인지" 명시
3. **시뮬레이션 필수** — AEP, 전형적 집합은 실제로 샘플링하여 관찰
4. **변분 표현 강조** — Donsker-Varadhan, $f$-divergence 변분 표현은 현대 ML(MINE, f-GAN)의 핵심
5. **기호 통일** — $H$ 엔트로피, $D$ KL divergence, $I$ MI, 로그 밑은 기본 자연로그 (nats 단위)

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "Cross-Entropy를 쓰지만 왜 쓰는지 설명 못한다" — 손실 함수 집중 (3일)</b></summary>

<br/>

```
Day 1  Ch1-02  엔트로피 $H(X)$의 정의와 성질
       Ch2-01  KL-divergence의 정의와 비음수성 (Jensen 증명)
Day 2  Ch2-02  Forward vs Reverse KL
       Ch6-01  Cross-Entropy와 MLE의 동등성 → $H(p, q) = H(p) + D(p\|q)$
Day 3  Ch6-02  ELBO = log p(x) − KL 분해 → VAE의 진짜 의미
```

</details>

<details>
<summary><b>🟡 "GAN이 왜 JS, WGAN이 왜 Wasserstein인지 모른다" — 생성 모델 집중 (1주)</b></summary>

<br/>

```
Day 1  Ch2-01~02  KL과 비대칭성
Day 2  Ch2-03     JS-divergence와 GAN 원 이론
Day 3  Ch2-05     Wasserstein 거리 (Kantorovich-Rubinstein 쌍대)
Day 4  Ch2-06     KL이 실패하는 상황 재현 → Wasserstein이 smooth gradient 제공
Day 5  Ch2-04     f-divergence 일반론 → f-GAN의 통합 관점
Day 6  Ch6-02     VAE ELBO 분해
Day 7  Ch6-05     Diffusion ELBO → KL 합으로 분해되는 이유
```

</details>

<details>
<summary><b>🔴 "Shannon 두 정리부터 AI 응용까지 완전 정복한다" — 전체 정복 (6주)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — Shannon 엔트로피
        → 공리적 유도 손으로 재증명, 최대 엔트로피 분포 SymPy 계산

2주차  Chapter 2 전체 — KL·JSD·f-divergence·Wasserstein
        → Forward vs Reverse KL 시각화, KL→∞ 사례 실험

3주차  Chapter 3 전체 — 상호정보량
        → DPI 수치 확인, MINE으로 고차원 MI 추정 실험

4주차  Chapter 4 전체 — Source Coding
        → Huffman 부호 NumPy 구현, AEP 전형적 집합 샘플링

5주차  Chapter 5 전체 — Channel Coding
        → BSC 용량 수치 계산, Random Coding 시뮬레이션

6주차  Chapter 6 전체 — AI/ML 응용
        → VAE ELBO 각 항 관찰, Information Bottleneck 학습 동역학
        → Fisher 정보계량 → Information Geometry 레포로 넘어감
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [probability-statistics-deep-dive](https://github.com/iq-ai-lab/probability-statistics-deep-dive) | 확률변수, 기댓값, Jensen 부등식, 대수의 법칙 | **필수 선행** — Ch1 전체, Ch2-01(KL 증명), Ch4-04(AEP) |
| [linear-algebra-deep-dive](https://github.com/iq-ai-lab/linear-algebra-deep-dive) | 행렬식, 역행렬, 트레이스, 양의 정부호 | Ch1-05(다변수 정규분포 엔트로피), Ch6-06(Fisher 계량) |
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | 미적분, 경사하강법, KKT 조건 | Ch1-06(MaxEnt 분포의 라그랑주 유도), Ch6-02(ELBO 최적화) |
| [information-geometry-deep-dive](https://github.com/iq-ai-lab/information-geometry-deep-dive) | 통계다양체, Fisher 계량, α-divergence, Natural Gradient | **후속 레포** — Ch6-06에서 이어짐 |
| [generative-models-deep-dive](https://github.com/iq-ai-lab/generative-models-deep-dive) | VAE, GAN, Diffusion, Normalizing Flow | **후속 레포** — Ch2, Ch6의 ML 응용 |
| [llm-alignment-deep-dive](https://github.com/iq-ai-lab/llm-alignment-deep-dive) | RLHF, DPO, KL 제약 강화학습 | **후속 레포** — Ch2 KL이 핵심 손실 |

> 💡 이 레포는 **정보 이론의 수학적 기반과 ML 손실 함수의 정보이론적 해석**에 집중합니다. ML 경험이 없어도 Chapter 1~5는 수학 레포로 학습 가능합니다. Chapter 6은 딥러닝 기초(Cross-Entropy, VAE, GAN 사용 경험)가 있을 때 연결이 더욱 깊어집니다.

---

## 📖 Reference

- **Elements of Information Theory** (Cover & Thomas, 2006) — 표준 교과서, 본 레포의 증명 대부분의 출처
- **Information Theory, Inference, and Learning Algorithms** (MacKay, 2003) — ML 관점의 입문 명저, 무료 PDF 공개
- **A Mathematical Theory of Communication** (Shannon, 1948) — 정보 이론의 원전, Ch1의 공리적 유도의 출처
- **Pattern Recognition and Machine Learning** (Bishop, 2006) Chapter 1.6 — ML에서의 정보 이론
- **Deep Learning and the Information Bottleneck Principle** (Tishby & Zaslavsky, 2015) — Ch6-04의 기반
- **Mutual Information Neural Estimation** (Belghazi et al., 2018) — MINE, Ch3-04
- **Representation Learning with Contrastive Predictive Coding** (van den Oord et al., 2018) — InfoNCE의 원전, Ch3-05
- **Optimal Transport: Old and New** (Villani, 2008) — Wasserstein 이론의 표준, Ch2-05
- **Wasserstein GAN** (Arjovsky et al., 2017) — WGAN, Ch2-05~06
- **Denoising Diffusion Probabilistic Models** (Ho et al., 2020) — Ch6-05의 ELBO 분해

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"Cross-Entropy 손실을 쓰는 것과, 그것이 '평균 최적 부호 길이'라는 Shannon의 근본 메시지를 아는 것은 다르다"*

</div>
