---
tags:
  - Computer_Vision
  - Deep_Learning
  - Retrieval
  - Contrastive_Learning
---
<mark style='background:var(--mk-color-orange)'>기존 방법론</mark>
<mark style='background:var(--mk-color-green)'>이 논문에서 제시한 방법</mark>
<span style='color:var(--mk-color-teal)'>제시한 방법의 장점</span>
<span style='color:var(--mk-color-red)'>제시한 방법의 단점</span>

TASK: Cross-modal retrieve


# Abstract
* Cross-modal retrieval method 는 여러 modality 에 대해 일반적인 representation 을 만들어 낸다.
	* 특히 vision, language modal 에서
* <span style='color:var(--mk-color-red)'>image 와 그 caption 에 대해  대응관계가 복잡한 점</span>은 <span style='color:var(--mk-color-red)'>Cross-modal retrieve 를 더 challenging</span> 하게 만든다.
	* image-caption 은 1-1 이 아닌 1-N 관계이기 때문

* 위와 같은 이유로 이 논문에서는 <span style='color:var(--mk-color-red)'>deterministic function 은 one-to-many 를 탐지할 수 없기에</span> 좋지 않은 모델이라고 말한다.
* 따라서 이 논문에서는 <mark style='background:var(--mk-color-green)'>Probabilistic Cross-Modal Embedding(PCME) 을 제안</mark>한다.
* 또한 <span style='color:var(--mk-color-red)'>기존의 데이터셋이 non-exhaustive annotation 을 가지기에</span> <mark style='background:var(--mk-color-green)'>새로운 dataset 인 CUB dataset 을 제안</mark>한다.

* REF:
	* [deterministic model vs. probabilistic model](https://velog.io/@cardy20/%EC%A7%80%EC%8B%9D-%EC%A1%B0%EA%B0%81-Activation-function)
		* deterministic model: 확률적인 요인이 없어 (동일 input)-(동일 output) 을 보장하는 모델
		* probabilistic model: 확률적인 요인이 첨가되어 (동일 input)-(동일 output) 을 보장하지 못하고 random 한 요소가 첨가되는 모델
# 1. Introduction
* cross-modal retrieval 은 어려운 task이다.
	* cross-modal retrieval 의 예시
		* query: image, database: text -> image 에 해당하는 문장을 찾는 task
		* query: text, database: image -> text 를 주어주고 그에 해당하는 image 를 찾는 task
* 위의 task 에서 어려움을 주는 대부분의 이유는 multi modal 에 맞는 representation 을 생성하는 것.
* 즉, multiple modality 를 위한 common representation 을 만드는 것은 chanllenging 하다.
	* e.g. 아래 이미지와 그에 대한 caption 을 생각해보자
	* 이미지는 고정되어 있기에 1
	* 하지만 이에 대한 caption 은 달라질 수 있다.
	* 즉 image-to-text 나 text-to-image 는 그 대응이 one-to-many 가 될 수 밖에 없다.

![[Pasted image 20240426134439.png|500]]
* one-to-many representation 을 만들기 위해 PVSE(Polysemous Visual-Semantic Embeddings) 가 제안됨.
* 이 방식은 given input 에 대해 K 개의 proposal 을 제안하는 방법 제안
* 이후에 이러한 방식은 많은 발전을 이루었다.
* 하지만 computational cost 가 매우 컸다.

* 이에 이 논문에서는 <mark style='background:var(--mk-color-green)'>Probabilistic Cross-Modal Embedding (PCME)</mark> 를 제안한다.
* 이 방법은 아래 특징을 가진다.
	* effective representation tools
	* explicit 한 many-to-many representation 이 필요 없다.
* 또한 이 방법론은 아래와 같은 장점을 가진다.
	* 첫째, PCME는 쿼리의 실패 가능성이나 난이도를 추정하는 등 유용한 애플리케이션으로 이어질 수 있는 불확실성 추정치를 제공합니다. 
	* 둘째, probabilistic model 은 더 풍부한 임베딩 공간으로 이루어진다
		* deterministic representation은 유사성 관계만을 표현할 수 있습니다. 
	* 셋째, PCME는 deterministic retrieval systems 을 보완합니다.

* 또한 이 논문에서는 <span style='color:var(--mk-color-red)'>non-exhaustive annotation 을 가지는 MS-COCO</span> 대신 <mark style='background:var(--mk-color-green)'>새로운 cross modality retreival benchmark(Using CUB) 를 제안</mark>한다.


# 2. Related Work

# 3. Method

$$
\mathcal{D=(C,I)}
$$
* $\mathcal{D}$ : vision and language dataset
* $\mathcal{C}$ : set of captions
* $\mathcal{I}$ : set of images

$$
\begin{align*}
c\in\mathcal{C},i\in\mathcal{I} \\
\tau(c) \subseteq \mathcal{I}, \tau(i) \subseteq \mathcal{C}\\
\end{align*}
$$

* $\tau(c)$ : caption $c$ 와 관련된 image(image 이므로 image set $\mathcal{I}$ 에 속함)
* $\tau(i)$ : image $i$ 와 관련된 caption(caption 이므로 caption set $\mathcal{C}$ 에 속함)

* 모든 query $q$ 에 대해 cross-modal match 는 2개이상 존재한다.
* 또한 $q$ 는 image 나 caption 이 될 수 있다.
$$
|\tau(q)>1|
$$
* cross-modal retrieval method 는 embedding space $\mathbb{R}^D$ 를 학습하여 
* 두 vector $f_\mathcal{V}, f_\mathcal{T}$ 의 similarity 를 구하는 것이 주요 목표이다. 

## 3.1. Building blocks for PCME
* 이 section 에서는 PCME 의 주요 구성요소인 아래 2가지에 대한 설명제시
	1. joint visual-textual embeddings 
	2. probabilistic embeddings

### 3.1.1. Joint visual-textual embeddings
![[Pasted image 20240426150657.png]]
* **visual encoder** $f_\mathcal{V}$
	1. image $i$ 를 ResNet image encoder 로 encoding
		* $i$ : input image
		* $g_\mathcal{V}$ : ResNet image encoder
		* $z_v=g_\mathcal{V}(i)$ : image encoder $g_\mathcal{V}$ 를 거치고 나온 image $i$ 의 feature map
	2. distribution predictor $h_\mathcal{V}$ 를 통해 distribution 을 예측
		* $h_\mathcal{V}$ : GAP(Global Average Pooling) 이후의 linear layer
		* 참고 : 위의 그림은 PCME 를 적용하여 평균과 표준편차까지 구한 그림
	
![[Pasted image 20240426151547.png]]
* **Textual encoder** $f_\mathcal{T}$
	1. caption $c$ 를 pre-trained 된 GloVe 로 encoding
		* $c$ : input caption
		* $g_\mathcal{T}$ : pre-trained GloVe
		* $z_t=g_\mathcal{T}(c)$ : caption encoder $g_\mathcal{T}$ 를 거치고 나온 caption $c$ 의 word-level descriptor
		* $z_{t}\in\mathbb{R}^{L(c)\times d_t}$ 
			* $L(c)$ : number of words in $c$
	2. bidirectional GRU $h_\mathcal{T}$ 로 부터 sentence-level feature 추출
		* $h_\mathcal{T}$ : bidirectional GRU
		* 참고 : 위의 그림은 PCME 를 적용하여 평균과 표준편차까지 구한 그림

* **Losses used in prior work**
	* joint embedding 의 loss 를 구하기 위해 contrastive 나 triplet loss 사용
* **Polysemous visual-semantic embeddings (PVSE)**
	* PVSE: one-to-many matches for cross modal retrieval 을 modeling 함
	1. visual case 와 textual case 2개로 나뉜다.
	2. 각 case 에 대해 $K$ 개의 proposal 을 추출한다.
	3. textual, visual 에서 추출된 embedding 을 이용하여 MIL(Multiple Instance Learning) 진행
		* $K^2$ 개의 possible visual-textual embedding pair 중 가장 좋은 pair 선택
		* 각 pair 는 supervised 되어 있다.
		* embedding 추출 관련 수식은 아래 PVSE: visual case, PVSE: textual case 참고

> 참고:  Multiple Instace Learning 이란
> MIL 이란 Multiple Instance Learning의 약자로써, Instance 각각에 Label이 붙어 그것을 예측하는 것이 아니라 **여러개의 Instance가 bag을 이루고 이 bag에 Label이 붙어 그것을 예측하는 것**을 말한다.
> REF: [MIL 이란?](https://jellyho.com/blog/117/)

1. PVSE: visual case
$$
v^k = \mathsf{LN}(h_\mathcal{V}(z_v) + s(w^1 \mathsf{att}_\mathcal{v}^k(z_v) z_v))
$$
* $v^{k}\in\mathbb{R}^{D},k\in\{1,...,K\}$ : visual embedding
* $w^1 \in \mathbb{R}^{d_v \times D}$ : Fully connected layer 의 weight
* $s$ : sigmoid function 
* $\mathsf{LN}$ : LayerNorm
* $\mathsf{att}_\mathcal{V}^k$ : visual self-attention $\mathsf{att}_\mathcal{V}$ 의 k-th attention head 

2. PVSE: textual case
$$
t^k = \mathsf{LN}(h_\mathcal{T}(z_t) + s(w^1 \mathsf{att}_\mathcal{T}^k(z_t) z_t))
$$
* $t^{k}\in\mathbb{R}^{D},k\in\{1,...,K\}$ : visual embedding
* $w^1 \in \mathbb{R}^{d_v \times D}$ : Fully connected layer 의 weight
* $s$ : sigmoid function 
* $\mathsf{LN}$ : LayerNorm
* $\mathsf{att}_\mathcal{T}^k$ : textual self-attention $\mathsf{att}_\mathcal{V}$ 의 k-th attention head 


### 3.1.2. Probabilistic embeddings for a single modality
(아래에서 설명할 내용들은 대부분 [Modeling Uncertainty With Hedged Instance Embedding](https://openreview.net/pdf?id=r1xQQhAqKX) 논문에서 나오는 내용이다. 자세한 내용은 위의 논문을 참고하라)


---
![[Pasted image 20240426175510.png]]
### HIB 에 대한 간단한 설명
* 2 종류의 input 에 대해 similarity 는 유지하되 불확실성을 포함한 representation 을 만들어 내는 모델
### pipeline of HIB 
1. CNN 을 통해 K 개 의 sampling 추출 -> branch 1
2. 위의 과정을 동일하게 다른 branch 에서 진행 -> branch 2
	* 위 과정은 contrastive learning 으로 학습(아래 참고)
	* $\hat{m}=1$ : matching, $\hat{m}=1$ : matching, $\hat{m}=0$ : non-matching
$$L_{\text{softcon}} = -\log p(m = \hat{m}|z_1, z_2) = \begin{cases} 
-\log p(m|z_1, z_2) & \text{if } \hat{m} = 1, \\
-\log (1 - p(m|z_1, z_2)) & \text{if } \hat{m} = 0, 
\end{cases}$$
3. 만들어진 branch 1 의 K개의 sample 과 branch 2 의 K 개의 branch 간의 distance 계산
4. 각 point 간의 similarity 를 구하기 위해 probabilistic expression 으로 치환한다.
	* $\sigma$ 는 sigmoid
$$p(m|z_1, z_2) := \sigma(-a||z_1 - z_2||_2 + b)$$
5. Match probability for probabilistic embedding(embedding 에 stochastic mapping 추가)
$$
p(m|x_1,x_2)=\int{p(m|z_1,z_2)p(z_1|x_1)p(z_2|x_2)dz_{1}dz_{2}}
$$
* 위의 수식을 직접 계산하는 것은 어려움
* 따라서 아래 방식으로 근사치를 구함
6. 이후 monte carlo estimation 을 진행하여 위의 수식에 대한 근사치를 구한다.
$$p(m|x_1, x_2) \approx \frac{1}{K^2} \sum_{k_1=1}^{K} \sum_{k_2=1}^{K} p(m|z_1^{(k_1)}, z_2^{(k_2)})$$
---
* 이 논문에서 제시하는 PCME 는 **각 sample** 을 **distribution 으로 modeling** 한다.
* 이 논문에서는 이러한 modeling 을 HIB(Hedged Instance Embedding)을 이용한다.
* HIB 는 probabilistic mapping $p_\theta(z|x)$ 를 train 한다. 이는 아래 효과를 가진다.
	* pairwise semantic similarity 보존
	* 고유한 불확실성을 보존(random 한 영역을 보존한다.)

* **Soft contrastive loss**
$$
L_{\alpha\beta}(\theta) =
\begin{cases}
-\log p_\theta(m|x_\alpha, x_\beta) & \text{if } \alpha, \beta \text{ is a match} \\
-\log (1 - p_\theta(m|x_\alpha, x_\beta)) & \text{otherwise}
\end{cases}
$$
* $L_{\alpha\beta}(\theta)$ : soft-contrastive loss for pair of samples $(x_\alpha,x_\beta)$
* $(x_\alpha,x_\beta)$ : pair of samples

* **Factorizing match probability**
* match 확률인 $p_\theta(m|x_\alpha, x_\beta)$ 아래 2가지로 분리함
	* embedding $p_\theta(m|z_\alpha, z_\beta)$ 
	* encoder $p_\theta(z|x)$ 에 기반한 match probability
* 그 후 Monte Carlo estimation 으로 추정한다.
$$
p_\theta(m|x_\alpha, x_\beta) \approx \frac{1}{J^2} \sum_{j} \sum_{j'} p(m|z_\alpha^j, z_\beta^{j'})
$$
* $p_\theta(m|x_\alpha, x_\beta)$ : match 확률
* $p_\theta(m|z_\alpha, z_\beta)$ : embedding
* $z^j$ : embedding distribution $p_\theta{(z|x)}$ 로 부터 sampling 한 sample

* **Match probability from Euclidean distance**
* sample-wise match probability 를 아래와 같이 연산한다.
$$
p(m|z_\alpha, z_\beta) = s(-a||z_\alpha - z_\beta||_2 + b)
$$
* $a, b$ : learnable scalar
* $s(\cdot)$ : sigmoid function

## 3.2. Probabilistic cross-modal embedding(PCME)
![[Pasted image 20240426205401.png]]
* 앞선 3.1. 절에서 <mark style='background:var(--mk-color-green)'>embedding space 를 만드는 방법</mark>과 <mark style='background:var(--mk-color-green)'>single modality 에서 probabilistic embedding 을 만드는 방법</mark>을 기술하였다.
* 3.2. 절에서는 multi modality embedding space(joint embedding space) 를 probabilistic representation with PCME 로 학습하는 방법을 설명한다.

### 3.2.1. Model architecture
* PCME 는 아래 2가지 embedding 을 동일한 embedding space $\mathbb{R}^D$ 에서 normal distribution 으로 parameterize 하여 표현한다.
* 이 과정에서 평균 vector 와 diagonal covariance matrices 를 구한다.
	* $p(v|i)$ : image embedding space
	* $p(t|c)$ : textual embedding space
$$
\begin{align*}
p(v|i) \sim \mathcal{N} (h_\mathcal{V}^\mu (z_v), \text{diag}(h_\mathcal{V}^\sigma (z_v)))\\
p(t|c) \sim \mathcal{N} (h_\mathcal{T}^\mu (z_t), \text{diag}(h_\mathcal{T}^\sigma (z_t)))
\end{align*}
$$

* $z_v=g_\mathcal{V}(i)$ : feature map of image $i$
* $z_t=g_\mathcal{T}(c)$ : feature sequence of caption $c$
* $h^\mu,h^\sigma$ : mean, variance vector

* **Local attention branch**
![[Pasted image 20240426213514.png]]
![[Pasted image 20240426213913.png]]
* 3.1.1 절의 PVSE architecture 에서 영감을 받아 그 구조를 차용한다.
* PVSE architecture 에서 image 에서는 GAP & FC, caption 에서는 Bi-GRU 로 향하는 방향은 완전히 같은 구조를 차용한다.
* 단, local attention branch 에서는 self-attention based aggregation 을 연산한다.
* (위의 이미지에서는 $v^{\mu},t^{\mu}$ 만을 구했지만 본 논문에서는 아래처럼 $v^\sigma,t^\sigma$ 를 모두 구한다.)
![[Pasted image 20240426214359.png|500]]

* **Module for $\mu$ versus $\sigma$** 
* $h^\mu_\mathcal{V},h^\mu_\mathcal{T}$ 를 구할 때는 sigmoid, LayerNorm 과 L2 projection 을 사용한다.
* 하지만, $h^\sigma_\mathcal{V},h^\sigma_\mathcal{T}$ 를 구할 때 <mark style='background:var(--mk-color-red)'>sigmoid, LayerNorm 과 L2 projection 을 사용</mark>할 때 <span style='color:var(--mk-color-red)'>과도하게 representation 을 제한</span>하는 경향이 있었다. 
* 따라서 $h^\sigma_\mathcal{V},h^\sigma_\mathcal{T}$ 를 구할 때는 **sigmoid, LayerNorm 과 L2 projection 을 모두 사용하지 않는다**.

* **Soft cross-modal contrastive loss**
$$L_{\alpha\beta}(\theta) =
\begin{cases}
-\log p_\theta(m|x_\alpha, x_\beta) & \text{if } \alpha, \beta \text{ is a match} \\
-\log (1 - p_\theta(m|x_\alpha, x_\beta)) & \text{otherwise}
\end{cases}$$
* joint probabilistic embedding 은 image 와 caption 의 probabilistic embedding 의 mapping 의 parameter 를 학습하는 것이다.
	* Probabilistic embedding of image : $p(v|i) = p_\theta(v|i)$ 
	* Probabilistic embedding of caption : $p(t|c) = p_\theta(t|c)$
* loss function 은 3.1.2. 절에서 소개한 loss function 을 그대로 사용한다.(위의 수식)
* match probabilities 는 이제 아래를 따른다.
	* cross-modal pair $(i,c):\mathcal{L}_{emb}(\theta_v,\theta_t;i,c)$
	* $\theta=(\theta_v,\theta_t)$
$$p_\theta(m|i, c) \approx \frac{1}{J^2} \sum_j^J \sum_{j'}^J s(-a||v^j - t^{j'}||_2 + b)$$
* 위 수식의 $v^j,t^{j'}$ 은 아래 수식을 따른다.
$$
\begin{align*}
p(v|i) \sim \mathcal{N} (h_\mathcal{V}^\mu (z_v), \text{diag}(h_\mathcal{V}^\sigma (z_v)))\\
p(t|c) \sim \mathcal{N} (h_\mathcal{T}^\mu (z_t), \text{diag}(h_\mathcal{T}^\sigma (z_t)))
\end{align*}
$$
* 최종 loss 를 이 논문에 맞게 바꾸면 아래와 같다.
$$
L_{ij}(\theta) =
\begin{cases}
-\log p_\theta(m|i,c) & \text{if } i, c \text{ is a match} \\
-\log (1 - p_\theta(m|i,c)) & \text{otherwise}
\end{cases}$$
# 4. Experiments
* 이 section 에서는 아래 항목을 다룬다.
	* experimental protocol
	* 현재 사용되고 있는 cross-modal retrieval benchmark 와 evaluation metric 에 대한 문제점
	* 그 후, CUB cross-modal retrieval task, COCO 에 대한 실험 결과
	* embedding space 에 대한 실험

## 4.1. Experimental protocol
* visual encoders: ResNet pre-trained on ImageNet
* text encoders: GloVe pre-trained with 2.2M vocabulary

1. COCO
* backbone: ResNet-152
* embedding dimension: $D=1024$
* optimizer: AdamP
2. CUB
* backbone: ResNet-50
* embedding dimension: $D=512$
* optimizer: AdamP

### 4.1.1. Metrics for cross-modal retrieval

---
![[Pasted image 20240427215558.png]]
#### 참고: Recall@k
* 높을수록 좋은 값
$$
Recall@k=\frac{\#\ of\ recommended\ items\ @k\ that\ are\ relevant}{total\ \#\ of\ relevant\ items}
$$
* Recall 은 실제 모든 1 중에서 내가 1로 예측한 것이 얼마나 되는지 비율을 나타낸다. **Recall@K는 사용자가 관심있는 모든 아이템 중에서 내가 추천한 아이템 K개가 얼마나 포함되는지** 비율을 의미한다. [그림1]의 예시를 보면, 사용자가 좋아한 모든 아이템이 6개, 그 중에서 나의 추천에 포함되는 아이템이 3개다. 따라서, Recall@5=0.5가 된다.
#### 참고: Precision@k
$$
Precision@k=\frac{\#\ of\ recommended\ items\ @k\ that\ are\ relevant}{\#\ of\ recommended\ items\ @k}
$$
* Precision은 내가 1로 예측한 것 중에 실제 1이 얼마나 있는지 비율을 나타낸다. **Precision@K는 내가 추천한 아이템 K개 중에 실제 사용자가 관심있는 아이템의 비율**을 의미한다. [그림1]의 예시에 따르면, 추천한 아이템은 5개, 추천한 아이템 중에서 사용자가 좋아한 아이템은 3개다. 따라서, Precision@5=0.6이 된다.

#### 참고: R-precision
 * R-precision이란, 간단하게 설명하자면 **모든 pair 가운데 ground-truth인 pair가 R개 존재**한다고 할 때 **cosine similarity를 기준으로 pair들을 정렬한** 후 **상위 R개의 pair 가운데 존재하는 알맞은 pair r개에 대해 r/R을 계산**하는 것입니다

---
* Recall@k in retrieval task
* query에 대해서 K개의 nearest neighbor중 같은 category의 image가 k 개 포함되면 TP 로 취급. 
* 이후 recall 계산
* $Recall=\frac{TP}{TP+FN}$: 모델이 정답을 맞춘 것 중 True 라고 분류한 수치
* $Precision=\frac{TP}{TP+FP}$

* Recall@k 는 k 값이 크면 클수록 COCO 에서 잘못된 예측에 대해 관대해진다.
* 이는 잘못 검색된 sample 에 대한 penalty 를 부여하지 못한다.
* 이러한 잘못된 검색에 대한 penalty 는 precision@k 로 보완한다.
* R-precision 을 도입
	* 각 query q 에 대해, 상위 R 개의 retrieval 항목에서 positive 의 비율을 측정
	* $R=|\tau{(q)}|$ : GT match 의 수
* 이러한 R-precision 이 의미를 가지기 위해서는 모든 positive pair 에 대해 annotating 이 되어 있어야 한다.
* 따라서 이 논문에서는 extra information(MS-COCO 에서는 class label) 을 이용하여 positive pair 에 대한 추가적인 annotating 을 진행한다.
* pair $(i,c)$ 에 대하여 binary label vector 를 생성한다. $y^{i},y^{c}\in\{0,1\}^{d_{label}}$
* 이렇게 주어진 binary label vector 는 $\zeta$ 라는 margin 을 주어 여러개의 기준을 생성한다.
* $\zeta=\{0,1,2\}$ 일 때를 모두 구한 후 평균을 내는 것을 최종적인 metric 으로 사용한다.
	* 0,1,2 는 **embedding 간의 Hamming distance** 를 의미함
	* 예시로 라벨이 {새, 고양이, 강아지, 기린, 얼룩말} 형태인 1D vector 라 해 보자
	* 아래 사진으로 $\zeta$ 의 경우를 설명하면
		* $GT-vector=\{1,0,0,1,1\}$
		* $y^i$: 아래사진, $y^c$: 기린과 새와 얼룩말이 있다, binary label vector: $\{1,0,0,1,1\}$
			* 이는 $\zeta=0$ 인 경우에만 positive pair 로 취급한다.
			* 즉, 완전히 동일해야만 positive 로 취급
		* $y^i$: 아래사진, $y^c$: 기린과 새가 있다, binary label vector: $\{1, 0, 0, 1, 0\}$
			* 이는 $\zeta\geq1$ 인 경우에만 positive pair 로 취급한다.
			* 즉, $\zeta=0$ 인 경우에는 $GT:\{1,0,0,1,1\},predict:\{1,0,0,1,0\}$ 간의 hamming distance 가 1이므로 negative pair 로 취급한다.
* 위와 같은 $\zeta$ 를 둔 평가지표를 
![[Pasted image 20240427162252.png]]

* **이 논문에서의 예시**
* class label vector = {선수, 배트, 글러브, 공, 밴치}
* 위를 간략화하여 binary vector 로 표시한다.
* 아래 GT 를 이 vector 로 표시하면 {1,1,1,1,0} 이다.({선수O,배트O,글러브O,공O,밴치X})
* 차례로 $\zeta=0$ 인 경우는 {1,1,1,1,0}, 즉, GT 와의 hamming distance 가 0 인 경우이다.
	* hamming distance {1,1,1,1,0}<->{1,1,1,1,0} == 0
* $\zeta=1$ 인 경우는 {1,1,1,0,0}, 즉, GT 와의 hamming distance 가 1 인 경우이다.
	* hamming distance {1,1,1,<span style='color:var(--mk-color-red)'>1</span>,0}<->{1,1,1,<span style='color:var(--mk-color-red)'>0</span>,0} == 0
* $\zeta=2$ 인 경우는 {1,1,1,0,1}, 즉, GT 와의 hamming distance 가 2 인 경우이다.
	* hamming distance {1,1,1,<span style='color:var(--mk-color-red)'>1,0</span>}<->{1,1,1,<span style='color:var(--mk-color-red)'>0,1</span>} == 2
![[Pasted image 20240427222758.png]]
### 4.1.2. Cross-modal retrieval benchmarks

#### COCO Captions
* image-to-text: 1-5
	* image 가 1개 주어졌을 때 caption 은 5개
* text-to-image: 1-1
	* caption 이 주어졌을 때 그에 상응하는 이미지는 1개
* 위의 조건에 만족하지 않는 pair 들은 그 유사성과 관련없이 모두 negative 로 취급
* 예시
![[Pasted image 20240427163556.png|500]]
* 위의 그림에서 모든 image 와 모든 caption 은 서로 유사성이 있어 positive 처럼 보인다.
* 하지만 COCO 에서 labeling 은 가능한 16개의 경우의 수 중 12개를 무시하고 단 4개의 경우만을 positive 로 취급한다.
* 이러한 경우 train 시<span style='color:var(--mk-color-red)'> 잡음이 많고</span>, <span style='color:var(--mk-color-red)'>신뢰성이 떨어지는 문제</span> 발생한다.

* 이에 이 논문에서는 CUB200-2011 을 제안한다.
#### CUB captions
* CUB 는 한 image 에 10개의 caption 을 부여한다.
* 또한 11,788개의 image 에 대해 200개의 세밀한 class 를 부여한다.
* 이를 통해 caption 과 image class 내의 균형을 맞추어 false-positive 를 억제한다.

## 4.2. Result on CUB
![[Pasted image 20240427164247.png]]
* 위의 표는 match probabilistic 에 대한 정당성을 보여주는 표이다.
* 2개의 cross embedding(text-imgae) 간의 유사성을 판단할 때 distance function 으로 어떠한 방법론을 사용했는지에 대한 결과이다.
* 이 논문에서 제시한 Match Prob 이 서능이 제일 좋음을 알수 있다.(단, 공간복잡도는 일부 손해를 본다.)
* 각 column 에 대한 설명은 아래 참고
	* sampling: Monte-Carlo sampling 사용 여부
	* i2t: image to text, t2i: text to image

## 4.3. Result on COCO
![[Pasted image 20240427164614.png]]

* 위의 결과를 보면 PMRP 수치는 SOTA 이다. 하지만 R@1 은 수치가 비교적 낮다.
* 이러한 이유는 아래와 같다.
	* PMRP: 완전히 정확하진 않더라도 문맥상으로 폭 넓게 일치하는지 판단이 가능
	* R@1: 정량적으로 자연어를 정확히 판단함
	* 즉, 실제로 이미지와 자연어를 자연스럽게 matching 하는 것은 해당 논문에서 제시한 모델이다.
* 전체에 대한 결과는 아래와 같다.

![[Pasted image 20240427165322.png]]

## 4.4. Understanding the learned uncertainty








---


* limitation:
	* section 4.3. Result on COCO 에서 R@k 에 대한 성능이 SOTA 가 아니다. 이 논문에서 새롭게 제시한 평가지표만 SOTA 를 달성하였다. 만약 정말로 자연어와 이미지간의 matching 을 PMRP 가 더 잘 평가한다면 PMRP 가 높거나 낮을때, R@k 가 높거나 낮을때의 visualized image 가 제시하여 확실하게 보여줬어야 할 것 같다.


For each query 𝑞 measure the proportion of positives in the top 𝑅 retrieval items.
