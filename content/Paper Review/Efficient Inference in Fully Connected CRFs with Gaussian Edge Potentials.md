---
dg-publish: true
tags:
  - Computer_Vision
  - Deep_Learning
  - Semantic_segmentation
---


REF : 
* [DenseCRF(Fully connected CRF) 관련 글](https://pseudo-lab.github.io/SegCrew-Book/docs/Appendix/DenseCRF.html)
* https://shining-programmer.tistory.com/1
* [CRF 와 관련된 정리글](https://www.datasciencecentral.com/conditional-random-fields-crf-short-survey/)
* https://towardsdatascience.com/conditional-random-fields-explained-e5b8256da776
* [[CRF 의 계보도]]

이 논문은 2011 년 논문이다. 이 글을 읽을 때 최근의 기법들에 대한 소개는 참고만 하는 것이 좋다.

# Abstract
* 최근 multi-class image segmentation and labeling task 에서 CRF(conditional random field) 를 사용한다.
	* region-level model 은 밀집된 연결을 사용
	* pixel-level model 더 큰 데이터를 처리하기 위해 sparse 한 그래프 연결을 사용
* 이 논문에서는 fully connected CRF 를 제안한다.
* output graph 는 수십억개의 edge 를 가지기에 전통적인 inference algorithm 적용이 힘들다.
* 따라서 Gaussian kernel 의 linear combination 으로 pairwise edge potential 을 정의한다.


# 1. Introduction
* Multi-class image segmentation and labeling
	* 한 image 안에 존재하는 모든 pixel 에 대해 어떠한 object 인지 표시하는 task
* 일반적인 접근법
	* pixel 이나 image 위에 정의된 CRF 에서 MAP(Maximum A Posterior)추론

* Adjacency CRF 구조는 이미지 내의 장거리 연결의 모델링에 어려움을 겪는다.
* 이로 인하여 <span style='color:#eb3b5a'>object 의 edge 에서 과도한 smoothing 이 발생</span>한다.
* 위의 문제를 해결하기 위해 <span style='color:#3867d6'>아래 2가지를 제시</span>
	* incorporate hierarchical connectivity 
	* higher-order potentials defined on image regions
* 위의 방식은 <span style='color:#eb3b5a'>region-base approach 이 복잡한 경계에 대해서 라벨링의 성능을 저하시키는 문제가 발생</span>한다.

* <mark style='background:#3867d6'>이에 이 논문에서는 Fully-connected CRF 를 제시한다.</mark>
	* 이전의 방식은 수백개의 image region 의 관계를 unsupervised 하게 예측하였다.
	* 하지만 우리는 <span style='color:#3867d6'>모든 pixel 을 서로 연결</span>하여 <span style='color:#3867d6'>segment 의 성능을 올렸다</span>.
* 위의 방식은 많은 수의 edge 와 node 를 가지게 된다.
* 이를 해결하기 위해 아래 제시
	* 임의의 feature space에서 Gaussian kernel의 선형 조합으로 정의된 쌍별 edge potential을 가진 fully connected CRF 모델을 위한 매우 효율적인 추론 알고리즘
	* mean field approximation to the CRF distribution 을 base 로 둔다.
	* 반복적으로 message passing step 으로 optimize 한다.
* 위의 과정을 통해 $n^2$ 을 가지는 공간복잡도에서 $n$(linear) 로 줄일 수 있다.


# 2. The Fully Connected CRF Model
### Conditional random field - characterized by Gibbs distribution 

$$
P(X|I) = \frac{1}{Z(I)} \exp\left(-\sum_{c \in \mathcal{C}_\mathcal{G}} \phi_c(X_c|I)\right)
$$

* $P(X|I)$ : Gibbs distribution
* $X=\{X_1,X_2,...,X_N\}$ : Random field $X$
	* $X_{j}$ : $j$ 번째 pixel 에 대한 variable
* $\mathcal{L}={l_1,l_2,...,l_k}$  
* $I=\{I_1,I_2,...I_N\}$ : Random field $I$
	* $I_j$ : $j$ 번째 pixel 에 대한 color vector
* $\mathcal{G}=(\mathcal{V},\mathcal{E})$ : Graph($\mathcal{V}=Vertex,\mathcal{E}=Edge$)
* $\mathcal{C}_\mathcal{G}$ : 그래프 $\mathcal{G}$ 의 clique $c$ 들의 집합
* $\phi_c$ : clique $c$ 에 potential(해당 라벨이 부여될 확률) 
	* 여기서 Potential 은 distribution 이다.

$$E(\mathrm{x}|I)=\sum_{c\in C_\mathcal{G}}{\phi_c{(\mathrm{x}_c|I)}}$$
*  $E(\mathrm{x}|I)$ : labeling $\mathrm{x}\in\mathcal{L}^N$ 의 Gibbs energy
$$
\mathrm{x}^*=argmax_{\mathrm{x}\in\mathcal{L}^N}P(\mathrm{x}|I)
$$
* $\mathrm{x}^*$ : Maximum a Posterior labeling of random field
* $\psi_c(x_c)$ : $\phi_c(\mathrm{x}_c|I)$ 와 같다. notation 의 편의를 위하여 이 식을 사용한다.

### Fully Connected Pairwise CRF Model
1. Gibbs energy
$$
E(\mathbf{x}) = \sum_{i} \psi_u (x_i) + \sum_{i<j} \psi_p (x_i, x_j),
$$
* $i$~$j$ 의 범위 : $1$~$N$
* $\psi_u(x_i)$ : unary potential
	* label assignment $x_{i}$ 에 대해 각 픽셀이 독립적으로 classifier 에 의해 계산된다.
	* 이러한 unary potential 은 질감, 위치, 색상 descriptor(feature map 과 유사)를 포함한다.
	* unary potential 은 각 픽셀을 독립적으로 prediction 되므로 <span style='color:#eb3b5a'>noisy 하고, inconsistent(일관성이 없다)하다</span>.(아래 그림 참고)
	* 따라서 이 논문에서는 <span style='color:#2d98da'>pairwise potential 을 제시</span>한다.
* $\psi_p{(x_i,x_j)}$ : pairwise potential

![[Pasted image 20240314195011.png|500]]


2. pairwise potential
$$E(\mathbf{x}) = \sum_{i} \psi_u (x_i) + \textcolor{Green}{\sum_{i<j} \psi_p (x_i, x_j)},$$
$$
\psi_p (x_i, x_j) = \mu(x_i, x_j)  \underbrace{\sum_{m=1}^{K} w^{(m)} k^{(m)} (f_i, f_j)}_{k(f_i,f_j)}
$$
$$
k^{(m)}(f_i, f_j) = \exp\left(-\frac{1}{2}(f_i - f_j)^T \Lambda^{(m)} (f_i - f_j)\right)
$$
* $k^{(m)}$ : gaussian kernel
* $f_i,f_j$ : pixel $i, j$ 에 대한 feature vector
* $w^{(m)}$ : linear combination weights
* $\mu$ : label compatibility function
	* $\mu(x_i,x_j)=\delta(x_i,x_j)$ 
	* Pott model 을 부여
	* $\mu(x_i,x_j)=[x_{i}\neq x_j]$ 
	* 인접한 유사한 픽셀이 서로 다른 라벨을 할당받을 때 패널티를 부여
	* 라벨간의 연관성에는 둔감하다
	* 즉, 하늘-새 의 차이와 고양이-하늘 과 차이를 두지 않고 학습한다.
	* 이 parameter 는 학습을 통해 갱신된다.
* $\Lambda^{(m)}$ : symmetric positive definite precision matrix   

* Multi-class image segmentation and labeling 작업을 위하여 contrast-sensitive two-kernel potential 을 정의한다.
* 이 two-kernel 은 color vector $I_{i},I_{j}$ 와 position $p_i,p_j$ 의 항을 가진다.
$$
k(f_i, f_j) = \underbrace{w^{(1)} \exp \left( -\frac{|{p}_i - {p}_j|^2}{2\theta_\alpha^2} -\frac{|{I}_i - {I}_j|^2}{2\theta_\beta^2} \right)}_{\text{appearance kernel}} + \underbrace{w^{(2)} \exp \left( -\frac{|{p}_i - {p}_j|^2}{2\theta_\gamma^2} \right)}_{\text{smoothness kernel}}
$$
1.  appearance kernel
	* 인접한 pixel 들이 비슷한 색을 가질 경우 같은 class 일 확률이 높다는 것에 영감을 받음
	* nearness($p$ 와 관련) 와 similarity($I$ 와 관련) 의 강도는 learnable parameter 인 $\theta_\alpha$ 와 $\theta_\beta$ 로 조정된다.
2. smoothness kernel
	* 작은 isolate region 을 제거하기 위함.
* 위의 parameter 는 data 로 부터 학습을 한다.
---
> 참고

### Gibbs distribution 이란
$$
P(X)=\frac{1}{Z}exp(-E(X))
$$
* $P(X)$ : state $X$ 의 확률
* $E(X)$ : state $X$ 의 energy 를 나타내는 함수
	* 본 논문에서는 Gibbs energy function 이다.
* $Z$ : 모든 가능한 상태에 대한 에너지의 지수함수를 합한 값
	* 즉 본 논문에서는 모든 가능한 clique 에 대한 에너지를 더한 값
	* $Z=\sum_{X'}{exp(-E(X'))}$

### Clique 란?
* 특정 ==graph 의 부분집합을 추출==하였을 때 그 부분집합이 ==완전 graph 일 경우==를 ==clique== 라고 한다.

![[Pasted image 20240314162853.png]]

* [[Energy 와 Potential 의 차이]]

---

# 3. Efficient Inference in Fully Connected CRFs

* 이 논문은 mean field approximation to the CRF distribution 을 base 로 한다.
* 이러한 approximation 기법은 iterative 한 message passing algorithm 을 제공한다.
---
> 참고 : messge passing algorithm 이란?
* 그래프에서 인접한 노드로 부터 'message'를 받고 그 message 를 기반으로 자신의 상태를 업데이트 하는 algorithm
* GNN 에서 쓰이는 기본적인 방법
![[Pasted image 20240314205719.png]]

* 위의 A Node 를 업데이트 하기 위해서는 인접한 B, C, D 로부터 message 를 전달받아 update 를 한다.
* 각 B, C, D 는 또한 인접한 노드로부터 message 를 받아 update 한다.
* 이러한 인접한 node 로 부터 message 를 전달받아 해당 node 를 update 하는 방식을 message passing algorithm 이라고 한다.
* [Message Passing Neural Network](https://process-mining.tistory.com/164)

---

* 이 논문에서는 message passing 이 feature space 에서 gaussian filtering 을 사용하여 수행 될 수 있음을 제시
* 이러한 방식은 message passing 의 시간복잡도를 $N^2$ 에서 $N$(linear)로 줄일 수 있다.
* 즉 우리의 모델은 노드의 수를 N 이라 보았을 때 edge 에 대해 sublinear 하게 연산을 줄일 수 있다.


## 3.1. Mean Field Approximation

* 정확한 distribution 인 $P(X)$ 를 예측하는 대신 아래 방식을 제시
* mean field approximation 을 통해 $Q(X)$ 를 계산하고 KL-divergence $D(Q||P)$ 를 minimize 하는 방식으로 학습을 진행한다.

### Compute $Q(X)$
$$
Q(X) = \prod_i Q_i(X_i)
$$
* $Q_i$ : independent marginal distribution
	* 변수 $X_i$ 에 대한 근사적인 independent distribution


$$
Q_i(x_i = l) = \frac{1}{Z_i} \exp\left\{-\psi_u(x_i) - \sum_{l' \in \mathcal{L}} \mu(l, l') \sum_{m=1}^{K} w^{(m)} \sum_{j \neq i} k^{(m)}(f_i, f_j) Q_j(l')\right\}
$$

- $Q_i(x_i = l)$: 변수 $X_i$ 가 라벨 $l$ 을 가질 근사 확률
- $\psi_u(x_i)$: 변수 $X_i$ 의 unary potential로, $X_i$ 의 단독 확률을 나타낸다
- $\mu(l, l')$: 라벨 $l$ 과 $l'$ 사이의 호환성을 측정하는 함수
- $\sum_{l' \in L}$: 모든 가능한 라벨 $l'$ 에 대해 합산
- $K$: 커널 함수의 개수
- $w^{(m)}$: $m$-번째 커널 함수의 가중치
- $k^{(m)}(f_i, f_j)$ : m-번째 커널 함수로, 노드 $i$ 와 $j$ 의 feature $f_i$ 와 $f_j$ 사이의 유사성을 측정
- $Q_j(l')$: 다른 변수 $X_j$ 가 라벨 $l'$ 을 가질 근사 확률
- $Z_i$: 정규화 상수로, $Q_i(x_i = l)$ 가 확률 분포가 되도록 합이 1이 되게 만든다.

* 이 업데이트 방정식은 주어진 CRF 모델 내의 각 변수에 대한 근사 확률 분포를 업데이트하는 데 사용된다. 
* 각 반복에서 모든 변수에 대한 $Q_i$ 가 업데이트되면, 전체 분포 $Q(X)$ 는 점점 원래 분포 $P(X)$ 에 가까워지는 근사치로 수렴한다.
![[Pasted image 20240314212609.png]]

* compatibility transform, local update 2개의 task 는 linear time 으로 실행된다.
	* $Q$ 를 계산하기 위해 $N$ 개의 $Q_i$ 만을 연산하면 된다.
* 하지만 각 변수는 다른 모든 변수의 합의 연산을 진행하여야 한다.
	* (위의 알고리즘처럼)
	* $X_i$ 에 대해 $X_j$($i$ 를 제외한 모든 수) 에 대한 message passing 연산을 해야 함.
* 아래에서는 위의 $N^2$ 연산을 어떻게 처리할 것인지를 제시할 것이다.

## 3.2. Efficient Message Passing Using High-Dimensional Filtering

$$
\tilde{Q}_i^{(m)}(l) = \underbrace{\sum_{j \in V} k^{(m)}(f_i, f_j)Q_j(l)- Q_i(l)}_{\text{message passing}}  = \underbrace{\left[G_{\Lambda^{(m)}} \otimes Q(l)\right](f_i)}_{\overline{Q}_i^{(m)}(l)} - Q_i(l)
$$
* 위의 수식은 Algorithm 1 에서 제시한 Message Passing 부분에서 i 와 j 간의 합을 모두 구하는 수식과 유사하다.
* 위의 수식에서 $Q_i(l)$ 을 빼 주는 이유는 아래의 수식에서 sigma 부분에서 $i\neq j$ 를 적용했지만 위의 수식에서는 $i=j$ 인 경우가 있기에 이러한 부분을 빼 주었다.
![[Pasted image 20240314231713.png]]
* 또한 이 논문에서는 Gaussian Kernel 을 이용하여 연산량을 줄였다.
![[Pasted image 20240314234518.png]]
1. Downsample
	* Gaussian Kernel 을 적용하기 전에 먼저 연산량을 줄이기 위해 downsampling 을 진행한다.
	* 이 때 표준편차의 두배를 넘기는 값들은 모두 0으로 처리한다.
	* 이렇게 할 수 있는 이유는 sampling theorem 덕분이다.(아래 그림 참고)
2. Convolution
	* Downsampling 된 feature 들에 gaussian kernel $G_{\Lambda(m)}$ 로 conv 연산진행
	* Gaussian kernel 을 적용하여 연산을 하더라도 아직 $O(Nd)$ 의 overhead 가 존재
	* 이에 저자는 [permutohedral lattice](_media-sync_resources/20240417T162552/20240417T162552_62316.pdf)를 통하여 이 연산을 줄여내었다.
3. Upsampling


---
> 참고 : sampling theorem

* 표본을 sampling 할 시 표준편차의 2배이상이 되면 완전한 복원이 가능하다고 하는 이론
* 원래는 신호처리 쪽에서 나오는 이론이지만 ML 에서도 쓰인다.
![[Pasted image 20240314235809.png|400]]

---

# 4. Learning


