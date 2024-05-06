REF:
* [SphereFace, CosFace, ArcFace 에 대해 간단한 비교와 요약이 잘 되어 있음](https://jiryang.github.io/2020/06/05/Openset-face-recognition/)

# Abstract

* Large-scale face recognition task 에서 주요한 작업 중 하나는 적합한 loss function 을 정의하는 것이다.
* center loss(centre loss) 는 특정 데이터의 **deep feature 와 그 데이터의 class 간의 거리**에 대한 **penalty 를 부여**하므로써 intra class compactness 를 달성한다.
* SphereFace 는 각도에 대한 penalty 를 부여하여 similarity 가 작으면 더 작게 크면 그 차이가 더 커지게 만든다.(L2 Norm 과 비슷한 역할)
* 최신 연구에서는 class sperability 를 올리기 위해 margin 을 도입
* 따라서 이 논문에서도 margin 을 도입하며, ArcFace(Additive Angular Margin Loss) 를 제안

### REF:
---
#### 1. SphereFace
아래 링크를 참고하고 3종류의 loss 정리할 것
https://minimin2.tistory.com/157

---

# 1. Introduction

* Face recognition task 에서는 DCNN representation 을 사용한다.
* DCNN representation 은 아래 성질을 가지도록 설계된다.
	* small intra-class distance
	* large inter-class distance
![[Pasted image 20240503151433.png]]

* face recognition task 에서는 2가지 큰 줄기가 있다.
1. multi-class classifier: loss function 으로 softmax loss 사용
	* 단점
		1. identity 의 수 $n$ 이 증가할 때마다 transformation matrix $W\in\mathbb{R}^{d\times n}$ 의 크기가 선형적으로 증가
		2. learned feature 가 closed set 에서는 충분히 separable 하지만 open set 에서는 분리가 되지 않는다.(open set, closed set 은 아래 ref 참고)
2. learn directly an embedding: loss function 으로 triplet loss 사용
	* 단점
		1. large-dataset 에 대해 연산량이 combinationally 하게 증가한다.
		2. 적합한 semi-hard sampling 이 어렵다.([semi-hard negative mining 참고 link](https://mic97.tistory.com/16))

### Introduction-REF
#### open set, closed set
**Closed-set**은 Training Set과 Test Set의 분류해야 할 Class가 같은 것인데요, 예를 들어 강아지/고양이를 분류하는 Training set으로 학습했다면 Test set도 강아지/고양이로만 되어있습니다.

반면에, **Open-set**은 강아지/고양이로 학습했지만 Test는 호랑이/사자를 분류하게 되는 경우입니다. 즉, 특정 class의 feature를 학습한다기 보다는 서로 다른 class의 feature끼리는 유사도가 낮고, 같은 class의 feature는 유사도가 높도록 학습해야 합니다. 이런 학습을 **Metric Learning** 이라고 합니다.

---

* softmax 의 차별화된 특성을 강화하기 위해 여러 변형이 이루어짐
* ==center loss(centre loss)== 가 그 중 하나.
* center loss 란 각 데이터의 feature 와 그 feature 가 속한 class 의 중심과의 Euclidean distance 를 구하는 방식을 취함
* 하지만 이러한 방식도 <span style='color:var(--mk-color-red)'>최근 face class 가 급증함에 따라 어려움을 겪음</span>

* 위의 문제를 해결하기 위해 SphereFace 에서는 margin 을 도입
* 이를 통해 모델이 intra-class 를 줄이고 inter-class 를 키우는 효과를 보았다.
* 하지만 SphereFace 는 loss function 을 계산하기 위해 근사치가 필요했다.
* 이러한 과정 때문에 학습이 불안정한 문제 발생
* 이러한 margin 의 문제점을 일부 해결한 CosFace 가 등장

* 이 논문에서는 모델의 disciminative 능력을 향상시키고, 훈련 과정을 안정시키기 위한 Additional Angular Marigin Loss(ArcFace) 를 제안한다.
![[Pasted image 20240503164649.png]]
* ArcFace 는 아래 장점을 가진다
	* **Engaging.** ArcFace는 정규화된 구면상에서 각도와 호 사이의 정확한 대응을 통해 측지 거리 margin을 직접 최적화한다. 우리는 512-D 공간에서 feature과 가중치 간의 각도 통계를 분석함으로써 직관적으로 어떤 일이 일어나는지 설명한다.

	* **Effective.** ArcFace는 대규모 이미지 및 비디오 데이터셋을 포함한 열 개의 얼굴 인식 벤치마크에서 State-of-the-art 성능을 달성한다.

	* **Easy.** ArcFace는 알고리즘 1에서 제시된 몇 줄의 코드만 필요하며, MxNet [5], Pytorch [23], Tensorflow [2]와 같은 계산 그래프 기반 딥 러닝 프레임워크에서 매우 쉽게 구현할 수 있다. 또한, [15, 16]의 연구와 달리, ArcFace는 안정적인 성능을 위해 다른 loss 함수와 결합할 필요가 없으며, 어떠한 훈련 데이터셋에서도 쉽게 수렴할 수 있다.

	* **Efficient.** ArcFace는 훈련 중에 무시할 만한 계산 복잡성만을 추가한다.

# 2. Proposed Approach
## 2.1. ArcFace
* 가장 널리 사용되는 classification softmax loss function 은 아래와 같은 형식을 가진다.
$$
L_1 = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{W_{y_i}^T x_i + b_{y_i}}}{\sum_{j=1}^n e^{W_j^T x_i + b_j}}
$$
* $x_i\in\mathbb{R}^d$ : $i$ 번째 sample 의 deep feature
	* $d$ : embedding dimension. 많은 논문에서도 그러하듯 512 로 설정함
* $y_i$ : $i$ 번째 class. deep feature $x_{i}$ 에 대응되는 label
* $W_j\in\mathbb{R}^d$ : weight matrix $W$ 의 $j$ 번째 열.
* $b_j \in \mathbb{R}^n$ : $W_j$ 에 대응되는 bias term
* 위와 같은 softmax loss 는 feature embedding 을 명시적으로 최적화하지 않기에 <span style='color:var(--mk-color-red)'>deep face recognition 에서 큰 성능 차이</span>를 가져온다.
* 이러한 성능차이는 특히 large-scale face recognition 에서 두드러지게 나타난다.


* simplicity 를 위해 bias term $b_j$ 를 0으로 고정
* logit 을 $W_j^T x_i = \|W_j\| \|x_i\| \cos \theta_j$ 로 변환
	* $\theta_j$: 가중치 $W_j$ 와 feature $x_i$ 사이의 각도
* 이후, 개별 가중치 $||W_j||$ 를 L2 normalization 을 통해 $1$ 로 정규화
* $||x_i||$ 를 L2 normalization 을 적용하여 $s$ 로 정규화
* 이를 통해 $W_j$ 와 $x_i$ 가 그 사이의 각도에만 의존하도록 한다
* 위와 같은 과정을 통해 학습된 embedding feature 가 hypershere 위에 분포하게 만든다.
$$
L_2 = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s \cos \theta_{y_i}}}{e^{s \cos \theta_{y_i}} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
$$
* intra-class compactness 를 증가시키고 inter-class discrepency 를 위해 margin $m$ 을 추가한다.
* 이러한 margin 이 추가된 loss 를 ArcFace loss 라 한다.
$$
L_3 = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
$$
![[Pasted image 20240504132800.png|500]]
* 위 그림에서도 알 수 있듯, softmax loss 는 class 간의 경계에서 모호성이 강하다.
* 하지만 ArcFace loss 는 가까운 class 사이에서 경계를 명확하게 분리한다.

## 2.2. Comparison with SphereFace and CosFace
### Numerical Similarity
* SphereFace, ArcFace, CosFace 에서는 각각 다른 종류의 margin penalty 가 제시되었다.
	* SphereFace : multiplicative angular margin $m_1$
	* ArcFace : additive angular margin $m_2$
	* CosFace : additive cosine margin $m_3$ 
* 위와 같은 margin penalty 는 numerical analysis 관점에서 아래 2가지 속성을 증가시키는 target logit 을 적용하게 된다는 점에서 유사하다.
	* intra-class compactness
	* inter-class diversity
![[Pasted image 20240504133503.png|600]]
* Figure 4.(b) 에서의 그림은 target logit 과 angle between feature and center 에 대한 그래프이다.
* angle between feature and center 값이 커질수록 ArcFace 는 target logit 값이 -0.6 에 근접한다.
* 즉, logit 값이 더 작으므로 class 분류를 더 잘한 것(==이거 한 번 질문해 볼 것. 이해 잘 안됨==)

* 이제 이 논문에서는 ShereFace, ArcFace, CosFace 의 각 penalty term 을 결합하여 사용한다.
$$
\cos(m_1\theta + m_2) - m_3
$$
$$
L_4 = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s(\cos(m_1\theta_{y_i}+m_2)-m_3)}}{e^{s(\cos(m_1\theta_{y_i}+m_2)-m_3)} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
$$

### Geometric Difference
* 이 논문에서 제시한 additive angular margin 은 다른 방식보다 뛰어난 geometric attribute 를 가진다.
![[Pasted image 20240504145351.png]]
* 위의 그림은 binary classification 을 각 방법론들에 대해 시각화한 자료이다.(회색은 margin)
	* **Softmax** 의 경우 <mark style='background:var(--mk-color-red)'>decision boundary 가 너무 tight</mark> 하여 <span style='color:var(--mk-color-red)'>decision boundary 부근에서 잘못된 예측</span>을 하는 경우가 많다.
	* **SphereFace** 의 경우 <mark style='background:var(--mk-color-red)'>margin 이 지나치게 크게 잡혀</mark> <span style='color:var(--mk-color-red)'>모델의 학습이 발산</span>하는 경우가 많다.
	* **CosFace** 의 경우 SphereFace 에 비해 margin 이 작지만<mark style='background:var(--mk-color-red)'> boundary 경계가 선형이 아니기에</mark> <span style='color:var(--mk-color-red)'>각도에 따라 학습이 발산</span>하는 경우가 존재한다.
	* **ArcFace** 의 경우 <mark style='background:var(--mk-color-teal)'>전체 각도에 대해 선형적</mark>이고 <mark style='background:var(--mk-color-teal)'>margin 역시 적절하게 존재</mark>하여 <span style='color:var(--mk-color-teal)'>가장 좋은 geometric attribute 를 가진다</span>.

* 또한, SphereFace 의 경우 훈련 초기의 발산을 억제하기 위해 joint super vision 을 이용하여 penalty 를 약화시킨다.
* 이 논문에서는 integer requirement 없이 복잡한 두배각 공식 대신 arc-cosine function 을 이용하여 SphereFace 의 margin 을 새롭게 적용하는 방식을 제안한다.

---

# 정리.
* 이 논문에서는 SphereFace, ArcFace, CosFace 의 margin penalty 를 융합하는 것을 제시한다.
* 따라서 각 loss 의 차이점을 짚고 넘어가고자 한다.

## Basic Softmax 
$$
L = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s \cos \theta_{y_i}}}{e^{s \cos \theta_{y_i}} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
$$
## SphereFace $m_1$
* 이 논문에서는 SphereFace 의 penalty term 을 $m_1$ 으로 표기한다.
$$
L = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s(\cos\textcolor{red}{m_1}\theta_{y_i})}}{e^{s(\cos\textcolor{red}{m_1}\theta_{y_i})} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
$$
* softmax 와의 차이점이라고 하면 penalty term $m_1$ 을 추가한 것을 들 수 있다.
* 여기서 penalty term 은 **$\theta_{y_i}$ term 앞**에 **곱해지는** 형식으로 추가된다.(위 수식의 빨간 부분)

## ArcFace $m_2$
* 이 논문에서는 ArcFace 는 penalty term 을 $m_2$ 로 표기한다.
$$
L = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s(\cos(\theta_{y_i} + \textcolor{red}{m_2}))}}{e^{s(\cos(\theta_{y_i} + \textcolor{red}{m_2}))} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
$$
* softmax 와의 차이점이라고 하면 penalty term $m_2$ 를 추가한 것을 들 수 있다.
* 여기서 penalty term 은 **$\theta_{y_i}$ term 뒤**에 **더해지는** 형식으로 추가된다.(위 수식의 빨간 부분)

## CosFace $m_3$
* 이 논문에서는 CosFace 의 penalty term 을 $m_3$ 로 표기한다.
$$
L = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s(\cos\theta_{y_i}-\textcolor{red}{m_3})}}{e^{s(\cos\theta_{y_i}-\textcolor{red}{m_3})} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
$$
* softmax 와의 차이점이라고 하면 penalty term $m_3$ 를 추가한 것을 들 수 있다.
* 여기서 penalty term 은 **$cos(\cdot)$ term 뒤**에 **빼지는** 형식으로 추가된다.(위 수식의 빨간 부분)

---

## 2.3. Comparison with Other Losses
* 이 논문에서는 아래 Figure 1. 과 같이 3가지 Loss 와 비교를 진행한다.
![[Pasted image 20240504152737.png|500]]
### Intra-Loss 
* 샘플과 기준 중심 사이의 angle/arc 를 감소시켜 intra-class compactness을 향상시키도록 설계됨
$$
\begin{align*}
L_5 &= L_2 + \frac{1}{\pi N} \sum_{i=1}^N \theta_{y_i}\\
L_2 &= -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s \cos \theta_{y_i}}}{e^{s \cos \theta_{y_i}} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
\end{align*}
$$
### Inter-Loss
* 다른 중심들 사이의 angle/arc 를 증가시켜 inter-class 차이를 강화함
$$
\begin{align*}
L_6 &= L_2 - \frac{1}{\pi N (n - 1)} \sum_{i=1}^N \sum_{j=1, j \neq y_i}^n \arccos(W_{y_i}^T W_j)\\
L_2 &= -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s \cos \theta_{y_i}}}{e^{s \cos \theta_{y_i}} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
\end{align*}
$$
* 여기서의 Inter-Loss는 최소 구형 에너지(Minimum Hyper-spherical Energy, MHE) 방법의 특별한 경우이다. 
* MHE를 통해 숨겨진 계층과 출력 계층이 규제되었다. 
* 이 논문에서는 SphereFace loss과 네트워크의 마지막 계층에서의 MHE loss을 결합하여 제안된 loss 함수의 특별한 경우를 제시한다.

### Triplet-loss
 * triplet 샘플들 사이의 angle/arc margin을 확대하는 것을 목표
$$\arccos(x_i^{pos} \cdot x_i) + m \leq \arccos(x_i^{neg} \cdot x_i)$$


# 3. Experiments

## 3.1. Implementation Details
### Dataset
* CASIA [41], 
* VGGFace2 [3], 
* MS1MV2, 
* DeepGlint-Face (MS1M-DeepGlint 및 Asian-DeepGlint 포함) 
![[Pasted image 20240504160456.png|500]]

### Experimental Settings
* 5개의 facial point 를 통해 nomalised face crop(112x112) 를 생성
* embedding network: ResNet50, ResNet100
* 마지막 conv 이후 BN-Dropout-FC-BN 구조로 512 Dimension embedding 추출
* 이 논문에서는 실험 설정을 이해하기 쉽도록 (\[training dataset, network structure, loss\]) 로 구조화

## 3.2. Ablation Study on Losses
![[Pasted image 20240504155156.png|500]]
* ArcFace 에서 최적의 margin 은 0.5
* SphereFace 에서 최적의 margin 은 1.35
* CosFace 에서 최적의 margin 은 0.35
* **ArcFace(0.5)** 가 3개의 데이터셋 모두에서 가장 좋은 결과 
* Joint margin(CM1(1,0.3,0.2), CM2(0.9,0.4,0.15)) 는 SphereFace, CosFace 보다는 좋은 성능을 보이지만, ArcFace 보다는 성능이 낮음

* ArcFace 의 우수성의 이해를 위해 Table 3. 제시 
* 다양한 loss 하에서 훈련 데이터(CASIA) 및 테스트 데이터(LFW)에 대한 자세한 각도 통계제시
![[Pasted image 20240504160756.png|500]]
* $\mathrm{W-EC}^\Downarrow$: mean of angle between $W_j$ and corresponding embedding feature centre
	* $W_j$ 와 그에 대응되는 embedding feature 중심간의 평균각도
* $\mathrm{W-Inter}^\Uparrow$: minimum angle between $W_j$'s 
	* $W_j$ 들 간의 최소 각도
* $\mathrm{Intra\ 1}^\Downarrow$ : mean of angles between $x_i$ and embedding feature centre on CASIA
	* $x_i$ 와 embedding feature 중심간의 평균각도(Dataset : CASIA)
* $\mathrm{Inter\ 1}^\Uparrow$ : minimum angles between embedding feature centres on CASIA
	* embedding feature 중심간의 최소 각도(Dataset : CASIA)
* $\mathrm{Intra\ 2}^\Downarrow$ : mean of angles between $x_i$ and embedding feature centre on LFW
	* $x_i$ 와 embedding feature 중심간의 평균각도(Dataset : LFW)
* $\mathrm{Inter\ 1}^\Uparrow$ : minimum angles between embedding feature centres on LFW
	* embedding feature 중심간의 최소 각도(Dataset : LFW)


---

(1) ArcFace의 경우 $W_j$는 임베딩 feature 중심과 거의 동기화되어 있으며 각도는 14.29도입니다. 그러나 Norm-Softmax의 경우 $W_j$와 임베딩 feature 중심 사이에는 명확한 편차(44.26도)가 있습니다. 따라서 $W_j$ 간의 각도는 훈련 데이터에서 inter-class 차이를 절대적으로 나타낼 수 없습니다. 대신, 훈련된 네트워크에 의해 계산된 임베딩 feature 중심이 더 대표적입니다. 
(2) Intra-Loss는 intra-class 변동을 효과적으로 압축할 수 있지만, inter-class 각도도 더 작게 만듭니다. 
(3) Inter-Loss는 $W$ (직접적으로)와 임베딩 네트워크 (간접적으로) 둘 다에서 inter-class 차이를 약간 증가시킬 수 있지만, intra-class 각도도 증가시킵니다. 
(4) ArcFace는 이미 매우 좋은 intra-class compactness과 inter-class 차이를 가지고 있습니다. 
(5) Triplet-Loss는 intra-class compactness은 ArcFace와 비슷하지만 inter-class 차이는 ArcFace에 비해 열등합니다. 또한, 그림 6에서 보여주듯이, ArcFace는 테스트 세트에서 Triplet-Loss에 비해 더 뚜렷한 margin을 가지고 있습니다.

---

## 3.3. Evaluation Result
### Results on LFW, YTF, CALFW and CPLFW
* 아래 표의 칼럼은 Accuracy
![[Pasted image 20240504162935.png|500]]
* LFW, YTF 에서 ArcFace 가 SOTA.

![[Pasted image 20240504163133.png|500]]
* CALFW, CPLFW 는 기존의 데이터셋보다 더 많은 연령대와 더 많은 자세를 가짐
	* 즉, 더 challenging 한 benchmark
	* 아래 Figure 7. 에서도 볼 수 있듯, CALFW, CPLFW 가 negative 와 positive 가 혼재되어 있는 영역이 더 크다. 이는 두 데이터셋이 더 challenging 한 benchmark 임을 보이는 결과이다.
* LFW, CALFW, CPLFW 에서도 ArcFace 가 SOTA
![[Pasted image 20240504163508.png|500]]

### Results on MegaFace
![[Pasted image 20240504163657.png|500]]
* $\mathrm{Id}$ : rank-1 face identification accuracy
* $\mathrm{Ver}$ : face verification TAR at $10^{-6}$ FAR
	* TAR (True Acceptance Rate): 시스템이 올바른 개인을 정확하게 인식하는 비율. 
		* 예를 들어, 얼굴 인식 시스템에서 올바른 사람의 얼굴을 맞게 인식하는 경우.
	* FAR (False Acceptance Rate): 시스템이 잘못된 개인을 올바른 개인으로 잘못 인식하는 비율.
		* 예를 들어, 한 사람의 얼굴을 다른 사람의 것으로 잘못 인식하는 경우.
	* 즉 FAR 을 매우 strict 하게 잡아 정확도를 계산하는 방법
* 이 표에서도 ArcFace 가 명확한 SOTA
* 위의 표에서 CASIA, R50, ArcFace, R 에서 R 은 data refinement 를 진행한 MegaFace benchmark 에서 실험을 진행했다는 의미
![[Pasted image 20240504164854.png|500]]

### Results on IJB-B and IJB-C
![[Pasted image 20240504165038.png|500]]
* SOTA

### Results on Trillion-Pairs
![[Pasted image 20240504165226.png|500]]
* 어떠한 Dataset 이 이 모델의 성능을 올려줬는지에 대한 평가
* 제안된 MS1MV2 데이터셋은 CASIA와 비교하여 성능을 분명히 향상시키며, 심지어 신원 수가 두 배인 DeepGlint-Face 데이터셋보다 약간 더 나은 성능을 보인다.

### Results on iQIYI-VID
![[Pasted image 20240504165537.png|500]]

* iQIYI-VID challenge 에서의 결과이다
* 위 challenge 는 video 에서의 정확도를 측정한다.


