---
dg-publish: "true"
tags:
  - backbone
  - Computer_Vision
  - Semantic_segmentation
  - Keypoint__Detection
---


# Abstract

* High resolution representation 은 여러 vision task 에서 매우 중요하게 다루어지는 문제이다.

* 기존의 방법론
	1. image -> high to low resolution feature map 생성 : (a)
	2. (a) 과정에서 high resolution feature 를 (b) 과정으로 convolution 으로 연결
	3. low resolution 을 high resolution 으로 recover : (b)

![[Pasted image 20240319164110.png]]

# 1   Introduction

* 기존 CNN base model
	1. 점진적으로 feature 의 size 를 줄여나가면서 결과적으로는 low resolution feature 를 만들어 낸다.
		* 이 map 은 sematically strong 하다.
		* 하지만 <span style='color:#eb3b5a'>high resolution 이 가지는 성질을 사용할 수 없다</span>(e.g. 위치정보)
	2. 만약 high resolution 이 포함하는 정보(위치정보등)가 필요할 경우 low resolution 에서 high resolution 으로 복원하는 방식을 취한다.

* 위의 방식 대신 우리는 <span style='color:#2d98da'>HRNet 을 제시</span>한다.
* 이 방식은 <span style='color:#2d98da'>high resolution 임에도 semantically strong</span> 하다.
* HRNet 은 아래 과정을 따른다
	1. high resolution 에서 low resolution 연산을 연속적인 연산이 아닌 **병렬로 처리**한다.(아래 사진 참고)
		*  위의 이유로 low resolution 에서 high resolution 으로의 복원을 하는 것이 아니라 <span style='color:#2d98da'>high resolution 을 지속적으로 가질 수 있게 됐다</span>.
	2. **low resolution 의 semantic 정보를 high resolution 에 전달**한다. 이를 <span style='color:#2d98da'>multi-resolution</span> 이라 한다.
		*  기존의 fusion method 는 high resolution-low level 과 low resolution-high level 을 병합하는 방식이였다. 
		*  위의 과정을 통해 high resolution 도 semantically strong 한 성질을 가지게 된다.
![[Pasted image 20240320153008.png]]

* HRNet 은 아래 3가지 version 이 존재한다.
1. HRNetV1
	* high resolution conv steam 에서 연산된 high-resolution representation 출력사용
	* human pose estimation 에서 사용됨
2. HRNetV2
	* high-to-low resolution parallel stream 으로부터 추출된 representation 을 combine
	* semantic segmentation 에서 사용됨
3. HRNetV2p
	* HRNetV2 에서 추출된 high-level representation 에서 multi-level representation 을 구축한다.
	* object detection, instance segmentation 에서 사용됨


# 2   Related Work
### Learning Low-Resolution Representation
* **high resolution(low semantic info) -> low resolution(high semantic info)**
* FCN(Fully Convolutional Network)은 fully connected layer 를 제거하여 convolution 연산으로만 이루어진 model
* 이러한 model 은 low-level medium-resolution representation 을 가진다.
	* 위의 방법론은 face align, edge detection, holistic edge detection 등에 사용됨
* 또한 위의 방식에서 조금 더 나아간 방식이 FPN(Feature Pyramid Network)이다.
* FPN 을 통해 multi-scale contextual representation 을 얻을 수 있다.
	* 위의 방법론은 instance segmentation 등에 사용됨

### Recovering High-Resolution Representation
* **low resolution -> high resolution 으로의 복원을 점진적으로 진행하는 방법**
* down sampling 을 통해 low resolution 생성
* up sampling 을 통해 high resolution 으로 복원
* up/down sampling network 가 대칭인경우
	* 또한 mirrored layer 를 복사하거나 계층을 건너뛰는 경우가 많음
	* e.g. SegNet, DeconvNet, U-net, Hourglass
* up/down sampling network 가 비대칭인경우
	* light upsampling process : 
		* 계산부담을 줄이면서도 해상도를 향상시킴
	* light downsample + heavy upsample : 
		* 가벼운 down sampling 을 통해 정보소실을 최소화
		* 무거운 upsampling 을 통해 세밀한 정보복원
	* skip connection between row/high resolution
		* low resolution -> high resolution connection 진행
	* FPN(Feature Pyramid Network)
		* multi scale feature extractor

### Maintaining High-Resolution Representations
* 이 논문에서 제시하는 방법론과 유사하다.
* high resolution 을 유지하는 방법론이다.
	* e.g. convolutional neural fabrics, interlinked CNNs, GridNet, and multiscale DenseNet
* 하지만 위의 예시로 들었던 방법론 들은 <span style='color:#eb3b5a'>병렬처리 위치, low-high info 교환등에 대한 세심한 설계가 부족</span>하다.
* 또한 <span style='color:#eb3b5a'>batch norm, residual connection 등을 사용하지 않았기에 충분한 성능을 보이지 못한다</span>.

### Multi-Scale Fusion
* upsampling 과정과 downsampling 과정에서 점진적으로 두 resolution 을 연결한다.
	* e.g. Hourglass, U-Net, SegNet
* Pyramid Pooling module atrous spatial pyramid pooling 을 통해 얻은 pyramid 특징을 fusion 한다.
	* e.g. PSPNet, DeepLabV2/3 
	* 이 논문에서 제시하는 pooling 도 위 방식과 유사하다.
	* 차이점은 아래와 같다.
		1. 단일한 resolution representation 이 아닌 4개의 resolution representation 사용
		2. fusion module 을 여러번 반복해서 사용하여 deep fusion 진행

### Our Approach
* high -> low convolution stream 을 병렬로 연결한다.
	* 위의 과정을 통해 <mark style='background:#2d98da'>전체 과정에서 high-resolution representation 을 유지</mark>한다.
	* 또한 multi resolution stream 에서 지속적으로 representation 을 fusion 함으로써 <mark style='background:#2d98da'>position sensitivity 를 유지</mark>할 수 있다. 

---

# 3   High-Resolution Network

![[Pasted image 20240320200637.png]]
1. stage1 : high-resolution covolution
2. stage2~4 : `two~four`-resolution block

* two stride-2 3x3 convolution 을 통해 resolution 을 $\frac{1}{4}$ 로 축소
* 각 main body 는 아래로 이루어져 있다.(아래 그림 참고)
	* parallel multi-resolution convolution
	* repeated multi-resolution fusions
	* representation head
![[Pasted image 20240321100848.png]]

## 3.1   Parallel Multi-Resolution Convolution
![[Pasted image 20240320200637.png]]
* High resolution 에서 시작하여 다음 스테이로 넘어갈 때 마다 low resolution 을 병렬로 추가
* 즉 뒤에 있는 블록일수록 low resolution representation 을 더 포함한다.(아래 이미지 참고)
![[Pasted image 20240321154725.png]]
* $N_{sr}$ : 
	* $s$ : s-th stage
	* $r$ : resolution index
		* $r=\frac{1}{2^{r-1}}$ 

## 3.2   Repeated Multi-Resolution Fusion

* 여러 종류의 resolution 을 fusion 하기 위해 <mark style='background:#2d98da'>fusion 을 하기 위한 경우의 수 3개를 제안</mark>한다.
* 또한 그 fusion 을 할 때 <mark style='background:#2d98da'>resolution 을 맞춰주기 위해 fusion function 을 제안</mark>한다.
![[Pasted image 20240321160345.png]]
* **Three representation**
$$\{R^i_r,r=1,2,3\}$$
* $r$ : resolution index

* **Output representation**
$$\{R^o_r,r=1,2,3\}$$
* 각 output representation($R^o_r$) 은 3개의 input 의 합으로 이루어진다.
$$R^o_r=f_{1r}(R^i_1)+f_{2r}(R^i_2)+f_{3r}(R^i_3)$$
* 단, stage3->stage4 로 갈 땐 아래처럼 4번재 representation 이 추가된다.
$$
R^o_r=f_{14}(R^i_1)+f_{24}(R^i_2)+f_{34}(R^i_3)
$$
![[Pasted image 20240321162119.png]]
![[Pasted image 20240321162138.png]]
![[Pasted image 20240321162149.png]]

### Transform Function : $f_{xr}(\cdot)$
$$f_{xr}(\cdot)$$
* if $x=r$ : 
	* $f_{xr}(R)=R$
* if $x<r$ : downsampling
	* ($r-s$)stride-2 3x3 conv 를 적용
	* 1 개의 stride-2 3x3 conv : 2배 downsampling
	* 2 개의 stride-2 3x3 conv : 4배 downsampling
* if $x>r$ : upsampling
	* bilinear upsampling + 1x1 conv
![[Pasted image 20240321164031.png]]

## 3.3   Representation Head
* Representation Head 란 이 논문에서 최종적으로 특정 task 에서 사용할 feature map 을 의미한다.
* 이 Representation Head 는 총 3가지 종류를 가지며 각 head 는 vision task 에 따라 달리 사용된다.
![[Pasted image 20240321164617.png]]

* HRNetV1 - (a)
	* 오로지 High resolution representation(연노랑 박스)만 사용한다.
	* 다른 3개의 Representation 은 무시한다.
* HRNetV2 - (b)
	* 4개의 representation 을 모두 이용한다.
	* low resolution representation 들은 전부 bilinear upsampling + 1x1 conv 를 이용하여 resolution 을 맞춘 뒤 concat 하여 사용한다.
* HRNetV3 - (c)
	* 4개의 representation 을 모두 이용한다.
	* HRNetV2 에서 추출된 feature map 을 downsampling 하여 사용한다.

## 3.5   Analysis
![[Pasted image 20240321180059.png]]
* Multi-resolution parallel convolution - (a)
	* group conv 와 유사
	* 여러개의 subset 으로 채널을 분리, 그 subset 은 각각 다른 resolution 을 가지도록 설계
	* 위를 통해 group conv 의 일부 이점을 지닐 수 있다.
> 참고
> Group Convolution 이란?
> 입력의 채널을 여러개의 그룹으로 나누어 각각 convolution 연산을 진행하는 방법

![[Pasted image 20240321193348.png]]

* Multi-resolution fusion - (b)
	* 일반적인 conv 의 multibranch full-connection(위의 그림의 (c)) 과 유사
	* 입력과 출력 모두 채널로 나누어서 연산

# 4    Human Pose Estimation
![[Pasted image 20240322124924.png]]
* HRNet 을 Human pose estimation 에 적용한다.
	* Human pose estimation 이란
		* WxHx3 이미지 I 에서 K 개의 keypoint(e.g. elbow, wrist, arm ...) 를 예측하는 task
		* K 개의 heatmap : $\{H_1,H_2,...,H_K\}$
		* heatmap 의 size : $\frac{W}{4}\cdot\frac{H}{4}$  
* HRNetV1 을 적용하여 high-resolution representation 을 이용하여 heatmap regression 을 진행한다.
![[Pasted image 20240322131829.png|300]]
### Loss function
* Predicted heat map 과 GT heat map 간의 MSE(Mean Square Error)
* GT heatmap 은 각 keypoint 의 실제 위치에 중심을 둔 2 pixel 표준편차의 2D gaussian 을 적용하여 생성

### Evaluation Metric
* [reference of OKS](https://glee1228.tistory.com/7)
* Based on OKS(Object Keypoint Similarity)
$$
\sum_{i} \exp\left(-\frac{d_i^2}{2s^2 k_i^2}\right) \delta(v_i > 0) \Bigg/ \sum_{i} \delta(v_i > 0)
$$
* $d_i$ : GT keypoint 와 predict keypoint 사이의 Euclidean distance
* $v_i$ : GT 의 visibility flag 
	* $v=0$ : labeling 되어 있지 않음
	* $v=1$ : labeling 되어 있지만 보이지 않음
	* $v=2$ : labeling 되어 있고 보임
* $s$ : object 의 scale(머리는 크게 눈은 비교적 작게)
* $k_i$ : keypoint의 가중치
	* keypoint 의 중요도를 기준으로 부여하는 가중치
![[Pasted image 20240322110651.png|300]]

# 5    Semantic Segmentation
![[Pasted image 20240322125510.png]]
* input image 를 HRNetV2 에 넣고 
* 15C-dimensional representation 에서 각 포인트에 대해 linear classifier 적용
* softmax loss 로 segmentation map 예측
![[Pasted image 20240322131857.png|300]]

### Evaluation metric
* mIOU 사용
![[Pasted image 20240322125701.png|300]]

### Dataset : PASCAL-Context
* training: 4998 / testing : 5105
* class : 59 + 1(background)

### Dataset : Cityscape
* 5000 data
* training: 2975 / validation: 500 / testing : 1525
* class num : 30(or 19)
* eval metric : mIOU

### Dataset : LIP(Human parsing dataset)
![[Pasted image 20240322130301.png]]
* training: 30462 / validation: 10000
* class : 19(human part level) + 1(background)

# COCO Object Detection
* object detection task 를 위해 COCO dataset 을 사용함
* HRNetV2p 사용
![[Pasted image 20240322131930.png|300]]
* 추가적인 작업으로 Instance segmentation 도 진행

### COCO dataset
* training data: 118k / validation data: 5k / test data: 20k