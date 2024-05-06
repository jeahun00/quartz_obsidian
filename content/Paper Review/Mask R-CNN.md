---
dg-publish: "true"
tags:
  - Computer_Vision
  - Deep_Learning
  - Instance_Segmentation
---
>Highlight, 혹은 글자색이 파란 것은 장점 혹은 단점을 극복한 방법, 빨간색은 제시한 방법론의 단점을 의미한다.

# Abstract

* Task : Object Instance Segmentation
* Method : ==Faster R-CNN== with ==Mask==
* Mask R-CNN 은 아래의 장점을 가진다.
	1. simple to train
	2. small overhead to Faster R-CNN(Fater R-CNN 에서 조금의 overhead 만 추가)
	3. 다른 task 에 접목하기가 좋음

# Introduction

* **Instance Segmentation**
	* **Object Detection** + **Semantic Segmentation**
		* Object Detection : 객체의 종류와 그 위치를 판단하는 task
		* Semantic Segmentation : 각 pixel 의 Class 를 구분해야 하는 task
	* 위의 2가지 작업을 동시에 진행해야 하기에 상당히 challenging 한 작업이다.

* **Mask R-CNN**
	* 각 RoI(Region of Interest) 에서 ==segmentation mask 를 예측하는 branch 를 추가==한다.
	* 이 작업은 classification 과 bounding box regression 작업과 동시에 진행된다.
	* <span style='color:#eb3b5a'>Faster R-CNN 에서 사용하는 RoIPool 은 pixel to pixel alignment 에 적합하지 않다</span>.
		* Object Detection 에서는 bounding box 를 이용하기에 pixel 단위의 정확도가 필요하지 않다.
		* 하지만 segmentation 작업은 애초에 pixel 단위의 작업이므로 적합하지 않다.
		* RoI Pooling Ref Link : https://inhovation97.tistory.com/69
	* <span style='color:#3867d6'>이에 본 논문에서는 RoI Align 이라는 기법을 제시한다. 이 기법 사용시 아래 장점을 가진다.</span>
		1. strict 한 localization metric 에서 10% -> 50% 의 성능향상을 보인다.
		2. mask prediction 과 class prediction 을 분리하는 것이 필수적이라는 것을 발견했다.

# Related Work

### R-CNN : 

---
* 참고
* R-CNN : 
	1. Selective Search 를 통한 RoI 추출 
	2. 각 RoI 에 대해 ConvNet 적용
	3. 추출된 ConvNet 으로 Bounding Box Regression 과 Classification 작업 진행
	4. Limitation : 
		1. <mark style='background:#eb3b5a'>각 RoI 에 대해 모두 ConvNet 을 학습해야 하므로 높은 연산량 요구</mark>
		2. -> 이를 해결하기 위해 Fast R-CNN 제안
![[Pasted image 20240222155524.png]]
* Fast R-CNN
	1. <mark style='background:#3867d6'>Image 에서 pretrain 된 ConvNet 으로 Feature Map 을 먼저 추출</mark>
	2. ConvNet 의 Feature map 에서 Selective Search 를 이용하여 RoI 추출
		* 이 연산은 CPU 연산이다(DL 학습과정이 아님)
	3. CNN 학습 -> Bounding Box Regeression, Classification 진행
	4. Limitation : 
		1. <mark style='background:#eb3b5a'>RoI 의 추출을 Selective Search 를 통해 추출하므로 bottleneck 발생</mark>
		2. -> 이를 해결하기 위해 Faster R-CNN 제안
![[Pasted image 20240222155602.png]]
* Faster R-CNN
	1. ConvNet 으로 feature map 추출
	2. <mark style='background:#3867d6'>Region Proposal Network 의 적용으로 RoI 추출</mark>
	3. 2번째 단계에서 얻어진 RoI 로 Classification 과 Bounding-box regression 을 진행
![[Pasted image 20240222155843.png]]
---

### Instance Segmentation
* 기존의 방법론들은 segmentation -> recognition 순으로 진행
* <mark style='background:#eb3b5a'>이는 느리고 부정확하다.</mark>
* <mark style='background:#3867d6'>최근에는 Object Detection System 에 segment proposal sytem 을 접목하는 방법이 제시되었다.</mark>
	* 이러한 방법론의 예시로 fully convolutional instance segmentatio(FCIS) 가 있다.
	* 이 방법론은 object class, box, mask 를 동시에 예측한다.
	* <mark style='background:#eb3b5a'>하지만 FCIS 는 Instance 가 겹치는 부분에 error 를 발생시킨다.</mark>
![[Pasted image 20240222163609.png]]
* 이러한 관점에서 다른 instance segmentation 방법론을 제시한다.
* 같은 카테고리를 다른 instance 로 자른다. 즉 segmentation 이후 객체를 구분하려 시도한다.
* <mark style='background:#3867d6'>이러한 방법들과 다르게 Mask R-CNN 은 instance 를 먼저 구분한 이후 segmentation 을 시도한다.</mark>

> Logical Flow : 방법론제시->장점->단점->다른방법론제시 가 반복되는 구조라서 헷갈리기에 정리함
> 1. Object Detection -> Segmentation 진행
> 2. 위 방법은 여러 단점 존재
> 3. 이에 Segmentation -> Object Detection 진행
> 4. 하지만 우리의 논문은 instance(Object Detection) -> Segmentation 을 진행할 것
> 5. 위의 2가지 방법은 점점더 combine 될 것을 이야기함


# Mask R-CNN

* Faster R-CNN 은 2가지 ouput 을 가진다.
	1. Class Label
	2. Bounding Box offset
* Instance Segmentation 작업을 위하여 하나의 branch 를 더 추가한다.
	1. object mask
* Mask R-CNN 은 위의 3가지의 branch 를 이용하며 pixel level 에서의 세밀한 segmentation 성능을 위하여 Fast/Faster R-CNN 에서 특정 기능을 추가한다.

### Faster R-CNN
* Fast R-CNN 은 Region Proposal Network(RPN) 을 통해 Proposal Region 을 추출한다.
* 위의 Related Work Part 에서 설명했으므로 간단히 짚고 넘어간다.

### Mask R-CNN
* Mask R-CNN 은 2단계의 procedure 가 있다.
	1. RPN
	2. parallel 하게 class, box offset 예측. 이 단계에서 각 RoI 별로 mask 를 출력한다.
		* 최근 방법론들에서는 classification 이 mask prediction 에 의존적이다.
		* 즉, Mask prediction -> Classification 진행
		* 하지만 우리는 Fast R-CNN 과 유사하게 classification 과 bounding-box regression 을 동시에 진행한다.

* multi-task loss
$$
L = L_{cls}+L_{box}+L_{mask}
$$

* $L_{cls}$ : Classification Loss
* $L_{box}$ : Bounding-box Loss
* $L_{mask}$ : Avarage binary cross-entropy loss
	* Binary mask which has $Km^{2}$ dimension for each RoI
	* $K$ : number of class
	* $m^2$ : resolution of mask
	* 위의 $L_{mask}$ 는 각 클래스별로 따로 연산되어 class 간의 경쟁을 하지 않도록 유도

### Mask Representation
* Box offset, class label 같은 경우 spatial information 을 압축한다.
* 그렇기에 FC(Fully Connected Layer) 로 정보를 압축하더라도 충분한 학습이 가능하다.
* 하지만 Mask 는 input image 와 같은 사이즈를 가지며, pixel 단위로 공간적인 정보를 보존하여야 한다. 
* 따라서 spatial information 이 무너지는 FC Layer 가 아닌 spatial Information 을 유지하는 CNN 기반의 학습을 진행하여야 한다.
* 즉, 우리는 각 RoI 의 $m$ x $m$  mask 를 예측하기 위해 Fully Convolution Network(FCN) 를 이용한다.
* Convolution Layer 를 사용함으로써 아래의 장점을 가진다.
	* 더 적은 parameter
	* 공간적인 정보 유지

### RoI Align
#### RoI Pool
* 기존에 사용하던 RoI Pool 은 classification, box offset 을 위한 것이였다.
![[Pasted image 20240222195702.png]]
* 위의 사진과 같이 RoI Pool 은 4개의 misalign 된 좌표를 quantize 한다.
	* $[x/16]$ : x / 16 한 이후의 결과를 반올림(rounding)한 결과
* 즉 소수값(Float)이 나온다면 이를 정수(Integer)로 반올림하여 정수화한다.
* 이러한 좌표의 변환은 Classification 에서는 큰 문제가 되지 않는다.(Classification 은 공간정보에 Robust 하기 때문)
* <mark style='background:#eb3b5a'>하지만 Segmentation 작업에서는 이러한 misalign 은 잘못된 결과를 도출한다.</mark>
* <mark style='background:#3867d6'>따라서 우리는 RoI Align 이라는 방법을 제시한다.</mark>

#### RoI Align
* RoI Pooling 에서 하던 작업을 유지하되 실수값이 나온다면 그대로 유지한다.
* 즉, 원래 $[x/16]$ 를 $x/16$ 로 사용한다.
* 하나의 값을 만들기 위해 주변 4개의 픽셀을 이용하여 Bilinear Interpolation 으로 값을 구한다.
![[Pasted image 20240222200553.png]]
* 이후 구한 픽셀값들을 max(or avarage) pooling 을 진행한다.

### Network Architecture
* Backbone network : 
	* Resnet-50-c4(Resnet50 의 4번째 layer 에서 추출된 Feature map)
	* ResNet-FPN(Feature Pyramid Network)