---
dg-publish: "true"
---
#Deep_Learning #Computer_Vision #Object_Detection 

# Abstract

* Feature pyramid는 여러 scale의 object를 탐지하기 위한 기본 구성 요소이다.
* 최근의 deep learning object detector에서는 과도한 메모리 사용으로 인해 pyramid 를 피하는 경향이 있다.
* 하지만 이 논문에서는 Feature Pyramid Network 라는 방법을 제시하여 computing 자원을 적게 차지하며 다양한 object의 크기를 detecting하는 방법을 제시한다.
* 기존 방식의 문제점과 FPN의 핵심 아이디어, Faster R-CNN과 결합된 FCN의 학습 과정과 성능에 대해 살펴본다.

# Introduction & Related Work

* 이 논문에서 Introduction 과 Related Work 는 대동소이하므로 둘을 합쳐서 기술할 것이다.
![[Pasted image 20240224211836.png]]

**(a) Featurized image pyramid**

* 입력 이미지의 크기를 resize하여 다양한 scale의 이미지를 네트워크에 입력하는 방법입니다.
* Overfeat 모델 학습 시 해당 방법을 사용했습니다. 
* 다양한 크기의 객체를 포착하는데 좋은 결과를 보여줍니다. 
* 하지만 이미지 한 장을 독립적으로 모델에 입력하여 feature map을 생성하기 때문에 추론 속도가 매우 느리며, 메모리를 지나치게 많이 사용한다는 문제가 있습니다. 

**(b) Single feature map**

* 단일 scale의 입력 이미지를 네트워크에 입력하여 단일 scale의 feature map을 통해 object detection을 수행하는 방법입니다. 
* YOLO v1 모델 학습 시 해당 방법을 사용했습니다. 
* 학습 및 추론 속도가 매우 빠르다는 장점이 있지만 성능이 떨어진다는 단점이 있습니다. 

**(c) Pyramidal feature hierarchy**

* 네트워크에서 미리 지정한 conv layer마다 feature map을 추출하여 detect하는 방법입니다. SSD 모델 학습 시 해당 방법을 사용했습니다. 
* multi-scale feature map을 사용하기 때문에 성능이 높다는 장점이 있지만 feature map 간 해상도 차이로 인해 학습하는 representation에서 차이인 **semantic gap**이 발생한다는 문제가 있습니다. 
* 모델이 얕은 layer에서 추출한 feature map에서 저수준 특징(low-level feature)까지 학습하면 representational capacity를 손상시켜 객체 인식률이 낮아진다고 합니다.

* SSD는 위에서 언급한 문제를 해결하기 위해 low-level feature를 사용하지 않고, 전체 convolutional network 중간 지점부터 feature map을 추출합니다. 
* 하지만 FPN 논문의 저자는 높은 해상도의 feature map은 작은 객체를 detect할 때 유용하기 때문에 이를 사용하지 않는 것은 적절하지 않다고 지적합니다.

# 3. Feature Pyramid Network

* 우리의 목표는 ConvNet 의 pyramidal feature hierarchy 를 활용하여 low-level 부터 high-level 의 semantic 을 가진 feature pyramid 를 구축하는 것이다.
* 그 결과로 나온 FPN 은 범용적이다.
* 특히 Region Proposal Method, Fast R-CNN 에 초점을 맞춘다.
* 또한, FPN 을 Instance Segmentation 으로 변화하는 방법도 제시한다.
* 우리의 방법론은 단일 scale 의 image 를 입력으로 받아 fully convolutional 한 방법으로 multi-level 에서 크기가 조정된 feature map 을 출력한다. 
* 우리의 Pyramid 구성은 bottom-up, top-down, lateral connection 을 포함한다.

### Bottom-up pathway
![[Pasted image 20240224220022.png]]
* **Bottom-up pathway** 과정은 이미지를 convolutional network에 입력하여 forward pass하여 **2배씩 작아지는** feature map을 추출하는 과정이다. 
* 이 때 각 stage의 마지막 layer의 output feature map을 추출한다. 
* 네트워크에는 같은 크기의 feature map을 출력하는 layer가 많지만 논문에서는 이러한 layer를 모두 같은 **stage**에 속해있다고 정의한다.
* 각 stage에서 별로 마지막 layer를 pyramid level로 지정하는 이유는 더 깊은 layer일수록 더 강력한 feature를 보유하고 있기 때문이다. 

* ResNet의 경우 각 stage의 마지막 residual block의 output feature map을 활용하여 feature pyramid를 구성하며 각 output을 **{c2, c3, c4, c5}**라고 지정한다**. 
* **이는 conv2, conv3, conv4, conv5의 output feature map임을 의미하며, 각각 {4, 8, 16, 32} stride**를 가지고 있다. 
* 여기서 {c2, c3, c4, c5}은 각각 원본 이미지의 1/4, 1/8, 1/16, 1/32 크기를 가진 feature map이다. 
* conv1의 output feature map은 너무 많은 메모리를 차지하기 때문에 피라미드에서 제외시킨다.

### Top-Down pathway and lateral connections
*  **Top-down Pathway**는 각 pyramid level에 있는 feature map을 2배로 upsampling하고 channel 수를 동일하게 맞춰주는 과정이다.
* 각 pyramid level의 feature map을 2배로 upsampling해주면 바로 아래 level의 feature map와 크기가 같아집니다. 가령 c2는 c3와 크기가 같아진다.
* 이 때 **nearest neighbor upsampling** 방식을 사용한다. 
* 이후  모든 pyramid level의 feature map에 1x1 conv 연산을 적용하여 channel을 256으로 맞춘다.
![[Pasted image 20240224220552.png]]
* 그 다음 upsample된 feature map과 바로 아래 level의 feature map과 element-wise addition 연산을 하는 **Lateral connections** 과정을 수행한다. 
* 이후각각의 feature map에 3x3 conv 연산을 적용하여 얻은 feature map을 각각 **{p2, p3, p4, p5}이다.**
* 이는 **각각 {c2, c3, c4, c5} feature map의 크기와 같다**. 
* 가장 높은 level에 있는 feature map c2의 경우 1x1 conv 연산 후 그대로 출력하여 p2를 얻는다.

* 위의 과정을 통해 FPN은 single-scale 이미지를 입력하여 4개의 서로 다른 scale을 가진 feature map을 얻는다. 
* 단일 크기의 이미지를 모델에 입력하기 때문에 기존 방식 (a) 방식에 비해 빠르고 메모리를 덜 차지한다. 
* 또한 multi-scale feature map을 출력하기 때문에 (b) 방식보다 더 높은 detection 성능을 보여준다.
![[Pasted image 20240224220845.png]]
* Detection task 시 **고해상도 feature map은 low-level feature를 가지지만 객체의 위치에 대한 정보를 상대적으로 정확하게 보존**하고 있다. 
* 이는 저해상도 feature map에 비해 downsample된 수가 적기 때문이다. 
* **이러한 고해상도 feature map의 특징을 element-wise addition을 통해 저해상도 feature map에 전달하기 때문에 (c)에 비해 작은 객체를 더 잘 detect한다.**

|                             | spatial information of object's position | feature information                          |
| --------------------------- | ---------------------------------------- | -------------------------------------------- |
| high resolution feature map | 낮음 : 공간정보가 많이 붕괴됨                        | 높음 : feature 가 conv 를 통해 많이 압축이 되어 많은 정보를 저장 |
| low resolution feature map  | 높음 : 공간정보를 잘 유지함                         | 낮음 : feature 정보를 비교적 적게 저장                   |

# 4. Applications

### 4.1. Feature Pyramid Network for RPN
* 기존의 RPN 은 Faster R-CNN 에서 제시하였다.
* RPN 의 구조는 3x3 conv layer, 2개의 1x1 conv layer 로 이루어져 있다.
* 위의 구조를 네트워크 헤드라고 한다.
![[Pasted image 20240225102638.png]]
* 또한 bouding box 의 유추를 위해 사전에 정의된 anchor box 를 사용한다.(아래 그림 참고)
![[Pasted image 20240225102811.png]]
* 우리는 위의 네트워크 헤드를 FPN 으로 대체한다.
![[Pasted image 20240225103059.png]]
* 원래는 위의 초록박스(네트워크 헤드)가 단일하게 anchor box 를 유추했다.
* 하지만 FPN 에서는 각 레벨로 네트워크 헤드를 적용하며, 각 네트워크 헤드에는 3개의 anchor box 를 부여한다. -> {1:1, 1:2, 2:1} 3개
* 위의 그림에서는 P6 가 없지만 원래는 {P2, P3, P4, P5, P6}에서 각각 {32², 64², 128², 256², 512²} 픽셀의 영역을 가진 앵커를 정의한다.
* 따라서 3개의 Anchor box 종류 * 5개의 레벨 = 15 개의 anchor box 가 나온다.
* 기존의 anchor box 를 비율과 크기별로 모두 선언한 것과 달리, FPN+RPN 방식은 고정된 3개의 비율과 여러 크기의 feature map 을 사용하여 동일한 효과를 낸다.

### 4.2. Feature Pyramid Networks for Fast R-CNN
* FPN을 통해 얻은 **multi-scale feature map {p2, p3, p4, p5}** 과 RPN(+FPN) 과정을 통해 얻은 **1000개의 region proposals**를 사용하여 **RoI pooling**을 수행한다.
* Fast R-CNN은 single-scale feature map만을 사용한 반면, FPN을 적용한 Faster R-CNN은 multi-scale feature map을 사용하기 때문 **region proposals를 어떤 scale의 feature map과 매칭**시킬지를 결정해야한다.
* 본 논문에서는 아래와 같은 공식으로 결정한다.
$$
k = \left\lfloor k_0 + \log_2\left(\sqrt{wh}/224\right) \right\rfloor
$$
* 논문에서는 위와 같은 공식에 따라 region proposal을 $k$번째 feature map과 매칭한다. 
* $w, h$ 는 RoI(=region proposal)의 width, height에 해당하며, $k$ 는 pyramid level의 index, $k_0$ 은  target level을 의미한다. 
* 논문에서는 $k_0=4$ 로 지정했다. 
* **직관적으로 봤을 때 RoI의 scale이 작아질수록 낮은 pyramid level, 즉 해상도가 높은 feature map에 할당하고 있음을 알 수 있다.**

* 위와 같은 공식을 사용하여 region proposal과 index가 $k$ 인 feature map을 통해 RoI pooling을 수행합니다. 이를 통해 고정된 크기의 feature map을 얻을 수 있습니다. 
	- **Input** : multi-scale feature map {p2, p3, p4, p5} and 1000 region proposals
	- **Process** : RoI pooling
	- **Output** : fixed sized feature maps