---
dg-publish: "true"
---
#Deep_Learning #Computer_Vision #Object_Detection 

# Abstract
* 기존의 YOLO detection model 에서 개선된 논문이다.
* Faster R-CNN with ResNet and SSD 보다 높은 성능을 보인다.
* 또한 speed 와 accuracy 는 trade-off 관계인데 이 논문에서는 균형을 잘 맞추어 좋은 성능을 보인다.

# 1. Introduction
* 기존의 object detection model 은 빠르고 정확하며 여러 object 를 구분할 수 있다.
* 하지만 대부분의 detection method 는 적은 수의 object 갯수만을 detecting 한다.

* 기존의 detection dataset 은 classification dataset 에 비해 그 규모가 너무 작다.
* 따라서 이 논문에서는 classification dataset 을 detection dataset 으로의 확장을 꾀한다.

* 많은 수의 카테고리를 구분하는 Object detection 을 위해 아래 2가지를 진행한다
	1. State-of-the-art object detection model : yolov2
	2. dataset combination method 와 joint training algorithm 을 통해  더 많은 카테고리(over 9000)를 구분한다.

# Proposed Method
# 2. Better
* 기존의 YOLO 는 (R-CNN base method 에 비해) 아래의 단점을 가진다.
	* 꽤 많은 수의 localization error 를 가진다.
	* low recall 을 가진다. 

* 또한 대부분의 computer vision 의 dl network 는 더 큰 network, 혹은 많은 모델의 앙상블을 통해 더 좋은 성능을 낼 수 있다.
* 하지만 yolo v2 에서는 다른 방식을 취할 것이다.

![[Pasted image 20240224153914.png]]
* Two-stage Detector : Fast/Faster R-CNN
* One-stage Detector : YOLO
### Batch Nromalization
* YOLO 의 모든 CNN 계층에서 batchnorm 을 추가한다.
* 이를 통해 mAP 에서 **2%** 의 개선효과를 얻는다.
* 또한 batchnorm 을 진행하면 dropout 을 제거하더라도 overfitting 이 발생하지 않는다.

### High Resolution Classifier
* 원래 YOLO 는 아래를 따른다.
	* Classification Network : darknet 을 224 x 224 image 로 pretrain
	* Detection Network : 448 x 448 image
* 즉 classification 에서 detection 단계로 넘어갈 때 모델은 학습 전환을 감지하고 새로운 해상도에 맞게 조정되어야 한다.

* 하지만 이 논문에서 제시하는 YOLOv2 는 다른 방법을 사용한다.
	* Classification Network : ImageNet 448 x 448 image 를 10 epoch 동안 fine-tuning 한다.
		* 이 과정을 통해 high resolution input image 에 대해서도 잘 적응할 수 있도록 한다.
	* Detection Network : Fine-tune 을 하여 network 를 detection 작업으로 전환한다.
* 이러한 방식은 mAP 에서 **4%** 의 증가를 보인다.

### Convolutional With Anchor Boxes
* 기존의 YOLO 는 각 grid cell 의 bounding box 의 좌표가 0~1 사이의 값을 가지도록 random initialize 하여 bounding box 의 최적값을 찾는다.
* 반면 Faster R-CNN 은 사전에 9개의 anchor box 를 설정해 두고 bounding box regression 을 통해 x, y 좌표와 offset 을 조정하는 과정을 거친다.
* 좌표 대신 offset 을 예측함으로써 문제를 단순화하여 네트워크의 학습을 용이하게 해 준다.
* ==YOLOv2 에서는 이러한 anchor box 를 도입한다.==
	1. Conv layer 의 ouput 이 높은 resolution 을 가지도록 pooling layer 를 제거한다.
	2. 또한 원래 입력을 448x448 로 주었다고 언급했지만 나중에 ==conv layer 이후 추출될 output feature map 의 크기가 홀수(13x13) 이 되도록 하기 위해== <mark style='background:#3867d6'>416x416 으로 crop</mark>한다.
		* 이러한 작업을 하는 이유는 크기가 큰 객체의 경우 이미지 내에서 중심을 차지하는 경향이 있기에 하나의 중심 Cell 을 추출 할 수 있는 홀수의 크기로 조정한 것
	3. 416x416 -> ConvNet -> 13x13(downsample ratio = 1/32).
	4. YOLO v1 은 98개의 box 를 예측 / YOLO v2 는 1000개 이상을 예측

|                     | Recall | mAP   |
| ------------------- | ------ | ----- |
| YOLOv2 w\\o. Anchor | 81%    | 69.5% |
| YOLOv2 w\\. Anchor  | 88%    | 69.2% |

* 위 표에서도 볼 수 있듯 Anchor Box 도입시 mAP 가 떨어짐을 알 수 있다.
* 하지만 Recall 이 높으므로 네트워크 개선의 여지가 있음을 알 수 있다.
	* Object Detection 에서 Recall 이 높다는 것 의 의미 : 실제 object 의 위치를 예측한 비율이 높다.

* YOLOv1 vs YOLOv2
	* YOLOv1 은 98개의 bounding box 로 비교
	* YOLOv2 는 1000개가 넘는 bounding box 로 비교
	* 따라서 YOLOv2 가 더 세밀하게 bounding box 를 예측가능

* <mark style='background:#eb3b5a'>anchor box 를 사용함에 있어 2가지 issue가 발생</mark>
	1. anchor box 크기와 aspect ratio 를 사전에 정의를 해야 한다.
	2. YOLO + anchor box 학습시 초기 iter 에서 모델이 불안정하다.
* 위의 2가지 문제를 아래 <mark style='background:#3867d6'>Dimension Cluser 와 Direct location prediction 에서 해결</mark>한다.
### Dimension Clusters
* anchor box 의 크기와 aspect-ratio 를 hand-pick 하는 것 보다 처음에 더 나은 조건(prior)을 선택한다면 detection 성능이 오를 것이다.
* 이를 위해 k-means clustering 을 통해 최적의 prior 를 탐색한다.

$$
d(box,centroid)=1-IOU(box,centroid)
$$

* $d$ : distance function
* $IOU$ : intersection over union
* 일반적인 k-means clustering 은 Euclidean distance 를 통해 $centroid$ 와 $sample$ 간의 거리를 계산한다.
* 하지만 단순한 pixel-wise distance 를 구하면 큰 bounding box 는 작은 bounding box 에 비해 큰 error 를 가진다.
* 이를 IOU 를 사용하여 pixel-wise 비교가 아닌 비율의 비교를 통해 해결한다.
![[Pasted image 20240224165215.png]]
* k = 5 일때 가장 균형있는 model 이 추출된다.
* hand-picked anchor box 9개 사용시 평균 IoU : 61.0
* k-mean cluster prior 5개 사용시 평균 IoU : 60.9
* 9개에 비해 더 적은 5개의 사전설정만으로 비슷한 성능을 냄
![[Pasted image 20240224165257.png]]

### Direct location prediction
$$
\begin{align*}
x = (t_x * w_a) - x_a\\
y = (t_y * h_a) - y_a\\
\end{align*}
$$
* YOLO 와 anchor box 를 함께 사용했을 때 초기 iter 에서 학습이 불안정한 문제 발생
* 이러한 불안정성은 박스의 (x,y) 위치를 예측하는 데서 발생한다.
* RPN(Region Proposal Network) 에서 네트워크는 $t_x$, $t_y$ 를 예측하고, 그 중심좌표를 위의 공식과 같이 계산한다.
* 하지만 위의 $t_x$, $t_y$, $w_a$, $h_a$ 의 값의 제한된 범위가 없다.
* 처음 random initialize 에서 적절한 값까지 가는데 많은 시행착오를 거쳐야 한다.

* 이러한 문제를 해결하기 위해 YOLO 의 grid cell 에 대한 상대적인 위치 좌표를 예측하는 방법을 채택한다
* bounding box coordinate : $t_x,t_y,t_w,t_h,t_0$ 를 예측
* example
	* $(c_x,c_y)$ : 왼쪽 상단 offset
	* bouding box prior width, height : $p_w,p_h$
$$
\begin{align*}
b_x = \sigma(t_x) + c_x \\
b_y = \sigma(t_y) + c_y \\
b_w = p_w e^{t_w} \\
b_h = p_h e^{t_h} \\
Pr(\text{object}) IOU(b, \text{object}) = \sigma(t_o)\\
\end{align*}
$$
![[Pasted image 20240224172025.png]]

* Dimension Cluster 와 Directly location prediction 을 동시에 사용하면 YOLO+anchor box 에 비해 **5%** 의 성능향상이 이루어진다.

### Fine Grained Features
 * YOLOv2 에서 사용하는 13x13 Feature map 은 큰 객체에 대해서는 탐지를 잘 하지만 작은 객체에 대해서는 조금 더 세밀한 feature map 이 필요하다.
 * 위와 같은 문제를 해결하기 위해 26x26x512 feature map 을 가져온다.
 * 이후 channel(512)은 유지한채로 26x26을 4등분하여 13x13x4 개를 만들어 concat 한다.(아래 그림 참고)
![[Pasted image 20240224172504.png]]
* 이렇게 생성된 13x13x2048 의 feature map 을 기존의 feature map 과 결합한다.
![[Pasted image 20240224173153.png]]
![[Pasted image 20240224195522.png]]
* 이러한 방법론을 통해 1% 의 성능향상을 이루었다.

### Multi-Scale Training
* 논문의 저자는 YOLO v2 모델을 보다 강건하게 만들기 위해 다양한 입력 이미지를 사용하여 네트워크를 학습시킨다.
* 논문에서는 10 batch마다 입력 이미지의 크기를 랜덤하게 선택하여 학습하도록 설계했다.
* 모델은 이미지를 1/32배로 downsample시키기 때문에 입력 이미지 크기를 32배수 {320, 352, ..., 608} 중에서 선택하도록 했다. 320x320 크기의 이미지가 가장 작은 입력 이미지이며, 608x608 크기의 이미지가 입력될 수 있는 가장 큰 이미지이다.

* 이를 통해 네트워크는 다양한 크기의 이미지를 입력받을 수 있고, 속도와 정확도 사이의 trade-off를 제공한다.
* 위의 표에서 확인할 수 있듯이 입력 이미지의 크기가 작은 경우 더 높은 FPS를 가지며, 입력 이미지의 크기가 큰 경우 더 높은 mAP 값을 가지게 된다.
![[Pasted image 20240224173826.png]]


# 3. Faster
### DarkNet-19
* YOLOv2 는 DarkNet-19 라고 하는 독자적인 classification model 을 backbone network 로 사용한다.
* 이는 기존의 다른 모델(e.g. VGG-16)을 사용할 경우 accuracy 가 올라가는 것에 비해 속도가 현저히 느려지기 때문이다.
![[Pasted image 20240224174145.png]]
* 위 사진은 DarkNet-19 의 Architecture 이다.
* 마지막 단계는 global average pooling 을 이용하여 fc layer 를 제거하여 parameter 수를 줄이고 detection 속도를 개선하였다.

### Training for classification
* 1000개의 class를 가지는 ImageNet을 학습한다.
* (위의 표에서 마지막 Convolution 의 filter 가 1000개인 이유임)
* (각 hyper parameter 와 성능지표는 논문 참고)

### Training for detection
* Classification 을 위해 만들어진 DarkNet-19 를 detection task 를 위해 수정한다.
* 마지막 Conv Layer 를 제거하고 3x3(x1024) 의 conv layer 로 대체.
* 이 때 1x1 conv layer의 channel 수는 예측할 때 필요한 수로, 앞서 살펴보았듯이, 각 grid cell마다 5개의 bounding box가 5개의 값(confidence score, x, y, w, h)과, PASCAL VOC 데이터셋을 사용하여 학습하기 때문에 20개의 class score를 예측한다. 
* 따라서 1x1 conv layer에서 channel 수를 125(=5x(5+20))개로 지정합니다

# 4. Stronger
* 논문에서는 YOLO v2를 classification 데이터와 detection 데이터를 함께 사용하여 학습시킴으로써 보다 많은 class를 예측하는 YOLO 9000을 소개한다. 
* 이를 위해 학습 시 classification 데이터와 detection 데이터를 섞어 학습시키는 방법을 생각해볼 수 있다. 
* 하지만 detection 데이터셋은 일반적이고 범용적인 객체에 대한 정보를 가지고 있는 반면, classification 데이터셋은 보다 세부적인 객체에 대한 정보를 가지고 있다. 
* 예를 들어 detection 데이터셋은 모든 개 이미지를 "개"라는 하나의 class로 분류하는 반면, classification 데이터셋은 "요크셔 테리어", "불독" 등 개를 종류별로 세부적인 class로 분류한다. 
* 이처럼 두 데이터를 섞어 학습시킬 경우 모델이 "개"와 "요크셔 테리어"를 별개의 배타적인 class로 분류할 가능성이 있습니다.
* 이러한 개와 요크셔 테리어의 class 를 합치는 과정이 필요하다.
* 이를 아래에서 다룬다.
### Hierarchical classification
* 위와 같은 의미가 같은 클래스의 분리를 방지하기 위해 WordNet 이라는 language dataset 을 가져와 적용한다.
* 원래의 WordNet 은 언어 자체가 매우 복합적이기에 Graph 형태를 띈다.
* 하지만 우리는 전체 그래프가 아닌 ImageNet의 개념에서 Tree(Hierarchical) 형태로 단순화한다.

![[Pasted image 20240224192110.png]]

* 위의 예시에서 처럼 "요크셔 테리어"를 "물리적 객체 - 동물 - 포유류 - 사냥개 - 테리어" 를 거쳐 도달 할 수 있도록 만든다.
* ImageNet 을 통해 WordTree 를 구성할 경우 최상위~최하위 노드까지의 category는 1369개이다.
* 특정 object 가 특정 category 에 속할 확률은 최상위노드에서 해당범주의 노드까지의 조건부 확률의 곱으로 나타낼 수 있다.
$$
\begin{align}
Pr(\text{Norfolk terrier}) = Pr(\text{Norfolk terrier}|\text{terrier}) \\
\cdot Pr(\text{terrier}|\text{hunting dog}) \\
\cdot \dots \cdot \\
\cdot Pr(\text{mammal}|Pr(\text{animal})) \\
\cdot Pr(\text{animal}|\text{physical object})
\end{align}
$$


### Dataset combination with WordTree
* 위에서 예시로 들었던 ImageNet -> WordTree 방식 뿐 아니라 다른 dataset 에서도 적용가능하다.
* WordNet 자체가 매우 방대한 자연어를 다룰 수 있기에 categorize 는 여러 Dataset 에서도 적용이 가능하다.
* 아래 사진은 COCO -> WordTree 에 대한 예시이다.
![[Pasted image 20240224195258.png]]

# 5. Conclusion
* 논문에서 YOLOv2와 YOLO9000에 대해 소개했는데, 다양한 크기의 image를 다양한 class로 구분하고 detection할 수 있는 정확하고 빠른 모델이다. 
* 계층적인 classification을 사용한 데이터셋 병합 아이디어는 Classification과 Segmentation 영역에서도 유용하게 사용될 수 있을것이고, 앞으로 더 많은 개선을 통해 Computer vision을 개발시킬거라고 기대하고 있다.