---
dg-publish: "true"
tags:
  - Deep_Learning
  - Computer_Vision
  - Object_Detection
---


<mark style='background:#eb3b5a'>빨간 형광펜은 이전의 방법론(단점을 가지는) 혹은 여러 단점들이나 극복해야 할 점들.</mark>
<mark style='background:#3867d6'>파란 형광펜은 빨간 형광펜을 극복했거나 이 논문에서 제시하는 장점들.</mark>
==노란 형광펜은 강조==
REF : 
* https://gaussian37.github.io/vision-detection-fcos/
* https://eehoeskrap.tistory.com/624 


# Abstract
* 이 논문에서는 Fully Convolutional 한 one-stage object detection 모델을 제시한다.
* 이 모델은 <mark style='background:#3867d6'>semantic segmentation 과 유사하게 pixel level 에서의 탐지를 진행</mark>한다.
* <mark style='background:#eb3b5a'>최근의 SOTA 논문들 같은 경우 pretrain 된 anchor box 에 의존적</mark>이다.
* 하지만 <mark style='background:#3867d6'>이 논문에서는 그러한 box 가 없고, region proposal 역시 없다</mark>.
* Anchor box 를 제거함으로써, 훨씬 더 단순하고 성능이 좋은 모델을 만들 수 있음을 제시한다.

# 1. Introduction
* 최근 ==Object Detection 의 main stream== 이 되는 기법들은 ==anchor box 에 의존적==이다.
* 이러한 <mark style='background:#eb3b5a'>Anchor box based detector 들은 아래의 문제점을 가진다</mark>.
	1. ==detector 의 성능이 box 의 size, aspect ratio 에 매우 민감==하다. 따라서 hyper parameter 를 매우 조심스럽게 tuning 을 해야 한다.
	2. ==Anchor box 의 scale 과 aspect ratio 가 고정되어 있기에== 큰 변화를 가진 object 와 작은 size 의 object 에 대해 탐지기의 generalize 능력을 저해한다.
	3. ==높은 recall 수치를 위해 많은 수(높은 밀도)의 anchor box 들이 필요==하다. 이러한 anchor box 들은 대부분 training 도중 negative(object 가 없음)으로 분류되는데, ==과도한 수의 negative sample 은 negative-positive 간 불균형을 야기==한다.
	4. Anchor box 는 그 특성상 GT 와의 IoU 수치로 비교하는데 ==이러한 방법론들은 많은 Computational overhead 를 야기==한다.

* 최근 FCN(Fully Convolutional Network)는 dense prediction task 에서 많은 발전을 이루었다.
	* dense prediction task : semantic segmentation, depth estimation ...
* 하지만 <mark style='background:#eb3b5a'>object detection 은 깔끔한(neat) FCN 이 아니다</mark>.
	* Anchor box prediction 이 존재하기 때문
* 만약 이러한 <mark style='background:#3867d6'>anchor box 를 제외</mark>시키고 <mark style='background:#3867d6'>fully convolutional 하게 task 를 진행</mark>한다면 <mark style='background:#3867d6'>더 간단하며, 다른 task 에도 유연하게 적용이 가능</mark>해진다.

* 여러 논문에서 object detection 을 위해 FCN 기반 framework 를 활용하려는 시도(DenseBox)가 있었다.
* 즉 feature map 의 공간 위에서 4차원 벡터($l,t,r,b$) 와 $class$ 를 예측하려 시도한다.
![[Pasted image 20240229201411.png]]
* 위와 같은 방법론(e.g. Dense Box)는 bbox 를 다루기 위해 training data 를 고정된 크기로 crop, resize 를 해야 한다.
* 또한 여러 사이즈의 이미지를 처리해야 하기에 <mark style='background:#eb3b5a'>Image Pyramid 구조를 차용하는데 이는 FCN의 철학에 어긋난다</mark>.(FCN 은 모든 convolution 연산을 한번에 계산해야 한다고 말함)

* figure 1의 오른쪽에서 볼 수 있듯 중첩이 상당히 일어난 bounding box는 겹치는 영역의 픽셀에 대해 regress 할 bounnding box를 결정하기 어렵다는 문제가 있다.   
  
* 이 논문에서는 이러한 문제를 자세히 다루고, FPN을 사용하여 이러한 어려운 문제를 해결할 수 있음을 보여주며, 결과적으로 Anchor 기반 검출기와 유사한 정확도를 얻을 수 있다는 것을 보여준다. 
* 또한 **이러한 방법이 대상 객체 중심에서 멀리 떨어진 위치에서 low-quality predicted bounding box를 생성할 수 있다는 것을 발견하였고, 이러한 문제를 해결하기 위해 해당 bounding box의 중심에 대한 픽셀의 편차(deviation)를 예측하는 “centerness” 개념을 소개한다.** 

* 정리하자면, FCOS 의 장점은 아래와 같다.  
	1.  객체 검출 문제는 semantic segmentation 등과 같이 FCN 프레임워크를 사용할 수 있게 되므로, <mark style='background:#3867d6'>다양한 작업들과 통합하여 재사용 될 수 있다</mark>.   
	2.  <mark style='background:#3867d6'>Proposal Free 및 Anchor Free 라서 파라미터의 수가 크게 줄어든다</mark>. 이러한 파라미터들은 heuristic tuning 이 필요하며, 좋은 성능을 달성하기 위해서 많은 trick이 필요하다. 또한 Anchor Free 기반 검출기는 학습이 비교적 단순해집니다!   
	3.  Anchor box를 제거함으로써 <mark style='background:#3867d6'>IoU 계산과 같은 Anchor box와 관련된 복잡한 계산들을 더 이상 하지 않아도 되고</mark>, 학습 중 <mark style='background:#3867d6'>Anchor box와 GT-box 간의 매칭을 할 필요가 없으므로 더 빠른 학습과 테스트를 수행</mark> 할 수 있습니다.   
	4.  FCOS는 SOTA를 달성합니다. 또한 <mark style='background:#3867d6'>이 FCOS는 two-stage 검출기에서 RPN(Region Proposal Networks)로 사용 될 수 있으며</mark>, <mark style='background:#3867d6'>Anchor 기반 RPN 방법 보다 훨씬 더 나은 성능을 보장</mark>합니다.   
	5.  제안된 검출기는 <mark style='background:#3867d6'>instance segmentation 및 keypoint detection을 포함하여 최소한의 수정으로 다른 vision 관련 task들을 해결 할 수 있도록 확장이 가능</mark>합니다. 그래서 FCOS는 many instance-wise prediction 문제에 대한 새로운 baseline이 될 것 이다.

# 2. Related Work
### Anchor-based Detectors
* Anchor 기반 검출기는 Fast R-CNN과 같은 전통적인 ==sliding-window 및 proposal 기반 검출기의 아이디어에서 시작==한다. 
* Anchor box는 bounding box 위치 예측을 개선시키기 위해 extra offsets regression과 함께 positive 또는 negative patch로 분류되는 pre-defined sliding windows 또는 proposal으로 정의 된다. 
* 따라서 이러한 검출기의 Anchor box는 training sample로 볼 수 있으며, 각 sliding window/proposal의 image feature를 반복적으로 계산하는 Fast RCNN과 같은 검출기와는 달리 ==Anchor box는 CNN의 feature map을 사용하고 반복적인 feature 계산을 하지 않고 검출 프로세스 속도를 크게 높이게 된다==.
* Anchor box를 설계하는 것은 Faster R-CNN의 RPN, SDD, Yolo v2에 의해 대중화되었으며 검출 모델의 baseline이 되어왔다.

* 하지만 위에서 언급한 바와 같이 Anchor box 는 지나치게 많은 hyper-parameter 를 생성하며, 일반적으로 우수한 성능을 내기 위해서는 미세하게 이를 조정해야 한다.
* Anchor shape을 설명하는 하이퍼 파라미터 외에도 Anchor 기반 검출기는 Anchor box에 positive, ignored 또는 negative sample로 레이블을 지정하는 하이퍼 파라미터도 필요요하다. 
* 이전에는 Anchor box의 레이블을 결정하기 위해 Anchor box와 GT-box 사이의 IoU를 사용하게 된다. 
* <mark style='background:#eb3b5a'>이러한 파라미터는 최종 정확도에 큰 영향을 미치게 되며 heuristic tuning이 필요하게 됩니다</mark>.   
**Anchor-free Detectors**   
* 가장 널리 알려진 Anchor-free 검출기는 YOLO v1 일 수 있다. 
* Anchor box를 사용하는 대신 YOLO v1은 객체 중심 근처 지점에서 bounding box를 예측하게 된다. 더 높은 품질의 검출을 생성할 수 있는 것으로 간주되기 때문에 중앙에서 가까운 지점만을 사용하게 된다. 
* 그러나 이는 중심 근처의 지점만 bounding box를 예측하는데 사용되기 때문에 YOLO v1은 YOLO v2 에서 언급한 것 처럼 low recall을 보인다. 
* 결과적으로 <mark style='background:#eb3b5a'>YOLO v2는 Anchor box를 사용</mark>한다. 
* YOLO v1과 비교하여 <mark style='background:#3867d6'>FCOS는 GT box의 모든 지점을 이용하여 bounding box를 예측하고</mark>, <mark style='background:#3867d6'>low-quality의 bounding box는 제안된 “centerness”로 처리</mark> 된다. 
* <mark style='background:#3867d6'>결과적으로 FCOS는 Anchor 기반 검출기와 비슷한 recall을 제공</mark>한다.
  
* <mark style='background:#eb3b5a'>CornerNet</mark>은 bounding box의 pair of corners를 검출하고 그룹화 하여 객체를 검출하는 SOTA Anchor-free 검출기이다. 이는 <mark style='background:#eb3b5a'>동일한 인스턴스에 속하는 pair of corners를 그룹화 하기 위해 훨씬 더 복잡한 post-processing을 거치게 된다</mark>. 
* 또한 <mark style='background:#eb3b5a'>DenseBox</mark> 라는 것도 있다. 이는 <mark style='background:#eb3b5a'>overlapping bounding box를 처리하기 어렵고 recall이 상대적으로 낮기 때문에 일반 객체 검출에 적합하지 않은 것</mark>으로 나타났습니다.





# 3. Our Approach
* FCOS는 <mark style='background:#3867d6'>per-pixel prediction 방식으로 Object Detection을 재구성</mark>하며, <mark style='background:#3867d6'>multi-level prediction을 사용하여 recall을 개선</mark>하고 <mark style='background:#3867d6'>overlapped bounding box로 인한 모호성을 해결하는 방법을 제안</mark>한다. 
* 마지막으로 <mark style='background:#3867d6'>low-quality detected bounding boxes를 억제</mark>하고 <mark style='background:#3867d6'>전체 성능을 크게 향상시키는 “centerness” 개념을 소개</mark>한다.

### 3.1. Fully Convolutional One-Stage Object Detector

*  $F_i \in \mathbb{R}^{\mathbb{H} \times \mathbb{W} \times \mathbb{C}}$  
	* backbone CNN의 layer $i$의 feature map
* $s$ 를 layer 까지의 stride 라고 정의
* 입력 이미지에 대한 ground-truth bounding boxes는 $\{B_i\}$ 라고 정의
	* $B_i = (x_0^{(i)}, y_0^{(i)}, x_1^{(i)}, y_1^{(i)}, c^{(i)}) \in \mathbb{R}^4 \times \{1, 2, \ldots, C\}$
	* $(x_0^{(i)}, y_0^{(i)})$  , $(x_1^{(i)}, y_1^{(i)})$ 는 bouding box 의 left-top, right-bottom 의 corner 좌표
	* $c^i$ : bounding box 에 속하는 객체의 class 를 뜻한다.
	* $C$ : class 의 갯수(MS-COCO dataset 에서는 80)

* feature map $F_i$ 의 $(x,y)$ 위치의 좌표는 실제 이미지에서 다음과 같은 식을 따른다.
$$
\left( \left\lfloor \frac{s}{2} \right\rfloor + xs, \left\lfloor \frac{s}{2} \right\rfloor + ys \right)
$$
![[Pasted image 20240229213421.png]]
- 예를 들어 어떤 layer에서의 좌표가 (23, 30) 이고 stride가 8이 었다면 실제 이미지에서는 (4 + 23 * 8, 4 + 30 * 8) = (188, 244)가 된다. 
- 각 좌표에서 $\lfloor\frac{s}{2}\rfloor$ 는 stride 연산을 통해 발생하는 오차를 stride의 반 만큼만 더해줘서 보상을 해주는 역할
- 이러한 방법을 통하여 기존의 <mark style='background:#3867d6'>anchor box를 통하여 추정 하지 않고</mark> <mark style='background:#3867d6'>직접적으로 각 위치를 추정하게 된다</mark>. 즉, 각각의 좌표 위치(coordinate -> $(x,y)$)를 학습해야 할 대상으로 바라보게 된다. 이는 anchor 기반의 detector와는 차이점을 보인다.

$$
\begin{align}
t^* = (l^*, t^*, r^*, b^*) \\
\end{align}
$$


* 위의 식을 이용하여 prediction 과 GT 간의 오차를 $t^*$ 벡터로 구할 수 있다.
* 벡터의 각 값은 아래와 같다.
	* $l^* = x - x_0^{(i)}$ : 왼쪽 방향의 벡터 
	* $t^* = y - y_0^{(i)}$ : 위쪽 방향의 벡터
	* $r^* = x_1^{(i)} - x$ : 오른쪽 방향의 벡터
	* $b^* = y_1^{(i)} - y$ : 아래쪽 방향의 벡터
![[Pasted image 20240229221133.png|400]]
* 만약 한 점 $(x,y)$ 가 여러 개의 bounding box 에 속하게 된다면 영역이 가장 작은 bounding box 를 선택한다.
* 이와 같은 방법을 채택하는 이유는 뒤에서 다룰 Multi-level prediciton 과 관련이 있으므로 아래에서 자세하게 다룰 예정이며 위의 방법을 채택한다 하더라도 성능에는 거의 영향을 주지 않는다.

- FCOS 에서는 $l, t, r, b$를 추정하면서 <mark style='background:#3867d6'>가능한한 많이 foreground에 대하여 regressor를 학습하려고 한다</mark>. 
- <mark style='background:#eb3b5a'>anchor 기반의 모델에서는 GT bounding box와 anchor box의 IoU가 충분히 높은 경우에만 foreground인 positive sample로 학습하는 것</mark>과 차이점이 있다. 
- 이 점이 <mark style='background:#3867d6'>FCOS가 anchor 기반의 모델에 비해 높은 성능을 내는 이유 중의 하나로 설명</mark>한다.

- FCOS에서 사용하는 **네트워크의 마지막 layer**는 **클래스 갯수 (MS COCO의 경우 80개)의 차원을 가지는 벡터와 4차원인 $t^*=(l^*,t^*,r^*,b^*)$ 을 예측**한다. 
- feature map의 모든 픽셀에 대하여 해당 픽셀이 속한 클래스 $c_i$와 bounding box를 추측하기 위한 4개의 값 $t^*=(l^*,t^*,r^*,b^*)$를 추정하도록 네트워크를 구성한다. 
- 따라서 **한 픽셀 당 $(l,t,r,b,c)$ 5개의 값을 예측한다.
- 이 때, feature map 상의 픽셀 $(x,y)$ 가 GT bounding box(class = $c^*$) 내부에 있는 픽셀이라면 target 의 class 는 $c^*$ 로 정의하고 positive sample 로 간주한다.
- 반면 픽셀 $(x,y)$ 가 bouding box 에 속하지 않는 픽셀이라면 $c^*=0$ 으로 정의하여 negative sample 로 간주한다. 

#### Network Outputs
 * 학습 시에는 multi-class classifier 가 아닌 클래스 갯수 C 에 대한 ==binary classifier== 를 사용한다.
 * 아래 그림의 $(x, y)$ 좌표에서 뽑아낸 1x1xC 블록에서 <mark style='background:#3867d6'>각 채널(각 클래스)별로 그 클래스에 속하는지 안속하는지를 binary classification 을 진행</mark>한다.
 * 즉, input image 상의 모든 좌표에 대해 1x1xC 에 대한 binary classification 을 진행한다.
![[Pasted image 20240301104149.png]]

---
> 참고

* 원래 FCN 의 per-pixel prediction 은 위의 그림의 1x1xC 블록에서 각 클래스를 선택할 확률을 softmax 로 측정한다. 
* 즉 클래스가 23개이면 1x1xC 의 모든 element 의 합이 1이 되도록 학습을 진행한다.
![[Pasted image 20240301104747.png]]
---
![[Pasted image 20240301232706.png]]
* 또한 4개의 convolutional layer 를 backbone 으로 하는 모델(위의 빨간 박스)에서 추출된 feature map 에 각각 추가하여 classification 과 regression 을 진행한다.
* regression 진행 시 regression 의 target 은 항상 positive 로 분류된다.
	* target $t^*=(l^*,t^*,r^*,b^*)$
* 따라서 regression branch 의 마지막 output $x$ 를 $exp(x)$ 로 대체한다.
* 이를 통해 regression 의 값의 범위를 $(0 \sim \infty)$ 로 mapping 하는 효과를 볼 수 있다.

* 위와 같은 기법등을 통해  FCOS는 location 당 9개의 <mark style='background:#eb3b5a'>Anchor box가 있는 Anchor 기반 검출기</mark> 보다 <mark style='background:#3867d6'>신경망 출력 변수가 9배 더 적은 장점을 가진다</mark>. 

#### Loss Function
$$
L\left(\left\{ p_{x,y} \right\}, \left\{ t_{x,y} \right\}\right) = \frac{1}{N_{\text{pos}}} \sum_{x,y} L_{\text{cls}} \left(P_{x,y}, c^*_{x,y}\right) + \frac{\lambda}{N_{\text{pos}}} \sum_{x,y} \mathbb{1}_{\{c^*_{x,y}>0\}} L_{\text{reg}}\left(t_{x,y}, t^*_{x,y}\right),
$$
* $L_{cls}$ : classification loss
	* Focal Loss 를 사용한다. ([REF](https://gaussian37.github.io/dl-concept-focal_loss/),아래 블록 참고)
* $L_{reg}$ : regression loss
	* IoU Loss 를 사용한다.
* $N_{pos}$ : Normalize Term
	* positive sample 의 갯수이다.
* $\lambda$  : regression loss weight. 보통은 1.
* $\mathbb{1}$ : 
	* class index 가 0 보다 클 경우(if $c^*_i>0$) 는 1 -> positive sample 일 경우
	* class index 가 0 보다 작을 경우($otherwise$) 는 0 -> negetive sample 일 경우
		* 이 경우 regression loss 가 0 이 되어 연산이 안됨
* 인퍼런스를 할 때에는 이미지를 네트워크에 feedforward한 후에 각 feature map인 $F_i$의 각 $(x, y)$ 위치에 대하여 classification score인 $p_{x,y}$ 와 regression prediction 인 $t_{x,y}$ 를 얻는다.
* $p_{x,y}>0.05$ 일 경우 positive sample 로 간주 위에서 구한 $l^*, t^*, r^*, b^*$ 의 식의 역과정으로 bounding box coordinate 인 $\hat{x}_l,\hat{y}_l,\hat{x}_r,\hat{y}_r$ 을 구한다.(아래 수식 참고)
$$
\begin{align}
\hat{x}_l = x - l^*\\
\hat{y}_l = y - t^*\\
\hat{x}_r = x + r^*\\
\hat{y}_r = y + b^*\\
\end{align}
$$
---
> 참고 : 이 부분은 Focal Loss 에 대해 "간단히"만 설명할 것이다. 추후 필요에 의해 다른 포스트에서 정리를 할지 말지는 미정이다.

* One-Stage Object Detection(e.g. YOLO, SSD 등) 에서 negative sample(object 가 없는 경우)이 positive sample(object 가 있는 경우)에 비해 지나치게 많다.
* 즉, 클래스간 불균형이 문제가 된다.
* 이러한 task에  Cross Entropy Loss 를 사용할 경우 negative sample 에 큰 수치를 부여하게 된다.
![[Pasted image 20240301112744.png]]

* 위와 같은 문제를 해결하기 위해 Balanced Cross Entropy 가 나왔다.
![[Pasted image 20240301133541.png]]
* 각 클래스별 entropy 에 가중치 $w_t$ 를 부여하여 클래스간 불균형을 해소하고자 하는 방법이다.

* 하지만 위의 Balanced Cross Entropy 도 easy / hard negative 를 구분하지 못한다.
* 이에 Focal Loss 가 등장한다.
![[Pasted image 20240301113017.png|300]]
1.  잘못 분류되어 $p_t$ 가 작아지게 되면 $(1-p_t)^\gamma$도 1에 가까워지고 $\log(p_t)$ 또한 커져서 Loss에 반영된다.
2.  $p_t$ 가 1에 가까워지면 $(1-p_t)^\gamma$은 0에 가까워지고 Cross Entropy Loss와 동일하게 $\log(p_t)$ 값 또한 줄어들게 된다.
3.  $(1-p_t)^\gamma$ 에서 $\gamma$ 를 `focusing parameter` 라고 하며 **Easy Example에 대한 Loss의 비중을 낮추는 역할**을 한다.
---

### 3.2. Multi-level Prediction with FPN for FCOS

FCOS 는 FPN + multi-level prediction 으로 2가지 문제를 해결할 수 있다.
1. stride 가 크면 recall 이 낮아질 수 있다.
	* stride 가 클 경우 낮은 BPR(Best Possible Recall) 을 얻을 수 있다.
	* 이는 <mark style='background:#eb3b5a'>stride 가 클 경우 작은 물체들은 feature map 에서 그 정보가 과도하게 생략될 수 있어 찾기가 어렵다</mark>.
	* 이로 인해 recall 이 낮아지게 된다.
	* 이 때 Anchor 기반의 Detector 는 IoU threhold 를 낮추어 recall 을 조정할 수 있다.
	* 하지만 FCOS 는 large stride 로 인하여 발생한 low recall 문제를 다른 방식으로 해결한다.
	* <mark style='background:#3867d6'>FPN(Feature Pyramid Network)를 도입</mark>하여 <mark style='background:#3867d6'>작은 물체는 큰 크기의 feature map 에서</mark>, <mark style='background:#3867d6'>큰 물체는 작은 크기의 feature map 에서 추출</mark>한다.(아래 그림의 빨간 박스 부분)
	* 위와 같은 방법으로 FCOS 는 Anchor 기반의 모델인 Retina 보다 더 좋은 BPR 을 얻을 수 있었다.
![[Pasted image 20240301135924.png]]

2. GT box 끼리 겹칠 때 모호성 발생
	* 아래 그림처럼 한 픽셀이 다른 2개(혹은 그 이상)의 객체에 overlap 되어 있을 때, 어떤 bouding box 와 class 를 대상으로 할지 "모호" 하다.
	* 이러한 모호성 문제는 FPN 구조를 이용하여 개선할 수 있다. 
	* 이는 위에서 언급했던 것과 마찬가지로 객체의 크기가 상이하게 다르면 다른 level의  feature map 에서 객체를 검색하기 때문이다.
![[Pasted image 20240301140714.png|400]]

* 기존의 anchor 기반의 detector 와는 다르게 각 level 의 feature 에서 bouding box 를 regression 할 때, 범위를 제한한다.
* 예를 들어 bounding box의 중심점에서 각 bounding box 까지의 좌/상/우/하 까지의 거리를 $l^*, t^*, r^*, b^*$ 라고 하면 각 $i$ 번째 feature 에서의 거리 범위는 아래와 같다.
$$
m_{i-1} \leq \max(l^*, t^*, r^*, b^*) \leq m_i
$$
* $\{m_2,m_3,m_4,m_5,m_6\}$ = $\{0,64,128,256,\infty\}$ 
* 위의 식을 벗어나면 negative sample 로 간주.
* 위의 식을 이용하면 작은 object 는 큰 feature map 에서 찾고 큰 object 는 작은 feature map 에서 찾을 수 있다.
* 만약 overlap 된 객체가 위의 수식을 이용하였는데도 여러 GT bbox 가 할당이 된다면 ==면적이 가장 작은 Gt box 를 사용하는 것으로 모호함을 해결==하였다.

* 마지막으로 더 좋은 성능을 위하여 마지막 head 부분(아래 그림의 빨간 박스) 의 Parameter 를 공유하는 것이 좋다.
* 하지만 다른 크기의 범위를 regression 하기 위해 동일한 feature head 를 쓰지는 않는다.
![[Pasted image 20240301143654.png]]

* 추가로 네트워크에서 추정한 $(l,r,t,b)$ 를 사용하지 않고 개선된 출력인 $(l^*,r^*,t^*,b^*)$ 를 사용한다.
* 이는 $exp(x)$ 에서 learnable parameter $s_i$ 를 적용하여 $exp(s_ix)$ 를 계산하는 것을 의미한다.
* 이 $s_i$ 는 일종의 scale parameter 로 $l,r,t,b$ 의 값이 지나치게 커지는 것을 방지하여 모델에 약간의 개선이 이루어진다.

### 3.3. Center-ness for FCOS
FCOS에서 multi-level prediction을 사용한 후에도 FCOS와 Anchor 기반 검출기 사이에는 여전히 성능 차이가 있다.
그 이유는 <mark style='background:#eb3b5a'>객체의 중심에서 멀리 떨어진 위치에 의해 생성된 low-quality predicted bounding box 들 때문이라고 생각</mark>했기에 <mark style='background:#3867d6'>하이퍼 파라미터를 도입하지 않고 문제를 해결 할 수 있는 “centerness” 개념을 제안</mark>한다.

* <mark style='background:#eb3b5a'>물체의 중앙에서 멀리 떨어진 위치의 bounding box 의 예측값의 품질이 좋지 않아 오인식 되는 경향이 있다</mark>.

$$
\text{centerness}^* = \sqrt{\frac{\min(l^*, r^*)}{\max(l^*, r^*)} \times \frac{\min(t^*, b^*)}{\max(t^*, b^*)}}.
$$
* 중심점과의 거리를 정규화하여 ==중심점에 가까울수록 1== ==중심점에 멀수록 0==을 부여. 
* $centerness^*$ 의 값은 0과 1사이
* binary cross entropy 로 학습
* 이렇게 학습한 $centerness^*$ 를 classification score 출력층에 곱해주면 <mark style='background:#3867d6'>마지막 layer 의 NMS 에서 객체의 중앙과 멀리 떨어져 위치를 추정한 box 는 걸러지게 된다</mark>. 
* 이를 위한 branch 를 따로 분기하여 $centerness^*$ 를 예측한다.(아래 빨간 박스)
![[Pasted image 20240301145930.png]]
* sqrt 로 값의 scale 조정. 
	* 이는 sqrt 하기 전의 값은 0~1 사이의 값인데 그 값을 바로 적용하게 되면 classification score 를 너무 줄이는 효과를 낸다.
	* 이러한 효과를 방지하기 위해 sqrt 로 값을 키운다.
![[Pasted image 20240301150428.png]]
(f: 기존의 값, g: sqrt 적용값)
* centerness 를 위한 binary cross entropy loss 는 앞서 정의한 loss 에서 추가로 더해진다.

