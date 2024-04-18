# Abstract

* 최근 point cloud 를 이용한 3d object detection 에서는 아래 2가지 방법이 주로 쓰인다.
	1. fixed encoder: <mark style='background:#3867d6'>빠르지만</mark> <mark style='background:#eb3b5a'>정확도가 낮다</mark>.
	2. encoder that learned from data : <mark style='background:#eb3b5a'>느리지만</mark> <mark style='background:#3867d6'>정확도가 높다</mark>.

* 이에 이 논문에서는 <mark style='background:#3867d6'>빠르면서도 정확도가 높은</mark> 새로운 방법인 **PointPillar** 를 제시한다.

# 1. Introduction

* 기존의 computer vision 은 DL method 로 엄청난 발전을 이루었다.
* 따라서 point cloud 에 이러한 DL method 를 적용하여 object detection 을 하고자 했다.
* 하지만 기존 dl 에서 이용하는 일반적인 이미지와 point cloud 는 상이하다.

|             | density | dimension |
|:-----------:|:-------:|:---------:|
|    image    |   ⬆️    |    2D     |
| point cloud |   ⬇️    |    3D     |

* 위의 차이점으로 인해 기존의 convolution 기반 DL method 를 바로 point cloud 에 적용하기는 어려움이 존재했다.

* 이에 최근 연구에서는 BEV(Bird Eye View) 를 이용하고자 했다.
* <mark style='background:#3867d6'>BEV 는 아래 장점을 가진다.</mark>
	1. object 의 scale 을 보존할 수 있다.
	2. BEV 를 convolution 연산을 하더라도 거리정보가 보존된다.
* <mark style='background:#eb3b5a'>하지만 BEV 는 아래 한계를 가진다.</mark>
	1. 극단적으로 sparse 하다
	2. conv 연산시 매우 비효율적이다.
* 위의 문제를 해결하기 위해 <mark style='background:#3867d6'>hand-crafted feature encoding 을 사용</mark>했다.
* <mark style='background:#eb3b5a'>하지만 이는 매우 소모적이다.</mark>

* BEV 의 이러한 문제를 해결하기 위해 <mark style='background:#3867d6'>VoxelNet 이 제시</mark>되었다.
	* VoxelNet 은 <mark style='background:#3867d6'>매우 좋은 성능</mark>을 보였지만
	* 4.4 Hz 로 <mark style='background:#eb3b5a'>매우 느린 inference time</mark> 을 보인다.

* 이러한 문제들을 해결하기 위해 <mark style='background:#3867d6'>PointPillars 를 제시</mark>한다.
* PointPillars 는 새로 제시하는 encoder 를 사용하여 point cloud 를 vertical pillar 로 치환하여 encoding 한다.
* <mark style='background:#3867d6'>PointPillars 는 아래 장점을 가진다.</mark>
	1. End-to-end 학습을 2D conv 연산만으로 가능하다.
	2. 고정된 encoder 를 사용하는 것이 아닌 feature 를 학습함으로써 point cloud 에 의해 생성된 전체 정보를 학습할 수 있다.
	3. Voxel 대신 pillar 를 사용하므로 수직방향의 분할을 수동으로 작업할 필요가 없다.
	4. 매우 빠르며 효율적이다(2D conv 연산만 하기에).
	5. 여러 다른 센서들에 대한 hand-tuning 이 필요 없다.

## 1.1. Related Works
### 1.1.1. Object detection using CNNs
* CNN 은 object detection 에 적용되기 좋은 구조이다.
* two-stage, one-stage detector 가 존재
* Focal Loss 가 중요하게 사용됨

### 1.1.2. Object detection in lidar point clouds
* point cloud 는 3d architecture 이다.
* 이러한 point cloud 에 3D conv 를 적용하는 것은 당연하다.
* 하지만 3D conv 는 너무 느리다.

* 따라서 이러한 문제를 해결하기 위해 point cloud 를 ground plane 이나 image plane 에 projection 하는 방법을 제안한다.
* 가장 일반적인 방법으로는 point cloud 를 voxel 로 치환하는 것이다.
* 최근의 연구의 흐름은 아래와 같다.
	* PointNet : point cloud 를 end-to-end 로 학습
	* VoxelNet : 
		* PointNet 을 이용하여 point cloud 로 object detection 을 수행
		* voxel 을 이용하며, 3D conv 를 사용하기에 느림
	* Frustum PointNet 
		* [핵심개념 이해 안됨]
		* point cloud 를 이용한 segmentation, classification 을 수행
	* SECOND : 
		* VoxelNet 에 비해 성능과 속도 모두 개선됨
		* 하지만 여전히 많은 연산을 차지하는 3D conv 연산을 제거하지 못함


# 2. PointPillars Network
![](Pasted%20image%2020240306152210.png)
* PointPillars 는 아래 3가지 구조를 가진다.(위 사진 참고)
1. **Feature encoder network** : point cloud 를 sparse pseudo-image 로 치환
2. **2D convolutional backbone** : pseudo-image 를 high-level representation 으로 치환
3. **Detection head** : class 와 3D box 예측

## 2.1. Pointcloud to Pseudo-Image
![](Pasted%20image%2020240306152210.png)
### 2.1.1. Pointcloud to Pillar
* 이 과정은 아래 사진의 빨간 박스를 처리하는 과정이다.
![](Pasted%20image%2020240307152434.png)

![](Pasted%20image%2020240307152253.png)
* Pointcloud 에서 discrete 하게 만들어진 grid X-Y plane 으로 사영을 내린다. 
* 위 과정에서 생성된 grid, 즉 point 들의 집합을 pillar 라고 한다.
* 이 때, point $l$ 의 좌표는 다음과 같다 : $l = (x,y,z)$
* 여기서 각 point 에 pillar 의 정보를 추가하기 위해 아래 데이터를 추가한다.
	* $r,x_c,y_c,z_c,x_p,y_p$ 
	* 즉, 각 point $l$ 을 새로운 point $\hat{l}$ 으로 치환한다.
	* 이렇게 새로 만들어진 $\hat{l}$ 을 decorated lidar point 라 한다.
	* $\hat{l}=\{x,y,z,r,x_c,y_c,z_c,x_p,y_p\}$
		* $r$ : 센서에 되돌아온 빛의 세기
			* LiDAR 센서는 빛을 쏘고 그 되돌아온 빛의 세기를 측정할 수 있다.
			* 반사되어 되돌아온 빛의 세기가 0에 가까우면 고무나 가죽, 혹은 검은 물체처럼 빛의 흡수율이 좋은 표면을 가진 물체이고
			* 반사되어 되돌아온 빛의 세기가 1에 가까우면 금속이나 거울, 하얀 물체처럼 빛의 반사율이 좋은 표면을 가진 물체임을 알 수 있다.
		* $x,y,z$ : point 의 원래 좌표
		* $x_c,y_c,z_c$ : ==각 필러 안의 point 의 산술평균==과 ==원래 좌표==간의 거리
		* $x_p,y_p$ : pillar 의 위치좌표

### 2.1.2. Stacked Pillar and generate Learned Feature
![](Pasted%20image%2020240307152704.png)
* 위의 과정에서 추출된 각 pillar 들을 DxNxP 의 `stacked pillar` 로 치환한다.
	* $D$ : dimension of pillar
	* $P$ : sample 당 비어 있지 않은 pillar 의 갯수
	* $N$ : pillar 당 point 의 갯수
		* 각 pillar 가 너무 많은 점을 가질 때(N 이 지나치게 클 때) : N 중 random sampling
		* 각 pillar 가 너무 적은 점을 가질 때(N 이 지나치게 작을 때) : zero padding 부여
* 이렇게 생성된 `Stacked Pillar` 를 아래 기법들을 사용하여 새로운 feature map 생성
	* Linear layer(1x1 conv 와 유사하게 동작)
	* Batch Norm
	* ReLU
* 이렇게 생성된 새로운 feature 는 CxPxN 의 크기를 갖는다.
* 위의 feature 를 channel 에 대해 max pooling 하여 CxP 크기로 압축한다.
* 위의 과정으로 생성된 CxP feature 를 `Learned Feature` 라고 한다.

### 2.1.3. Generate Pseudo Image
![](Pasted%20image%2020240307153542.png)
* 앞선 Learned Feature 에서 압축된 정보를 Stacked Pillar 에서 저장하고 있는 Pillar 의 위치정보인 $x_p, y_p$ 로 되돌린다.
* 이렇게 생성된 새로운 image 를 Pseudo Image 라고 한다.
* Pseudo image 는 HxWxC 의 차원을 가진다.

### 2.1.4. 정리
* 앞선 `2.1.1`~`2.1.3` 의 과정을 통해 <mark style='background:#3867d6'>3D convolution 없이 학습가능한 feature map 을 생성</mark>해 내었다.

## 2.2. Backbone
![](Pasted%20image%2020240307154121.png)
### 2.2.1. Top-down Network(위의 사진에서 빨간 박스)
* Conv 연산을 연속적으로 적용하여 여러 size 의 resolution 을 만들어 낸다.
* 위 과정에서 $(S,L,F)$ 이 학습에 쓰인다.
	* $S$ : stride 의 크기
	* $L$ : 3x3 conv 의 갯수(각 conv 는 BN과 ReLU사용)
	* $F$ : 출력 채널의 크기

### 2.2.2. Upsampling
* $Up(S_{in},S_{out},F)$ : Upsampling 과정
	* 이 과정에서 [Transposed Convolution](https://velog.io/@hayaseleu/Transposed-Convolutional-Layer%EC%9D%80-%EB%AC%B4%EC%97%87%EC%9D%B8%EA%B0%80) 을 사용(링크 참고)
	* $S_{in}$ : initial stride
	* $S_{out}$ : final stride
	* $F$ : 출력 채널의 크기

### 2.2.3. Concat
* `2.2.1`~`2.2.2` 과정에서 생성된 각 conv feature 를 concat 하는 과정
* 이 과정을 통해 여러 resolution 을 모두 살릴 수 있다.
* 위의 과정 덕분에 여러 scale 에 대한 detection 이 가능하다.

## 2.3. Detection Head
* Detection Head 는 [SSD](https://herbwood.tistory.com/15) 를 사용한다.(링크 참고)
* Object detection 이 아닌 다른 task 를 하고 싶다면 그 task 에 해당하는 model 을 적용하면 된다.

# 3. Implementation Detail

## 3.1. Network
- PointPillars의 모든 가중치는 균등 분포를 사용하여 (uniform distribution)임의로 초기화된다.
- Encoder network에서 $C = 64$ 다.
- Car 클래스를 탐지할 때와 cyclist, pedestrain을 탐지할 때 각각 다른 hyparameter를 사용한다.
- Car 탐지 땐 backbone에서 $S = 2$, cyclist, pedestrian을 탐지할 때는 $S = 1$이다.

- 두 network 모두 3개의 block을 생성한다. 
- 이때 block을 만들기 위해 사용하는 2D convolution 연산은 $Block1(S, 4, C)$, $Block2(2S,6, 2C)$, $Block3(4S, 6, 4C)$로 정의된다.

- 그리고 upsampling할 때 사용되는 transposed 2D convolution 연산은 각각 $Up1(S, S, 2C)$, $Up2(2S, S, 2C)$, $Up3(4S, S, 2C)$로 정의된다.
- 3개의 block은 모두 concatenated되어 6C개 feature를 갖는 tensor가 된다. 이 tensor는 detection head로 전달된다.

## 3.2. Loss
* GT box 와 Anchor box 는 아래의 파라미터를 가진다.
* $(x,y,z,w,l,h,\theta)$
	* $x,y,z$ : box 의 중심좌표
	* $w,l,h$ : box 의 가로,세로,높이
	* $\theta$ : box 의 z 축에 대한 회전 각도
![400](Pasted%20image%2020240307161518.png)
* GT box 와 Anchor Box 간의 차이는 아래와 같이 구한다.
$$
\begin{align*}\\
\Delta x = \frac{x^{\text{gt}} - x^a}{d^a}, \quad \Delta y = \frac{y^{\text{gt}} - y^a}{d^a}, \quad \Delta z = \frac{z^{\text{gt}} - z^a}{h^a} \\
\Delta w = \log \left( \frac{w^{\text{gt}}}{w^a} \right), \quad \Delta l = \log \left( \frac{l^{\text{gt}}}{l^a} \right), \quad \Delta h = \log \left( \frac{h^{\text{gt}}}{h^a} \right) \\
\Delta \theta = \sin \left( \theta^{\text{gt}} - \theta^a \right),
\end{align*}
$$
* $d_a=\sqrt{(w_a)^2+(l^a)^2}$ 

* 위의 수식을 통해 $\mathcal{L_{loc}}$ 을 구한다.
* $\mathcal{L_{loc}}$ 은 localization loss 이다.
$$\mathcal{L}_{\text{loc}} = \sum_{b \in (x,y,z,w,l,h,\theta)} \text{SmoothL1} (\Delta b)$$
* $\mathcal{L_{cls}}$ 은 classification loss 이다.
$$
\mathcal{L}_{\text{cls}} = -\alpha_a (1 - p^{\gamma}) \log p^a
$$
* Classification loss 에서는 Focal Loss 를 사용한다.