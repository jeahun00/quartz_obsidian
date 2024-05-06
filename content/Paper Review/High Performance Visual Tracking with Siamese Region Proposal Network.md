---
dg-publish: "true"
tags:
  - Object_Tracking
  - One_Shot_Learning
  - Computer_Vision
  - Deep_Learning
---

* [correlation filter vs. convolution filter](https://velog.io/@syiee/Computer-vision-Linear-Filters)
* [한글논문리뷰 1](https://sonny-daily-story.tistory.com/27)
* [한글논문리뷰 2](https://velog.io/@kimkj38/%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-SiamRPN-High-Performance-Visual-Tracking-with-Siamese-Region-Proposal-Network)
# Abstract
* 대부분의 <span style='color:#eb3b5a'>기존 object detector 는 실시간으로 동작하지 못한다</span>.
* 따라서 이 논문에서는 대규모 image pair 를 이용한 <mark style='background:#2d98da'>Siamese-RPN</mark>(Region Proposal Network) 을 제시한다.
* Siamese-RPN 은 아래 구조를 따른다.
	1. feature extraction을 위한 Siamese subnetwork
	2. classification branch와 regression branch를 포함하는 region proposal subnetwork
* inference time 에서는 local one-shot detection 작업으로 구성

# 1. Introduction

* 기존의 object tracking 의 방법론은 크게 2가지로 나뉜다.
	1. **correlation filter base**
		* fourier domain 에서 학습(domain specific information 을 활용함)
		* 이 방식은 정확도를 향상시키기 위해 deep feature 를 사용한다.
		* 이로 인하여<span style='color:#2d98da'> 정확도는 높지만</span>,
		* <span style='color:#eb3b5a'>속도가 매우 느리다</span>.
	2. **deep learned feature base** 
		* 깊은 deep learning net 을 사용하여 학습된 feature 를 이용.
		* 한 번의 training 이후에 추가적인 연산없이 inference 진행(domain specific information 사용 X)
		* <span style='color:#eb3b5a'>정확도는 비교적 낮지만</span>(correlation base method에 비해),
		* 속도는 <span style='color:#2d98da'>실시간에 쓸 수 있을 정도로 빠르다</span>.

* 이 논문에서는 **잘 설계된 deep learning tacker** 가 최신 <span style='color:#2d98da'>correlation base tracker 와 비슷한 성능</span>과 <span style='color:#2d98da'>빠른 inference speed</span> 를 가질 수 있다고 함.

* Siamese-RPN 은 아래 2가지로 구성
	1. template branch
	2. detection branch
* faster R-CNN 에 영감을 받아 correlation feature map 에서 RPN 을 수행
* 표준 RPN 과는 달리 위의 두 branch 의 correlation feature map 으로 region proposal 진행
* 또한 이 모델은 one-shot detector 이다.
	* 이 논문에서 제시한 방법은 online trakning task 에서 one-shot learning을 처음으로 적용

---
# 참고: siamese network for object tracking
(이 부분은 본 논문이 input 과 task 에 대한 설명이 빈약하기에 추가적으로 적어둠)
(또한 이 부분은 추후에 글을 분리할 예정)
### Siamese network 란
* **siamses neural network** 는 **one-shot, few-shot learning** 을 위한 기법 중 하나
* siamese neural network 이 적용된 대표적인 논문에는 [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) 가 있다.
![[Pasted image 20240418104118.png|500]]
* [이 논문](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)은 학습 데이터로부터 1쌍의 image pair 가 같은지 다른지를 학습한다.
	* 위 사진에서 vertification tasks 에서 코끼리와 축구공 끼리 비교
* 이후 **학습데이터에 존재하지 않던 새로운 class**가 들어오더라도 **해당 데이터 쌍이 같은지 다른지를 판단**할 수 있게 해 준다(one-shot learning).
	* 농구공-북극곰은 다른 객체, 농구공-농구공은 같은 객체임을 인식 가능

### Siamese network for object tracking
* 정리하지만 ~~<mark style='background:#fa8231'>siamese network 의 주요 기능은 input data 쌍이 같은지 다른지를 판단할 수 있게 해 준다</mark>~~.
* 이러한 원리를 이용하여 object tracking task 에서 **"$T-1$ 시간에서 감지된 객체를 T 시간에서 찾을 수 있을 것이다"** 라는 아이디어를 적용한다.
* 이러한 아이디어를 적용한 논문이 SiamsesMask 이다.
![[Pasted image 20240418105446.png]]
### Fully convolutional 
### [Fully-convolutional siamese networks for object tracking](https://link.springer.com/chapter/10.1007/978-3-319-48881-3_56)


* Reference
	* [siamese network for object tracking 1](https://blog.hbsmith.io/siamesenetwork%EA%B3%BC-%EA%B0%9D%EC%B2%B4-%EC%B6%94%EC%A0%81-1-34882f533857)
	* [siamese network for object tracking 2](https://blog.hbsmith.io/siamesenetwork%EA%B3%BC-%EA%B0%9D%EC%B2%B4-%EC%B6%94%EC%A0%81-2-b0ace08e6232)

---

# 3. Siamese-RPN framework
![[Pasted image 20240417204237.png]]

* 위의 그림과 같이 Siamese-RPN frame work 는 2개의 network 로 이루어져 있음
	1. Siamese subetwork
	2. Region Proposal subnetwork
		* RPN subnet 은 2개의 branch 로 나뉨
		1. classification branch
			* foreground-background classification 에 이용됨
		2. regression branch
			* proposal refinement 에 이용됨


## 3.1. Siamese feature extraction subnetwork
$$
h(L_{k\tau}x)=L_{\tau}h(x)
$$
* $L_\tau$: translation operator 
	* $(L_{\tau}x)[u]=x[u-\tau]$
	* stride $k$ 를 가진 fully convolution 을 맞추기 위해 모든 padding 을 제거
![[Pasted image 20240417212423.png|400]]
* Siamese feature extraction subnetwork 는 두개의 branch 로 구성됨(위 그림 참고)
	1. template branch
		* target patch 를 이전 frame 으로부터 받아옴($z$ 로 표기)
		* 추출된 feature : $\varphi(z)$  
	2. detection branch($x$ 로 표기)
		* target patch 를 현재 frame 으로부터 받아옴($x$ 로 표기)
		* 추출된 feature : $\varphi(x)$
* 위의 2개의 CNN network 는 parameter 를 공유함
![[Pasted image 20240417213258.png|400]]