#Deep_Learning 

## Partial Convolution

* Image Inpainting 분야에서 쓰이는 기법 중 하나
* Image Inpainting 이란? : 
	영상 혹은 이미지에서 자신이 원치 않는 아이템을 삭제하고 그 부분을 자연스럽게 채워 넣는 분야
	
	![AltText|500](_media-sync_resources/20240417T162534/20240417T162534_61020.png)

---
## 1. 수식포함 설명

1. 기본 원리
	* Convolution 을 할 때 <mark style='background:#eb3b5a'>Masking Layer 와 1 Layer 의 Sum 의 비율</mark>로 <mark style='background:#eb3b5a'>0 으로 비어 있는 부분을 보강</mark>해준다.
2. Notation
	* $X$(Input Image):
		* 비어 있는 부분이 존재하는 input image
	* $M$(Masking Layer):
		* 그림처럼 Inpainting 을 진행해야 하는 부분은 0으로 나머지는 1로 채워져 있는 layer
	* $1$(1 elements Layer):
		* 모든 element 가 1로 채워진 layer
	* $\odot$  :
		* Convolution 연산

![600](_media-sync_resources/20240417T162534/20240417T162534_58582.png)

위의 수식에 대해 조금 더 자세히 알아보자

$$
\begin{align*}
W^T(X \odot M)\frac{sum(1)}{sum(M)}+b
\end{align*}
$$
$\frac{sum(1)}{sum(M)}$ : 1로 채워진 레이어의 elements 들의 합에서 hole 을 포함한 Masking 레이어의 elements 의 합으로 나눈 값. 
즉 이러한 값들을 곱해서 input image 의 hole 의 값을 일부 보강한다.

## 2. Padding 에의 적용

#### 2.1. 기본 개념 설명
* 위에서 partial convolution 은 Image Inpainting 의 기법중 1개였다.
* 이 부분을 Padding 을 처리하는 부분에 도입 
* 즉 <mark style='background:#eb3b5a'>Image Inpainting 의 Hole</mark> 과 일반적인 <mark style='background:#eb3b5a'>Convolution Layer 의 Padding</mark> 을 <mark style='background:#eb3b5a'>같은 개념</mark>으로 보자는 것!

![AltText](_media-sync_resources/20240417T162534/20240417T162534_48802.png)

자 이제 아래 이미지를 보자.
우리가 위에서 다루었던 수식과 크게 다르지 않다.
1. Image Inpainting 의 Partial Convolution
$$
\begin{align*}\\
x' &= W^T(X \odot M)\frac{sum(1)}{sum(M)}+b\\\
\end{align*}
$$
2. Partial Convolution Padding
$$
\begin{align*}\\
x'(i,j)=W^T(X^{p0}_{(i, j)} \odot 1^{p0}_{(i, j)})r_{(i, j)}+b=W^{T}X^{p0}_{(i, j)}r_{(i,j)}+b
\end{align*}
$$

![](_media-sync_resources/20240417T162534/20240417T162534_31156.png)


Partial Inpainting 에서는 Masking Layer 가 존재했다(Notation 에서 M)
> Masking Layer : Hole 이 존재하는 부분은 0 아닌 부분은 1

그런데 위의 Partial Convolution Padding 에서는 <mark style='background:#eb3b5a'>Padding 을 Hole 로 보기로 하였다</mark>.
즉 <mark style='background:#eb3b5a'>Padding 이 아닌 부분은 1 Padding 인 부분은 0 으로</mark> 채우는 것이다. 이러한 Layer 를 Masking Layer 로 볼 수 있으며 이를 $1^{p0}$ 로 표현하였다.

![](_media-sync_resources/20240417T162534/20240417T162534_53420.png)

위의 사진에서 볼 수 있듯 $r_{(i, j)}$ 부분은 Image Inpainting 의 $\frac{sum(1)}{sum(M)}$ 과 같다.

#### 2.2. Case of Big Padding
1. Different Image Sizes
	* input image 의 크기가 다 다른 경우가 있다. 이럴 경우 Padding Size 가 일관되지 않고 그 size 자체가 매우 커지는 경우가 있다.
	* 위와 같은 상황에서는 왜곡이 크게 발생할 수 밖에 없다.
2. Convolution at the border can be invalid
	* 위의 Case 와 같이 Padding 이 지나치게 커 지면Window Size < Padding 인 경우가 발생할 수 있다.
	* 이럴경우에도 큰 왜곡이 발생할 수 있다.
> 위의 2가지 Case 는 근본적으로 하나의 문제를 제시한다. 
> 지나치게 큰 Padding 처리에 관한 것이다. 이에 대한 해결책을 아래에서 제시한다.

![](_media-sync_resources/20240417T162534/20240417T162534_19194.png)

위의 수식에서 알 수 있듯 레이어를 쌓아나가는 과정에서 이전에 사용했던 마스크를 이용하여 새로운 마스크를 재정의한다. 이에 대한 수식이 위의 수식이다.

#### 2.3. Result

![AltText|500](_media-sync_resources/20240417T162534/20240417T162534_55498.png)


REF:
Guilin Liu, et al., "Image Inpainting for Irregular Holes Using Partial Convolutions", ECCV 2018
Liu, Guilin, et al. "Partial convolution based padding." _arXiv preprint arXiv:1811.11718_ (2018)
 



