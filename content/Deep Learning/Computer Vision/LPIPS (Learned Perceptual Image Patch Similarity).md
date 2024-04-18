#Deep_Learning 

* 2개의 image 의 유사도를 평가하기 위해 사용되는 지표 중 하나이다.
* PSNR 의 인간의 인식을 수치로 표현하지 못하는 문제를 SSIM 이 해결하였다.
* 하지만 이 SSIM 역시 인간의 인식, 즉, 사진의 Context 를 고려하지는 못한다.
* 이러한 문제를 해결하기 위해 사용되는 지표가 LPIPS 이다.

## Introduce

그렇다면 인간의 인식을 어떻게 평가지표에 녹여낼 수 있을까?
아래 논문에서 새로운 방법을 제시한다.
[The Unreasonable Effectiveness of Deep Features as a Perceptual Metric(2018)](_media-sync_resources/20240417T162536/20240417T162536_46696.pdf)

이 논문에서 high-level image classification 으로 학습된 conv network 가 representaion space 에서 매우 유용하다는 것을 보여준다.

즉 pre trained cnn base model 의 feature map 을 이미지의 유사도를 평가하는 지표로 이용한다.

![](_media-sync_resources/20240417T162536/20240417T162536_12756.png)


$$
\begin{align*}\\
d(\mathbf{x}, \mathbf{x}_0) = \sum_l \frac{1}{H_l W_l} \sum_{h,w} || \mathbf{w}_l \odot (\hat{\mathbf{y}}^{l}_{hw} - \hat{\mathbf{y}}^{l}_{0hw}) ||^2_2
\end{align*}
$$
$L$ : layer
$h,w$ : 해당 layer 의 width, height
$y,y_0$ : unit-nomarlize feature vector
	2개의 image 를 각각 pre train 된 cnn model 에 입력하여 뽑아낸 feature map
$d$ : distance functions

위의 수식을 살펴보면 모든 layer 의 $y,y_0$ 에 대해 채널 단위로 L2 Loss 를 구한 값을 더한 것.
위의 값을 이미지에 대한 유사도로 본 것!

REF  : 
paper : [The Unreasonable Effectiveness of Deep Features as a Perceptual Metric(2018)](_media-sync_resources/20240417T162536/20240417T162536_64162.pdf)
review blog : https://xoft.tistory.com/4