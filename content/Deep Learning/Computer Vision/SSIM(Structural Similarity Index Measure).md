#Deep_Learning 

PSRN 과 다르게 수치적인 에러가 아닌 인간의 시각적 화질 차이를 평가하기 위해 고안된 방법
## 1. 기본개념
SSIM 은 3가지 지표로 이미지의 품질을 평가한다.

1. Luminance : 두 이미지의 평균 밝기를 비교하여 밝기의 유사성을 측정
2. Constrast : 두 이미지의 표준 편차를 비교하여 대비의 유사성을 평가
3. Structure :두 이미지의 구조적 패턴을 비교(이는 이미지의 텍스처와 같은 세밀한 패턴을 평가하는 데 중요한 역할)
 
$$
\begin{align*}
SSIM(x,y)=[l(x,y)]^{\alpha} \cdot [c(x,y)]^{\beta} \cdot [s(x,y)]^{\gamma}
\end{align*}
$$

* 위의 수식에서 $l$ 이 Luminance, $c$ 가 Constrast, $s$ 가 Structure 이다.
* 각 수식에 대해 자세히 알아본다

---
#### 1.1. Luminance
* 각 픽셀의 이미지의 픽셀값으로 연산
 $$
\mu_{x}=\frac{1}{N} \sum_{i=1}^{N}x_i
$$

* $x_i$ : 각 픽셀의 값(밝기를 의미한다)
* $N$ : 전체 픽셀의 갯수
* $\mu_x$ : 이미지의 평균 luminance

* 두 이미지 $x,y$ 간의 luminance 에 대한 식은 아래와 같다
$$
\begin{align*}
l(x,y)=\frac{2\mu_x\mu_y+C_1}{\mu_x^2+\mu_y^2+C_1}\\
\end{align*}
$$
* $\mu_{x}, \mu_{y}$ 의 값이 같다면 1 이된다.
* $\mu_{x}, \mu_{y}$ 의 값의 차이가 크면 클수록 0에 가까워진다.
* $C_1$ 에 해당하는 값은 zero devision 을 예방하기 위한 값이다.
> cf) $C_{1}=(K_1L)^2$
> $K_1$ : 일반 상수 값이며 보통 0.01 을 주로 사용한다.
> $L$ : 픽셀값의 범위. 8bit 일 경우 255

---
#### 1.2. Constrast
* 이미지 내의 빛의 밝기가 바뀌는 정도를 나타내는 양. 
* 픽셀 간의 분포를 통해 정량화 하고 표준편차를 이용.
$$
\sigma_{x}=(\frac{1}{N-1}\sum_{i=1}^{N}(x_{i}-\mu_x)^2)^{1/2}
$$
> N 이 아닌 N-1 로 나눈 이유는 [[면접준비-DL]]의 모분산, 표본분산에서 확인
* $\sigma_x$ : 이미지의 픽셀간의 표준편차(표본값($x_i$) - 평균값($\mu_x$))

* 두 이미지 $x, y$ 간의 contrast 에 대한 식은 아래와 같다

$$
\begin{align*}
c(x,y)=\frac{2\sigma_x\sigma_y+C_2}{\sigma_x^2+\sigma_y^2+C_2}\\
\end{align*}
$$
* 위의 Luminance 와 마찬가지로 
* $\sigma_{x}, \sigma_{y}$ 의 값이 같다면 1 이된다.
* $\sigma_{x}, \sigma_{y}$ 의 값의 차이가 크면 클수록 0에 가까워진다.
* $C_2$ 에 해당하는 값은 zero devision 을 예방하기 위한 값
> cf) $C_2=(K_2L)^2$
> $K_2$ : 일반 상수이며 주로 0.03 을 사용
> $L$ : 픽셀값의 범위. 8bit 의 경우 255

#### 1.3. Structure
* 픽셀값의 구조적인 차이점을 나타내며 정성적으로 성분을 확인 시, edge 를 나타냄
* 이 값은 luminance 의 평균, constrast 의 표준편차로 Nomalize 된 픽셀 값의 분포에서 픽셀값을 재정의

$$
\frac{(X-\mu_x)}{\sigma_x}
$$
* $X$ : input image
* 두 이미지의 structure 성분의 유사성 확인은 두 이미지 간의 `correlation` 을 이용하는 것과 같은 의미이다.
$$
\begin{align*}
\text{corr}(X, Y) &= \frac{\sigma_{xy}}{\sigma_x \sigma_y} \\
&= \frac{E[(x - \mu_x)(y - \mu_y)]}{\sigma_x \sigma_y} \\
&= E\left[ \frac{(x - \mu_x)(y - \mu_y)}{\sigma_x \sigma_y} \right] \\
&= E\left[ \frac{(x - \mu_x)}{\sigma_x} \cdot \frac{(y - \mu_y)}{\sigma_y} \right]
\end{align*}
$$
> correlation 이란?
> ![](_media-sync_resources/20240417T162534/20240417T162534_69579.png)
> REF : https://darkpgmr.tistory.com/185

* 두 이미지 $x, y$ 의 correlation 을 구한다는 것은 structure 성분의 곱의 평균을 구하는 것
* structure 성분이 같은 방향으로 커지면(두 이미지가 같으면) 1에 가까워짐. 다른면(두 이미지가 다르면) 0에 가까워짐
* 이러한 성질은 위의 luminance 와 contrast 의 개념과 동일
$$
\begin{align*}
s(x,y)=\frac{\sigma_{xy}+C_3}{\sigma_{x}\sigma_{y}+C_3}
\end{align*}
$$
* $C_3$ : 일반적으로 이 값은 ${C_2}/{2}$ 를 주로 사용한다. 이는 값 추후 수식 전개의 용이성 때문!
* $\alpha,\beta,\gamma=1\, C_3=C_2/2$  일 때
$$
\begin{align*}
\text{SSIM}(x, y) = l(x, y) \cdot c(x, y) \cdot s(x, y) \\
= \frac{2\mu_x\mu_y + C_1}{\mu_x^2 + \mu_y^2 + C_1} \cdot \frac{2\sigma_{xy} + C_2}{\sigma_x^2 + \sigma_y^2 + C_2} \cdot \frac{\sigma_{xy} + C_2/2}{\sigma_x \sigma_y + C_2/2}\\
= \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}
\end{align*}
$$
![](_media-sync_resources/20240417T162534/20240417T162534_79853.png)


정리하자면 
* SSIM 은 luminance, contrast, structure 3가지 요소를 활용하여 이미지를 평가하는 지표이다.
* PSNR 과 마찬가지로 SSIM 의 값이 높을수록 더 좋은 이미지임을 나타낸다.

## 2. SSIM 의 한계
PSNR, SSIM 은 픽셀간의 단순한 값으로 이미지를 판단한다
이로 인해 인간의 인지로 이상하다고 판단하는 이러한 개념을 반영하지 못한다.
이러한 복잡한 인간의 인지를 Nueral Network의 힘을 빌려 수치로 나타내고자 한 값이  [[LPIPS (Learned Perceptual Image Patch Similarity)]] 이다.