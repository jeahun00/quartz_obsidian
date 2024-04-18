#Deep_Learning 
# PSNR 이란?

Super Resolution 분야에서 결과를 정량적인 수치로 측정하기 위해 사용되는 기법 중 하나이다.

### 1. 기본 개념
![](_media-sync_resources/20240417T162535/20240417T162535_77069.png)

1. 원본 사진(Real Value)과 Resolution 을 거친 사진(Predict Value)의 픽셀간의 차이를 MSE 로 구함
2. 위에서 구한 값을 픽셀이 가질 수 있는 최대값($MAX_i$)에서 나누고 그 비율을 log 를 취해줌
> log를 취하는 이유는?
> -> 인간의 인식은 선형적인 수치보다 로그수치 즉 비율의 변화에 더 민감하게 반응
> -> PSNR 의 목적이 이미지간의 차이를 수치적으로 보이기 위함
> -> 즉 log를 취함으로써 그 값을 직관적으로 이해하기 위함
3.  MSE 는 이미지간의 차이가 크다는것을 의미 -> 즉 MSE 가 작을수록 좋다
4. PSNR 은 MSE 와 반비례 -> 즉 좋은 이미지일수록 PSNR 이 크다

### 2. PSNR 의 한계
PSNR 은 인간의 인식에 대한 지표를 무시한다.
아래 사진을 보자
![](_media-sync_resources/20240417T162535/20240417T162535_00784.png)

우리는 직관적으로 오른쪽 사진이 더 화질이 좋다고 느낀다.
하지만 PSNR 수치는 오히려 왼쪽 사진이 더 높다.
이러한 현상이 나타나는 이유는 오른쪽 사진에 Blur 처리된 부분이 존재하기 때문이다.
이러한 단점을 보안하기 위해 [[SSIM(Structural Similarity Index Measure)]] 을 병행하여 사용하는 경우가 많다.

---

REF: https://m.blog.naver.com/mincheol9166/221771426327

![[Pasted image 20231217164034.png]]