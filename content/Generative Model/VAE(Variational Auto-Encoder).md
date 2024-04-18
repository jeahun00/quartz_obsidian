# VAE 란?

* Input Image X 를 설명할 수 있는 feature 를 추출
* 추출된 해당 feature 를 Latent Vector z 에 담음
* 이 Latent Vector z 를 통해 X 와 유사하지만 완전히 새로운 데이터를 생성하는 모델
* 이 때 각 feature 는 gaussian distribution 을 따른다고 가정
* Latent z 는 feature 의 평균($\mu$) 과 분산($\sigma^2$) 을 따름

![AlignText](img_store/Pasted%20image%2020240112135706.png)

* $p(z)$ : latent vector $z$ 의 확률밀도함수(probability density function). gaussian distribution 을 따름
* $p(x|z)$ : 주어진 $z$ 에서 특정 $x$ 가 나올 조건부 확률에 대한 확률밀도함수
* $\theta$ : 모델의 파라미터

---

# VAE 의 구조

![](img_store/Pasted%20image%2020240112150558.png)

* input image X 를 Encoder 에 통과시켜 Latent vector z 를 구하고
* Latent z 를 다시 Decoder 에 통과시켜 
* 기존 input image X 와 비슷한 특징을 공유하지만 새로운 image 를 생성하는 작업

* 들어오는 input image 에 대해 다양한 feature 들이 각각의 확률 변수가 되는 확률 분포를 생성
* 이러한 확률분포들 중 input image 의 feature 와 유사한 확률 분포를 생성
* 이러한 확률 분포를 이용하여 input image 와 유사한 image 를 새롭게 만들어냄
![[img_store/Pasted image 20240112172315.png]]

---

# AE vs VAE
### AE
![[img_store/Pasted image 20240112173530.png]]
* AE 의 목적은 위의 그림과 같이 feature extraction 이다.
* 반면 VAE 의 목적은 AE 에서 뽑아진 feature 들과 유사한 이미지를 새로 생성해 내는 것이다.

### VAE
![[img_store/Pasted image 20240112173712.png]]
* 위의 사진과 같이 Encoding 으로 평균, 분산의 vector 를 구하고 
* 그 값으로부터 sample latent vector 를 추출한다.
* 이 sample latent vector 와 유사한 이미지를 만들어 내는 것이 VAE 의 목적이다.

---

# VAE 의 학습방법과 수식적 이해

![[img_store/Pasted image 20240112173908.png]]
* 위의 사진처럼 $p_\theta(x)$ 를 최대화 하는 것이 목적이다.
* 여기서 Maximum Likelihood Estimation 을 사용한다.
	(MLE 와 관련된 내용은 아래 링크 참고)
	[[Maximum Likelihood Estimation]]

* 즉 VAE 의 궁극적인 목표는 x 의 likelihood 를 maximize 하는 distribution 을 만드는 것이다.

$$
p_{\theta}(x) = \int p_{\theta}(z)p_{\theta}(x|z)dz
$$
* 위의 수식은 Bayes 정리와 Marginalisation 을 통해 도출된 수식이다.

> **Marginalisation?**
> 1. y 가 주어졌을 때 x 의 확률을 표준화
> 2. x 이외의 y 를 조건으로 하였을 때 영향을 줄 수 있는 모든 z 를 결합 확률을 구해 적분한 것
> (아래 사진 참고)

![[img_store/Pasted image 20240112190050.png]]

* 다시 아래 수식으로 돌아오자
$$
p_{\theta}(x) = \int_{z} p_{\theta}(z)p_{\theta}(x|z)dz
$$
* 이 수식에서 우리는 $p_\theta(x)$ 를 직접적으로 구할 수 없다.
* why?
	* 위의 수식은 z 에 대한 단일한 integral 이 아닌
	* possible 한 모든 z 에 대한 integral 이기 때문 
* $p_\theta(x|z)$ 를 bayes rule 로 다시 정리(posterior 를 추출하기 위함)해 보아도 결국 우리가 구해야 하는 $p_\theta(x)$ term 을 구할 수 없다.
**Bayes Rule**
$$
p_\theta(z|x)=\frac{p_\theta(x|z)p_\theta(z)}{p_\theta(x)}
$$


![[img_store/Pasted image 20240112192118.png]]
* 이러한 문제를 해결하기 위해 VAE 는 $p_\theta(z|x)$ 에 해당하는 값이 Encoder 임을 주목한다.
* 이 Encoder 값을 네트워크를 통해 근사시킨 값 $q_\phi(z|x)$ 로 $p_\theta(z|x)$ 를 대체한다.

* 여기서 $p_{\theta}(x)$ 의 최대값을 구하기 위한 mathematical 한 trick 이 들어간다. I
![[img_store/Pasted image 20240112193539.png]]

* 제일 밑에 식의 3가지 Term 에 대해 알아보자
$$
\mathbb{E}_z \left[ \log p_{\theta}(x^{(i)} | z) \right] - D_{KL}(q_{\phi}(z | x^{(i)}) || p_{\theta}(z)) + D_{KL}(q_{\phi}(z | x^{(i)}) || p_{\theta}(z | x^{(i)}))
$$
1. $\mathbb{E}_z \left[ \log p_{\theta}(x^{(i)} | z) \right]$
	1. $q_\theta(z|x)$ 에서 x 를 입력으로 받아 z 를 sampling 한다
	2. sampling 된 z 를 통해 $p_{\theta}(x^{(i)} | z)$  라는 distribution 을 만들어낸다.
	3. 즉 $p_{\theta}(x^{(i)} | z)$ 는 sampling 된 값이므로 연산 가능하다
	4. 따라서 위의 Term 역시 연산이 가능하다
2. $D_{KL}(q_{\phi}(z | x^{(i)}) || p_{\theta}(z))$
	1. $q_{\phi}(z | x^{(i)})$ 는 network 를 통해 연산 가능
	2. $p_\theta(z)$ 는 단순한 gaussian distribution
	3. 따라서 이 2개의 distribution 간 KL-Divergence 역시 연산 가능
3. $D_{KL}(q_{\phi}(z | x^{(i)}) || p_{\theta}(z | x^{(i)}))$
	1. $p_{\theta}(z | x^{(i)})$ 이 값이 계산 불가.
	2. 따라서 전체 Term 도 계산 불가
	3. 하지만 이 Term 은 KL-Divergence 특성상 0보다 큰 값
	4. 궁극적인 우리의 목적은 $\log p_\theta(x^{i})$ 를 최대화 하는 것
	5. 그렇다면 이 Term 을 noise 로 취급하고 무시해도 됨

* 이제 우리가 신경써야할 2개의 Term 을 아래와 같이 표기한다.
$$
\begin{align*}\\
\mathcal{L}(x^{(i)}, \theta, \phi) &= 
\mathbb{E}_z \left[ \log p_{\theta}(x^{(i)} | z) \right] - D_{KL}(q_{\phi}(z | x^{(i)}) || p_{\theta}(z))
\end{align*}
$$

* 위 수식의 $\mathcal{L}$ 은 Tractable Lower Bound 이다.
* 이 $\mathcal{L}$ 를 최대화 하는 것이 VAE 의 목적이다
* 위와 같은 $\mathcal{L}$ 을 ELBO(Evidence of Lower BOund) 라고 한다.







REF : 
* VAE : https://di-bigdata-study.tistory.com/4
* Marginalisation : https://blog.naver.com/sw4r/220988265784
* Marginal Likelihood : https://m.blog.naver.com/sw4r/221380395720