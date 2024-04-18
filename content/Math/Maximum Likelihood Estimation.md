#Math 

우선 MLE 를 이해하기 이전에 Likelihood 와 Probability 의 차이를 이해해야 한다.

## 1. Likelihood vs Probabilty
$$
\begin{align*}
Probabilty : P(data|distribution)\\
Likelihood : L(distribution|data)
\end{align*}
$$
![](_media-sync_resources/20240417T174111/20240417T174111_62134.png)

* Probability : 주어진 **확률분포(probability distribution)가 고정**된 상태에서, 관측되는 **사건이 변화**될 때
* Likelihood : 관측된 **사건이 고정**되어 있을 때, **확률분포(probability distribution)가 변화**할 때
* 즉, 움직이는 값(변수) 가 probability 와 likelihood 의 가장 큰 차이점이다.

|      | Probability             | likelihood              |
| ---- | ----------------------- | ----------------------- |
| 변수 | data(or event, or case) | distribution            |
| 고정 | distribution            | data(or event, or case) |
자세한 예시와 설명은 [이 링크](https://xoft.tistory.com/30) 참고

---
## 2. MLE(Maximum Likelihood Estimation) 

MLE 란 위와 같이 특정 데이터들이 주어졌을 때 그 데이터들에 <mark style='background:#eb3b5a'>가장</mark> 적합한 distribution 을 찾는 것이다.
(즉 distribution 을 움직여 가며 데이터들에 가장 적합한 위치를 찾는 것)

![AltText|400](_media-sync_resources/20240417T174111/20240417T174111_96425.png)

* 위의 사진에서 빨간 점들을 포함할 수 있는 distribution 을 연산한다고 가정해 보자.
* 이들의 likelihood 를 전부 곱했을 때 최대가 되는 분포를 찾으면 된다.

![](_media-sync_resources/20240417T174111/20240417T174111_58015.png)

이러한 MLE 를 구하기 위해서는 계산의 편의성을 위해 log-likelihood function 을 미분하여 그 최댓값을 찾는다.
$$
\begin{align*}
\frac{\partial}{\partial \theta} \mathcal{L}(\theta | x) = \frac{\partial}{\partial \theta} \log P(x | \theta) = \sum_{i=1}^{n} \frac{\partial}{\partial \theta} \log P(x_i | \theta) = 0

\end{align*}
$$

MLE 는 그 특성상 Prior 가 배제되어 있다. 우리가 강력하게 알 수 있는 Prior(사전확률) 이 존재한다면 [[Maximum A Posterior]] 를 사용할 수 있다.

[MLE 관련 링크](https://xoft.tistory.com/30)