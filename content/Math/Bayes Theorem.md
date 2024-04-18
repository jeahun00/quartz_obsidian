#Math 

베이즈 정리는 두 확률변수의 사전확률(Prior)과 사후확률(Posterior) 사이의 관계를 나타내는 정리이다.

사전확률의 정보로 사후확률을 추론할 수 있다는 것이 핵심이다.

$$
\begin{align*}
P(A|B)=\frac{P(B|A)P(A)}{P(B)}
\end{align*}
$$

![](_media-sync_resources/20240417T162604/20240417T162604_04791.png)

Posterior(사후확률) : $P(A|B)$, $B$ 가 관측되었을 때 A 의 확률을 나타낸다.
Likelihood(우도) : $P(A|B)$, 해당 관측치 $B$ 가 나올 확률을 나타낸다.
Prior(사전확률) : $P(A)$, 관측을 하기 전 이미 알고 있는 확률을 나타낸다.

* ML 의 대표적 문제인 classification 을 예로 들어보자.
* B를 input data, A를 class 의 집합이라고 가정하자.
* 어떤 클래스 $A_i(i=1,2,3...n)$ 에 속할 확률 $P(A_i|B)$, 즉 posterior 를 계산하는 문제로 볼 수 있다.
* Classification 문제에서는 이러한 $P(A_1|B),P(A_2|B),P(A_3|B)...P(A_n|B)$ 중 가장 큰 값을 구하는 과정으로 볼 수 있다.

이러한 개념은 확률론에서 큰 지류 중 하나인 Bayesianism 이다.
이 정리는 많은 분야에서 매우 중요하게 다뤄지는 문제이므로 추후에도 계속 업데이트를 할 예정이다.

이후에는 [[Maximum Likelihood Estimation]] 에 대해 알아볼 것이다.
이 개념은 추후에 나올 평가지표중 하나인 mAP 에 주요하게 다뤄지는 개념이므로 정리를 할 것이다.