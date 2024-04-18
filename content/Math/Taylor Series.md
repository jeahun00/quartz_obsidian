#Math 

## 1. Taylor Series 와 그 수학적 의미

Taylor Series 란 given function $f(x)$ 를 다항함수로 근사시키는 기법이다.

$$
\begin{align*}\\
f(x) &\approx p_{n}(x) \\
p_{n}(x) &= f(a) + f'(a)(x-a) + \frac{f''(a)}{2!} (x-a)^2 + \cdots + \frac{f^{(n)}(a)}{n!} (x-a)^n \\
&= \sum_{k=0}^{n} \frac{f^{(k)}(a)}{k!} (x-a)^k
\end{align*}
$$

위의 수식에서 $n$ 이 커질수록 그 값이 원래의 함수 $f(x)$ 에 근사한다.

이러한 Taylor Series 의 진가는 미분이나 적분이 어려운 함수들에서 드러난다.

$$
\int_{0}^{1} \sin(x^2) \, dx = \int_{0}^{1} \left( x^2 - \frac{x^6}{3!} + \frac{x^{10}}{5!} - \cdots \right) \, dx
$$
원 함수인 $sin(x^{2})$ 은 적분이 어려운 함수이다.
이러한 함수를 다항함수로 근사시켜 연산을 쉽게 할 수 있다.

## 2. Taylor Series Expantion with ML

ML 의 Optimization 의 한 기법 중 Gradient Descent 가 있다.
이 기법은 1차 미분을 사용한다.

![](_media-sync_resources/20240417T174110/20240417T174110_97672.png)

그리고 2차미분을 이용하는 방법이 있다.

![](_media-sync_resources/20240417T174110/20240417T174110_76596.png)

위의 식은 Taylor Series 로 유도된다.
![](_media-sync_resources/20240417T174110/20240417T174110_84923.png)

REF : https://darkpgmr.tistory.com/149