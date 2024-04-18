---
tistoryBlogName: jeahun10717
tistoryTitle: "[Math] Information, Entropy, Cross-Entropy, KL-Divergence, Mutual-Information 정리"
tistoryVisibility: "3"
tistoryCategory: "1218439"
tistorySkipModal: true
tistoryPostId: "79"
tistoryPostUrl: https://jeahun10717.tistory.com/79
---
#Math #Deep_Learning 
## 1.  Information(정보량) : 주어진 임의의 이벤트에서 발생하는 놀라움(?)의 정도
$$I(E) = -log[Pr[E]] = -log(P)\tag{1}$$
* *$E$ : stochastic event
* *$Pr[E], P$ : Probability of Event E

 <mark style='background:#eb3b5a'>example</mark> : 주사위를 던질 때 6이 나올 확률이 90%라 하자. 나머지 1~5까지 나올 확률은 2.5%라 하자. 이 때 각 확률이 나올 확률은 아래와 같다.
$$
\begin{align*}
P(E(X = 1)) = 0.02 \quad P(E(X = 2)) = 0.02 \quad P(E(X = 3)) = 0.02\\
P(E(X = 4)) = 0.02 \quad P(E(X = 5)) = 0.02 \quad P(E(X = 6)) = 0.90

\end{align*}
$$
이 때 주사위를 굴릴 때의 정보량은 아래와 같다.

$$
\begin{align*} I(E(X = 1)) &= -\log(0.02) \approx 5.64 \\ I(E(X = 2)) &= -\log(0.02) \approx 5.64 \\ I(E(X = 3)) &= -\log(0.02) \approx 5.64 \\ I(E(X = 4)) &= -\log(0.02) \approx 5.64 \\ I(E(X = 5)) &= -\log(0.02) \approx 5.64 \\ I(E(X = 6)) &= -\log(0.90) \approx 0.15 \end{align*}
$$
이 때 $X=6$ 이 나오는 것보다 $X=1,2,3,4,5$ 가 나오는 것이 더 놀랍다. 이러한 정보를 나타내는 수치가 information 이다.

---

## 2. Entropy

$$
H(X) = -\sum_X P(X)log_2(P(X)) = \sum_X P(X)*(-log_2(P(X)))\tag{2}
$$
* *$P(X)$ : $X$ 가 일어날 확률
* *$(-log_2(P(X)))$ : $X$ 에서의 information 

위의 식에서도 알 수 있듯, <span style='color:#eb3b5a'>Entropy</span>는 <span style='color:#eb3b5a'>Information</span>의 기대값이다. 
(참고 : Machine Learning의 관점에서 Entropy는 class간의 불균형을 수치로 나타낸 것이다.)
예시를 들어서 생각해보자.

<mark style='background:#eb3b5a'>example</mark> : 
1. 동전을 던질 때 앞면과 뒷면의 확률이 같을 경우 : $P(X = H) = 0.5, P(X = T) = 0.5$
$$
\begin{align*}
H(X) &= P(X = H) \times -log(P(X = H)) + P(X = T) \times -log(P(X = T)) = 1
\end{align*}
$$
2. 동전을 던질 때 앞면과 뒷면의 확률이 다를 경우 : $P(X = H) = 0.25, P(X = T) = 0.75$ 
$$
\begin{align*}
H(X) &= P(X = H) \times -log(P(X = H)) + P(X = T) \times -log(P(X = T)) = 0.81\
\end{align*}
$$
위의 1번예와 2번예의 차이는 뭘까?
2번예는 1번예에 비해 정보량이 균일하지 않다. 2번 예 같은 경우 한쪽의 확률이 높으므로 <span style='color:#eb3b5a'>예측하기가 쉽다</span>고도 할 수 있다. 또한 수치적으로는 2번예는 1번예에 비해 Entropy라고 정의한 H의 값이 더 작다. 즉 <span style='color:#eb3b5a'>2번예는 1번예에 비해 Entropy가 더 작다</span>.  
이를 인지하고 아래 그래프를 보자.
![](_media-sync_resources/20240417T174109/20240417T174109_74990.png)


위의 그래프에서 H 는 Entropy 이다. 이 H값이 작을수록 중앙값에 더 몰려있는 것을 알 수 있다.
즉 엔트로피가 클수록 확률분포가 더 균일하다고 할 수 있다.

여기서 다시 Entropy 를 정리해 보자.

<mark style='background:#eb3b5a'>Entropy란</mark> : 
* 확률분포 $P(X)$에서 일어날 수 있는 모든 사건들의 정보량의 기대값으로 $P(X)$의 불확실 정도를 평가한 값
* 정보이론에서의 Entropy : 정보를 표현하는데 필요한 최소평균자원량
$$
\begin{align*}\\
H_{p} &= -\sum_{x}P(X)log_2P(X)\tag{3}\\
discrete : H_{p} &= -\sum_{x}P(X)log_2P(X)\tag{4}\\
continuous : H_{p} &= -\int P(X)log_2P(X)dX\tag{5}
\end{align*}
$$
* Machine Learning 에서 Entropy를 사용할 때 밑을 2가 아닌 e를 사용하는데 이는 미분과 적분이 용이하기 때문이다
* Entropy는 확률분포 P(X)가 `uniform distribution`일 때 최대화된다.
* Entropy는 확률분포 P(X)가 `delta function`일 때 `uniform distribution`일 때 최대화된다.

---

## 3. Cross Entropy

$$
\begin{align*}
discrete : H(P, Q) &= -\sum_{X} P(X)logQ(X) = \sum_{X} P(X)\frac{1}{logQ(X)}\tag{6}\\
continuous : H(P, Q) &= -\int P(X)logQ(X)dX = \int P(X)\frac{1}{logQ(X)}dX\tag{7}\\
\end{align*}
$$

(아래의 용어설명은 machine learning에서 쓰이는 용어로 기술하였다.)
$P(X)$ : real value distribution(실제 값의 분포)
$Q(X)$ : prediction value distribution(예측 값의 분포)

일반적으로 $H(P, Q) \geq H(P)$ 이다.

즉 예측된 값 $Q(X)$가 $P(X)$에 가까워질수록 $H(P, Q)$는 $H(P)$에 가까워진다.
반대로 말하면 예측값이 $Q(X)$가 $P(X)$에 멀어지면 엔트로피가 커지는 경향이 존재한다고 볼 수 있다.

Cross Entropy 는 예측값과 실제값의 차이를 수치로 나타낸다고 설명했다.
즉 예측값의 N개의 라벨과 실제값의 N개의 라벨의 분포의 차이를 나타내는 것이다.
그렇다면 이를 이용하여 우리는 2개의 라벨만 존재하는 Binary Classification 문제를 해결할 수 있다.
Binary Classification 이란 2개의 레이블(주로 0 또는 1)로 분류를 해야하는 Task 이다.
위의 Task 에서 Loss Function(Cost Function) 으로 주로 쓰이는 것이 Binary Cross Entropy 이다.

<mark style='background:#eb3b5a'>Binary Cross Entropy</mark>
$$
\begin{align*}\\
discrete : H(P, Q) &= -\sum_{X} P(X)logQ(X) = \sum_{X} P(X)\frac{1}{logQ(X)}\tag{6}\\
H(y) &= -\sum_{i = 1}^{N} y_{i}log{\hat{y_i}}\\
H(y - 1) &= -\sum_{i = 1}^{N} (1 - y_{i})log{(1-\hat{y_i})}\\
Cost(y,\hat{y}) &= H(y)+H(y - 1)\\
&= \sum_{i = 1}^{N} -y_{i}log{\hat{y_i}}-(1 - y_{i})log{(1-\hat{y_i})}
\end{align*}
$$

위의 식에서 극단적인 예를 들면 명확한 이해에 도움이 된다.
* 만약 실제값이 1이고  예측값이 0일 때는 아래와 같다.
$$
\begin{align*}\\
Cost(1,0) =& -y_{i}log{\hat{y_i}}-(1 - y_{i})log{(1-\hat{y_i})}\\
&=-1log0-0log1\\
&=\infty
\end{align*}
$$
* 만약 실제값이 0이고  예측값이 1일 때는 아래와 같다.
$$
\begin{align*}\\
Cost(0,1) =& -y_{i}log{\hat{y_i}}-(1 - y_{i})log{(1-\hat{y_i})}\\
&=-0log1-1log0\\
&=\infty
\end{align*}
$$
즉 위의 상황처럼 아예 예측을 반대로 했을 경우(Error 가 최대인 경우) 는 무한으로 발산한다.

---

## 4. KL-Divergence(Kullback-Leibler Divergence)

$$
\begin{align*}
discrete : D_{KL}(P||Q) &= H(P,Q) - H(P)\\
&= -\sum_{X} P(X)logQ(X) -(- \sum_{X} P(X)logP(X))\\
&= \sum_{X} P(X)log\frac{P(X)}{Q(X)} \tag{8}\\\\
continuous : D_{KL}(P||Q) &= H(P,Q) - H(P)\\
&= -\int P(X)logQ(X)dX - (-\int P(X)logP(X)dX)\\\\
&= \int P(X)log\frac{P(X)}{Q(X)}dX \tag{9}\\
\end{align*}
$$


Cross Entropy $H(P, Q)$는 예측값인 $Q(X)$의 값이 $P(X)$와 멀어질수록 Entropy가 커지는 경향을 가진다.
KL-Divergence는 위에 그 커진 값이 얼마나 커졌는지 수식으로 나타낸다. 

위의 수식(8)에서  $H(P, Q)$는 cross entropy, $H(P)$는 $P$에 대한 entropy 이다. 수식(8)은 이 두 수식의 차를 구했으므로 일종의 거리함수로 볼 수 있다.

---
## 5. Mutual Information

임의의 두 확률변수 $X, Y$가 <mark style='background:#eb3b5a'>독립</mark>이라면 <mark style='background:#eb3b5a'>Joint Distribution</mark>은 확률의 곱으로 표현할 수 있다.
$$
P(X, Y) = P(X)P(Y) \tag{10}
$$
만약 $X, Y$가 독립이 아니라면 <mark style='background:#eb3b5a'>확률 곱</mark>과 <mark style='background:#eb3b5a'>Joint Distribution간의 차이</mark>를 KL-Divergence로 측정이 가능하다.
$$
\begin{align*}
I(X;Y) &= \sum_{y\in Y} \sum_{x\in X} p(x, y) log(\frac{p(x, y)}{p(x)p(y)})\tag{11}\\
I(X;Y) &= D_{KL}(p(x,y)||p(x)p(y))\tag{12}
\end{align*}
$$
<mark style='background:#eb3b5a'>KL-Divergence</mark> 가 <mark style='background:#eb3b5a'>두 분포간의 entropy의 차이</mark>를 구했다면 <mark style='background:#fa8231'>Mutual Information</mark>은 <mark style='background:#fa8231'>두 분포가 얼마나 독립적</mark>인지를 구한다.

두 분포가 완전히 independent 하다면 0(아래 수식 참조), dependent 하면 할수록 값이 커진다(Divergence 가 크다).

$$
\begin{align*}
I(X;Y) &= \sum_{y\in Y} \sum_{x\in X} p(x, y) log(\frac{p(x, y)}{p(x)p(y)})\tag{11}\\
&= \sum_{y\in Y} \sum_{x\in X} p(x, y) log(\frac{p(x)p(y)}{p(x)p(y)})\\
&= \sum_{y\in Y} \sum_{x\in X} p(x, y) log(1)\\
&=0 
\end{align*}
$$

REF : 
* https://gaussian37.github.io/ml-concept-basic_information_theory/


| Feature              | MoCo                                                       | SimCLR                             |
|----------------------|------------------------------------------------------------|------------------------------------|
| Key Storage          | Running queue of keys (negative samples)                   | Within current mini-batch          |
| Encoder              | Separate query and key encoders with momentum update for the key encoder | Single encoder for all samples     |
| Gradient Update      | Only through queries                                       | Across all samples in a mini-batch |
| Negative Sample Size | Large number due to a decoupled queue mechanism            | Limited to current mini-batch size |
