#ReinforceLearning #Math
# Markov Reward Process

### Reward
* 앞서 살펴본 Markov Process 는 <mark style='background:#eb3b5a'>특정 state 에서 다른 state 로 transition 할 확률만 존재</mark>
* state 들 간의 전이의 가치를 평가하는 수치가 없다.
* 이러한 단점을 보완하기 위해 <mark style='background:#eb3b5a'>state 들 간의 전이에 대한 가치까지 부여</mark>한 게 <mark style='background:#eb3b5a'>Markov Reward Process</mark> 이다.

![](img_store/Pasted%20image%2020240106142912.png)

* 이제 특정 state 에서 다른 state 로 전이할 때 reward 값을 포함하여 기대값을 계산할 수 있다.
* 이렇게 state 와 reward 를 받아 기대값으로 mapping 하는 함수 $R_{s}$를 아래와 같이 표현할 수 있다.

$$
R_s=E[r_{t+1}|S_t=s]
$$

* 위의 수식은 t 에서 t+1 로 시간이 흐를 때 얻을 수 있는 reward 에 대한 식이다.
* t+1 에서 바로 reward 를 받았기 때문에 이러한 reward 를 **Immediate Reward** 라고 한다.

### Return (include "discounting factor")

현재의 시점 t 에서 미래의 특정 시점에 대한 total reward 를 고려해 보자.

$$
G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$
* $G_t$ : 시점 $t$ 에서 시작해 시간이 진행됨에 미래에 받을 수 있는 Reward(보상,기대값) 의 총합
* $R_{t+1}$ : $t$ 에서 $t+1$ 로 전이할 때의 Immediate Reward
* $\gamma$(discounting factor) : 미래의 정보에 대한 penalty. 0~1 사이의 값
	* 왜 이런 penalty 를 부여할까?
	* 1. 미래의 보상은 현재의 보상에 비해 그 확률이 더 낮으므로 일부 손해를 부여
	* 2. 만약 이러한 값이 없으면 t 가 무한히 커지게 되면 $G_t$ 의 값이 지나치게 커지는 문제가 발생. 이는 강화학습에 적용할 경우 학습 자체가 불가능해짐.

### Value Function of MRP

* 특정 시나리오의 기대값은 G_t 이다.
* 가능한 시나리오들을 sampling 하고 그 시나리오들의 기대값이 Value Function 이다.
$$
V(s) = E[G_t|S_t=s]
$$
![](img_store/Pasted%20image%2020240106152525.png)
위의 예시에 대한 Value Function 에 대한 계산은 아래와 같다.
![](img_store/Pasted%20image%2020240106152551.png)

위와 같은 MRP 의 개념에서 action 과 그 action 에 대한 policy 를 추가한 개념이 [[Markov Decision Process(MDP)]] 이다.