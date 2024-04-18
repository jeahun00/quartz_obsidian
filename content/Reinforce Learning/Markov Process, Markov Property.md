---
tistoryBlogName: jeahun10717
tistoryTitle: Markov Process, Markov Property
tistoryVisibility: "3"
tistoryCategory: "1054367"
tistorySkipModal: true
tistoryPostId: "81"
tistoryPostUrl: https://jeahun10717.tistory.com/81
---
 #ReinforceLearning #Math
# Markov Process(MP, Markov Chain)

* Probability Process(확률과정) : 
	* 시간이 진행함에 따라 상태(state)가 확률적으로 변하는 과정
	* 어떠한 확률 분포(probability distribution)를 따르는 확률변수(random variable)이 이산적(discrete)인 시간흐름(time interval)마다 값을 생성하는 것

* Markov Process : 
	* <mark style='background:#eb3b5a'>time interval 이 discrete</mark> 하고, <mark style='background:#eb3b5a'>현재의 state 가 이전 state 에만 영향을 받는</mark> Probability Process

# Markov Property

* 어떠한 시간에 특정 state 에 도달하든 그 이전에 어떠한 state 를 거쳐왔든 다음 state 로 갈 확률은 항상 같다는 성질. 
* t 시점의 state 에서 t+1 시점의 state 로 갈 확률과 t, t-1, t-2, t-3 ... 시점을 거친 state 의 확률은 같다는 성질.
* 즉, 시간 0 ~ 시간 t-1 까지의 정보는 이미 t 가 포함하고 있다는 뜻. 
* 이러한 성질로 인해 memoryless property 라고도 부름
$$
\Pr(S_{t+1} = s' | S_0, S_1, ..., S_{t-1}, S_t) = \Pr(S_{t+1} = s' | S_t)
$$

Example
![](img_store/Pasted%20image%2020240106141436.png)

* 위의 사진은 어떠한 학생의 일과에 대한 Markov chain 이다.
* 이러한 Markov Chain 은 아래와 같은 성질을 가진다.
	* 한 state 에서 다른 state 로 가는 확률의 합은 1이다.
	* terminal state 를 가진다. 이 terminal state 로 이행이 되면 더 이상 state 의 이동이 발생하지 않는다.
	* 무한한 process 를 거치면 결국 이 terminal state 로 이행이 되고 이를 수렴으로 볼 수 있으며 이렇게 terminal state 로 수렴하는 것을 stationary distribution 이라고 한다.

# State Transition Probability Matrix

* 위의 예시에서 살펴본 바와 같이 <mark style='background:#eb3b5a'>state 들 간의 이동</mark>을 <mark style='background:#eb3b5a'>전이(transition)</mark> 이라고 한다.
* 이를 확률로 나타나고 이를 <mark style='background:#eb3b5a'>state transition probability</mark> 라고 한다.
* 
$$
P_{ss'} = Pr(S_{t+1}=s'|S_t=s)
$$

* <mark style='background:#eb3b5a'>각 state 들 간의 state transition probability</mark> 를 <mark style='background:#eb3b5a'>matrix</mark> 로 나타낸 것이<mark style='background:#eb3b5a'> State Transition Probability Matrix </mark>이다.

![](img_store/Pasted%20image%2020240106142239.png)

이러한 Markov Process 에서 Reward 라는 개념을 포함한 것이 [[Markov Reward Process(MRP)]] 이다.

