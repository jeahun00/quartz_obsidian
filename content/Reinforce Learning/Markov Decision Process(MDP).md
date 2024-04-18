#ReinforceLearning #Math
# Markov Decision Process(MDP)

* MP 에서 Reward 의 개념을 추가한 것이 MRP
* MRP 에서 action 개념을 추가한 것이 MDP 이다.

### Action
* 기존에 state 의 변화는 transition probability 에 따라 임의로 변경이 되었다.
* 강화학습에서 State 의 변화는 Agent 가 action 이라는 기능을 수행함으로써 변하게 된다 
![AlignText|400](img_store/Pasted%20image%2020240106153041.png)
* 즉 MP, MRP 는 value 를 Environment 의 관점에서 매겼다면,
* MDP 는 Agent 의 관점에서 Value 를 매기게 됨
![](img_store/Pasted%20image%2020240106153247.png)


### Policy

* 기존의 MP, MRP 는 state 끼리 직접 연결되는 구조.
* MDP 에서는 action 이라는 개념이 추가되었으므로 state 와 action 사이를 정의하는 개념이 필요
* 이에 policy 를 이용하여 state 에서 action 을 mapping 을 정의하게 됨
* 즉 특정 state 에서 어떠한 action 을 취할지를 정의한 것이 policy
$$
\pi(a|s)=Pr(A_t=a|S_t=s)
$$
* 위의 수식에서 알 수 있듯 policy 는 state $s$ 에서 action $a$ 를 수행할 확률로 정의
* 강화학습의 궁극적인 목표는 state 에 도달할 때마다 <mark style='background:#eb3b5a'>return 을 maximize</mark> 하는 <mark style='background:#eb3b5a'>action 을 선택</mark>하는 함수를 찾는 것!

### Policy vs Transition Probability

* 특정 state 에서 다른 state 로 변하는 확률에 대한 mapping 은 아래와 같다.
$$
\Pr(S_{t+1} = s' | S_0, S_1, ..., S_{t-1}, S_t) = \Pr(S_{t+1} = s' | S_t)
$$
* 이러한 mapping 은 MDP 와 유사해 보이지만 MDP 는 action 이 추가된 것이다.
$$
P_{ss'} = \Pr(S_{t+1}=s' | S_t=s) \rightarrow P_{ss'}^a = \Pr(S_{t+1}=s' | S_t=s, A_t=a)
$$

![](img_store/Pasted%20image%2020240106154703.png)

* 즉 Policy 와 Transition Probability 의 차이는 아래와 같다.
* Policy : State $s_t$ 에서 Action $a$ 를 선택할 확률
* Transition Probability : State $s_t$ 에서 State $s_{t+1}$ 로 전이될 확률

* MDP 에서의 Transition Probability 는 state $s_t$ 에서 action $a$ 를 취하여 state $s_{t+1}$ 로 전이될 확률을 의미한다.
![](img_store/Pasted%20image%2020240106155451.png)

* 또한 이제는 Reward 역시 state 간의 직접적인 관계가 아니라 action a 가 포함되어야 하므로 Reward 에 대한 함수도 아래와 같이 바뀐다.
$$
R_s=E[r_{t+1}|S_t=s]\rightarrow R_s^a=E[r_{t+1}|S_t=s,A_t=a]
$$

### Value Function of MDP

* 이전 MP, MRP 의 경우 Value Function 은 state 간의 관계로만 정의되었다.
* MDP 는 State $s_t$ -> Action $a$, Action $a$ -> State $s_{t+1}$ 2개의 Value 가 존재한다.

![](img_store/Pasted%20image%2020240106160038.png)

1. agent 가 t 시점에 state s 에서 policy 를 따라 action a 를 수행
2. state s 에서 action a 를 수행하면 reward 를 받음
3. transition probability 에 따라 state s' 으로 전이

* **State-value Function**
	MDP 에서 <mark style='background:#eb3b5a'>state-value</mark> 는 state 에서 선택하는 policy 에 따라 선택하는 action 이 달라지고, 이후 state 는 그 action 에 따라 달라지기 때문에 <mark style='background:#eb3b5a'>policy 에 영향을 받는다</mark>.
$$
V_{\pi}(s)=E_{\pi}[G_t|S_t=s]
$$
	즉, MDP 에서 <mark style='background:#eb3b5a'>state s 의 가치</mark>는 해당 state 에서 <mark style='background:#eb3b5a'>policy 에 의해 얻게 되는 reward 들의 총합</mark>(return) 을 나타낸다.

* **Action-value Function**
	agent 가 하는 action 에 대해서 value 를 평가
$$
q_{\pi}(s,a)=E_{\pi}[G_t|S_t=s,A_t=a]	
$$
$$G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$

여기서 state  $s_t$ 의 가치는 그 $s_t$ 에서 선택하는 policy $\pi$ 에 따라 취하게 되는 action $a$ 의 action-value function 으로 판단할 수 있고(아래 사진의 빨간 박스), action $a$ 의 가치는 그 action 을 취함으로써 전이되는 state $s_{t+1}$ 들의 가치로 판단(아래 사진의 파란 박스)할 수 있다. 

![](img_store/Pasted%20image%2020240106161733.png)

즉, state 의 action 선택에 따른 가치를 판단하는 action-value function 과 선택한 action 에 따라 전이된 state 의 가치를 판단하는 state-value function 은 서로 긴밀하게 연결되어 있다.
이러한 연결을 수학적으로 나타낸 것이 [[Bellman Equation]] 이다.

Markov Property: The current state completely characterizes the state of the world. Rewards and next states depend only on current state, not history.