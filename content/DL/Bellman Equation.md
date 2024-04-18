#ReinforceLearning #Math  
* MDP 에서 살펴본 state-value function 과 action-value function 사이의 관계로 state $s_t$ 와 $s_{t+1}$ 로의 전이에 대한 관계식을 생성할 수 있다.
* 이러한 전이에 대한 관계식을 Bellman Equation 이라 한다.
* 강화학습은 이러한 Bellman Equation 을 푸는 과정이라고 볼 수 있다.

---

Bellman Equation 은 expectation 과 optimality, 2가지 종류가 있다.
아래부터는 2가지를 차례로 알아볼 것이다.

# Bellman Expectation Equation

### State-Value Function

* MDP 의 State-value function 은 아래와 같이 정의된다.

$$
V_{\pi}(s)=E_{\pi}[G_t|S_t=s]
$$

* 위 식에서 G_t 는 아래와 같이 정의된다.

$$
G_t = R_{t+1} + \gamma R_{t+2} + \dots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

* 위의 $V_\pi(s)$ 의 식에서 $G_t$ 를 아래와 같이 풀어 쓸 수 있다.

$$
\begin{align*}
V_{\pi}(s)
&= \mathbb{E}_{\pi}[ G_t | S_t = s ]\tag 1\\
&= \mathbb{E}_{\pi}[ R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots | S_t = s ]\tag 2\\
&= \mathbb{E}_{\pi}[ R_{t+1} + \gamma(R_{t+2}+\gamma R_{t+3} + \ldots) | S_t = s ]\tag 3\\
&= \mathbb{E}_{\pi}[ R_{t+1} + \gamma G_{t+1} | S_t = s ]\tag 4\\
&= \mathbb{E}_{\pi}[ \textcolor{magenta}{R_{t+1}} + \textcolor{skyblue}{\gamma V_{\pi}(S_{t+1})} | S_t = s ]\tag 5\\
\end{align*}
$$

* 이 결과가 Bellan Expectation Equation 이다. 
* 여기서 $(5)$ 식을 주목해보자.
$$\mathbb{E}_{\pi}[ R_{t+1} + \gamma V_{\pi}(S_{t+1}) | S_t = s ]$$
* $\textcolor{magenta}{R_{t+1}}$ : t+1 시점에 받는 **immediate reward**
* $\textcolor{skyblue}{\gamma V_\pi(S_{t+1})}$ : discounted factor 가 곱해진 **future reward** 

* 여기서 주목해야 할 점은 시점 t 의 state-value function 이 다음 시점 t+1 의 state-value function 으로 표현할 수 있다는 점이다.
* 위의 식은 직접 구현에 어려움이 있다.
* 이에 computational 하게 식을 수정해 줄 필요가 있다.
* 아래 이미지를 보자
**이미지 1**
![](../../../img_store/Pasted%20image%2020240106173710.png)

* 현재 시점 $s_t$ 에서 가능한 action 이 $a_1, a_2, a_3$ 라 가정. 이러한 action 들을 취할 확률인 policy 를 계산
	* $\pi(a_1|s)=Pr(A_t=a_1|S_t=s)$
	* $\pi(a_2|s)=Pr(A_t=a_2|S_t=s)$
	* $\pi(a_3|s)=Pr(A_t=a_3|S_t=s)$
* agent 는 $a_1, a_2, a_3$ 를 선택할 확률이 정해지고 그 때의 action-value function 을 구할 수 있다.
	* $q_{\pi}(s,a_1)=E_{\pi}[G_t|S_t=s,A_t=a_1]$
	* $q_{\pi}(s,a_2)=E_{\pi}[G_t|S_t=s,A_t=a_2]$
	* $q_{\pi}(s,a_3)=E_{\pi}[G_t|S_t=s,A_t=a_3]$
* 이제 우리가 구하고자 하는 $V_\pi(s)$ 는 action-value function 과 policy 간의 곱 즉, expectation 임을 알 수 있다.
$$
\begin{align*}\\
V_\pi(s)=\pi(a_1|s)*q_{\pi}(s,a_1)+\pi(a_2|s)*q_{\pi}(s,a_2)+\pi(a_3|s)*q_{\pi}(s,a_3)
\end{align*}
$$

* 위의 식을 일반화하면 아래와 같다.
$$
V_{\pi}(s) = \sum_{a \in A} \pi(a|s) * q_{\pi}(s,a)
$$

* 이제 우리는 expectation 을 computable 한 형태로 표현하였다.
* 하지만 위와 같은 형태로는 next state-value function 과의 관계가 직접적으로 드러나지 않는다.
* 이에 q-function(action-value function) 에 대해 식을 전개해보자.
**이미지 2**
![](../../../img_store/Pasted%20image%2020240106180146.png)
* 이제 위의 수식을 우리가 구한 $V_\pi(s)$ 에 대입하면 아래와 같은 수식이 도출된다.
$$
V_{\pi}(s) = \sum_{a \in A} \pi(a|s) \left( R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V_{\pi}(s') \right)

$$
* 위의 식을 통해 이제 $V_\pi(s)$ 와 $V_\pi(s')$  의 관계식을 직접적으로 드러낼 수 있게 되었다.

### Action-Value Function

$$
V_{\pi}(s) = \sum_{a \in A} \pi(a|s) \left( R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V_{\pi}(s') \right)
$$

위의 **이미지 2** 의 수식에 $V_\pi(s)$ 를 우리가 구한 위의 수식을 대입해 보자.

$$
\begin{align*}
\textcolor{red}{V_{\pi}(s)} = \sum_{a \in A} \pi(a|s) \left( R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a \textcolor{red}{V_{\pi}(s')} \right)\\
\textcolor{red}{q_{\pi}(s, a)} = R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a \sum_{a' \in A} \pi(a'|s') \textcolor{red}{q_{\pi}(s', a')}
\end{align*}
$$

1. 처음 agent가 경험하는 state s나 action에 대한 값들은 모두 의미없는 값들입니다(0 혹은 random number).
2. 그리고 최적의 policy도 모른 상태로 random요소를 가미하여 이것저것 action을 선택해봅니다. 
3. 그런 과정에서 env.과 상호작용을 하면서 얻은 reward와 state에 대한 정보들을 통해서 어떤 state에서 어떤 action을 하는 것이 좋은지(최대의 reward를 얻을 수 있는지)를 판단합니다.
4. 이때 이를 판단하는 수단이 state-value func.과 action-value func.이고, 이것을 Bellman eqn.을 가지고 update하며 점점 높은 reward를 얻을 수 있는 state를 찾아가고 action을 하도록 배워나갑니다. 
5. 이러한 과정 속에서 최대의 reward를 갖는 action들을 선택하게 되는 optimal policy를 찾아나갑니다.

위의 5번 과정의 maximize 된 reward 를 만들어 낼 수 있는 식을 **Bellman Optimality Equation** 이라 한다.

---

# Bellman Optimality Equation

* 강화학습의 궁극적인 목표는 reward 를 최대화 할 수 있는 policy 를 찾는 것이다.
* 이러한 policy 를 optimal policy 라고 부른다.
* 이러한 optimal policy 를 따르는 Bellman Equation 을 **Bellman Optimality Equation** 이라 한다.

### Optimal Value Function

* Optimal state-value function
$$
\begin{align*}
V^*(s) = \max_{\pi} V_{\pi}(s) \\
\end{align*}
$$
* Optimal action-value function
$$
\begin{align*}
q^*(s, a) = \max_{\pi} q_{\pi}(s, a)\\
\end{align*}
$$

* 위의 2개의 value function 은 최대의 reward 를 가질 때의 value function 이다.
* 이 중 q function 에 대한 optimal value function 을 구할 수 있다면 
* 주어진 state 에서 q-value 가 가장 높은 action 을 선택할 수 있다
* 따라서 위의 과정을 통해 우리는 optimal policy 를 구할 수 있게 된다.
* 이러한 과정을 수식으로 표현하면 아래와 같다.
$$
\pi^*(a|s) = 
\begin{cases} 
1 & \text{if } a = \underset{a}{\mathrm{argmax}} \ q_{\pi}(s, a) \\
0 & \text{otherwise} 
\end{cases}
$$
* 즉, q-function 을 최대화하는 action 만을 취하고 나머지는 0으로 만들겠다는 뜻이다.

### Bellman Optimality Equation

위의 2개의 optimal value function 의 관계를 아래의 Bellman Optimality Equation 으로 나타낼 수 있다.

$$
\begin{align*}
V_*(s) = \max_{\pi} V_{\pi}(s) &= \max_{a} q_{\pi}(s, a)\\
&= \max_{a} \left( R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V_{\pi}(s') \right)\\
&= \max_{a} R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V_{\pi}(s')\\
\end{align*}
$$

$$
\begin{align*}
q_*(s, a) = \max_{\pi} q_{\pi}(s, a)
&= \max_{\pi} \left( R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V_{\pi}(s') \right)\\
&= R_s^a + \gamma \sum_{s' \in S} P_{ss'}^a V_{\pi}(s')
\end{align*}
$$

