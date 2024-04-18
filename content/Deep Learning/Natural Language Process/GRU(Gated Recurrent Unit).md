 #Deep_Learning #NLP #RNN
# GRU

* LSTM 구조를 경량화 하여 적은 메모리로 빠른 연산이 가능한 모델이다.
* LSTM 에 비해 모델이 가볍지만 성능은 크게 떨어지지 않는다.

* 전체적인 동작은 LSTM 과 유사하지만 GRU 는 Cell State 를 사용하지 않고 RNN 모델처럼 hidden state 인 $h_t$ 만을 이용한다.

**LSTM 수식**
$$
\begin{align*}
\{C_t,h_t\}=LSTM(x_t,C_{t-1},h_{t-1})
\end{align*}
$$
**GRU 수식**
$$
\begin{align*}
h_{t}=GRU(x_{t},h_{t-1}) 
\end{align*}
$$

# GRU 의 구조

![[Pasted image 20231217194225.png]]

* $z_t$ : update gate 라 부른다
* $r_t$ : reset gate 라 부른다

### Gate
![[Pasted image 20231217200312.png]]
$r_t$ : 이전의 정보를 얼마나 지울지 결정
$z_t$ : 이전의 정보를 얼마나 유지시킬지를 결정
![[Pasted image 20231217195129.png]]

### Candidate Hidden State

![[Pasted image 20231217202659.png]]
![[Pasted image 20231217201417.png]]
* $\tilde{h_t}$ : Candidate Hidden State
* 이전에 연산한 $r_t$ 와 $Uh_{t-1}$ 을 곱하여 이전 time step 에서 무엇을 지워야 할지 결정
* $tanh$ 을 통과하므로 1 ~ -1 사이의 값을 가진다.
* update gate 를 반영하지 않았기에 Candidate 라 명명됨
* 즉 LSTM 의 $\tilde{C_{t}}$ 랑 비슷한 역할이다

### Hidden State Update

![[Pasted image 20231217201030.png]]
![[Pasted image 20231217201427.png]]
* $z_t$ : update gate 로 이전의 정보를 얼마나 포함시킬지를 결정
* $z_t$ 가 0 에 가까우면 이전 state 인 h_{t-1} 의 값을 크게 유지, 1 에 가까우면 새로운 state 인 $\tilde{h_{t}}$ 를 크게 유지하는 것이다.

# conclusion

![[Pasted image 20231217201457.png]]

* Reset Gate 는 short-term dependency : 현재의 정보를 반영
* Update Gate 는 long-term dependency : 과거의 정보를 반영



REF : 
* https://hyen4110.tistory.com/26