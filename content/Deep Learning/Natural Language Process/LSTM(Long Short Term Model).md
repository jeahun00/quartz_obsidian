#Deep_Learning #NLP #RNN

# Introduction

* RNN 모델은 이전 Gradient Vanishing, Gradient Exploding 의 위험을 가지고 있다.
* 여기서 Gradient Exploding 은 Gradient Clipping 으로 어느 정도 해결이 가능하다.
* 하지만 Gradient Vanishing 은 모델의 구조상 바뀔수가 없다.
* 이에 이러한 문제를 해결하기 위해 LSTM 을 제안한다.

# LSTM(Long Short Term Model)


![LSTM](_media-sync_resources/20240417T162531/20240417T162531_67962.png)

기본적인 모델의 구조는 위와 같다.
위에서 볼 수 있듯 4개의 Gate 를 통해 과거의 기억을 Cell State 에 전달하여 과거기억 소실 문제를 해결한다.

기존 RNN 의 수식은 아래와 같다.

$$
\begin{align*}
h_{t} = RNN(h_{t-1},x_t)
\end{align*}
$$
LSTM 의 수식을 살펴보자
$$
\begin{align*}
\{C_t,h_t\}=LSTM(x_t,C_{t-1},h_{t-1})
\end{align*}
$$

* 즉 RNN 은 이전 상태를 저장하는 $h_{t-1}$ 과 $x_t$ 로 부터 새로운 상태인 $h_t$ 를 출력한다.
* 하지만 LSTM 은 C_t 라고 하는 새로운 정보가 추가된다.
* 이러한 이전 Cell State 인 $C_{t-1}$ 과 현재의 정보 $x_t$, 와 RNN 과 같은 이전 상태인 $h_{t-1}$ 로 부터 새로운 Cell State 인 $C_{t}$ 를 출력한다.
* 이러한 C_t 는 4개의 gate 를 통해 구해진다.
* 그렇다면 이제 $C_t$ 를 조절하는 4개의 gate 에 대해 알아보자
### 0. LSTM's 4 gates
![](_media-sync_resources/20240417T162531/20240417T162531_33394.png)

* 위의 이미지에서 볼 수 있듯 LSTM 은 4래의 Gate 가 존재한다.
* input gate, forget gate, output gate, gate for $\tilde{C_t}$ (이 gate 는 CS231n 에 따라 gate gate 라 칭하겠다.)
* 이러한 gate 를 조절하여 과거의 정보와 현재의 정보의 세기를 결정하는 것이 LSTM 의 핵심이다.

![](_media-sync_resources/20240417T162531/20240417T162531_30879.png)


### 1. Forget Gate

![](_media-sync_resources/20240417T162531/20240417T162531_80643.png)

![](`_media-sync_resources/20240417T162531/20240417T162531_88206.png)

![](_media-sync_resources/20240417T162531/20240417T162531_22511.png)

* 이전 hidden state($h_{t-1}$) 와 현재 time step 의 input($x_t$) 을 가지고 이들 중에서 어떠한 정보를 버릴지를 결정한다.
* 여기서 $f_t$ 의 값은 0~1 사이의 값을 가진다.
* 또한 $f_t$ 의 값은 multiple 로 Cell State 에 전달이 된다.
* 이는 어느정도의 세기로 이전 정보를 Cell State 로 전달할지를 정하게 된다.
* 즉 0에 가까우면 이전 정보를 아예 전달하지 않겠다는 뜻이고 1에 가까우면 이전 정보를 모두 전달하겠다는 뜻이다.

### 2. Input Gate(and Gate Gate)

![](_media-sync_resources/20240417T162531/20240417T162531_19054.png)

![[Pasted image 20231217164151.png]]

![[Pasted image 20231217164158.png]]

* $i_t$ : 이전 hidden state 인 $h_{t-1}$ 과 현재 time step 의 input 인 $x_t$ 를 가지고 이들 중 어떤 정보를 현재 state 로 지정할지를 정한다.
* $\tilde{C_t}$  : $h_{t-1}$, $x_t$ 를 이용하여 새로운 Cell State 인 $\tilde{C_t}$ 를 만들고 이를 얼마만큼의 크기로 전달할지를 정한다.

* forget gate 에서 연산한 ${f_t}$ 와 위에서 연산한 $i_t$ 와 $\tilde{C_{t}}$ 를 이용하여 Cell State 를 갱신한다.
![[Pasted image 20231217183922.png]]

### 3. Output Gate

![[Pasted image 20231217184054.png]]
![[Pasted image 20231217184106.png]]
![[Pasted image 20231217184111.png]]

* $o_t$ : 이전 hidden state 인 $h_{t-1}$ 과 현재 input 인 $x_t$ 를 이용하여 $o_t$ 를 연산한다.
* $h_t$ : input gate 에서 마지막으로 갱신한 Cell State 인 $C_t$ 를 위에서 연산한 $o_t$ 와 연산하여 현재 hidden state 인 $h_t$ 를 갱신한다.
	* 즉 이전 정보를 포함한 $C_t$ 에서 마지막으로 $h_t$ 를 갱신함으로써 다음 셀에 얼마만큼의 정보를 줄지를 결정하는 gate 이다.


# conclusion

위와 같은 LSTM 모델은 RNN 의 고질적인 문제인 Gradient Vanishing 을 해소한다.
즉 이전 정보가 단계를 거침으로 인해 정보가 소실되는 문제를 해결한다.
하지만 4개의 Gate 를 통한 연산 때문에 연산량이 증가하는 단점이 존재한다.
이를 간소화하지만 성능은 유사하게 나오는 GRU 라는 모델도 존재한다.
이에 대한 설명은 아래 링크를 통해 볼 수 있다.


[[GRU(Gated Recurrent Unit)]]

REF : 
* https://glanceyes.com/entry/LSTM%EA%B3%BC-GRU%EC%9D%98-Gate%EB%B3%84-%ED%8A%B9%EC%A7%95%EA%B3%BC-%EA%B5%AC%EC%A1%B0-%ED%95%9C%EB%B2%88%EC%97%90-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0
* https://colah.github.io/posts/2015-08-Understanding-LSTMs/
