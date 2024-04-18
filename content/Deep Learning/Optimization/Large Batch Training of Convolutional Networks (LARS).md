#Deep_Learning 

# overview

* 크고 깊은 CNN 모델을 학습시키는데 많은 시간이 걸림.
* 그렇기에 많은 수의 GPU 를 사용해야 하고 그렇게 되려면 Batch Size 가 커져야 한다. 
* 일반적으로 큰 Batch Size 는 Iteration 이 적다.
* 그렇다면 LR(Learning Rate)를 초기에 크게 줘야 한다.
* 하지만 큰 LR 은 모델이 발산할 수 있다.
* 이러한 문제 일부를 해결하기 위해 초기 LR은 작게 부여하고 epoch이 진행될수록 LR을 linear 하게 올려주는 Linear Scaling LR with warm-up 이 제시되었다.

> LR warm-up : lr이 $p$일때 warm-up을 $n$ epoch동안 한다면, 첫 번째 iteration에서 lr은 $1×p/n$, 두 번째는 $2xp/n$ 을 사용

* LARS 는 weight 와 gradient 간의 norm 사이의 비율을 측정
* 해당 비율이 작으면 학습이 불안정한 문제 발생
* 이에 위의 비율을 이용한 Layer-wise Adaptive Rate Scaling 을 제안
* ADAM 과의 차이점?
	* Weight 가 아닌 Layer 마다 다른 LR 을 사용
	* Update 의 정도가 Weight Norm 에 의해 결정


# detail
weight 와 gradient update의 비율인 ${||w||}/{ ||\lambda \times \nabla E(w_t) \|}$ 가 weight 와 bias 마다, 각 층마다 다르고, 이 비율이 낮으면 학습이 불안정하고 느려짐을 발견함

> 기존 SGD 는 모든 층에 동일한 LR $\lambda$ 를 사용해 $w_{t+1}=w_t-\lambda\nabla L(w_t)$ 를 수행한다.
> 이 때 $\lambda$ 가 크다면 $||\lambda \times \nabla E(w_t) ||$ 가 커지게 되고 이렇게 된다면 $||w_t||$ 의 값보다 더 커져($w_{t+1}$ 이 0보다 작아짐 -> 0이 되지 못하고 값을 벗어남) 모델이 발산하게 된다.
> 따라서 학습 조건에서 초기 weight initialize 가 매우 중요하게 작용한다.

위와 같은 문제를 해결하기 위해 아래 방안을 제시한다.

$$
\begin{align*}
\Delta w_t = \gamma \times \lambda' \times \nabla L(w_t)\\
\lambda' = \eta \times \frac{\|w'\|}{\|\nabla L(w')\| + \beta \times \|w'\|}
\end{align*}
$$

$\lambda^l$ : Local Learning Rate -> $l$ 레이어 에서의 learning rate 
$\eta$ : 한번의 업데이트동안 weights 의 변화가 얼마나 믿을 수 있는지 판단하는 계수
$\beta$ : weight decay의 크기를 적용하는 상수. weight decay 사용 안할 시 무시 가능

* 위와 같은 수식으로 weight update 를 진행하게 되면 weight update 가 gradient 에 depend 하지 않게 된다.
* 즉 $\frac{\|w'\|}{\|\nabla L(w')\|}$ 가 커지면 LR 을 키우고 작아지면 LR을 줄여 학습을 원할하게 진행하게 된다.

REF :
https://creamnuts.github.io/paper/LARS/
https://2-chae.github.io/category/1.papers/19