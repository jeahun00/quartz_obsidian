#Deep_Learning 

L1, L2 Loss 는 Loss 함수를 구할 때 Regularization 을 위해 더해주는 Term 이다.
# Loss
### L1 Loss
* 실제값 $y_i$ 와 예측값 $f(x_i)$의 차의 절대값 들의 합이다.
$$
L = \sum_{i=1}^{n} |y_i - f(x_i)|
$$
### L2 Loss
* 실제값 $y_i$ 와 예측값 $f(x_i)$ 의 차이의 제곱들의 합이다.
$$
L = \sum_{i=1}^{n} (y_i - f(x_i))^2
$$

### L1 vs L2

1. Robustness
Robustness 란 oulier 인 데이터가 있을 때 그 oulier 가 원함수에 어느정도에 영향을 받았는지를 나타낸다.

L1 Loss 의 경우 아주 큰 outlier 가 있을 때 그 차이가 linear 하다.
L2 Loss 의 경우 아주 큰 outlier 가 있을 때 그 차이가 제곱이 된다.
즉 L2 Loss 가 Outlier 에 훨씬 민감하다.

---

# Regularization

### L1 Regularization
$$
\text{cost}(W, b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}_i, y_i) + \lambda \frac{1}{2} \| w \|
$$
### L2 Regularization
$$
\text{cost}(W, b) = \frac{1}{m} \sum_{i=1}^{m} L(\hat{y}_i, y_i) + \frac{\lambda}{2} \| w \|^2
$$

* oulier : 
* L1 Reg 는 oulier 에 대해 Regulization Term 이 비교적 작으므로 전체 Term 은 민감하게 반응한다.
* L2 Reg 는 oulier 에 대해 Regulization Term 이 비교적 크므로 전체 Term 에서 덜 민감하게 반응하여 부드러운 곡선을 만들어 낸다.

* Sparcity
* L1 Reg 는 많은 데이터를 0으로 만드는 특성이 있다. 이를 sparcity 라고 한다. 

![](_media-sync_resources/20240417T162529/20240417T162529_83672.png)

위의 그림에서 우리가 적절한 w 를 학습을 할 때 $H_{0}$ 가 $||w||_1$ 을 만날 때 많은 상황에서 0 혹은 1로 만나게 된다. 
이러한 이유로 L1 Regularization 은 sparcity 를 가진다고 할 수 있다.

* Blur
일부 의미 없는 데이터(지나치게 작은 값들) 이 모델에 영향을 미치는 것
* L1 Regularization 의 경우 위와 같은 값들을 0으로 만들어 Blur 에 강하다.
* L2 Regularization 의 경우 위와 같은 값들을 0으로 만들지 못하기 때문에 Blur 에 약하다.

---

### Regularization 의 역할


