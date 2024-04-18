#Deep_Learning 

# Adam 의 한계
1. 하이퍼파라미터에 지나치게 민감하다
Adam 은 learning rate 를 이전의 learning 로 부터 update 하는 RMSProp 의 특징을 가지고 있는데 초기 learning rate 를 잘못 지정한다면 학습률이 떨어지는 경향이 있다.
2. 과거 오정보의 누적
Adam 은 이전 Gradient로 Momentum 과 scale parameter를 조절한다.
이로 인하여 처음 입력된 잘못된 데이터가 지속해서 누적될 수 있다.
3. 메모리 이슈
과거의 Gradient 를 저장하고 있어야 하므로 다른 optimizer 에 비해 메모리 요규량이 높다.
4. Test(Prediction)에 약함
Train data 의 학습률은 좋지만 Test data 의 학습률이 떨어지는 Overfitting 을 야기할 수 있다.

# ADAM 이후의 Optimizer
[[Large Batch Training of Convolutional Networks (LARS)]]
[[Layer-wise Adaptive Moments optimizer for Batch training (LAMB)]]
[[AdamW]]
[[Lion]]
[[CosineAnnealingLR]]