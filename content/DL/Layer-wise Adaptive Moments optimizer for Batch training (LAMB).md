#Deep_Learning 

# LAMB Optimizer

Layer-wise Adaptive Moments optimizer for Batch training (LAMB)는 LARS에서 영감을 받아 개발된 최적화 알고리즘으로, 특히 대규모 모델과 데이터셋에서 효율적인 학습을 위해 설계되었습니다. LAMB는 LARS의 기본 아이디어를 채택하고, Adam 최적화 알고리즘의 기법을 결합하여 더욱 발전시킨 형태입니다.
LARS 에 대한 자세한 설명은 아래 링크 참고
[[Large Batch Training of Convolutional Networks (LARS)]]
## LAMB의 핵심 개념

LAMB 최적화기는 다음의 두 가지 주요 개념을 결합합니다:

1. **Layer-wise Adaptive Rate Scaling (LARS)**: LARS와 마찬가지로, LAMB는 각 레이어의 가중치에 대해 개별적인 학습률 조정을 제공합니다. 이는 레이어마다 다른 학습 속도를 허용하고, 큰 배치 학습에서 발생할 수 있는 불안정성을 감소시킵니다.

2. **Adaptive Moment Estimation (Adam)**: LAMB는 Adam의 모멘텀 및 RMSProp과 유사한 적응적 2차 모멘트 추정을 사용합니다. 이는 각 가중치 업데이트에 대한 그래디언트의 방향과 크기 모두를 고려하여, 더 안정적이고 효과적인 학습을 가능하게 합니다.

## LAMB의 수식

LAMB의 업데이트 규칙은 다음과 같습니다:

1. **모멘텀 및 2차 모멘트 계산**:

$$
\begin{align*}
   m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla L(w_{t-1})\\
   v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla L(w_{t-1}))^2\\
\end{align*}
$$
   여기서 $m_t$​와 $v_t$​는 각각 모멘텀과 2차 모멘트, $β_1​$과 $β_2$​는 각각 모멘텀 및 2차 모멘트에 대한 감쇠율입니다.
   
2. **편향 수정**:
$$
\begin{align*}
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}\\
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
\end{align*}
$$
이 단계에서는 초기 단계에서의 편향을 교정합니다.

3. **레이어별 학습률 조정 및 업데이트**
$$
\begin{align*}\\
\text{Ratio} &= \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}\\
\text{Layerwise LR} &= \eta \times \frac{\|w_t\|}{\|\text{Ratio}\|}\\
w_{t+1} &= w_t - \text{Layerwise LR} \times \text{Ratio}

\end{align*}
$$
$\eta$ : Global Learning Rate
$\epsilon$ : zero devision 을 방지하기 위한 작은 값의 상수

## LAMB의 장점 및 한계

- **장점**: LAMB는 큰 배치 크기와 대규모 모델에서 빠른 수렴을 달성할 수 있습니다. 또한, 레이어별 학습률 조정으로 인해 학습 과정이 더 안정적이고 효율적입니다.
    
- **한계**: LAMB는 하이퍼파라미터 설정에 민감하며, 특히 $β_1​$, $β_2$​, 및 $\eta$ 값의 선택에 따라 성능이 크게 달라질 수 있습니다. 또한, 모든 유형의 아키텍처나 문제에 대해 동일하게 잘 작동하는 것은 아닙니다.
