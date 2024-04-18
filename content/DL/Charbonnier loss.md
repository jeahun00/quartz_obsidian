#Deep_Learning 

L1, L2 Loss 의 장점을 모두 가지는 loss function 이다.
$$L_{Charbonnier}​(e)=\sqrt{e^2+ϵ^2​}$$

* $e$ : Real Value 와 Predict Value 의 차이 즉 에러
* $\epsilon$ : 위의 값을 계산할 때 0 지점에서의 미분을 용이하게 하는 역할. 대체로 아주 작은 상수

효과
1. 오차가 적을 때 : L2 Loss 와 유사하게 동작하며 작은 오차에 대해 부드럽게 대응
2. 오차가 클 때 : L1 Loss 처럼 손실 값의 급격한 변화를 막아줌. 즉 이상치에 덜 민감하게 반응

[[L1, L2 Regularization]] 에 대한 Note는 앞에 링크를 참고하라