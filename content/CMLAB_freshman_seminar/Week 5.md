# Lecture 6 : CNN Architecture 

- [ ] 정리
- [ ] ppt


# Lecture 7 : Training Neural Networks

- [ ] 정리
- [ ] ppt 

# HW 
1. LPIPS논문리뷰
2. regularizer의 역할, regularizer가 w 작아지게 만드는 이유 (L1&L2, softmax와 연관)
3. pytorch를 이용해서 backpropagation 각 w의 gradient를 확인할 수 있는 방법
4. MLP Layer를 Conv layer로 구현하는 방법 (MLP와 Conv filter의 관계 & 구현해야 하는 이유 - hint: input사이즈)
5. CNN의 특징 - translation equivariance(invariance)을 중심으로


$$
w \rightarrow w - \frac{\eta \lambda}{n} \text{sgn}(w) - \eta \frac{\partial \mathcal{C}}{\partial w}


$$

- - Stationarity of statistics과 locality of pixel dependencies