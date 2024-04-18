#Math #Deep_Learning

CS231n 을 정리하던 중 Gradient, Jacobian 에 대한 개념이 부족함을 느껴 정리를 하게 되었다.

#### CS231n 해당 슬라이드
CS231n Lecture 4. Neural Network and Backpropagation 118p 
![](_media-sync_resources/20240417T162604/20240417T162604_55869.png)

## 1. Scalar to Scalar
* 우리가 흔히 접하는 $y = f(x)$ 꼴일 때를 미분하는 형식이다.
* 입력과 출력 모두가 scalar 인 경우이다.
* 이에 대한 설명은 생략하겠다.

## 2. Vector to Scalar
* 스칼라를 벡터로 미분하는 경우
* 이러한 형태의 벡터를 Gradient Vector 라고 한다.
* 즉 $x$가 vector, $y$ 가 scalar 이다.
$$
\begin{align*}\\
\nabla{\mathbf{y}} = \frac{\partial \mathbf{y}}{\partial \mathbf{x}} = 
\begin{bmatrix}
\frac{\partial y}{\partial x_1} \\
\frac{\partial y}{\partial x_2} \\
\vdots \\
\frac{\partial y}{\partial x_N} \\
\end{bmatrix}

\end{align*}
$$
* $x$ 의 각 요소가 변할 때 $y$ 가 얼마나 변하는지를 수치로 나타냄

## 3. Vector to Vector
* 벡터를 벡터로 미분하는 경우
* 이러한 형태의 Matrix 를 Jacobian Matrix 라고 한다.
* x 의 각 요소와 y 의 각 요소 각각의 미분값을 행렬로 나타냄
* $x = [x_1, x_2, ... ,x_N]$, $y=[y1,y2,...y_M]$
$$
\begin{align*}
J = \frac{d\mathbf{y}}{d\mathbf{x}} = 
\begin{bmatrix}
\frac{\partial \mathbf{y}_1}{\partial \mathbf{x}}^T \\
\vdots \\
\frac{\partial \mathbf{y}_M}{\partial \mathbf{x}}^T \\
\end{bmatrix}
=
\begin{bmatrix}
\nabla \mathbf{y}_1^T \\
\vdots \\
\nabla \mathbf{y}_M^T \\
\end{bmatrix}
=
\begin{bmatrix}
\frac{\partial y_1}{\partial x_1} & \cdots & \frac{\partial y_1}{\partial x_N} \\
\vdots & \ddots & \vdots \\
\frac{\partial y_M}{\partial x_1} & \cdots & \frac{\partial y_M}{\partial x_N} \\
\end{bmatrix}

\end{align*}
$$

