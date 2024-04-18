# Sung Kim 모두의 딥러닝
https://hunkim.github.io/ml/

- [x] 머신러닝의 개념과 용어 ✅ 2023-11-07
- [x] Linear Regression 의 개념 ✅ 2023-11-07
- [x] Linear Regression cost함수 최소화 ✅ 2023-11-07
- [x] 여러개의 입력(feature)의 Linear Regression ✅ 2023-11-07
- [x] Logistic (Regression) Classification ✅ 2023-11-08
- [x] Softmax Regression (Multinomial Logistic Regression) ✅ 2023-11-09
- [x] ML의 실용과 몇가지 팁 ✅ 2023-11-09
- [x] lab7 metal gpu 돌릴 수 있는지 확인 + ppt 마무리 📅 2023-11-09 ✅ 2023-11-13
# Stanford CS231n

$W := W - \alpha \frac{\partial}{\partial W} cost(W)$ 
$W := W - \alpha \frac{1}{m} \sum_{i=1}^{m} (Wx^{(i)}-y_{(i)})x^{(i)}$ 

$$
\begin{align*}\\
\underset{x\_train}{\\
\underbrace{
\begin{bmatrix}
{73}&{80}&{75}\\
{93}&{88}&{93}\\
{89}&{91}&{80}\\
{96}&{66}&{70}\\
{73}&{66}&{70}
\end{bmatrix}}}
\cdot
\underset{W}{\\
\underbrace{
\begin{bmatrix}
{W_1}\\
{W_2}\\
{W_3}\\
\end{bmatrix}}}
=
\underset{y\_train}{\\
\underbrace{
\begin{bmatrix}
{152}\\
{185}\\
{180}\\
{196}\\
{142}\\
\end{bmatrix}}}
\end{align*}
$$