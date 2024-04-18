# CS231n Assignment 1

REF :
* assignment1 solution : 
	* https://github.com/mantasu/cs231n/tree/master/assignment1
	* https://codestudyisdiff.tistory.com/10
	* https://velog.io/@clayryu328/weekly%EC%8B%A4%EC%8A%B51-cs231n-assignment-1-KNN
* KNN algorithm : https://firework-ham.tistory.com/27

* Q4 : https://lionkingchuchu.tistory.com/39 
# Seminar Assignment
1. median filter, bilateral filter implementation
2. relationship between disparity, depth 
3. darkprogrammer review -> 1~3
https://darkpgmr.tistory.com/77
https://darkpgmr.tistory.com/78
https://darkpgmr.tistory.com/79


**Inline Question 3**

Which of the following statements about $k$-Nearest Neighbor ($k$-NN) are true in a classification setting, and for all $k$? Select all that apply.

1. The decision boundary of the k-NN classifier is linear.
2. The training error of a 1-NN will always be lower than or equal to that of 5-NN.
3. The test error of a 1-NN will always be lower than that of a 5-NN.
4. The time needed to classify a test example with the k-NN classifier grows with the size of the training set.
5. None of the above.

$\color{blue}{\textit Your Answer:}$ 2, 4

$\color{blue}{\textit Your Explanation:}$
1. F
The boundary of k-NN classifier is not linear. Because k-NN found nearest points, and that points are not in linear.
2. T
Generally true. Because 1-NN just check nearest 1 point. That mean if some points positioned in boundaries, 1-NN can't classify that point.
3. F
This statement is problem dependented. Not always.
4. T
This statement is clearly true. Because when datasets are big, the time of compute distance are grows.

$$
\sum_{i}||X_{trian} - X_{test}||^2=X_{train}^2+X_{test}^2-2X_{train}X_{test}
$$