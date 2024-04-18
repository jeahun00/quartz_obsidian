이 노트는 cs231n assignment 를 해결하며 생기는 의문과 그 해답을 작성하는 노트이다. 주로 python 기본 문법과 numpy 관련한 자료들을 업로드 할 예정이다.

# Numpy

---

### 1. numpy list 비교
* numpy list 가 같은지 다른지 비교해야 하는 경우가 있다.
* 이럴 경우 사용할 수 있는 코드이다.
* 또한 일반적인 python list 비교코드도 포함하였다.
**예제**
```python
import numpy as np

list_a = [[1,2],[3,4]]
list_b = [[1,2],[3,4]]
list_c = [[1,2],[-3,4]]

# python list 간 비교
print(list_a == list_b)
print(list_a != list_c)
    
# numpy list 간 비교
list_a = np.array(list_a)
list_b = np.array(list_b)
list_c = np.array(list_c)

print(np.array_equal(list_a, list_b))
print(np.array_equal(list_a, list_c))
```
**출력**
```bash
True
True
True
False
```

---
### 2. numpy axis
* numpy 에서 axis 순서가 항상 헷갈렸다.
* 이를 정리하고자 한다.

**2.1. 2-dimension**
* 2차원에서 축의 방향은 아래와 같다.
![[../../../../img_store/Pasted image 20240119122258.png]]
* 2차원 matrix 에서 element 에 대한 접근은 H, W 이다.(혹은 Row, Column)
**예제**
```python
list = np.array([[0,4], [8,12]])
print(list)
print(list[1,0])
```
**출력**
```bash
[[ 0  4]
 [ 8 12]]
8
```

**2.2. 3-dimension**
* 3차원에서 축의 방향은 아래와 같다.
![[../../../../img_store/Pasted image 20240119122534.png]]
* 3차원 tensor 에서 element 에 대한 접근은 C, H, W 이다.(혹은 Channel, Row, Column)
**예제**
```python
list = np.array([
    [[0,1,2,3],
     [4,5,6,7]],
    [[8,9,10,11],
     [12,13,14,15]],
])

print(list)
print(list[1][0][2])
```

**출력**
```bash
[[[ 0  1  2  3]
  [ 4  5  6  7]]

 [[ 8  9 10 11]
  [12 13 14 15]]]
10
```

REF : 
* https://m.blog.naver.com/PostView.naver?isHttpsRedirect=true&blogId=sooftware&logNo=221577171997

---
### 3. np.sum()
* `np.sum()` 함수는 input tensor 에 대해 모든 element 를 다 더한 값이다.
* 여기에 axis option 을 추가할 수 있는데 axis 는 더하는 방향에 대한 것이다.
**예제**
```python
list_a = np.array([
    [1,2,3,4],
    [5,6,7,8],
    [9,10,11,12]
])
print(np.sum(list_a))
print(np.sum(list_a, axis=0))
print(np.sum(list_a, axis=1))
```

**출력**
```bash
78
[15 18 21 24]
[10 26 42]
```

REF : 
* http://taewan.kim/post/numpy_sum_axis/

---
### 4. np.argmax()
* 특정 배열 안의 element 중 가장 큰 element 의 index 를 출력
* 2차원 이상의 배열인 경우 flatten 된 index 부여
**예제**
```python
list_1D = np.array([5,3,7,3425,2])

list_2D = np.array([
    [1,2,3,4,5],
    [6,7,8,9,10],
    [11,12,23242,14,15],
    [16,17,18,19,20]
])
print(np.argmax(list_1D))
print(np.argmax(list_2D))
```

**출력**
```bash
3
12
```

REF : 
* https://powerdeng.tistory.com/135

---

### 5. np.bincount()
* `list 안에서 제일 큰 값` + `1`이 배열의 크기이다.
* 해당 배열은 0 부터 위에서 구한 최대값 + 1 까지의 값들의 갯수를 해당 index 에 저장한다.

**예제**
```python
import numpy as np

given_array = [3,2,2,6,7,4,8,9,9,9]
answer = np.bincount(given_array)
print(answer)
```

**출력**
```
[0 0 2 1 1 0 1 1 1 3]
```

REF : 
* https://nurilee.com/2020/05/10/bincount-%EB%9E%80/

---

### 6. np.concatenate()
* 매개변수로 여러개의 list 를 입력받는다. 
* 이러한 배열들을 axis 기준으로 병합해 준다.

**예제**
```python
A2=np.array([ [1,2,3],[10,20,30] ])
B2=np.array([ [4,5,6],[40,50,60] ])

print(np.concatenate((A2,B2),axis=0))
print(np.concatenate((A2,B2),axis=1))
```

**출력**
```
array([[ 1,  2,  3],
       [10, 20, 30],
       [ 4,  5,  6],
       [40, 50, 60]])
array([[ 1,  2,  3,  4,  5,  6],
       [10, 20, 30, 40, 50, 60]])
```

---

### 7. np.compress()

```
numpy.compress(condition, array, axis=None, out=None)
```

- `condition`: 선택 조건을 나타내는 불리언 배열이나 조건식입니다.
- `array`: 조건에 따라 선택될 배열입니다.
- `axis` (선택적): 선택 조건을 적용할 축을 지정하는 옵션입니다. 기본값은 `None`으로 전체 배열에 대해 조건을 적용합니다.
- `out` (선택적): 결과를 저장할 출력 배열입니다. 기본값은 `None`으로 결과를 새로운 배열로 반환합니다.
**Example**
```python
import numpy as np

# 입력 배열
arr = np.arange(1, 11)
print("입력 배열:", arr)

# 홀수값 필터링
filtered_arr = np.compress(arr % 2 != 0, arr)
print("홀수값 필터링 결과:", filtered_arr)
```

**output**
```
입력 배열: [ 1  2  3  4  5  6  7  8  9 10]
홀수값 필터링 결과: [1 3 5 7 9]
```