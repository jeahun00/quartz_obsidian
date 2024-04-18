#Deep_Learning #Programming 

딥러닝을 공부하던 중 `pytorch` 로 구현을 하고 있는데 `torch.nn.Linear` 라는 메소드를 아무런 이해 없이 사용을 해 왔다는 것을 깨달았다.
이를 이해하기 위해 간랸한 노트를 남긴다.

### 1. nn.Linear 란
---
신경망의 맥락에서 nn.Linear 는 선형 변환을 의미한다.
즉 아래 식과 같은 형태를 의미한다.
$$y=XW^{T}+ b$$
`X` : 입력값(행렬일수도 있고 단일값일수도 있다.)
`W` : 가중치(선형레이어에서의 기울기)
`b` : 편향
`y` : 출력


### 2. 사용법
---
```python
import torchfrom torch import nn 
linear_layer = nn.Linear(in_features=3, out_features=1)
```

`in_feature` : 입력노드의 수
`out_feature` : 출력노드의 수
주의 : <mark style='background:#eb3b5a'>입력, 출력노드는 가중치가 아니다</mark>. 가중치는 입력노드 수 * 출력노드 수 이다.
아래 사진을 보면 Previous Layer 가 위에서 설명한 입력노드 수 이고, Fully-Connected Layer 부분이 위에서 말한 출력 레이어의 수이다.
아래 사진에서 간선으로 표현된 부분이 Weight 이다.
즉 Weight Matrix 의 크기는 입력레이어 노드 * 출력레이어 노드 = 7 * 5 = 35 이다.
![200](_media-sync_resources/20240417T162528/20240417T162528_26093.png)


---
### Ref :
https://ecoagi.ai/ko/topics/Python/nn-linear