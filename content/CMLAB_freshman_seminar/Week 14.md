# Assignment 

* torch.device, os.environment 로 gpu 설정 및 조정 하는 방법 조사

# 오교수님
### 1. FMA-Net: Flow-Guided Dynamic Filtering and Iterative Feature Refinement with Multi-Attention for Joint Video Super-Resolution and Deblurring

기존 연구 한계점 : 
* HOFFR -> 2D CNN 을 이용하는데 spatial information(공간 정보) 처리가 어려움
* 아래 논문에 따르면 large size filter 필요하다.
![[../../../img_store/Pasted image 20240202161218.png]]

* 본 저자는 small size filter 로도 처리가 가능하다.
### 2. DyBluRF: Dynamic Deblurring Neural Radiance Fields for Blurry Monocular Video,



# 엄교수님: 
### 1. Disentangled Representations for Short-Term and Long-Term Re-Identification

### 2. Bi-directional Contrastive Learning for Domain Adaptive Semantic Segmentation


1. REF : https://sukzoon1234.tistory.com/72
2. TODO : 
	1. Pseudo labeling : https://lv99.tistory.com/79
	2. Domain Adaption : https://lucastorming.tistory.com/5
	3. Unsupervised Domain Adaption : https://zooyeonii.tistory.com/21
	4. Prototypical learining : https://rhcsky.tistory.com/9
	5. contrastive learning : https://www.blossominkyung.com/deeplearning/contrastive-learning



---

* 발표자료를 만들 때 motivation 작성할 때 확실하게 언급
* 또한 선행연구를 설명할 땐 아래에 작게 인용문구 넣을 것
* 논문에서 해결하고자 하는 task 에 대한 설명도 포함해야 함(ex: sr, semantic segmentation)
* 한 슬라이드 안에 너무 많은 글이 들어가지 않도록. 최대한 이 슬라이드를 읽는 사람이 확실하게 이해할 수 있도록 그림이나 시각자료도 첨부할 것
* Related work 는 <mark style='background:#fa8231'>기존의 방법론에 대한 문제점 제시</mark>. 이 문제를 <mark style='background:#fa8231'>본 논문에서는 어떻게 해결했는지</mark>를 반드시 포함할 것
* proposed method 에서 notation 에 대한 설명을 할 것
* figure 를 사용할 때 집중하고자 하는 부분만 선명하게 나머지는 숨겨서 삽입할 것
* 평가지표에 대한 설명도 간단히 포함할 것
* figure 를 가져올 때 그림은 그대로 놔두되 그 그림에 대한 설명은 ppt 글씨체로 맞춰 다시 삽입할 것

