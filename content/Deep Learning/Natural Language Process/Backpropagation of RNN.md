#Deep_Learning 

RNN 의 Backpropagation 은 일반적인 NN 에 비해 복잡하지만 그렇다고 그 개념을 크게 벗어나진 않는다.
하지만 RNN Backpropagation 을 볼 때마다 헷갈려서 이렇게 정리를 한다.
또한 아래 예시는 CS231n Lecture 8. RNN part 에서의 그림과 일치 시키기 위한 작업도 병행하였다.


![AltText|500](_media-sync_resources/20240417T162533/20240417T162533_49160.png)


![AltText|500](_media-sync_resources/20240417T162533/20240417T162533_53344.png)

* 위의 2개의 그림은 매우 다르게 보이지만 아래 그림처럼 결국은 같은 그림이다.
* 아래 그림은 위의 2개의 그림에 같은 기능들을 하는 것들을 서로 연결시켜 둔 것이다.
* 설명은 첫번째 그림으로 진행하겠지만 CS231n 을 듣는 사람들을 위해 이러한 작업을 하였다.

![](_media-sync_resources/20240417T162533/20240417T162533_98475.png)


아래 그림을 보면 이해가 쉬울 것
![](_media-sync_resources/20240417T162533/20240417T162533_06565.png)
