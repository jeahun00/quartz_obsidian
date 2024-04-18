#Deep_Learning

# Census Loss (Census Transform Loss)

target pixel 의 주변영역의 밝기 변화를 target pixel 보다 밝으면 0 어두우면 1 로 인코딩 하고 그 결과를 bit string 으로 추출한 값

Census Loss 는 위의 과정을 통해 구해진 값을 Hamming Distance 를 이용하여 그 차이를 구한 것이다.([[Hamming Distance]])

![](_media-sync_resources/20240417T174216/20240417T174216_32109.png)
