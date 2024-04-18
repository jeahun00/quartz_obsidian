#Math 

 stochastic process(확률 과정)는 시간에 따라 무작위로 변하는 값의 시퀀스. 
		1. ex> 주사위 던지기
		2. State Space(상태 공간) : 확률 과정이 취할 수 있는 모든 상태의 집합
			1. ex> 주사위 던지기에서 {1,2,3,4,5,6}
		3. index Set(인덱스 집합) : 보통은 시간을 나타낸다. 확률 과정의 변화를 추정할 때 사용된다.
			1. ex> 주사위 던지기에서 n 번째 시행에서의 n
			4. Probability Distribution(확률 분포) : 상태공간과 인덱스 집합에 기반하여 확률 과정의 확률 분포를 정의


docker container run -it -d -p 55555:8888 --gpus all --shm-size=128G --name dockerTest 4157de9bccb1 /bin/bash