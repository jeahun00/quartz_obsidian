---
tags:
  - Deep_Learning
  - Super_Resolution
  - "#Programming"
  - "#Docker"
  - "#mmagic"
---
* 최근 학교 project 와 관련하여 BasicVSR++ 코드를 구동시켜 봐야 했다.
* 코드를 실행하는 과정에서 상당히 많은 시행착오를 거쳤고 기존의 코드를 구동하는 방식과 조금 달라 기록으로 남기려고 한다.

# 환경
* OS: ubuntu 20.04
* python env: docker with torch
	* [docker image-pytorch/pytorch cuda11.3](https://hub.docker.com/r/pytorch/pytorch/tags?page=&page_size=&ordering=&name=11.3)
```
docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
```

# 환경 설정
## 1. docker 환경 설정
```bash
# docker 이미지 가져오기
$ docker pull pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime

# docker container 생성
$ docker container run -it -d -p 7777:8888 --gpus all --shm-size=128G --name BasicVSRpp fa50f7fed43a /bin/bash
```
* docker run 의 flag 에 대한 설명은 [이 링크](https://jeahun10717.tistory.com/43) 참고
* 위의 `fa50f7fed43a` 는 docker image uid
## 2. mmagic 환경 설정
* 우선 생성된 docker container 에 접근
```bash
$ docker exec -it BasicVSRpp /bin/bash
```

* git 설치
* git 이 설치되어 있다면 이 과정은 생략해도 됨
```bash
$ apt update
$ apt install git
```

* docker container 에 접근 한 후 BasicVSR++ git repository clone 진행
```bash
$ git clone https://github.com/ckkelvinchan/BasicVSR_PlusPlus
```

### 관련된 mmagic 동작을 위한 라이브러리 설치
1. openmim 설치(mim 동작을 위한 것)
```bash
$ pip install openmim
```

2. mmcv 설치
	* 주의! : 
		* 공식 깃 문서에서는 `mim install mmcv-full` 로 설치한다.
		* 이 경우 mmcv 가 2.0.x 이상 버전이 깔리게 된다.
		* 이 repository 에서는 mmcv 가 1.3.x >=, 1.6.0<= 버전만 호환된다.
		* 따라서 이 버전에 맞는 mmcv 버전을 설치하여야 한다.
```bash
$ pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.12.0/index.html
```

3. repo 에 접근
```bash
$ cd BasicVSR_PlusPlus
```

4. requirement lib 설치
```bash
$ pip install -v -e .
```



# 3. inference code 동작확인

1. 우선 BasicVSR++ 의 checkpoint 를 설치한 후 chkpts 폴더를 만들어 그 안에 해당 checkpoint 를 옮긴다.
	* [checkpoint link](https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth)
```bash
$ mkdir chkpts
$ cd chkpts

# curl 이 설치되어 있다면 아래는 생략 가능
$ apt install wget

# checkpoint 설치
$ wget https://download.openmmlab.com/mmediting/restorers/basicvsr_plusplus/basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth

# repo 의 예시에 맞게 checkpoint 이름 변경
$ mv basicvsr_plusplus_c64n7_8x1_600k_reds4_20210217-db622b2f.pth basicvsr_plusplus_reds4.pth
```

2. runs inference code
```bash
$ python demo/restoration_video_demo.py configs/basicvsr_plusplus_reds4.py chkpts/basicvsr_plusplus_reds4.pth data/demo_000 results/demo_000
```
* 만약 위의 명령어를 실행했는데 아래 에러가 발생하면 이는 video 나 image 의 입출력을 관리하는 ffmpeg 이 설치되어 있지 않기 때문이다.
```bash
Traceback (most recent call last):
  File "demo/restoration_video_demo.py", line 5, in <module>
    import cv2
  File "/opt/conda/lib/python3.7/site-packages/cv2/__init__.py", line 8, in <module>
    from .cv2 import *
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
* 이는 ffmpeg 을 설치하면 간단하게 해결된다.
```bash
$ apt update
$ apt install ffmpeg
```