---
tistoryBlogName: jeahun10717
tistoryTitle: "[Docker] Ubuntu Server Docker Container - 외부 pc vscode 연결"
tistoryVisibility: "0"
tistoryCategory: "1151035"
tistorySkipModal: true
tistoryPostId: "82"
tistoryPostUrl: https://jeahun10717.tistory.com/82
---
* 최근 연구실에서 서버를 하나 받게 되었다. 원래는 아래 방식으로 원격 연결을 했었다.
1. ubuntu server 안에 docker container 생성
2. docker container 안에서 anaconda, jupyter 설치후 jupyter lab 개방
3. ubuntu server 와 docker container 간 연결
4. 이후 외부에서 ubuntu sever 의 docker container 와 연결된 특정 port 에 접근(jupyer lab 으로 연결)

> 위의 내용은 [이 링크](https://jeahun10717.tistory.com/44)를 참고하라

* 그런데 최근 프로젝트를 진행하며 Python 에서 디버깅을 해야 하는 상황이 많았는데 jupyter lab 에서의 디버깅이 너무 불편하다고 느꼈다.
* 이에 외부 pc(본인은 mac os) 의 vscode 와 ubuntu server 안의 docker container 를 연결하는 방법을 정리할 것이다.

---

## 0. Overview

![[img_store/Pasted image 20240215132347.png]]

* 연결과정은 아래와 같다.
* mac vscode -> ubuntu server -> docker container
* 나는 mac 을 사용하므로 아래 글의 pc 는 mac 을 의미한다.

## 1. Server 설정

### 1.1. port 외부 개방 설정
* Server 는 내가 pc 에서 접근하고자 하는 port 를 우선적으로 개방해야한다.
* port 개방과 관련해서는 아래 링크를 참고하라

### 1.2. docker 설치 및 container 구동
* Docker 설치 : https://jeahun10717.tistory.com/42
* Docker 명령어 : https://jeahun10717.tistory.com/43
* Docker 외부 연결(jupyter) : https://jeahun10717.tistory.com/44 / (이 링크는 참고만 하면 됨)
## 2. PC 설정

* VScode 는 기본적으로 깔려 있어야 한다. VScode 에 대한 설명은 생략한다.
### 2.1. VScode - remote ssh 

* 이 부분은 아래 링크를 참고하라.
* https://dev-taerin.tistory.com/16

## 3. 연결과정

### 3.1. remote ssh 로 ubuntu 환경으로 연결
* vscode 실행후 `f1` 입력
* `Remote-SSH: 호스트에 연결` 클릭
![[img_store/Pasted image 20240215133810.png]]
* 연결하고자 하는 server 선택
![[img_store/Pasted image 20240215133741.png]]
* 정상적으로 진입하면 왼쪽 아래처럼 `ssh: [서버 이름]` 형식으로 출력됨
![[img_store/Pasted image 20240215134044.png]]

### 3.2. Server 에 Docker extension 설치
![[img_store/Pasted image 20240215134238.png]]
* 성공적으로 설치되면 왼쪽 사이드바에 docker 그림이 생성된다.
![[img_store/스크린샷 2024-02-15 오후 1.43.16.png]]

### 3.3. Server 에 Remote Developer extention 설치
![[img_store/Pasted image 20240215134839.png]]

### 3.4. Ubuntu Server 와 Dokcer Container 연결
* Remote Developer icon 클릭
![[img_store/Pasted image 20240215134958.png]]
* 원격 탐색기에 개발 컨테이너 탭으로 변경
![[img_store/Pasted image 20240215135047.png]]
* 정상적으로 컨테이너와 연결되면 아래와 같은 창이 뜬다
![[img_store/Pasted image 20240215140233.png]]