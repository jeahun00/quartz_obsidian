# Abstract

- 비디오 NeRF는 딥러닝 기술을 사용하여 비디오 프레임에서 3D 공간을 재구성하는 기술
- *Neural Radiance Field(NeRF) : 3차원 장면을 모델링하는 방법. 씬을 표현하기 위해 사용되는 딥러닝 모델. 입력으로 점(x,y,z)와 방향을 받아 해당 점에서 바라보는 장면의 색상과 반사도를 예측한다.
- Dynamic Deblurring : 블러링된 영상을 선명하게 복원하는 기술. DyBluRF에서는 NeRF 모델을 기반으로 동적인 장면을 복원하기 위해 다음의 방법을 사용한다.
	- 블러링된 영상에서 카메라의 위치와 방향을 추정한다.
	- 추정된 카메라 정보를 이용하여 NeRF 모델을 학습한다.
	- 학습된 NeRF 모델을 이용하여 블러링된 영상을 선명하게 복원한다.

초기에 NeRF는 static scenes를 위해 개발되었다.
이후 이를 기반으로 **video view synthesis** 기법들이 활발히 개발되었다.
비디오의 경우, 객체나 카메라의 움직임으로 인한 motion blur 문제가 발생한다. 이로인해 sharp spatio-temporal views(선명한 시공간 뷰)를 합성하는 것에 어려움이 있다.
따라서 본 논문은 a novel dynamic deblurring NeRF framework for blurry monocular video, DyBluRF 를 제안한다. 

대략적인 모델의 구성은 다음과 같다.

1. **Interleave Ray Refinement (IRR) stage**
	다음 두 가지 작업을 동시에 진행한다.
	- reconstruct dynamic 3D scenes
	- 주어진 blurry 프레임에서 추출된 incorrect 카메라 포즈 정보를 refine해서 정확도를 높인다.
2. **Motion Decomposition-based Deblurring (MDD) stage**
	a novel incremental latent sharp-rays prediction (ILSP) approach for the blurry monocular video frames by **decomposing the latent sharp rays into global camera motion and local object motion components.**
		흐릿한 단안 비디오 프레임을 위한 새로운 점진적 잠재적인 선명한 광선 예측(ILSP) 방법으로
		**latent sharp ray를 global camera motion과 local object motion 컴포넌트로 분해한다.**

DyBluRF는 blurry 단안 비디오에 대한 새로운 뷰 합성 문제를 해결하는 첫 번째 모델로 기존 sota를 능가한다.

# 1. Introduction
static scene을 위한 NeRF가 처음으로 등장한 이후 비디오 뷰 합성을 위한, 즉 dynamic scenes를 위한  다양한 유형의 neural rendering method가 활발히 개발되었다.

*(왜 단안 비디오를 사용하게 되었는지)*
*'Dynamic video'* 의 *'free viewpoint rendering'* 을 위해 주로 multi-view-based 방법을 사용해왔다. 하지만 이 방법은 동기화된 캡쳐 방식이 필요한데 이것은 일반 사용자들에게는 impractical 하다. 따라서 casually captured ==monocular video==에 대한 dynamic view synthesis 를 진행한다. 

한편, 비디오는 촬영 과정에서 빛의 축적 *(셔터가 열려있는 동안)* 에 의한 물체의 움직임이나 카메라 흔들림의 결과로 ==motion blur==가 발생하게 된다. 이러한, 프레임에 blur가 있는 단안 비디오에서 *'sharp spatio-temporal views'* 를 합성하는 것은 여러 어려움이 있다.

(i) Apply 2D video deblurring as a preprocessing step
직관적인 해결책으로 optimizing video NeRF 이전에 2D video deblurring을 전처리 단계로 적용하는 방법이 있다. 하지만 이렇게 픽셀 도메인에서 독립적으로 프레임을 디블러링하면 일관성없는 geometry(기하학적 구조)가 발생할 수 있다. 그리고 이것은 이후 단계인 video NeRF 최적화를 통해 복구할 수 없다.

(ii) 기존의 NeRFs 
- Deblurring NeRFs for **static multi-view images**
	- 시간 차원에 대한 motion-aware deblurring module이 없어서 시간 정보를 캡처하지 못한다.
- Existing SOTA monocular video NeRFs
	- 디블러링 컴포넌트가 없어서 흐릿한 프레임을 처리할 수 없다.
	

(iii) 기존 알고리즘의 한계
deformable 객체를 포함하는 blurry 단안 비디오로부터 *'SfM(Structure-from-Motion)'* 알고리즘으로 추출한 카메라 포즈의 정확도가 낮기 때문에 detecting and matching salient keypoints에 어려움이 있다. 
> *'SfM(Structure-from-Motion)'* : 2D 이미지에서 특징점을 추출하고, 이를 이용해 3D 공간에서의 객체의 위치와 방향을 추정하는 알고리즘. 비디오 프레임에서 객체의 위치와 방향을 추정하는 데 사용되지만, 블러 현상으로 인해 정확도가 떨어질 수 있다. 특히 변형 가능한 객체가 있는 경우에는 더욱 어렵다.
> 
> 카메라 포즈가 부정확하다가 무슨 말 : 카메라가 촬영한 이미지를 3D 공간에 매핑하기 위해서는 카메라 포즈, 즉 각도와 위치가 정확해야 한다. 카메라의 흔들림, 객체의 움직임, 조명의 변화 등이 카메라 포즈를 부정확하게 만드는 요인이다.


이 어려움들을 해결하는 DyBluRF를 제안한다.
- can render the sharp novel spatio-temporal views from blurry monocular videos with imprecise camera poses    
- **IRR stage**에서 dynamic 3D scenes의 재구성과 카메라 포즈의 refinement를 동시에 수행해서 부정확한 카메라 포즈 정보를 극복한다.
- **MDD stage**에서 latent sharp-rays prediction(ILSP)를 통해 단안 비디오에서 global camera와 local object motions로 인한 blurring을 효과적으로 처리한다.
- 실험을 위해, 흐릿한 버전의 iPhone 데이터 세트를 합성했다. 흐릿한 데이터 세트를 사용하여 훈련된 DyBluRF는 정확한 카메라 포즈 정보로 선명한 데이터 세트에서 훈련된 SOTA 방법과 비슷한 결과를 보여준다.

# 2. Related Work

## 2.1. Conventional Video Deblurring
*(기존 deblurring method의 한계)*
모션 블러는 비디오 촬영 중에 빠르게 움직이는 물체나 카메라의 흔들림으로 인해 발생하는데, 이는 빛이 센서나 필름에 셔터가 열려 있는 동안 지속적으로 모이기 때문이다. 비디오 디블러링을 위한 다양한 딥러닝 메소드가 개발되었지만 기존 비디오 디블러링을 비디오 NeRF 최적화 이전에 전처리 단계에 적용하면 ==inconsisntent geometry in 3D space를 야기한다. 이것은 비디오 NeRF 최적화로 correct할 수 없다.==
`따라서, 비디오 디블러링과 비디오 NeRF를 함께 사용할 때는 이러한 문제를 고려해야 한다.`

## 2.2. Deblurring NeRFs for Static Scenes
*(기존 static scene을 위해 개발된 NeRF기반 방법들의 한계, 시간 차원을 고려하지 못함)*
blurry한 multi-view static images에서 **일관된 3D geometry를 갖는** visually appealing한 프레임을 생성하기 위해 여러 NeRF기반 방법이 등장했다. DeblurNeRF는 픽셀 수준에서 spatial blur kernel과 latent sharp radiance field를 추정하기 위해 end-to-end 볼륨 렌더링 프레임워크를 사용한다. BAD-NeRF는 이미지 노출 촬영 시간 동안 가상 카메라 궤적을 공동으로 예측한다. DP-NeRF는 물리적 제약 조건을 활용하여 3D 일관성을 유지하기 위해 rigid blurring kernel을 도입한다. ExBluRF는 6-DOF 카메라 포즈의 차원을 줄이기 위한 MLP 기반 프레임워크를 소개하고, 복셀 기반 방사형 필드를 사용한다. ==그러나 위의 방법들 중 어느 것도 시간적 차원에 대한 motion-aware deblurring이 없기 때문에 *'non-rigid'* 비디오 뷰 합성에 적용할 수 없다.==


## 2.3. NeRFs for Dynamic Scenes
*(기존 dynamic scene을 위해 개발된 NeRF기반 방법들의 한계, deblurring component가 없고 포즈 최적화를 하지 못함)*
최근의 비디오 뷰 합성은 static NeRF 프레임워크를 기반으로 확장되었다. 이들은 non-rigid deformable 변환이나 4D 시공간 방사 현장을 모델링하기 위해 씬 플로우 기반 프레임워크 또는 정규 필드를 통합하여 동적 NeRF를 나타낸다. NSFF, DynamicNeRF, DynIBaR은 주로 단안 비디오를 위한 새로운 시공간적인 뷰를 생성하기 위해 두 종류의 NeRF를 결합한다 : time-invariant NeRF and time-variant NeRF. 그러나 이 방법들은 움직이는 물체에 대한 사전 학습된 모션 마스크 추출과 3D scene flows에 대한 다양한 정규화 손실에 크게 의존하기 때문에 비디오 뷰 합성의 디블러링 효과가 떨어진다. D-NeRF, HyperNeRF, TiNeuVox같은 방법은 초기에 관측 공간의 광선을 표준 공간의 구부러진 광선으로 변환하는 변형 또는 오프셋 필드를 학습합니다. ==그러나 위에서 언급한 기존의 SOTA 단안 비디오 NeRF 방법 중 어느 것도 블러링으로 인해 손상된 포즈 정보를 효과적으로 복원할 수 있는 디블러링 요소와 강력한 포즈 최적화가 부족하다.==

## 2.4. NeRFs with Pose Optimization
*(기존 pose optimization의 한계)*
디테일한 것들은 잘 캡쳐하기 위해 캡쳐과정에서 NeRF는 정확한 카메라 포즈, 고정 장면이 필요하다. 하지만 **SfM** 알고리즘으로 얻은 카메라 포즈에는 본질적으로 픽셀 수준의 부정확성이 포함되어 있다. 이러한 부정확성은 ==특히 글로벌 카메라와 로컬 객체의 움직임이 모두 있는 경우 고유한 키포인트를 감지하고 일치시키는 데 어려움이 있기 때문에 흐릿한 프레임을 처리할 때 더욱 악화될 수 있다.== 결과적으로 렌더링된 NeRF의 출력은 ground truth 이미지와 상당한 오차를 보인다. NeRF 파라미터와 포즈 정보를 공동으로 최적화하기 위해 여러 NeRF 방법들이 개발되었다. 그러나 이런 방법은 주로 rigid 특성이 있는 시나리오에 집중되어 있으며, blurry 단안 비디오에서 자주 관찰되는 non-rigid 특성과는 상당히 다르다.

# 3. Proposed Method: DyBluRF

## 3.1. Design Considerations

제안하는 DyBluRF는 'blurry' monocular video에서 'sharp' dynamic neural radiance field를 표현하는 것이 목표.
DyBluRF는 Interleave Ray Refinement (IRR) stage 와 Motion Decomposition-based Deblurring (MDD) stage로 구성된다.

**[IRR stage]**
: dynamic radiance field를 ==대략적으로== 재구성하고 부정확한 카메라 포즈를 최적화한다.

- Dynamic NeRF Rendering
	- radiance field를 static net과 dynamic net으로 분해한다.
	separately estimate the rendered colors of static scene component and dynamic scene component via continuous volume rendering
	
	- Static Net
		- use predicted binary motion mask to prevent learning the dynamic scene component and make the optimization stable
	- Dynamic Net
	
	 rendered colors of static scene component and dynamic scene component의 output을 DyBluRF가 volume density blending factor로 통합해서 full rendered color를 얻는다.
	
- Ray Refinement
	refine imprecise camera pose via learnable embedding for screw axis

- interleave optimization 전략을 사용한다.
	- ray refinement와 static&dynamic net optimization을 번갈아 진행한다.
	-  DyBluRF의 학습 안정성을 향상시키는 동시에 3D dynamic reconstruction을 학습하고 camera pose를 개선한다.

**[MDD stage]**
글로벌 카메라 모션과 로컬 객체 모션을 고려한 물리적 블러 프로세스를 시간 축을 따라 점진적인 방식으로 효과적으로 합성하는 새로운 incremental latent sharp-rays prediction(**ILSP**) 방법을 사용한다.

## 3.2. Preliminaries

**Dynamic Neural Radiance Fields.**

DyBluRF는 동적 환경에서 움직이는 물체를 포함하는 장면을 재구성하기 위해 static NeRF를 확장한 모델이다. DyBluRF는 NeRF 모델을 기반으로 동적인 장면을 재구성한다.
기존 static NeRF모델은 static scene 재구성을 위해 사용되지만, DyBluRF 모델은 시간에 따라 변하는 장면을 재구성하기 위해, 시간 t당 한 프레임으로 구성된 단안 비디오의 프레임들을 사용한다.
DyBluRF는 신경망을 사용하여 비디오 장면의 연속적인 래디언스를 표현하는 것을 학습한다.
이때 단안 비디오의 $N_f$ 개의 프레임 집합을 $\{I_t\}^{N_f}_{t=1}$, 해당 카메라 포즈는 $\{P_t\}^{N_f}_{t=1}$로 표기한다.
래디언스 표현을 Static Net $F_{\theta^s}$과 Dynamic Net $F_{\theta^d}$, 2개의 신경망으로 나눠서 표현한다. 


- Static Net $F_{\theta^s}$
	공간 위치 정보 인코딩을 입력으로 받아 ==static scene component의 색==, 부피 밀도, 부피밀도 블렌딩 요소 b를 추정한다.
$$F_{\theta^s} : \gamma(X, d) \rightarrow (c^s, \sigma^s, b) $$
	-  $\gamma$로 spatial positional encoding
	- $c^s$ : color
	- $\sigma^s$ : volume density of static scene component
	- **$b$ : volume density blending factor**

- Dynamic Net $F_{\theta^d}$
	시간에 따라 변하는 임베딩을 입력으로 받아 ==dynamic scene component의 색==과 볼륨 밀도에 매핑한다.
$$F_{\theta^d} : (\gamma(X, d), l(t)) \rightarrow (c^d, \sigma^d) $$
	- $\gamma$ : positional encoding
	- $l$ : $t$를 인코딩하는 Generative Latent Optimization (GLO) 
	- $c^d$ : color
	- $\sigma^d$ : volume density of dynamic scene component


$$r_{p;t}(s) = o_t + sd_{p;t}$$
- $r_{p;t}$ (casted ray from o_t with d_p;t) : 카메라 원점 $o_t$에서 주어진 영상 평면의 픽셀 $p$를 통과하는 ray
- s : A sampling ray distance
- $d_{p;t}$ : A viewing direction of a ray of pixel p at time t, 시간t에서 픽셀p를 통과하는 ray의 방향

광선 $r_{p;t}$를 따라 N개의 불연속적인 세그먼트 ${[s_n, s_{n+1}]}^N_{n=1}$에 대해 적분을 계산하여 $C^s(r_{p;t})$,  $C^d(r_{p;t})$ 즉, ==static scene component와 dynamic scene component의 렌더링된 색==상을 각각 추정한다:
![[Pasted image 20240207203839.png]]
Tn은 누적 투과율, αn은 alpha-compositing 가중치로 다음과 같이 정의된다:
![[Pasted image 20240207203908.png]]

$δ_n = s_{n+1} - s_n$은 세그먼트 길이. 카메라 포즈 $P_t$를 가진 픽셀 p의 전체 렌더링 색상 $C^{full}(r_{p;t})$을 예측하기 위해 DyBluRF는 볼륨 밀도 블렌딩 계수 b로 $F_{θ_s}$와 $F_{θ_d}$의 출력을 결합한다:
![[Pasted image 20240207204151.png]]

**Binary Motion Mask Prediction.** 
*'motion decomposition'* 학습은 dynamic NeRF에서 ==static scene component의 재구성을 안정화하기 위해== 이전 논문들에서 많이 채택된 방법이다. 

DyBluRF에서 모션 분해는 IRR 단계와 MDD 단계에서 모두 필수적이다. dynamic scene component의 알파컴포지팅 가중치 $α^d_n$를 $r_{p;t}$를 따라 $(1 - b_n)$으로 누적하여 motion uncertainty $M_{uncert}(r_{p;t})$를 다음과 같이 계산한다:
![[Pasted image 20240207204440.png]]

**Deblurring Neural Radiance Fields.** 
선명한 래디언스 필드를 재구성할 때 블러 문제를 해결하기 위해, 픽셀 단위의 블러 커널과 선명한 픽셀 색상을 예측하여 기존의 디블러링 static NeRF 방법과 유사한 물리적 블러 프로세스를 시뮬레이션한다. 이 물리적 블러 프로세스는 시간 t에서 픽셀 p의 흐릿한 색상 $B_{p;t}$를 생성하는 과정으로, 선명한 픽셀 색상 $I_{p;t}$에 unknown 모션 블러 커널 $k_{p;t}$를 적용하여 형성된다.
![[Pasted image 20240207205117.png]]
여기서 ∗는 컨볼루션 연산. 

주어진 흐릿한 단안 비디오 프레임 ${B_t}^{N_f}_{t=1}$과 부정확한 카메라 포즈 ${P_t}^{N_f}_{t=1}$로 DyBluRF를 학습시킨다. 블러 프레임을 사용하여 DyBluRF를 최적화하기 위해, 목표 광선 $r_{p;t}$를 기반으로 캐스팅된 latent sharp rays ${ \dot{r}_{p;t;q}}^{N_b}_{q=1}$집합을 예측하여 monocular dynamic radiance fields에 대한 블러 프로세스를 모델링한다. 그런 다음 해당 광선에 대한 볼륨 렌더링된 픽셀 색상을 평균내어 블러 픽셀 색상을 생성한다. 이때 q는 인덱스, $N_b$는 잠재적인 선명한 광선의 수를 나타낸다. 이 블러 프로세스는 다음과 같다:
![[Pasted image 20240207205644.png]]
- $\hat{C}_B(r_{p;t})$ : ray $r_{p;t}$의 블러 픽셀 색상
- A(·, ·) : ray $r_{p;t}$의 렌더링 색 $\hat{C}(r_{p;t})$와 latent sharp rays  ${r_{p;t;q}}^{N_b}_{q=1}$의 렌더링 색 집합 ${\hat{C}(\dot{r}_{p;t;q})}^{N_b}_{q=1}$의 평균 함수

>(블러과정정리)
>- ray casting : 목표 광선 $r_{p;t}$를 기반으로 latent sharp rays ${ \dot{r}_{p;t;q}}^{N_b}_{q=1}$ 를 cast
>- ray tracing : 각 latent sharp rays를 추적해서 해당 위치에서의 래디언스 값 계산
>- 블러 픽셀 색상 생성 : 각 latent sharp rays에서 계산된 래디언스 값을 평균내어 블러 픽셀 색상을 생성

## 3.3. Interleave Ray Refinement Stage
### 3.3.1 Ray Refinement
>*(카메라의 부정확한 포즈 추정치로 인해 발생하는 광선의 왜곡을 보정한다)*

${\tilde{r}_p;t}$가 시간 t 에서 부정확한 카메라 포즈 $P_t$로 픽셀 p에서 촬영된 부정확한 ray라고 가정했을 때, ${\tilde{r}_p;t}$를 ray wraping으로 더 정확한 광선 $\hat{r}_{p;t}$ 로 개선한다:
![[Pasted image 20240207212340.png]]
### 3.3.2 Interleave Optimization
dynamic radiance field와 카메라 포즈를 동시에 최적화하는 것은 위험성이 있고 로컬 미니마에 빠질 수 있다. 이를 위해 다양한 loss로 ray refinement와 static 및 dynamic net을 번갈아 최적화하는 **interleave optimization** 전략을 사용한다. 다양한 loss들은 다음과 같다.

**Photometric Loss.** 광학적 손실
DyBluRF 아키텍처를 안정적으로 최적화하기 위해 렌더링된 색상에 대한 모델의 photometric loss를 최소화한다. 입력 ray $r_{p;t}$ (=$\hat{r}_{p;t}$ or $\dot{r}_{p;t;q}$)가 주어지면 **dynamic scene component**의 색상 $\hat{C}^d(r_{p;t})$와 전체 렌더링 색상 $\hat{C}^{full}(r_{p;t})$을 렌더링한다. 그런 다음, 렌더링된 각 렌더링된 색상 $\hat{C}(r_{p;t})$와 블러 처리된 GT 색상 $B_{p;t}$의 L2 손실을 다음과 같이 계산한다:
`이 광학적 손실은 모델이 렌더링한 색상과 실제 장면의 색상 간의 차이를 측정한다. 이 손실을 최소화함으로써, 모델은 실제 장면과 더 유사한 색상을 생성하도록 학습된다.`
![[Pasted image 20240207212850.png]]

여기서 $\hat{C}(r_{p;t})$는 $\hat{C}^d(r_{p;t})$ 또는 $\hat{C}^{full}(r_{p;t})$일 수 있다. **static scene component**의 렌더링 색상 $\hat{C}^s(r_{p;t})$의 경우, 앞서 예측한 $M(r_{p;t})$를 사용하여 dynamic scene component를 학습하지 못하도록 masked photometric loss를 적용한다:
![[Pasted image 20240207213301.png]]

**Surface Normal Loss.** 
DynIBaR은 사전 훈련된 단안 깊이 추정 네트워크를 적용하여 NeRF를 정규화했다. 스케일이 모호한 깊이 추정의 어려움을 완화하기 위해 DynIBaR은 사전 훈련된 monocular disparity를 SfM pointcloud와 정렬할 것을 제안했다. 그러나 모션 분할 priors 없이 흐릿한 비디오 프레임에서 SfM을 재구성하면 신뢰할 수 없는 결과가 나올 수도 있다. 그림 3에서 볼 수 있듯이, 우리는 DynIBaR과 달리 surface normal supervision을 채택하여 동적 네트워크 $F_{θ^d}$ 의 밀도 예측을 정규화한다. 1차 finite difference를 사용하여 사전 훈련된 깊이 추정 모델로부터 시간 t에서 픽셀 $p = (p_u, p_v)$의 실측 표면 법선 $n_{p;t}$를 계산한다. 그런 다음 예측된 표면 법선 ^$n_p;t$를 다음과 같이 계산한다:
![[Pasted image 20240207214106.png]]
>NeRF는 3D 공간에서 물체의 모양과 재질을 추정하는 기술로, 이 때 표면 법선 정보를 활용하여 정확도를 향상시킨다. 이를 위해 사전 학습된 깊이 추정 모델을 사용하여 기준이 되는 표면 법선을 계산하고, 이를 예측된 표면 법선과 비교하여 손실을 계산한다. 이 손실을 활용하여 NeRF 모델의 성능을 향상시킬 수 있다.

**Static Blending Loss.** 
움직이는 물체 영역을 감독하기 위해 사전 학습된 세그먼테이션 모델에 의존하는 기존의 동적 NeRF 방법과 달리, DyBluRF는 비지도 방식으로 바이너리 모션 마스크 예측을 최적화한다. 정적 네트워크와 동적 네트워크는 서로 다른 수렴 속도를 보이기 때문에 장면 분해는 종종 동적 구성 요소에 유리한 경향이 있다. 이 문제를 해결하기 위해 λsb에 가중치를 부여한 정적 블렌딩 손실($L_{sb} = -λ_{sb}\sum_{r_{p;t}} \log(b)$)을 도입하여 d블렌딩 계수 b가 정적 특성을 유지하도록 유도한다.

**Distortion Loss.** 
MipNeRF 360의 왜곡 손실 $L_{dist}$를 사용한다. 왜곡 손실은 정적, 동적 및 블렌딩 렌더링의 밀도 추정치에 적용되며, 각각 Ls dist, Ld dist 및 Lfull dist로 표시된다.

## 3.4. Motion Decomposition-based Deblurring Stage
MDD 단계에서는 흐릿한 단안 비디오 프레임에 대한 새로운 incremental latent sharp-rays prediction(ILSP) 접근 방식을 제안한다.
![[Pasted image 20240207214620.png]]
![[Pasted image 20240207214712.png]]
이 접근 방식은 global camera-motion-aware ray prediction and a local object-motion-aware ray refinement로 구성된다.

**Global Camera-motion-aware Ray Prediction (GCRP).** 
정적 및 동적 장면 구성 요소에서 모두에서 발생하는 카메라 모션 블러 과정을 모델링하기 위해 정제된 광선 $\hat{r}_{p;t}$를 기반으로 여러 개의 잠재적 선명한 광선 ${\dot{r}^g_{p;t;q}}^{N_b}_{q=1}$을 다음과 같이 추정한다:
![[Pasted image 20240207214944.png]]
$S^g_{t;q}$는 글로벌 카메라 모션 인식 스크류 축으로,
`전역 카메라 움직임을 인식하는 스크류 축(Screw Axis)`
q번째 latent sharp 광선 $\dot{r}^g_{p;t;q}$에 대한 t의 학습 가능한 임베딩이다. GCRP는 IRR의 출력인 정제된 광선  $\hat{r}_{p;t}$를 글로벌 카메라 모션을 고려하여 Nb 예측된 잠재적 선명한 광선 ${ ̇r^g_{p;t;q}}^{N_b}_{q=1}$에 매핑한다.


**Local Object-motion-aware Ray Refinement (LORR).**
잠재적인 선명한 광선을 추정하기 위해 단일 모션 인식 광선 예측, 즉 GCRP만 채택하면 모델은 글로벌 카메라 모션과 로컬 오브젝트 모션을 결합하여 다양한 모션의 외부 혼합을 학습하는 경향이 있다. 이로 인해 흐릿한 훈련 이미지에 잔상이 퍼지는 등 부자연스러운 아티팩트가 발생할 수 있다. 디테일한 motion을 처리하기 위해, 픽셀 단위의 로컬 객체 모션을 고려하여 q번째 예측된 잠재 선명 광선 ̇$r^g_{p;t;q}$를 개선하여 블러 처리된 ray를 글로벌 카메라 모션을 따라 로컬 객체 모션으로 분해한다:
![[Pasted image 20240207215809.png]]

여기서 Sl p;t;q = F θ(⌈φ(̇ rg p;t;q, l(t))는 로컬 객체 모션 MLP F θ에 의해 학습된 로컬 객체 모션 인식 스크류 축으로 이산화된 광선 임베딩 시간(̇ rg p;t;q)과 인코딩된 φ l(t)을 입력으로 사용합니다. ⌈·⌋은 채널별 연결을 나타냅니다. 로컬 객체 모션 인식 광선 개선(LOR)은 각각의 예측된 잠재 선명한 광선 ̇ rg p;t;q를 로컬 객체 모션(onto-one 매핑)을 고려하여 단일 대응하는 정제된 잠재 선명한 광선 ̇ rl p;t;q에 매핑합니다. 구체적으로, LORR은 Algo. 2의 12행과 같이 이진 모션 마스크 M(ˆ rp;t)(=1)으로 표시되는 동적 장면 구성 요소에만 적용됩니다. 흐릿한 색상 ˆ CB를 얻기 위해, 우리는 Algo. 2에 설명된 대로 정적 장면 구성 요소의 경우 Eq 14에서 또는 동적 장면 구성 요소의 경우 Eq 15에서 예측된 잠재 선명한 광선에 Eq 7을 적용합니다.
> LORR은 영상에서 객체의 움직임을 감지하고, 이를 기반으로 블러 처리된 영상을 선명하게 복원하는 데 사용된다. 이를 위해 GCRP라는 단일 모션 인식 레이 예측을 사용하며, 추가적으로 LORR이라는 기법을 사용하여 지역 객체의 움직임을 더욱 섬세하게 처리한다. 이를 통해 영상의 블러링을 효과적으로 제거하고, 선명한 영상을 얻을 수 있다. 

# 4. Experiments

**Blurry iPhone Dataset**.
DyBluRF를 평가하기 위해, 카메라 포즈가 부정확한 흐릿한 프레임만 주어졌을 때 새로운 뷰 합성을 위한 디블러링을 평가하기 위해  부정확한 카메라 정보로 새로운 Blurry iPhone Dataset를 합성한다. 이 데이터 셋에는 훈련 데이터 셋에 대한 부정확한 카메라 포즈 정보 ${\tilde{P}_t}^{Nf}_{t=1}$와 함께 합성된 흐린 RGB 비디오 프레임 ${B_t}^{N_f}_{t=1}$과 아이폰 데이터 세트에서 파생된 평가 데이터 세트에 대한 정확한 카메라 포즈 정보가 있는 원본 sharp RGB 비디오 프레임이 포함된다. 이러한 흐릿한 RGB 훈련 비디오 프레임을 생성하기 위해 먼저 VFI 메소드를 사용하여 원본 아이폰 데이터 세트에서 인접한 sharp 비디오 프레임의 각 쌍 사이에 7개의 중간 프레임(x8)을 보간한다. 그 후 8프레임 보폭으로 7개의 연속 프레임을 평균하여 이전 및 이후 순간의 원본 프레임과 3개의 보간 프레임을 통합한다. 이 과정은 각 흐린 RGB 프레임을 생성하며, 카메라 이동이 노출 시간 내에서 일정 속도로 일어난다고 가정한다. 각 흐린 프레임에 대한 해당 카메라 회전을 계산하기 위해 구형 선형 보간을 활용하여 각 보간 프레임에 대한 보간 회전 행렬을 생성한다. 그런 다음 이러한 행렬은 평균화된 RGB 프레임의 해당 쿼터니언의 평균을 위한 쿼터니언으로 변환된다. 그런 다음 결과 평균화된 쿼터니언은 회전 행렬로 재변환된다. 해당 카메라 변환을 계산하기 위해 원본 변환 벡터를 선형으로 보간하고 평균화된 변환 벡터를 평균화한다.

**Metrics.** 
단안 비디오 세팅에서 DyBluRF를 평가하고 다른 SOTA 방법과 비교하기 위해,  Dycheck에서 소개한 접근 방식에 따라 mPSNR, mSSIM 및 mLPIP를 포함한 공동 가시성 마스크 이미지 메트릭을 활용한다. 이러한 메트릭은 훈련 카메라에서 관찰되지 않는 테스트 비디오 프레임의 영역을 마스킹한다. 또한 재구성된 비디오 프레임의 시간적 일관성을 측정하기 위해 tOF를 활용한다.

## 4.1. Comparisons with State-of-the-Art Methods
DyBluRF의 단안 비디오 새로운 뷰 합성 블러링 제거 품질을 검증하기 위해, 기존의 dynamic novel view synthesis methods와 deblurring novel view synthesis methods를 비교한다. 모든 메소드들은 새로 합성된 Blurry iPhone Dataset으로 최적화했다. 원래 static novel view synthesis만을 위해 설계된 기존의 deblurring novel view synthesis methods는 시간 인스턴스를 추가 입력으로 포함시켜 공정한 비교를 위해 동적 구성 요소를 합성할 수 있도록 했다. 표 1은 다양한 방법에 대한 정량적 비교를 보여줍니다. 표에서 볼 수 있듯이, 우리의 DyBluRF는 모든 메트릭에서 기존의 SOTA 방법을 크게 능가합니다. 특히, 우리 모델은 다른 방법보다 깊은 메트릭, 예를 들어 지각적 품질 (mLPIPS) 및 시간적 일관성 (tOF)을 통해 구조적 품질이 훨씬 우수합니다. 그림 1과 4는 SOTA 방법에 비해 우리 DyBluRF의 우수한 시각화를 보여줍니다.

DyBluRF의 비디오 디블러링 능력의 효과를 더 입증하기 위해, DyBluRF를 선명한 비디오 프레임과 정확한 카메라 포즈를 사용하여 최적화된 기존의 단안 비디오 NeRF 방법과 비교한다. 이때 DyBluRF는 새로운 Blurry iPhone 데이터 세트로 훈련하는 반면, 다른 dynamic novel view synthesis methods는 원래의 iPhone 데이터 세트로 훈련되었다. 부정확한 포즈의 흐릿한 데이터 세트에 대해 훈련된 DyBluRF는 정확한 카메라 포즈 정보를 가진 sharp 데이터 셋으로 훈련된 SOTA 방법과 견줄 만한 결과를 달성한다.
![[Pasted image 20240208101408.png]]

## 4.2. Ablation Study
![[Pasted image 20240208101838.png]]

**IRR Stage** 
IRR 단계를 사용한 포즈 최적화는 특히 부정확한 포즈 정보와 함께 제공되는 주어진 흐릿한 프레임으로 훈련할 때 디블러링으로 새로운 뷰 합성을 효과적으로 처리하는 데 중요하다. 주어진 흐릿한 프레임에 대해 인터리브 최적화가 아닌 naive pose optimization을 채택하면, 최적화가 전혀 없는 것에 비해 더 노이즈가 많은 래디언스 필드가 발생한다.

**Surface Normal Loss**
표 3은 또한 NeRF를 정규화하기 위한 일반적인 깊이 감독(d)에 비해 표면 수직 손실(e)(Sec. 3.3.2)의 개선을 강조한다. 사전 훈련된 단안 깊이 추정기는 스케일이 모호하기 때문에 사전 훈련된 깊이 맵과 래디언스 필드의 스케일을 일치시키기가 어렵다. 안정성을 위해 감독 전에 깊이 지상 진실에 중앙 스케일링을 적용하지만 깊이 감독으로 훈련된 변형(d)은 여전히 비합리적인 렌더링 결과를 생성한다. 마지막으로, Lsn(e)으로 훈련된 변형은 기하학 정규화가 없는 변형(c)에 비해 mPSNR 및 mSSIM 측면에서 더 나은 결과를 보여준다. 

**MDD Stage** 
최종 DyBluRF(f)를 (e)과 비교함으로써 MDD Stage(3.4항 및 그림 2-(b))는 특히 지각 품질(mlPIPS)을 향상시킨다. 또한, 그림 5는 'Block' 장면에 대한 디블러링 접근 방식에 대한 ablation study의 시각적 비교 결과를 보여준다. 특히, 그림 5-(a)는 MDD Stage로 인한 비디오 디블러링에 대한 새로운 뷰 합성 측면에서 최종 DyBluRF의 효과를 보여준다. MDD Stage는 객체의 가장자리 또는 복잡한 텍스처 영역에서 흐릿함을 효과적으로 처리한다. 또한, 그림 5-(b)와 같이, 저희 모델은 훈련 시간 지수 전반에 걸쳐 글로벌 카메라 모션과 로컬 객체(즉, 흰색 벽돌) 모션의 혼합물을 더 잘 분해합니다. 이는 LORR(Eq. 15)로 훈련되지 않은 변종에 비해 훈련 중 객체가 통과하는 영역에 대한 강력한 새로운 뷰 합성을 초래한다.

# 5. Conclusion

흐릿한 단안 영상을 위한 새로운 dynamic deblurring NeRF인 DyBluRF를 제안한다.

- 카메라 포즈가 정확하지 않은 흐릿한 단안 프레임에서 선명한 새로운 시공간 뷰를 효과적으로 렌더링할 수 있다.
- IRR 단계는 동적 3D 장면을 동시에 재구성하고 부정확한 카메라 포즈 정보를 최적화한다.
- MDD 단계는 latent sharp ray를 전역 카메라 모션 및 로컬 객체 모션 구성 요소로 분해하는 새로운 ILSP (Incremental Latent Sharp Rays Prediction) 접근방식을 도입한다.

실험 결과 DyBluRF가 both qualitatively and quantitatively sota를 모두 능가한다.