# Abstract

* human-instruction(사람의 지시) 을 통해 image 를 편집하는 방법을 제시
	* 사람이 직접적으로 text 로 지시를 내리면 모델이 그 지시를 이해하고 image 를 편집한다.
* 위 모델을 위한 Dataset:
	* large pre-trained model: GPT-3, stable diffusion model(text-to-image model)
* 이 논문의 장점
	* pre-example fine-tuning 이나 inversion 이 필요 없다. 이로 인해 빠른 inference 가능
	* 다양한 입력이미지와 written instruction 에 대해 설득력 있는 출력 생성

# 1. Introduction
* human instruction 을 따를 수 있는 생성모델을 만드는 것이 목적
* 이를 위한 대규모 dataset 을 확보하는 것은 어려운 일
* 따라서 다양한 modality 에서 pre-train 된 대형 모델을 결합한 paired dataset 사용
	* LLM(GPT-3)
	* text-to-image model(Stable Diffusion)

* 이 논문의 generative model 은 forward pass 에서 directly 하게 edit 진행
* 직접적으로 생성된 image 와 written instruction 만으로도 실제와 비슷한 image 를 생성가능
* 즉, zero-shot generalization 이 가능
![[Pasted image 20240430142656.png]]

# 2. Prior work

### Composing large pretrained models
* 최근 연구들에서 Large pre-trained model 은 multi modal task 를 좋은 성능으로 처리할 수 있음을 보임
	* e.g. 
		* image captioning: 이미지를 주어졌을 때 그 이미지에 대한 설명을 하는 task 
		* VQA(Visual Question Answering): 이미지가 주어지고 그 이미지에 대한 질문이 주어졌을 때, 그에 대한 응답을 내놓는 task
* 또한 이러한 multi modal large pre-trained model 을 결합하는 방식에는 아래와 같은 예시가 있다.
	* joint finetuning on a new task [4, 33, 40, 67], 
	* communication through prompting [62, 69], 
	* composing probability distributions of energy-based models [11, 37], 
	* guiding one model with feedback from another [61], 
	* iterative optimization [34]
* 이 논문의 모델은 2개의 pre-trained model 의 상호보완적인 능력을 취한다는 공통점이 있지만,
* 이 모델들을 활용하여 paired multi-modal train data 를 생성하는 차이점이 존재.

### Diffusion-based generative models
* Diffusion-based generative model 은 최근 multi-modal generation 도 훌륭히 해냄
* 또한 임의의 text caption 에서 사실적인 image 를 생성해 내기도 함
	* text-to-image generative model

### Generative models for image editing
* 기존의 Image editing model 은 아래와 같은 task 를 처리하는 데 중점을 두었다.
	* style transfer
	* translation between image domains
* 또한 CLIP 을 사용하여 text 를 image 를 guide 하는 역할을 부여하기도 함.

* 최근, text-to-image diffusion model 을 image editing 에 사용함
* 일부 text-to-image model 은 태생적으로 image editing 능력을 가지고 있다.
	* DALLE-2는 이미지의 변형을 생성하고, 영역을 인페인팅하며, CLIP 임베딩을 조작할 수 있다.
* 하지만 이러한 모델을 image editing 에 사용하는 것은 힘들다.
	* 유사한 text prompt 가 유사한 image 를 생성해 낸다는 보장이 없기 때문
* 위의 성질을 다루기 위해 prompt-to-prompt method 를 적용
* 이러한 방법론으로 isolate 된 image editing 이 가능하도록 한다.
* 우리는 prompt-to-prompt method 를 train data 생성에 이용한다.

* SDEdit[38]은 사전 훈련된 모델을 사용하여 입력 이미지에 잡음을 추가하고 새로운 대상 프롬프트로 제거한다. 
* 우리는 SDEdit을 baseline으로 비교한다. 
* 다른 최근 작업들은 주어진 캡션과 사용자가 그린 마스크를 기반으로 지역적 인페인팅을 수행하거나[5, 48], 소수의 이미지에서 배운 특정 객체나 개념의 새로운 이미지를 생성하거나[13, 52], 단일 이미지를 역전시키고(미세 조정한 후) 새로운 텍스트 설명으로 재생성하여 편집을 수행한다[27]. 

* 이러한 접근 방식과는 대조적으로, <mark style='background:var(--mk-color-green)'>우리의 모델은 단일 이미지와 그 이미지를 편집하는 방법에 대한 지시(즉, 어떠한 이미지의 전체 설명도 아님)만을 받아 전방향 패스에서 직접 편집을 수행</mark>하며, <mark style='background:var(--mk-color-green)'>사용자가 그린 마스크, 추가 이미지, 또는 예시별 역전이나 미세 조정이 필요하지 않다</mark>.

* 정리
	* Baseline 은 SDEdit 을 사용
	* SDEdit 은 이미지에 추가적은 green masking 이나 noising 을 추가해야 한다.
	* 하지만 이 논문에서는 그러한 추가 작업이 필요 없다.
### Learning to follow instructions
* 기존의 text-based image edit 작업들과는 다르게 추가정보를 제공할 필요가 없다.
* 단순하게 text instruction 만 주어지면 모델을 스스로 대상 객체나 시각적 속성의 변경을 쉽게 분리하여 editing 을 진행한다.

### Training data generation with generative models
* Dataset 을 취합하고 annotating 을 하는 것은 <span style='color:var(--mk-color-red)'>매우 소모적인 작업</span>
* 최근에는 generative model 로 data 를 생성하는 것에 대한 관심 증가
* 이 논문에서는 LLM, text-to-image model 을 이용하여 image editing 에 사용할 데이터를 직접 생성하여 사용한다.

# 3. Method

* 이 논문에서는 image editing task 를 supervised learning problem 으로 취급한다.
1. editing 전,후의 이미지 + instruction 을 pair 로 하는 데이터 생성
![[Pasted image 20240430155409.png|500]]
2. image editing diffusion model 을 위에서 생성한 dataset 으로 학습
![[Pasted image 20240430155525.png|500]]
* image editing model 을 학습할 때, <mark style='background:var(--mk-color-green)'>합성된 이미지를 썼음에도</mark> <span style='color:var(--mk-color-green)'>Real image 를 편집하는 데 generalized 될 수 있다</span>.

## 3.1. Generating a Multi-modal Training Dataset
* 이 section 에서는 editing 전, 후의 text 와 그에 상응하는 editing 전, 후의 image 를 생성하는 방법에 대해 설명한다.

* Text generation(section 3.1.1)
	1. editing 전의 image 를 묘사하는 text 를 input 으로 넣는다.
	2. 위의 text 를 GPT-3 에 입력시켜 instruction 과 그 instruction 을 원래의 input 에 적용한 text 를 생성한다.
![[Pasted image 20240430162002.png|500]]
* Image generation(section 3.1.2)
	1. 원본 text, instructed text 를 text-to-image model(Stable diffusion+prompt2prompt) 을 통해 image pair 를 생성한다.
![[Pasted image 20240430162859.png|500]]
### 3.1.1. Generating Instructions and Paired Captions
* 입력: editing 전의 이미지를 설명하는 caption
* GPT-3 에 입력 데이터 삽입
* 출력: instruction + edited caption
* 예시:
	* 입력: "photograph of a girl riding a horse"
	* GPT-3 에 입력 데이터 삽입
	* 출력: Instruction-"have ger ride a dragon" / edited caption: "photograph of a girl riding a dragon"
![[Pasted image 20240430162002.png|500]]

* GPT-3 를 이 논문에 맞는 모델로 finetuning 하기 위해 아래 과정을 거침
	1. LAION-Aesthetics V2 6.5+ Dataset 에서 700개의 input caption 을 sampling
	2. 사람이 직접 instruction 과 output caption 을 작성함
* 위의 과정을 통해 caption, edit instruction, output caption 의 triplet data 를 만든다.
* 위 데이터를 통해 GPT-3 Davinci 모델을 단일 epoch 동안 fine-tuning 을 진행
* 예시
![[Pasted image 20240430163705.png]]

* LAION 을 사용한 이유?
	* LAION Dataset 은 그 내용이 다양하며, 다양한 매체(real image, digital art)를 포함하기 때문

### 3.1.2. Generating Paired Images from Paired Captions
* text-to-image generative model 은 아래 단점을 가짐
	* input text 가 매우 미세하게 달라지더라도 output image 들의 연속성을 보장할 수 없다.(아래 그림의 (a) 참고)
* 따라서 출력될 이미지의 연속성을 보장하기 위해 <mark style='background:var(--mk-color-green)'>prompt-to-prompt model 을 사용</mark>한다.(아래 그림의 (b) 참고)
![[Pasted image 20240430164237.png|500]]

* Prompt-to-prompt model 을 사용할 때, <span style='color:var(--mk-color-red)'>사용자가 이미지에 큰 변화를 주고자 한다면 연속성을 유지하는 것이 오히려 방해</span>가 될 수 있다.
* 다행히, propmt-to-prompt model 은 <span style='color:var(--mk-color-teal)'>두 image 간의 유사성을 제어할 수 있는 매개변수가 존재</span>한다.
	* the fraction of denoising steps $p$ with shared attention weights
* 하지만, <span style='color:var(--mk-color-red)'>caption 과 edited text 만으로는 최적의</span> $\textcolor{red}{p}$ <span style='color:var(--mk-color-red)'>를 찾는 것은 어려운 일</span>이다.
* 따라서 이 논문에서는 아래 과정으로 최적의 $p$ 를 찾는다.
	1. 각 caption pair 마다 100개의 image pair sample 을 생성하는데,
	2.  $p\sim \mathcal{U}(0.1,0.9)$ 에서 random 하게 $p$ 를 선택하고, 
	3. 위의 sample 들을 CLIP 기반 metric 을 사용하여 필터링한다.
		* 이 metric 은 두 image 간의 변화와 두 caption 간의 변화 사이의 일관성을 측정

* 위의 과정을 통해 <mark style='background:var(--mk-color-teal)'>image 의 다양성과 품질을 보존</mark>하고 <mark style='background:var(--mk-color-teal)'>Stable Diffusion 의 실패에 더욱 robust</mark> 하게 만들어준다.

## 3.2. InstructPix2Pix

* 이 논문에서 written instruction 에서 image editing 을 하기 위해 conditional diffusion model 을 훈련시킨다.
* 이 모델은 stable diffusion 을 base 로 한다.

* Stable Diffusion
$$
L_{LDM}:=\mathbb{E}_{\mathcal{E}(x),y,\epsilon\sim\mathcal{N}(0,1),t}[||\epsilon-\epsilon_\theta(z_t,t,\tau_\theta(y))||^2_2]
$$
* $\mathcal{E}(x)=z$ : input image $x$ 에 대한 latent representation
* $z_t$ : time step t 에서의 latent representation
* $y$ : prompt(modality 에 따라 달라짐)
* $\tau_\theta$ : prompt $y$ 에 대한 embedding(modality 에 따라 달라짐)
* stable diffusion 은 단일한 prompt y 에 대해서 diffusion process 를 진행한다. 

$$
L=\mathbb{E}_{\mathcal{E}(x), \mathcal{E}(c_I),c_T,\epsilon\sim\mathcal{N}(0,1),t}[||\epsilon-\epsilon_\theta(z_t,t,\mathcal{E}(c_I),c_T)||^2_2]
$$
* $\mathcal{E}(x)=z$ : input image $x$ 에 대한 latent representation. 이 논문에서는 editing 이 완료된 이미지(생성하고자 하는 image).
* $\mathcal{E}(C_I)$ : editing 이전의 이미지 의 latent representation.(모델이 edit 하고자 하는 image 의 latent representation)
* $c_T$ : prompt. 이 논문에서는 instruction text.
* $z_t$ : time step t 에서의 latent representation.

* [Pretraining is All You Need for Image-to-Image Translation](https://arxiv.org/abs/2205.12952)
* 위 논문에서는 image translation task 에서 pair training data 가 제한적일 때, 처음부터 다시 훈련하는 것보다 fine-tuning 하는 것이 성능이 더 좋다고 한다.

* 이미지 컨디셔닝을 지원하기 위해 추가 입력 채널을 첫 번째 convolution layer에 추가하여 $z_t$와 $\mathcal{E}(c_I)$를 concat한다. 
* Diffusion model의 모든 가중치는 사전 학습된 체크포인트에서 초기화되고 새로 추가된 입력 채널에서 작동하는 가중치는 0으로 초기화된다.

### 3.2.1 Classifier-free Guidance for Two Conditionings

#### Classifier-free guidance for single conditions
*  classifier free guidance은 암시적인 classifier $p_\theta (c | z_t)$가 conditional $c$ 에 높은 likelihood 할당하는 곳으로 probability mass를 효과적으로 이동시킬 수 있다.

![[Pasted image 20240430202834.png]]
* Classifier-free diffusion guidance 에서 training 은 unconditional model 과 conditional model 이 동시에 학습된다.
	* 위의 사진은 Classifier-free diffusion guidance 논문의 Joint training part 에 대한 설명이다.
* 위의 algorithm 의 핵심은 아래와 같다.
	* $c = \varnothing$ : random 하게 이 수식을 적용하여 unconditional training 을 진행한다.
	* $c \neq \varnothing$ : 즉, 이 부분이 conditional training 을 진행하는 부분이다.

* inference 시에는 아래 수식을 따른다(Classifier-free diffusion guidance 에서의 algorithm2 의 수식이다. 형태가 달라보이지만 결국 같은 형태임을 알 수 있다.) 
$$\tilde{e_\theta}(z_t, c) = e_\theta(z_t, \varnothing) + s \cdot (e_\theta(z_t, c) - e_\theta(z_t, \varnothing)) (2)$$
* s 가 1을 넘어갈 때 condition 이 적용되게 된다.(1 미만일 때 uncondition 에 가까워짐)
* 위 수식은 1개의 modal 을 다루기 위한 수식이다.
* 이 논문에서는 2개의 modal($c_I,c_T$) 를 가지기에 이를 다루기 위한 변형이 필요하다.

#### Classifier-free guidance for multi conditions
* 이 논문에서는 두가지 condition 을 갖는다.
	1. $c_I$ : edit 전의 이미지
	2. $c_T$ : instruction text
* 이 논문에서는 위의 2개의 condition 모두에 대해 classifier free guidance 를 활용하는 것이 효과적임을 언급하고 있다.
* [Compositional Visual Generation with Composable Diffusion Models](https://arxiv.org/abs/2206.01714) 에서는 conditional diffusion model 이 다양한 conditional 값에서 score estimate 을 조합할 수 있음을 보였다.
* training step 에서 아래 규칙을 따른다.
	* 5% 의 확률로 $c_I=\varnothing$ 부여 : $c_T$ 와 uncondition 학습
	* 5% 의 확률로 $c_T=\varnothing$ 부여 : $c_I$ 와 uncondition 학습
	* 5% 의 확률로 $c_I=\varnothing, c_T=\varnothing$ 부여 : $c_I$ 와 $c_T$(2개의 condition) 학습
* inference step(sampling step) 에서는 아래 규칙을 따른다.
$$
\begin{align*}
\tilde{e_\theta}(z_t, c_I, c_T) = &e_\theta(z_t, \varnothing, \varnothing) \\
&+ s_I \cdot (e_\theta(z_t, c_I, \varnothing) - e_\theta(z_t, \varnothing, \varnothing))\\
&+ s_T \cdot (e_\theta(z_t, c_I, c_T) - e_\theta(z_t, c_I, \varnothing))
\end{align*}
$$
* $s_I$ 를 증가시키면 input image 와 밀접하게 닮은 이미지가 생성됨
* $s_T$ 를 증가시키면 강렬한 edit 이 일어난다.

if $s_I$ is increased, an image closely resembling the input image is generated
if $s_T$ is increased, a more dramatic edit occurs


---
#### 참고: 
* 아래 사진은 Classifier free diffusion guidance 의 sampling 파트이다.
![[Pasted image 20240430203602.png]]
* 위 사진에서의 sampling step 을 살펴보면 아래 수식과 같다.
$$
\tilde{\epsilon_t}=(1+w)\epsilon_\theta(z_t,c)-w\epsilon_\theta(z_t)
$$
* 위 수식을 $w$ 에 대해서 묶고 다시 정리하면 아래와 같다.
$$
\tilde{\epsilon_t}=\epsilon_\theta(z_t,c)+w(\epsilon(z_t,c)-\epsilon_\theta(z_t))
$$
* 위 수식에서 $\epsilon_\theta(z_t)$ 를 $\epsilon_\theta(z_{t},\varnothing)$ 으로 고치면 이 논문에 적혀있는 수식이 된다.(아래는 이 논문에서 제시한 수식)
$$
\tilde{e_\theta}(z_t, c) = e_\theta(z_t, \varnothing) + s \cdot (e_\theta(z_t, c) - e_\theta(z_t, \varnothing))
$$

---
# 4. Result






---
REF:
[37] Nan Liu, Shuang Li, Yilun Du, Antonio Torralba, and Joshua B Tenenbaum. Compositional visual generation with composable diffusion models. arXiv preprint arXiv:2206.01714,2022