# Abstract

대규모 얼굴 인식을 위한 deep convolution Neural Network(Deep Convolutional Neural Networks, DCNNs)을 사용한 feature 학습에서 주요 도전 과제 중 하나는 차별적인 힘을 향상시킬 수 있는 적절한 loss 함수의 설계입니다. 중심 loss(Centre loss)은 유클리드 공간에서 deep feature과 해당 클래스 중심 간의 거리에 대한 패널티를 부과하여 intra-class compactness을 달성합니다. 
SphereFace는 마지막 완전 연결 계층에서의 선형 변환 행렬을 각 공간에서 클래스 중심의 표현으로 사용할 수 있다고 가정하고, 따라서 deep feature과 해당 가중치 간의 각도에 대해 곱셈적 방식으로 패널티를 부과합니다. 최근의 인기 있는 연구 경향은 얼굴 클래스 분리를 최대화하기 위해 잘 확립된 loss 함수에 margin을 포함시키는 것입니다. 본 논문에서는 얼굴 인식을 위해 매우 차별화된 feature을 얻기 위해 첨가 각 margin loss(Additive Angular Margin Loss, ArcFace)을 제안합니다. 제안된 ArcFace는 구면상에서의 측지 거리와 정확하게 일치함으로써 명확한 기하학적 해석을 가집니다. 우리는 새로운 대규모 이미지 데이터베이스와 대규모 비디오 데이터셋을 포함한 열 개의 얼굴 인식 벤치마크에서 최근의 최첨단 얼굴 인식 방법들에 대해 아마도 가장 광범위한 실험 평가를 제시합니다. 우리는 ArcFace가 일관되게 최고의 성능을 뛰어넘고 미미한 계산 오버헤드로 쉽게 구현될 수 있음을 보여줍니다. 향투 연구를 용이하게 하기 위해 코드는 다음에서 이용 가능합니다: https://github.com/deepinsight/insightface


# 1. Introduction

얼굴 인식을 위한 방법으로 선택된 deep convolution Neural Network(Deep Convolutional Neural Network, DCNN) 임베딩은 일반적으로 pose estimation 단계[42] 이후에 얼굴 이미지를 작은 intra-class distance와 큰 inter-class distance를 가지는 feature으로 매핑합니다.

얼굴 인식을 위해 DCNN을 훈련시키는 두 가지 주요 연구 분야가 있습니다. 하나는 다중 클래스 분류기를 훈련시켜 훈련 세트의 다른 정체성을 분리할 수 있게 하는 방법으로, 소프트맥스 분류기[31, 22, 3]를 사용하는 것이고, 다른 하나는 triplet loss[27]과 같은 임베딩을 직접 학습하는 것입니다. 대규모 훈련 데이터와 정교한 DCNN 아키텍처를 기반으로, 소프트맥스 loss 기반 방법[3]과 triplet loss 기반 방법[27] 모두 얼굴 인식에서 우수한 성능을 달성할 수 있습니다. 그러나 소프트맥스 loss과 triplet loss 모두 일부 단점이 있습니다. 소프트맥스 loss의 경우: (1) 선형 변환 행렬 $W \in \mathbb{R}^{d \times n}$의 크기는 정체성 수 $n$과 선형적으로 증가합니다; (2) 학습된 feature은 폐쇄 집합 분류 문제에 대해 분리 가능하지만 개방 집합 얼굴 인식 문제에 대해서는 충분히 차별적이지 않습니다. triplet loss의 경우: (1) 대규모 데이터셋의 경우 얼굴 triplet의 조합 폭발이 발생하여 반복 단계 수가 크게 증가합니다; (2) 반 정도 어려운 샘플 마이닝은 모델 훈련에 효과적인 문제가 되기 어렵습니다.

여러 변형들[36, 6, 43, 15, 35, 33, 4, 32, 25]이 소프트맥스 loss의 차별적인 힘을 강화하기 위해 제안되었습니다. Wen 등[36]은 각 feature 벡터와 그 클래스 중심 간의 유클리드 거리인 중심 loss을 개척하여, intra-class compactness을 얻으면서 inter-class 분산이 소프트맥스 loss의 공동 패널티에 의해 보장됩니다. 그러나 훈련 중에 실제 중심을 업데이트하는 것은 훈련 가능한 얼굴 클래스 수가 최근 급격히 증가함에 따라 매우 어렵습니다.

분류용 DCNN의 마지막 완전 연결 계층에서 나오는 가중치가 각 얼굴 클래스의 중심과 개념적 유사성을 가진다는 점을 관찰함에 따라, [15, 16]에서는 추가적인 intra-class compactness과 inter-class 차이를 동시에 강제하는 곱셈 각 margin 패널티를 제안하여 훈련된 모델의 차별적인 힘을 더욱 향상시켰습니다. 
SphereFace[15]는 중요한 각 margin 개념을 도입했지만, 그들의 loss 함수는 계산을 위해 일련의 근사치가 필요했으며, 이로 인해 네트워크의 불안정한 훈련이 발생했습니다. 훈련을 안정화하기 위해, 그들은 표준 소프트맥스 loss을 포함하는 하이브리드 loss 함수를 제안했습니다. 
경험적으로, 소프트맥스 loss이 훈련 과정을 지배하는데, 이는 정수 기반의 곱셈 각 margin이 대상 로짓 곡선을 매우 가파르게 만들어 수렴을 방해하기 때문입니다. CosFace[35, 33]는 대상 로짓에 직접 코사인 margin 패널티를 추가하여 SphereFace보다 더 나은 성능을 달성했지만, 훨씬 쉬운 구현을 가능하게 하고 소프트맥스 loss의 공동 감독 필요성을 완화했습니다.

본 논문에서는 얼굴 인식 모델의 차별적인 힘을 더욱 향상시키고 훈련 과정을 안정화하기 위해 첨가 각 margin loss(Additive Angular Margin Loss, ArcFace)을 제안합니다. 그림 2에서 설명한 바와 같이, DCNN feature과 마지막 완전 연결 계층 간의 점 곱(dot product)은 feature과 가중치 정규화 후 코사인 거리와 동일합니다. 우리는 아크코사인 함수를 사용하여 현재 feature과 대상 가중치 사이의 각도를 계산합니다. 이후, 대상 각도에 첨가 각 margin을 추가하고, 다시 코사인 함수를 통해 대상 로짓을 얻습니다. 그 다음, 모든 로짓을 고정된 feature 규범으로 재조정하며, 이후 단계는 소프트맥스 loss에서와 정확히 동일합니다. 제안된 ArcFace의 장점은 다음과 같이 요약할 수 있습니다:

**Engaging.** ArcFace는 정규화된 구면상에서 각도와 호 사이의 정확한 대응을 통해 측지 거리 margin을 직접 최적화합니다. 우리는 512-D 공간에서 feature과 가중치 간의 각도 통계를 분석함으로써 직관적으로 어떤 일이 일어나는지 설명합니다.

**Effective.** ArcFace는 대규모 이미지 및 비디오 데이터셋을 포함한 열 개의 얼굴 인식 벤치마크에서 최첨단 성능을 달성합니다.

**Easy.** ArcFace는 알고리즘 1에서 제시된 몇 줄의 코드만 필요하며, MxNet [5], Pytorch [23], Tensorflow [2]와 같은 계산 그래프 기반 딥 러닝 프레임워크에서 매우 쉽게 구현할 수 있습니다. 또한, [15, 16]의 연구와 달리, ArcFace는 안정적인 성능을 위해 다른 loss 함수와 결합할 필요가 없으며, 어떠한 훈련 데이터셋에서도 쉽게 수렴할 수 있습니다.

**Efficient.** ArcFace는 훈련 중에 무시할 만한 계산 복잡성만을 추가합니다. 현재의 GPU는 수백만의 정체성을 훈련에 쉽게 지원할 수 있으며, 모델 병렬 전략은 훨씬 더 많은 정체성을 쉽게 지원할 수 있습니다.

제시된 이미지를 번역해 드리겠습니다. 여기에서 설명하는 내용은 **ArcFace**에 관한 것입니다.

# 2. 제안하는 접근법
## 2.1. ArcFace
가장 널리 사용되는 분류 loss 함수인 소프트맥스 loss은 다음과 같이 제시됩니다:

$$
L_1 = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{W_{y_i}^T x_i + b_{y_i}}}{\sum_{j=1}^n e^{W_j^T x_i + b_j}}
$$

여기서 $x_i \in \mathbb{R}^d$는 $i$번째 샘플의 deep feature을 나타내며, $y_i$-번째 클래스에 속합니다. 임베딩 feature 차원 $d$는 이 논문에서 512로 설정되었습니다. $W_j \in \mathbb{R}^d$는 가중치 $W \in \mathbb{R}^{d \times n}$의 $j$-번째 열을 나타내며, $b_j \in \mathbb{R}^n$는 편향 항입니다. 배치 크기와 클래스 수는 각각 $N$과 $n$입니다. 전통적인 소프트맥스 loss은 deep 얼굴 인식에서 널리 사용됩니다. 그러나 소프트맥스 loss 함수는 feature 임베딩을 명시적으로 최적화하지 않아 intra-class 샘플의 유사성을 강화하고 inter-class 샘플의 다양성을 부여하는데, 이것이 deep 얼굴 인식에서 큰 성능 차이를 가져옵니다. 이러한 성능 차이는 대규모 얼굴 인식에서 큰 intra-class 외모 변화(예: 자세 변화, 연령 차이)와 대규모 테스트 시나리오(예: 백만[12, 37, 18] 혹은 조 단위[1] 페어)에서 나타납니다.

간단함을 위해, 우리는 편향 $b_j$를 [15]에서처럼 0으로 고정합니다. 그런 다음, 로짓을 $W_j^T x_i = \|W_j\| \|x_i\| \cos \theta_j$로 변환합니다. 여기서 $\theta_j$는 가중치 $W_j$와 feature $x_i$ 사이의 각도입니다. [15, 35, 34]에 따라, 개별 가중치 $\|W_j\|$를 L2 정규화를 사용하여 1로 고정합니다. 또한 feature $\|x_i\|$를 L2 정규화하여 $s$로 재조정합니다. feature과 가중치에 대한 정규화 단계는 예측이 feature과 가중치 사이의 각도에만 의존하게 만듭니다. 학습된 임베딩 feature은 반지름이 $s$인 초구체 위에 분포하게 됩니다.

$$
L_2 = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s \cos \theta_{y_i}}}{e^{s \cos \theta_{y_i}} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
$$

임베딩 feature이 초구체의 각 feature 중심 주위에 분포되어 있기 때문에, 우리는 $x_i$와 $W_{y_i}$ 사이에 첨가 각 margin 페널티 $m$을 추가하여 동시에 intra-class compactness을 증가시키고 inter-class 차이를 높입니다. 제안된 첨가 각 margin 페널티는 정규화된 초구체에서의 측지 거리 margin 페널티와 동등하므로, 우리의 방법을 ArcFace라고 명명합니다.

$$
L_3 = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s(\cos(\theta_{y_i} + m))}}{e^{s(\cos(\theta_{y_i} + m))} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
$$

우리는 8개의 다른 정체성에서 각각 약 1,500개의 이미지를 포함하는 충분한 샘플을 선택하여 소프트맥스 및 ArcFace loss로 2D feature 임베딩 네트워크를 훈련시킵니다. 그림 3에서 보여주듯이, 소프트맥스 loss은 대략적으로 분리 가능한 feature 임베딩을 제공하지만, 결정 경계에서 뚜렷한 모호성을 생성합니다. 반면에 제안된 ArcFace loss은 가장 가까운 클래스들 사이에 분명한 간격을 강제할 수 있습니다.

## 2.2. SphereFace 및 CosFace와의 비교
### Numerical Similarity.
SphereFace [15, 16], ArcFace, CosFace [35, 33]에서는 세 가지 다른 종류의 margin 페널티가 제안되었습니다. 예를 들어, multiplicative angular margin $m_1$, additive angular margin $m_2$, additive cosine margin $m_3$ 등입니다. 수치 분석의 관점에서, 이러한 다양한 margin 페널티는 각도 [15]에 추가하든, 코사인 공간 [35]에서 적용하든, intra-class compactness을 강화하고 inter-class diversity을 증가시키기 위해 대상 로짓 [24]을 패널티하는 방식으로 모두 동일하게 작용합니다. 그림 4(b)에서, 우리는 SphereFace, ArcFace 및 CosFace의 최적 margin 설정 하에서의 대상 로짓 곡선을 표시합니다. 우리는 $W_{y_i}$와 $x_i$ 사이의 각도가 초기에 약 90도에서 시작하여 ArcFace 훈련 중 약 30도에서 끝나는 것을 고려하여, [20°, 100°] 범위 내에서만 이 대상 로짓 곡선을 보여줍니다(그림 4(a) 참조). 직관적으로, 대상 로짓 곡선에서 성능에 영향을 미치는 세 가지 요소가 있습니다. 즉, 시작점, 종료점 및 기울기입니다.

모든 margin 페널티를 결합함으로써, 우리는 SphereFace, ArcFace 및 CosFace를 하나의 통합된 프레임워크로 구현하며, 여기서 $m_1$, $m_2$, $m_3$을 하이퍼파라미터로 사용합니다.

$$
L_4 = -\frac{1}{N} \sum_{i=1}^N \log \frac{e^{s(\cos(m_1\theta_{y_i}+m_2)-m_3)}}{e^{s(\cos(m_1\theta_{y_i}+m_2)-m_3)} + \sum_{j=1, j \neq y_i}^n e^{s \cos \theta_j}}
$$

그림 4(b)에서 보여주듯이, 위에서 언급한 모든 margin을 결합함($\cos(m_1\theta + m_2) - m_3$)으로써, 우리는 또한 높은 성능을 가진 다른 대상 로짓 곡선을 쉽게 얻을 수 있습니다.

### Geometric Difference.
ArcFace와 이전 작업들 사이의 수치적 유사성에도 불구하고, 제안된 첨가 각 margin은 측지 거리와 정확히 일치하는 각 margin으로 더 나은 기하학적 속성을 가집니다. 그림 5에서 보여주듯이, 우리는 이진 분류 케이스 하에서의 결정 경계를 비교합니다. 제안된 ArcFace는 전체 구간에 걸쳐 일정한 선형 각 margin을 가지고 있습니다. 반면, SphereFace와 CosFace는 비선형 각 margin만을 가집니다.

margin 디자인에서의 사소한 차이가 모델 훈련에 '나비 효과'를 미칠 수 있습니다. 예를 들어, 원래의 SphereFace [15]는 훈련 시작 시 발산을 피하기 위해 연화 최적화 전략을 사용합니다. 훈련 초기에 발산을 방지하기 위해, SphereFace는 소프트맥스로부터의 공동 감독을 사용하여 곱셈 margin 페널티를 약화시킵니다. 우리는 정수 요구 사항 없이 아크-코사인 함수를 사용하여 복잡한 두 배 각 공식 대신에 margin을 적용하는 SphereFace의 새로운 버전을 구현합니다. 우리의 구현에서, $m = 1.35$는 어떠한 수렴 문제도 없이 원래 SphereFace와 유사한 성능을 얻을 수 있음을 발견했습니다.
## 2.3. 다른 loss 함수와의 비교
다른 loss 함수들은 feature과 가중치 벡터의 각도 표현을 기반으로 설계될 수 있습니다. 예를 들어, 우리는 intra-class compactness을 강화하고 inter-class 차이를 증가시키기 위한 loss을 초구체 위에서 설계할 수 있습니다. 그림 1에서 보여주듯이, 이 논문에서는 세 가지 다른 loss과 비교합니다.

**Intra-Loss**는 샘플과 기준 중심 사이의 각도/호를 감소시켜 intra-class compactness을 향상시키도록 설계되었습니다.
$$L_5 = L_2 + \frac{1}{\pi N} \sum_{i=1}^N \theta_{y_i}$$

**Inter-Loss**는 다른 중심들 사이의 각도/호를 증가시켜 inter-class 차이를 강화하는 것을 목표로 합니다.
$$L_6 = L_2 - \frac{1}{\pi N (n - 1)} \sum_{i=1}^N \sum_{j=1, j \neq y_i}^n \arccos(W_{y_i}^T W_j)$$

여기서의 Inter-Loss는 최소 구형 에너지(Minimum Hyper-spherical Energy, MHE) 방법의 특별한 경우입니다. [14]에서는 MHE를 통해 숨겨진 계층과 출력 계층이 규제되었습니다. 이 논문에서는 SphereFace loss과 네트워크의 마지막 계층에서의 MHE loss을 결합하여 제안된 loss 함수의 특별한 경우를 제시합니다.

**Triplet-Loss**는 triplet 샘플들 사이의 각도/호 margin을 확대하는 것을 목표로 합니다. FaceNet[27]에서는 정규화된 feature에 유클리드 margin이 적용됩니다. 여기서, 우리는 feature의 각도 표현을 사용하여 triplet loss을 적용합니다.
$$\arccos(x_i^{pos} \cdot x_i) + m \leq \arccos(x_i^{neg} \cdot x_i)$$

# 3. 실험
## 3.1. 구현 세부사항
**Dataset.** 표 1에 나와 있는 바와 같이, 우리는 CASIA [41], VGGFace2 [3], MS1MV2, DeepGlint-Face (MS1M-DeepGlint 및 Asian-DeepGlint 포함) [1]를 훈련 데이터로 별도로 사용하여 다른 방법과의 공정한 비교를 수행합니다. 제안된 MS1MV2는 MS-Celeb-1M 데이터셋 [7]의 반자동 정제 버전입니다. 우리가 알기로는, 대규모 얼굴 이미지 주석에 특정 인종의 주석자를 사용하는 것은 처음입니다. 이는 경계 케이스(예: 어려운 샘플과 노이즈 샘플)를 구별하는 것이 주석자가 정체성을 잘 모를 경우 매우 어렵기 때문입니다. 훈련 중에는 효율적인 얼굴 검증 데이터셋(예: LFW [10], CFP-FP [28], AgeDB-30 [19])을 탐색하여 다양한 설정에서의 개선을 확인합니다. 가장 널리 사용되는 LFW [10]와 YTF [38] 데이터셋 외에도, 최근 대규모 포즈와 대규모 연령 데이터셋(예: CPLFW [44] 및 CALFW [45])에서의 ArcFace의 성능도 보고합니다. 또한, 대규모 이미지 데이터셋(예: MegaFace [12], IJB-B [37], IJB-C [18], Trillion-Pairs [1]) 및 비디오 데이터셋(iQIYI-VID [17])에서 제안된 ArcFace를 광범위하게 테스트합니다.

**Experimental Settings.** 데이터 전처리를 위해, 최근 논문들 [15, 35]을 따라 5개의 얼굴 점을 이용하여 정규화된 얼굴 크롭(112 × 112)을 생성합니다. 임베딩 네트워크로는 널리 사용되는 CNN 아키텍처인 ResNet50과 ResNet100 [9, 8]을 사용합니다. 마지막 컨볼루션 계층 이후, BN [11]-Dropout [29]-FC-BN 구조를 탐색하여 최종 512-D 임베딩 feature을 얻습니다. 
이 논문에서는 실험 설정을 이해하기 쉽도록 ([훈련 데이터셋, 네트워크 구조, loss])로 사용합니다. [35]를 따라 feature 스케일 $s$를 64로 설정하고 ArcFace의 각도 margin $m$을 0.5로 선택합니다. 이 논문의 모든 실험은 MXNet [5]을 사용하여 구현됩니다. 배치 크기를 512로 설정하고 네 개의 NVIDIA Tesla P40 (24GB) GPU에서 모델을 훈련합니다. CASIA에서는 학습률이 0.1에서 시작하여 20K, 28K 반복에서 10으로 나눕니다. 훈련 과정은 32K 반복에서 완료됩니다. MS1MV2에서는 학습률을 100K, 160K 반복에서 나누고 180K 반복에서 완료합니다. 모멘텀은 0.9로 설정하고, 가중치 감쇠는 5e-4로 설정합니다. 테스트 중에는 완전 연결 계층 없이 feature 임베딩 네트워크만 유지합니다(ResNet50의 경우 160MB, ResNet100의 경우 250MB) 및 각 정규화된 얼굴에 대해 512-D feature(ResNet50의 경우 8.9 ms/얼굴, ResNet100의 경우 15.4 ms/얼굴)을 추출합니다. 템플릿(예: IJB-B 및 IJB-C) 또는 비디오(예: YTF 및 iQIYI-VID)의 임베딩 feature을 얻기 위해, 템플릿의 모든 이미지 또는 비디오의 모든 프레임에서 feature 중심을 단순히 계산합니다. 훈련 세트와 테스트 세트 사이의 중복된 정체성은 엄격한 평가를 위해 제거되며, 모든 테스트에 단일 크롭만 사용됩니다.
## 3.2. loss에 대한 소거 연구
표 2에서, 우리는 먼저 CASIA 데이터셋에서 ResNet50을 사용하여 ArcFace의 각도 margin 설정을 탐구합니다. 실험에서 관찰된 최적의 margin은 0.5였습니다. 
식 4에서 제안된 결합 margin 프레임워크를 사용하면 SphereFace와 CosFace의 margin을 설정하기가 더 쉬워지는데, 각각 1.35와 0.35에서 최적의 성능을 발휘하는 것으로 나타났습니다. 
SphereFace와 CosFace 모두 우리의 구현을 통해 수렴에 어려움 없이 훌륭한 성능을 이끌어 낼 수 있습니다. 제안된 ArcFace는 모든 세 테스트 세트에서 가장 높은 검증 정확도를 달성합니다. 또한, 그림 4(b)의 대상 로짓 곡선에 의해 안내된 joint margin 프레임워크(CM1 (1, 0.3, 0.2) 및 CM2 (0.9, 0.4, 0.15)에서 최고의 성능이 관찰됨)로 광범위한 실험을 수행했습니다. 결합 margin 프레임워크는 개별 SphereFace 및 CosFace보다 더 나은 성능을 보여 주었지만, ArcFace의 성능에 의해 상한이 결정되었습니다.

margin 기반 방법과의 비교뿐만 아니라, 우리는 ArcFace와 intra-class compactness 강화(Eq. 5) 및 inter-class 차이증가(Eq. 6)를 목표로 하는 다른 loss들과의 추가 비교를 수행합니다. 기준으로 선택한 소프트맥스 loss을 사용하였을 때, CFP-FP 및 AgeDB-30에서 가중치와 feature 정규화 후 성능 하락을 관찰했습니다. 소프트맥스와 intra-class loss을 결합하면 CFP-FP와 AgeDB-30에서 성능이 향상됩니다. 그러나 소프트맥스와 inter-class loss을 결합하는 것은 정확도를 약간만 향상시킵니다. Triplet-loss가 Norm-Softmax loss보다 더 우수한 성능을 보이는 것은 margin이 성능 향상에 중요함을 나타냅니다. 그러나 triplet 샘플 내에서 margin 페널티를 사용하는 것은 ArcFace에서처럼 샘플과 중심 사이에 margin을 삽입하는 것보다 효과가 덜합니다. 마지막으로, 우리는 Intra-loss, Inter-loss 및 Triplet-loss를 ArcFace에 통합했지만 개선이 관찰되지 않았으며, 이는 ArcFace가 이미 intra-class compactness, inter-class 차이 및 분류 margin을 강제하고 있음을 시사합니다.

ArcFace의 우수성을 더 잘 이해하기 위해, 다양한 loss 하에서 훈련 데이터(CASIA) 및 테스트 데이터(LFW)에 대한 자세한 각도 통계를 표 3에서 제공합니다. 우리는 다음을 발견했습니다: 
(1) ArcFace의 경우 $W_j$는 임베딩 feature 중심과 거의 동기화되어 있으며 각도는 14.29도입니다. 그러나 Norm-Softmax의 경우 $W_j$와 임베딩 feature 중심 사이에는 명확한 편차(44.26도)가 있습니다. 따라서 $W_j$ 간의 각도는 훈련 데이터에서 inter-class 차이를 절대적으로 나타낼 수 없습니다. 대신, 훈련된 네트워크에 의해 계산된 임베딩 feature 중심이 더 대표적입니다. 
(2) Intra-Loss는 intra-class 변동을 효과적으로 압축할 수 있지만, inter-class 각도도 더 작게 만듭니다. 
(3) Inter-Loss는 $W$ (직접적으로)와 임베딩 네트워크 (간접적으로) 둘 다에서 inter-class 차이를 약간 증가시킬 수 있지만, intra-class 각도도 증가시킵니다. 
(4) ArcFace는 이미 매우 좋은 intra-class compactness과 inter-class 차이를 가지고 있습니다. 
(5) Triplet-Loss는 intra-class compactness은 ArcFace와 비슷하지만 inter-class 차이는 ArcFace에 비해 열등합니다. 또한, 그림 6에서 보여주듯이, ArcFace는 테스트 세트에서 Triplet-Loss에 비해 더 뚜렷한 margin을 가지고 있습니다.

## 3.3. 평가 결과
**LFW, YTF, CALFW 및 CPLFW 결과.** LFW [10] 및 YTF [38] 데이터셋은 이미지 및 비디오에서 제약이 없는 얼굴 검증을 위한 가장 널리 사용되는 벤치마크입니다. 이 논문에서는 제한 없는 외부 데이터 라벨링 프로토콜을 사용하여 성능을 보고합니다. 표 4에서 보고된 바와 같이, MS1MV2에서 ResNet100으로 훈련된 ArcFace는 LFW와 YTF에서 기준 모델(예: SphereFace [15], CosFace [35])을 큰 차이로 능가하며, 이는 첨가 각 margin 페널티가 깊게 학습된 feature의 차별력을 현저하게 향상시킬 수 있음을 보여주고, ArcFace의 효과성을 입증합니다.

LFW 및 YTF 데이터셋 외에도, 최근에 도입된 CPLFW [44] 및 CALFW [45] 데이터셋에서의 ArcFace 성능을 보고합니다. 이 데이터셋들은 LFW의 동일 정체성에서 더 높은 자세 및 연령 변화를 보여줍니다. 모든 오픈 소스 얼굴 인식 모델 중에서, ArcFace 모델은 **표 5**에서 보여주듯이 최고 순위의 얼굴 인식 모델로 평가되며, 뚜렷한 차이로 경쟁 모델을 능가합니다. 
**그림 7**에서는 MS1MV2에 ResNet100으로 훈련된 ArcFace 모델로 예측된 LFW, CFP-FP, AgeDB-30, YTF, CPLFW 및 CALFW의 **양성 및 음성 쌍의 각도 분포**를 보여줍니다. 우리는 자세 및 연령 차이로 인한 intra-class 변동이 양성 쌍 사이의 각도를 현저하게 증가시켜 얼굴 검증을 위한 최적의 임계값을 증가시키고 히스토그램에 더 많은 혼동 영역을 생성하는 것을 명확하게 확인할 수 있습니다.

**MegaFace 결과.** MegaFace 데이터셋 [12]은 갤러리 세트로 690K 다른 개인의 1M 이미지와 FaceScrub [21]에서 530개의 고유한 개인의 10만 장의 사진을 프로브 세트로 포함합니다. MegaFace에서는 두 가지 프로토콜(대규모 또는 소규모 훈련 세트) 아래에서 두 가지 테스트 시나리오(식별 및 검증)가 있습니다. 50만 개 이상의 이미지를 포함하는 경우 훈련 세트를 대규모로 정의합니다. 공정한 비교를 위해, 우리는 각각 소규모 프로토콜과 대규모 프로토콜에서 CASIA와 MS1MV2에 ArcFace를 훈련시킵니다. **표 6**에 따르면, CASIA에서 훈련된 ArcFace는 최고의 단일 모델 식별 및 검증 성능을 달성하여, 강력한 기준선(예: SphereFace [15], CosFace [35])뿐만 아니라 다른 발표된 방법들[36, 14]을 능가합니다.

식별과 검증 간에 뚜렷한 성능 차이가 관찰되었기 때문에, 우리는 전체 MegaFace 데이터셋에서 철저한 수동 검사를 수행하고 잘못된 라벨이 붙은 많은 얼굴 이미지를 발견했으며, 이는 성능에 상당한 영향을 미쳤습니다. 따라서, 우리는 전체 MegaFace 데이터셋을 수동으로 정제하고 MegaFace에서 ArcFace의 올바른 성능을 보고합니다. 정제된 MegaFace에서, ArcFace는 CosFace를 명확히 능가하며 검증 및 식별 모두에서 최고의 성능을 달성합니다.

대규모 프로토콜 하에서, ArcFace는 FaceNet [27]을 명확한 차이로 능가하며, CosFace [35]와 비교하여 식별에서는 비슷하고 검증에서는 더 나은 결과를 얻습니다. CosFace가 사적인 훈련 데이터를 사용하는 것에 대해, 우리는 ResNet100을 사용하여 MS1MV2 데이터셋에서 CosFace를 다시 훈련시킵니다. 공정한 비교 하에서, ArcFace는 CosFace를 능가하며, 그림 8에서 보여주듯이 식별과 검증 시나리오 모두에서 CosFace보다 우월한 성능을 나타내는 상한선을 형성합니다.

**IJB-B 및 IJB-C 결과.** IJB-B 데이터셋 [37]은 1,845명의 대상자, 21.8K개의 정지 이미지 및 7,011개의 비디오에서 나온 55K개의 프레임을 포함합니다. 총 12,115개의 템플릿이 있으며, 이 중 10,270개는 진짜 매치, 8M개는 가짜 매치입니다. IJB-C 데이터셋 [37]은 IJB-B를 확장한 것으로, 3,531명의 대상자, 31.3K개의 정지 이미지 및 11,779개의 비디오에서 나온 117.5K개의 프레임을 포함합니다. 총 23,124개의 템플릿이 있으며, 이 중 19,557개는 진짜 매치, 15,639K개는 가짜 매치입니다.

IJB-B 및 IJB-C 데이터셋에서, 우리는 최신 방법들 [3, 40, 39]과의 공정한 비교를 위해 훈련 데이터로 VGG2 데이터셋을 사용하고 임베딩 네트워크로 ResNet50을 사용하여 ArcFace를 훈련합니다. 표 7에서, 우리는 이전 최고 성능 모델들 [3, 40, 39]과 ArcFace의 TAR (@FAR=1e-4)를 비교합니다. ArcFace는 IJB-B 및 IJB-C 모두에서 성능을 분명히 향상시킬 수 있으며 (약 3~5%, 이는 오류의 상당한 감소입니다), 더 많은 훈련 데이터 (MS1MV2)와 더 deep Neural Network (ResNet100)을 활용함으로써 TAR (@FAR=1e-4)를 IJB-B에서 94.2%, IJB-C에서 95.6%까지 추가 향상시킬 수 있습니다. 그림 9에서, 우리는 제안된 ArcFace의 IJB-B 및 IJB-C에서의 전체 ROC 곡선을 보여주며, ArcFace는 FAR=1e-6 설정에서도 인상적인 성능을 달성하여 새로운 기준을 설정합니다.

**Trillion-Pairs 결과.** Trillion-Pairs 데이터셋 [1]은 갤러리 세트로 Flickr의 158만 장 이미지와 프로브 세트로 LFW [10]의 5.7K 신원에서 27.4만 장의 이미지를 제공합니다. 갤러리와 프로브 세트 사이의 모든 쌍이 평가에 사용되며 (총 0.4조 쌍입니다). 표 8에서, 우리는 다양한 데이터셋에서 훈련된 ArcFace의 성능을 비교합니다. 제안된 MS1MV2 데이터셋은 CASIA와 비교하여 성능을 분명히 향상시키며, 심지어 신원 수가 두 배인 DeepGlint-Face 데이터셋보다 약간 더 나은 성능을 보입니다. MS1MV2의 모든 신원과 DeepGlint의 아시아 유명인사를 결합했을 때, ArcFace는 최고의 식별 성능 84.840% (@FPR=1e-3)을 달성하고 리더보드의 최신 제출작(CIGIT IRSEC)과 비교할 만한 검증 성능을 보여줍니다.

**iQIYI-VID 결과.** iQIYI-VID 챌린지 [17]는 iQIYI의 다양한 쇼, 영화 및 TV 드라마에서 4934개의 신원으로부터 56만 5372개의 비디오 클립(훈련 세트 21만 9677, 검증 세트 17만 2860, 테스트 세트 17만 2835)을 포함합니다. 각 비디오의 길이는 1초에서 30초까지 다양합니다. 이 데이터셋은 얼굴, 의복, 목소리, 보행 및 자막을 포함한 다중 모달 단서를 제공하여 캐릭터 식별을 지원합니다. iQIYI-VID 데이터셋은 MAP@100을 평가 지표로 사용합니다. MAP(평균 정밀도)는 전반적인 평균 정확도율을 의미하며, 훈련 세트의 각 사람 ID(쿼리로 사용)에 대한 테스트 세트에서 검색된 해당 비디오의 평균 정확도율의 평균입니다.

표 9에서 보여주듯이, ResNet100으로 훈련된 MS1MV2 및 아시아 데이터셋과 결합된 ArcFace는 높은 기준선(MAP=79.80%)을 설정합니다. 각 훈련 비디오의 임베딩 feature을 기반으로, 우리는 분류 loss을 가진 추가적인 3단계 완전 연결 네트워크를 훈련하여 iQIYI-VID 데이터셋에서 맞춤형 feature 기술자를 얻습니다. iQIYI-VID 훈련 세트에서 학습된 MLP는 MAP를 6.60% 상당히 향상시킵니다. 모델 앙상블과 현장의 객체 및 장면 분류기 [20]에서의 컨텍스트 기능을 지원받아, 우리의 최종 결과는 러너업을 명확한 차이로 능가합니다 (0.99%).
# 4. 결론
본 논문에서는 얼굴 인식을 위한 DCNN을 통해 학습된 feature 임베딩의 차별적 힘을 효과적으로 강화할 수 있는 첨가 각 margin loss 함수를 제안합니다. 우리는 가장 포괄적인 실험을 통해 우리의 방법이 지속적으로 최신 기술을 능가함을 입증합니다. 이 논문에서 보고된 결과의 재현성을 촉진하기 위해 자세한 설명이 포함된 코드를 공개합니다.

**감사의 글.** Jiankang Deng은 Imperial President’s PhD 장학금과 NVIDIA로부터의 GPU 기부에 대한 재정적 지원을 인정합니다. Stefanos Zafeiriou는 EPSRC Fellowship DEFORM (EP/S010203/1), FACER2VM (EP/N007743/1) 및 Google Faculty Fellowship으로부터의 지원을 인정합니다.