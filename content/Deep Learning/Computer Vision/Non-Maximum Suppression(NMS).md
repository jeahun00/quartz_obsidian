Object Detection 을 공부하던 중 bounding box 를 선택하는 기법인 NMS 에 대한 것을 알게 되었는데 해당 기법에 대해 모호한 점이 많아 개념을 정리하고 직접 구현을 해보고자 한다.
# why are we using NMS?

* object detection task 에서 하나의 object 를 탐색할 때 여러개의 가능한 bounding box 가 존재한다.
* 이렇게 동일한 object 에 많은 bounding box 가 존재하면 이 모델을 사용할 때 어떤 box 가 최적인지 알아차리기가 힘들다.
* 따라서 이러한 bounding box 들 중 적합한 단일한 bounding box 만을 남기기 위해 NMS 를 사용한다.

![](img_store/Pasted%20image%2020240105173349.png)

# NMS pseudo code

![](img_store/Pasted%20image%2020231228145255.png)

* `line 1` : NMS procedure
	* $B$ : 초기 Bounding Box(이하 bbox) 후보들의 list
* `line 2` : 새로운 Bounding Box 후보 List 인 $B_{nms}$ 초기화
* `line 3` : bbox 후보 list 인 $B$ 에서 1 개 추출
	* for 문 통해 이 과정 반복
* `line 4` : 추출된 $b_i$ 를 삭제하는 과정을 False 로 초기화 
* `line 5` : line 3 에서 추출된 $b_i$ 와 비교할 $b_j$ 를 bbox 후보 list $B$ 에서 추출 
	* for 문 통해 위 과정 반복
* `line 6` : 
	* $same(b_i, b_j)$ : 후보로 뽑힌 2 bbox 간의 IoU 를 계산한다.
		* IoU 에 대한 설명은 아래에서 자세히 다룰 것이다.
	* $\lambda_{nms}$ : IoU 값에 대한 threshold 이다. 
		* IoU 가 일정 수준이상일 때만 판단하겠다는 뜻
		* 즉 임계치보다 작은 값들은 $b_i$ 와 비교하는 $b_j$ 가 너무 벗어난 bbox 이므로 무시하고 다음 bbox $b_j$ 를 보겠다는 뜻
* `line 7` ~ `line 8` : 
	* $b_i$ 보다 $b_j$ 의 confidence score 가 더 높을 때 $b_i$ 는 $b_j$ 보다 나쁜 후보이다.
	* 따라서 $b_i$ 를 삭제한다.
	* $score(c, b_j)$ : bbox $b_j$ 가 얼마나 객체 class $c$ 를 포함할 확률 값
		* yolo 같은 object detection model 에서 제시하는 confidence score 이다.
* `line 9` : $discard$ 가 $False$ 일 때 진입
* `line 10` :  새 bbox 후보 list 인 $B_{nms}$ 에 $b_i$ 추가
* `line 11` : 위의 `line 2` ~ `line 10` 의 과정이 끝나면 최종 bbox list 인 $B_{nms}$ return 

# Intersection over Unit(IoU)

2개의 box 가 있을 때 그 box 가 차지하는 전체 면적을 2개의 박스가 겹치는 면적에서 나눈 값이다.
![](img_store/Pasted%20image%2020231228153216.png)

# Algorithm

### 1. Pseudo Code 그대로 구현

1. **IoU Code**
```python
def IoU(box1, box2):
    """Compute the Intersection-Over-Union of two boxes.
    Args:
      box1: array of [cx, cy, width, height].
      box2: array of [cx, cy, width, height].
    Returns:
      iou: a float number in range [0, 1].
    """
    # Compute intersection
    x_left = max(box1[0] - 0.5 * box1[2], box2[0] - 0.5 * box2[2])
    y_top = max(box1[1] - 0.5 * box1[3], box2[1] - 0.5 * box2[3])
    x_right = min(box1[0] + 0.5 * box1[2], box2[0] + 0.5 * box2[2])
    y_bottom = min(box1[1] + 0.5 * box1[3], box2[1] + 0.5 * box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Compute union
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union_area = box1_area + box2_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou
```

2. **NMS code**
```python
def NMS(B, threshold):
    """Non-Maximum supression
    <below parameters are following the NMS paper notation>
    
    B : array of bboxes([cx, cy, w, h, c]) -> center format
        c : this is the confidence score of bbox

    B_nms : array of nms bboxes([True or False])
    discard : flag of keep or discard bbox
    """
    B_nms = []
    IoU_list = []
    discard = False
    discard_list = [False] * len(B)
    
    B = sorted(B, key=lambda x:x[4], reverse=True)

    for i in range(len(B) - 1):
        print("-----------------------------------")
        for j in range(i+1, len(B)):
            IoU_val = IoU(B[i], B[j])
            if IoU_val > threshold:
                discard_list[j] = True
            print('i, j, IoU, threshold : %d, %d, %.2f, %.2f' % (i, j, IoU_val, threshold))
        print(B)
        print("-----------------------------------")

    for i in range(len(B)):
        if discard_list[i] == False:
            B_nms.append(B[i])
    
    return B_nms
```

3. **plt 를 통한 사진 출력 함수**
```python
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

def draw_and_save_bounding_boxes(image_path, bboxes, save_path):
    """
    Draw bounding boxes on the image using matplotlib and save the result.

    Parameters:
    image_path (str): Path to the image.
    bboxes (list of lists): List of bounding boxes, each box is represented as [x, y, width, height].
    save_path (str): Path to save the resulting image.
    """
    # Load the image
    image = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    # Draw each bounding box
    for bbox in bboxes:
        if len(bbox) == 5:
            x, y, width, height = bbox[:-1]
        elif len(bbox) == 4:
            x, y, width, height = bbox
        rect = patches.Rectangle((x - width // 2, y - height // 2), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Remove axes for better visualization
    ax.axis('on')

    # Display the image
    plt.show()

    # Save the image
    fig.savefig(save_path, bbox_inches='tight')



```

4. **code 실행**
```python
bbox = [
    [200, 200, 140, 400, 0.67],
    [240, 260, 160, 440, 0.80],
    [270, 250, 140, 490, 0.70],
    [250, 250, 440, 170, 0.34],
    [240, 260, 180, 460, 0.98]
]


bbox_nms = NMS(bbox, 0.20)




image_path = './running.jpg'
save_path = './'
draw_and_save_bounding_boxes(image_path, bbox, save_path)
draw_and_save_bounding_boxes(image_path, bbox_nms, save_path)
print('->')
print(bbox_nms)
```

5. **결과**
![](img_store/Pasted%20image%2020231229133422.png)

REF :
* https://wikidocs.net/142645
* https://naknaklee.github.io/etc/2021/03/08/NMS/
* https://ctkim.tistory.com/entry/Non-maximum-Suppression-NMS

