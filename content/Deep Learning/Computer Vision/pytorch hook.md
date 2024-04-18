hook 이란 패키지를 만드는 코드에서 중간에 원하는 코드를 삽입하는 기능

# Tensor 에 Hook 적용
tensor.register_hook 으로 적용
```python
tensor = torch.rand(1, requires_grad=True)

def tensor_hook(grad):
    pass

tensor.register_hook(tensor_hook)

# tensor는 backward hook만 있다.
tensor._backward_hooks
```

# Module 에 Hook 적용
- `register_forward_hook`, `register_forward_pre_hook`, `register_full_backward_hook`으로 등록할 수 있다.
### register_forward_hook : 
forward pass를 하는 동안 (output이 계산할 때 마다) 만들어놓은 hook function을 호출.  (즉 forward 이후에 실행!)
이렇게 등록한 함수에 인자로 모듈이 실행되기 전 입력값과 실행 후 출력값을 받음 (input, output)
hook 에서 input 을 건드린다 하더라도 foward 과정에서 모델에 영향을 주지는 않는다.

```python
import torch
from torch import nn

# Add 모델을 수정하지 마세요! 
class Add(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, x1, x2):
        output = torch.add(x1, x2)
        return output

# 모델 생성
add = Add()


# TODO : hook를 이용해서 전파되는 output 값에 5를 더해보세요!
def hook(module, input, output):
  return output + 5

add.register_forward_hook(hook)


# 아래 코드는 수정하실 필요가 없습니다!
x1 = torch.rand(1)
x2 = torch.rand(1)

output = add(x1, x2)
```

### register_forward_pre_hook
forward pass를 하기 직전에 hook function을 호출.  (즉 forward 실행 이전에 실행됨)
이렇게 등록한 함수에 인자로 모듈이 실행되기 전 입력값만을 받음 (input)

```python
import torch
from torch import nn


# Add 모델을 수정하지 마세요! 
class Add(nn.Module):
    def __init__(self):
        super().__init__() 

    def forward(self, x1, x2):
        output = torch.add(x1, x2)

        return output

# 모델 생성
add = Add()

# TODO: 답을 x1, x2, output 순서로 list에 차례차례 넣으세요! 
answer = []

# TODO : pre_hook를 이용해서 x1, x2 값을 알아내 answer에 저장하세요
def pre_hook(module, input):
    answer.append(input[0])
    answer.append(input[1])
    # return input[0], input[1]


# 아래 코드는 수정하실 필요가 없습니다!
x1 = torch.rand(1)
x2 = torch.rand(1)

add.register_forward_pre_hook(pre_hook)
output = add(x1, x2)
print(x1, x2, answer)
```

### register_full_backward_hook
backward pass를 하는 동안 (gradient가 계산될 때마다) hook function을 호출.  
이렇게 등록한 함수에 인자로 backpropagation에서의 gradient 값들을 받음 (grad_input, grad_output)

```python
import torch
from torch import nn
from torch.nn.parameter import Parameter


# Model 모델을 수정하지 마세요! 
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.W = Parameter(torch.Tensor([5]))

    def forward(self, x1, x2):
        output = x1 * x2
        output = output * self.W

        return output

# 모델 생성
model = Model()


# TODO: 답을 x1.grad, x2.grad, output.grad 순서로 list에 차례차례 넣으세요! 
answer = []

# TODO : hook를 이용해서 x1.grad, x2.grad, output.grad 값을 알아내 answer에 저장하세요
def module_hook(module, grad_input, grad_output):
    answer.append(grad_input[0])
    answer.append(grad_input[1])
    answer.append(grad_output[0])

model.register_full_backward_hook(module_hook)
# 아래 코드는 수정하실 필요가 없습니다!
x1 = torch.rand(1, requires_grad=True)
x2 = torch.rand(1, requires_grad=True)

output = model(x1, x2)
output.retain_grad()
output.backward()
```

REF :
* https://ohsy0512.tistory.com/27
* https://velog.io/@ss-hj/hook%EC%9D%B4%EB%9E%80-feat.-PyTorch