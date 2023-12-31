---
layout: post
title:  "[이코테] Part 2"
date:   2023-09-25 20:20:00+0530
categories: Python
---
[이것이 코딩 테스트다 part 2 - 구현]      



이것이 코딩 테스트다 part 2 - 구현(Implementation)
---------------


코딩테스트에서 구현이란 '**머릿속에 있는 알고리즘을 소스코드로 바꾸는 과정**'   
즉, 모든 범위의 코테 문제 유형을 포함하는 개념이다.
(이 책에선 __완전탐색__ (모든 경우의 수를 다 계산) / __시뮬레이션__ (문제에서 제시한 알고리즘을 한단계식 차례로 직접 수행)도 구현의 범주에 들어가서 학습한다.)   
그러려면 문법을 숙지하는 것, 타자를 빨리치는 것 등에 대한 피지컬을 요구하는 것과 같다.

보통 이제 그러면 어려운 구현 문제의 유형은 무엇이냐 !
- 알고리즘은 간단하지만 코드가 지나치게 길어지는 문제
- 특정 소수점 자리까지 출력해야하는 문제
- 문자열이 입력으로 주어졌을 때 파싱을 해야하는 문제 
등 .. __사소한 조건 설정__ 이 많은 문제일수록 코드 구현이 까다롭다.
   
   
    
**구현시 고려할 메모리 제약 사항**
[파이썬 기준] 
변수의 표현 범위 - 직접 변수의 자료형을 지정할 필요가 없어 매우 큰 수의 연산 또한 기본으로 지원한다.

리스트 크기 - 수백만 개 이상의 데이터를 처리할 때는 메모리 제한을 염두에 두고 코딩해야 하는데, 그 중에서도 크기가 1000만 이상인 리스트가 있다면 메모리 용량으로 문제룰 풀 수 없게 될 수도 있다.

**구현문제 접근방법**
구현은 보통 문제 길이가 길지만 어렵진 않아서 문법만 익숙하다면 쉽게 풀 수 있다.
또 문제 제출 시 PYPY3가 파이썬3의 문법을 지원하면서 속도가 더 빠르기 때문에 PYPY3로 제출하면 실행시간을 단축시킬 수 있다.

***
예제 4-1) 상하좌우

_**내가 짠 코드**_
```python
#예제 4-1) 상하좌우
n = int(input())

data = input().split()

x = 1
y = 1

for i in data:
    if i == 'R':
        if y == 5:
            continue
        y += 1
    elif i == 'L':
        if y == 1:
            continue
        y -= 1
    elif i == 'U':
        if x == 1:
            continue
        x -= 1
    elif i == 'D':
        if x == 5:
            continue
        x += 1

print(x,y)

```

예제 4-2) 시각

_**내가 짠 코드**_
```python
#예제 4-2) 시각
n = int(input('시간 입력 : '))
count = 0
#00시 00분 00초 ~ 00시 59분 59초 사이에 3이 있을 확률 계산

for i in range(n+1):
    for j in range(0,60):
        for k in range(0,60):
            if i == 3 or i == 13 or i == 23:
                count += 1
            elif j == 3 or j == 13 or j == 23 or 30 <= j <= 39 or j == 43 or j == 53:
                count += 1
            elif k == 3 or k == 13 or k == 23 or 30 <= k <= 39 or k == 43 or k == 53:
                count += 1
            
print(count)
```
실전문제 2 왕실의 나이트

_**내가 짠 코드**_
```python
#실전문제 2) 왕실의 나이트

space = input()

count = 0
direction = [[2,1],[2,-1],[1,2],[-1,-2],[-2,-1],[-2,1],[-1,2],[1,-2]]

match = {
    'a' : 1,
    'b' : 2,
    'c' : 3,
    'd' : 4,
    'e' : 5,
    'f' : 6,
    'g' : 7,
    'h' : 8,
}

col = match[space[0]]
row = int(space[1])

for i in direction:
    if col + i[0] < 1 or col + i[0] > 8:
        continue
    elif row + i[1] < 1 or row + i[1] > 8:
        continue
    else:
        count += 1
        
print(count)
```


***
