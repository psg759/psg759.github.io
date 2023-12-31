---
layout: post
title:  "[이코테] Part 2"
date:   2023-09-04 20:20:00+0530
categories: Python
---
[이것이 코딩 테스트다 part 2 - Greedy]      



이것이 코딩 테스트다 part 2 - Greedy
---------------


그리디 알고리즘이란 어떤 문제가 있을때 단순 무식하게, 탐욕적으로 문제를 푸는 알고리즘이다.
쉽게말해, **현재 상황에서 지금 당장 좋은것만을 고르는 방법**   
정렬, 최단 경로 등의 알고리즘 유형은 그 알고리즘의 사용 방법을 정확히 알고 있어야만 해결 가능한 경우가 많지만 그리디 유형의 경우 외우지 않아도 풀 수 있는 가능성이 높다는 장점이 있다.

보통 코테에서 출제되는 그리디 알고리즘 유형 문제는 창의력을 요하기 때문에 특정 문제를 만났을 때 단순 현재 상황에서 가장 좋아보이는 것만을 선택해도 문제 풀 수 있는지가 관건이다!

그리디 알고리즘은 기준에 따라 좋은걸 선택하는 것이기 때문에
		_**가장 큰 순서대로, 가장 작은 순서대로**_
같은 힌트를 제시한다.
보통은 정렬 알고리즘을 사용할 때 만족하므로, 그리디와 정렬은 자주 짝으로 출제된다.😉

----
예제 3-1) 거스름돈

_**내가 짠 코드**_

```python 
a = int(input("거스름돈을 입력하세요 : "))
b = [500,100,50,10]
cnt = 0


for i in b:
        cnt += a // i
        if a % i == 0:
            break
        a = a % i
        
print(cnt)
```
이 문제를 그리디로 푼 핵심 관건은 **'가장 큰 화폐 단위부터'** 를 확인하고, 탐욕적으로 문제에 접근했을 때 정확한 답을 찾을 수 있다는 것
최소한의 아이디어를 떠올려 문제를 풀었다면, 이것이 정당한지에 대해 검토할 수 있어야 완벽한 답을 도출할 수 있다.
이것저것 아이디어를 떠올려보는 것이 중요함

----
실전문제 2) 큰 수의 법칙


_**내가 짠 코드**_
```python
n,m,k = map(int, input().split())
#n개의 데이터수, 총 m번 더해야 하고, 연속으로 더할 수 있는 횟수는 k번

data = list(map(int, input().split()))

data.sort()

result = 0

result = (m // k) * k * data[-1] + (m % k) * data[-2]

print(result) 
```
이 문제의 핵심은 특정 인덱스의 수가 연속해서 k번 더해질 수 있다는 것이다. 그 말은 가장 큰 수를 가진 특정 인덱스를 k번 더하고 다른 수를 1번 더한 후 다시 또 가장 큰 수를 가진 특정 인덱스를 k번 더할 수 있다는 것이다.

그렇다면 결국 필요한 것은 가장 큰 수, 두번째로 큰 수 이다.

----
실전문제 3) 숫자 카드 게임

_**내가 짠 코드**_
```python
n, m = map(int, input().split())
min = 0

for i in range(n): 
    data = list(map(int, input().split()))
    data.sort()
    if i == 0:
        min = data[0]
    if min < data[0]:
        min = data[0]
        
print(min)
```

이 문제는 파이썬 함수(max,min)를 숙지한 사람이 더 효율적으로 풀 수 있는 문제이다.
max,min 함수를 알고는 있었지만 막상 문제 풀 때 생각이 안나서 sort를 사용해서 문제를 풀었다.

----
실전문제 4) 1이 될 때까지

_**내가 짠 코드**_
```python
n, k = map(int, input().split())
result = 0

while n != 1:
    if n % k == 0:
        n = n / k
        result += 1
    else:
        n -= 1
        result += 1
        
    
print(int(result))
```
사실 책의 답변이랑 좀 달라서 맞게 푼건지는 모르겠지만..   
어쨌든 핵심은   
**n이 k의 배수가 될 때까지 1씩 빼기**   
**n을 k로 나누기**   
라고 한다.   
책이랑 비교해서 어느 부분을 다르게 짰는지 좀 더 봐야겠다 !


***
