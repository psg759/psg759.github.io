---
layout: post
title:  "[이코테] Part 2"
date:   2023-10-01 20:20:00+0530
categories: Python
---
[이것이 코딩 테스트다 part 2 - DFS/BFS]      



이것이 코딩 테스트다 part 2 - DFS/BFS
---------------


DFS / BFS 개념에 들어가기에 앞서 알아두어야 할 자료구조 기초지식 ! > **Stack / Queue**

Stack & Queue는 다음의 두 핵심 함수로 구성된다.
- 삽입(Push) : 데이터를 삽입한다.
- 삭제(Pop) : 데이터를 삭제한다.   
( ※ 이 두 함수를 사용할 때는 오버플로 / 언더플로를 주의! )

#### Stack 🍱
스택은 도시락통(보온병)과 같은 원리다. 도시락을 쌀 때도 아래서부터 차곡차곡, 꺼낼때도 위에서부터 !   
쉽게 말해 **선입후출**(처음에 넣은것을 제일 나중에 꺼낸다) or **후입선출** (마지막에 넣은것을 제일 처음 꺼낸다) 라는 뜻이다.

파이썬에서 stack을 이용할 때는 별도의 라이브러리를 사용할 필요 없이, 기본 리스트에서 append(), pop() 메서드를 이용하여 구현할 수 있다.

#### Queue 👥
큐는 콘서트 대기줄이다. 먼저 온 사람이 먼저 들어가고 나가는 것도 먼저 나간다 =3   
**선입선출**(처음에 넣은것이 제일 처음 꺼내진다)은 편의점 물건 진열할 때도 많이 쓰이는데 먼저 들어온 걸 먼저 나가도록 한다고 이해하면 된다.

파이썬으로 큐를 구현할 때는 collections 모듈에서 제공하는 deque 자료구조를 활용하면 된다
> from collections import deque

마찬가지로 삽입이나 삭제시에는 append(), popleft() 메서드를 이용한다.

***
#### DFS ⛏️
DFS는 Depth-First-Search (깊이 우선 탐색)는 그래프에서 <u>깊은 부분을 우선적으로 탐색</u>하는 알고리즘이다.
DFS의 스택 자료구조를 이용한 동작 과정은 다음과 같다.
- 탐색 시작 노드를 스택에 삽입하고 방문 처리를 한다. > visited[start] = True
- 스택의 최상단 노드에 방문하지 않은 인접 노드가 있으면 그 인접 노드를 스택에 넣고 방문 처리, 방문하지 않은 인접 노드가 없으면 스택에서 최상단 노드를 꺼냄
- 2번의 과정을 더 이상 수행할 수 없을 때까지 반복한다.   

**DFS 예제 코드**
```python
def dfs(graph, v, visited):
    visited[v] = True
    print(v, end=' ')
    
    for i in graph[v]:
        if not visited[i]:
            dfs(graph, i, visited)
            
graph = [
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]

visited = [False] * 9

dfs(graph, 1, visited) 
```

#### BFS 🕸️
BFS는 Breadth First Search(너비 우선 탐색)는 <u>가까운 노드부터 탐색</u>하는 알고리즘이다.
BFS의 큐 자료구조를 이용한 동작 과정은 다음과 같다.
- 탐색 시작 노드를 큐에 삽입하고 방문 처리를 한다. >> visited[start] = True
- 큐에서 노드를 꺼내 해당 노드의 인접 노드 중에서 방문하지 않은 노드를 모두 큐에 삽입하고 방문 처리를 한다.
- 2번의 과정을 더 이상 수행할 수 없을 때까지 반복한다.

**BFS 예제 코드**
```python
from collections import deque

def bfs(graph, start, visited):
    
    queue = deque([start])
    visited[start] = True
    
    while queue:
        v = queue.popleft()
        print(v, end=' ')
        
        for i in graph[v]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
                
graph = [
    [],
    [2,3,8],
    [1,7],
    [1,4,5],
    [3,5],
    [3,4],
    [7],
    [2,6,8],
    [1,7]
]

visited = [False] *9

bfs(graph, 1, visited)
```

__유형별 check__
DFS > 간선으로 연결된 애들을 한 뭉텅이로 처리하는 문제 유형 ..
BFS > 한 노드에서 특정 어디까지 최단 ~ (거리, 시간 등)의 문제 유형 ..

근데 문제를 풀다보니 DFS로 풀지 BFS로 풀지 고민하게 되는 유형들이 있는데,
DFS 같은 경우는 깊이로 파고 들어가다가 원하는 값을 찾았을 때 바로 그 값만을 return 하고 끝내고 싶은데 재귀함수들이 펼쳐져 있어서 이것들을 다 마무리 짓고 빠져나와야하는 점이 좀 해결하기 어려운 것 같다. > 해결방법을 더 찾아보자 !💡
그래서 여러 문제들 다 dfs로 시도 했다가 결국 bfs로 바꿔서 해결 성공한 경우가 많았음 .. !

***

실전문제 3) 음료수 얼려먹기

_**책 코드**_
```python
# 실전문제 3) 음료수 얼려먹기
#dfs각 자리에서 주변에 뭐가 있는지 조사할때 재귀적으로 좌표를 옮겨가면서 조사하는 문제에서 많이 찾는듯

n,m = map(int, input().split())
graph = []

check = 0

for i in range(n):
    graph.append(list(map(int, input())))

#좌표값이 1인 경우는 자동으로 False를 리턴하면서 다음 좌표로 넘어가게 됨
def dfs(x,y):
    #범위를 멋어난 경우도 False return
    if x < 0 or y < 0 or x > n-1 or y > m-1:
        return False
    #좌표값이 0인 경우 상하좌우를 체크하며 좌표값이 0인값이 있는지 체크, 있다면 재귀 안에 함수로써 한꺼번에 1로 바뀌게 됨
    #재귀로 펼쳐지는 함수값들의 return값은 상관이 없고 주변에 0이 있다면 1로 변경해주는게 point
    if graph[x][y] == 0:
        graph[x][y] = 1
        dfs(x-1, y)
        dfs(x+1, y)
        dfs(x,y+1)
        dfs(x,y-1)
        return True
    return False

#좌표를 돌면서 체크
for i in range(n):
    for j in range(m):
        #0인 음료수가 1개이든 여러개이든 묶어있다면 하나로 봄
        if dfs(i,j) == True:
            check += 1
            
print(check)
```

실전문제 4) 미로탈출

_**책 코드**_
```python
# 실전문제 4) 미로탈출
from collections import deque

#bfs를 사용하면 초기값에서 어디든지 최단 거리를 구할 수가 있음
n,m = map(int, input().split())

graph = []
for i in range(n):
    graph.append(list(map(int, input())))

#상하좌우 방향을 순회하기 위한 리스트
dx = [-1,1,0,0]
dy = [0,0,-1,1]    

def bfs(x,y):
    #초기값을 queue에 넣어줌
    queue = deque()
    queue.append([x,y])
    
    while queue:
        x,y = queue.popleft()
        print(x,y)
        
        #상하좌우를 탐색하며 1이 있는지 탐색
        for i in range(4):
            nx = x+dx[i]
            ny = y+dy[i]
            
            if nx < 0 or ny < 0 or nx > n-1 or ny > m-1:
                continue
            if graph[nx][ny] == 0:
                continue
            #1이 존재한다면 이전의 값에 1을 더해주고 해당 좌표값을 또 큐에 넣어 그 값의 상하좌우를 탐색할 수 있도록 함.
            if graph[nx][ny] == 1:
                graph[nx][ny] = graph[x][y] + 1
                queue.append([nx,ny])
    return graph[n-1][m-1]

#초기값 설정
print(bfs(0,0))
```

***
