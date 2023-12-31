---
layout: post
title:  "[이코테] Part 1"
date:   2023-08-17 20:20:00+0530
categories: Python
---
[이것이 코딩 테스트다 part 1]      



이것이 코딩 테스트다 part 1
---------------

   
**코딩 테스트 개념** 

코딩 테스트란 ? 
기업 / 기관에서 직원이나 연수생을 선발하기 위한 목적으로 시행되는 '__일종의 문제풀이 시험__'

**코딩 테스트의 유형**

온라인 코딩 테스트
정해진 시간에 응시자가 사이트에 접속해 문제를 읽고 해답을 소스코드 형태로 작성하여 제출하면 온라인 저지 서비스가 정답 여부를 알려주고 점수를 부여함.
(타인과 문제 풀이를 공유하지 않는 선에서 인터넷 검색을 허용하는 경우가 많아서 오프라인 코딩 테스트에 비해 높은 성적을 받을 확률이 높음)

오프라인 코딩 테스트
응시자가 시험장에 방문해서 치르는 시험이다. 대체로 인터넷 검색이 허용되지 않으며 회사에서 제공하는 컴퓨터 환경에서 바로 시험에 응시한다.
(대체로 오프라인으로 보는 경우 별도의 면접실로 안내되어 문제풀이에 대해 설명하기도 한다.)

**코딩 테스트 준비를 돕는 다양한 서비스**

코드업 > 난이도가 낮은 문제가 많아 처음 공부하는 사람에게 적합함

백준 온라인 저지 > 국내에서 가장 유명한 알고리즘 문제 풀이 사이트

프로그래머스 > 국내 알고리즘 학습 사이트, 카카오 문제를 제공함

SW Expert Academy > 삼성에서 공식적으로 제공하는 알고리즘 학습 사이트

**코딩 테스트에 유리한 언어는 ?**

파이썬 OR C++ (종류, 쓰임, 환경에서 사용 언어가 다 다르지만 대체적으로 이럼)

**온라인 개발 환경**

리플릿,파이썬 튜터, 온라인 GDB, 파이참 .. 등등 많지만
나는 vs code로 개발하는게 제일 편함

**복잡도**

시간복잡도 > 알고리즘을 위해 필요한 연산의 횟수
공간복잡도 > 알고리즘을 위해 필요한 메모리의 양

시간복잡도를 표현할 때는 보통 빅오 표기법을 사용한다.
(밑으로 갈수록 큰값)

**O(1)	:	상수시간
O(logN)	:	로그시간
O(N)	:	선형시간
O(NlogN)	:	로그선형시간
O(N^2)	:	이차시간
O(N^3)	:	삼차시간
O(2^n)	:	지수시간**


흔한 케이스는 아니지만 상수 차수의 값을 무시하면 안되는 경우도 있음.
3N^3 + 5N^2 + 1000000인 알고리즘의 경우 N의값이 작으면 상수를 무시하면 안됨.

좀 더 능숙해지면 문제를 해석하기 전에 조건을 먼저 보고 사용 가능한 알고리즘을 추릴 수 있다고 한다. 예를들어 데이터 개수 N이 1000만 개를 넘어가며 시간 제한이 1초라면 O(N)을 예상할 수 있고, 또는 데이터의 크기나 탐색 범위가 100억이나 1000억을 넘어가는 경우 O(logN)의 시간 복잡도를 가진 알고리즘을 작성 해야 할 것이다.

대표적인 문제 풀 때의 예시이다. ( 시간 제한이 1초인 문제들 )
+ N의 범위가 500인 경우 : 시간 복잡도가 O(N^3)인 알고리즘을 설계하면 문제를 풀 수 있다.
+ N의 범위가 2000인 경우 : 시간 복잡도가 O(N^2)인 알고리즘을 설계하면 문제를 풀 수 있다.
+ N의 범위가 100000인 경우 : 시간 복잡도가 O(NlogN)인 알고리즘을 설계하면 문제를 풀 수 있다.
+ N의 범위가 10000000인 경우 : 시간 복잡도가 O(N)인 알고리즘을 설계하면 문제를 풀 수 있다.

공간 복잡도를 표기할 때도 시간 복잡도처럼 빅오 표기법을 이용한다.
일반적으로 메모리 사용량 기준은 MB단위로 제시된다.

**시간과 메모리 측정**

파이썬에서는 프로그램 수행 시간과 메모리 사용량을 측정할 수 있는 코드가 있음.

time 라이브러리를 사용해서 time 함수로 걸린 시간을 출력할 수 있음.

***
