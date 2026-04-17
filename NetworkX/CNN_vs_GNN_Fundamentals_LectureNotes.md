# CNN vs GNN: 근본적 차이

## 강의 목표

1. CNN의 **Translation Invariance**를 정확히 이해한다
2. GNN의 **Permutation Invariance**가 왜 필요한지 이해한다
3. 두 불변성이 각 아키텍처에서 어떻게 구현되는지 배운다
4. WWAI의 theme_graph_builder에 이를 어떻게 적용하는지 본다

---

# 제1부: CNN의 Translation Invariance

## 1.1 정의

**Translation Invariance(공간적 시프트 불변성)**란:

입력 이미지를 어떤 방향으로 이동시켜도 (픽셀 값이 이동하는 것 자체는 다르지만), 
**같은 객체/패턴으로 인식**한다는 성질입니다.

### 수학적 표현

```
f(I) = f(I(·-Δ))

여기서:
- I: 입력 이미지
- f: CNN 모델
- Δ: 이동 벡터 (Δx, Δy)

의미: 이미지를 Δ만큼 평행이동해도 같은 결과 f를 반환
```

### 시각적 예시

```
원본 이미지:              오른쪽으로 이동:
┌─────────────────┐       ┌─────────────────┐
│ ■ ■ □ □ □ □ □  │       │ □ ■ ■ □ □ □ □  │
│ ■ ■ □ □ □ □ □  │  →    │ □ ■ ■ □ □ □ □  │
│ □ □ □ □ □ □ □  │       │ □ □ □ □ □ □ □  │
│ □ □ □ □ □ □ □  │       │ □ □ □ □ □ □ □  │
└─────────────────┘       └─────────────────┘
  "고양이 눈"               "고양이 눈"

CNN 출력:  같음! ✓
```

---

## 1.2 CNN에서 어떻게 구현되는가?

### 핵심: Weight Sharing (가중치 공유)

```python
# Conv2D 레이어의 원리

Kernel (3×3):
┌─────┬─────┬─────┐
│ w1  │ w2  │ w3  │
├─────┼─────┼─────┤
│ w4  │ w5  │ w6  │
├─────┼─────┼─────┤
│ w7  │ w8  │ w9  │
└─────┴─────┴─────┘

이 커널을 이미지의 모든 위치에서 반복 적용:

위치 (0,0):    위치 (0,1):      위치 (0,2):
┌───┬───┬───┐ ┌───┬───┬───┐   ┌───┬───┬───┐
│w1 │w2 │w3 │ │w1 │w2 │w3 │   │w1 │w2 │w3 │
├───┼───┼───┤ ├───┼───┼───┤   ├───┼───┼───┤
│w4 │w5 │w6 │ │w4 │w5 │w6 │   │w4 │w5 │w6 │
├───┼───┼───┤ ├───┼───┼───┤   ├───┼───┼───┤
│w7 │w8 │w9 │ │w7 │w8 │w9 │   │w7 │w8 │w9 │
└───┴───┴───┘ └───┴───┴───┘   └───┴───┴───┘

→ 같은 가중치 (w1~w9)를 모든 위치에서 재사용
```

### 구체적 계산

```
Input Image I:
┌────┬────┬────┬────┐
│ i1 │ i2 │ i3 │ i4 │
├────┼────┼────┼────┤
│ i5 │ i6 │ i7 │ i8 │
├────┼────┼────┼────┤
│ i9 │i10 │i11 │i12 │
└────┴────┴────┴────┘

위치 (0,0)에서의 convolution:
output[0,0] = w1*i1 + w2*i2 + w4*i5 + w5*i6 + ... (3×3 패치)

위치 (0,1)에서의 convolution:
output[0,1] = w1*i2 + w2*i3 + w4*i6 + w5*i7 + ... (3×3 패치, 오른쪽으로 1칸 이동)

→ 같은 가중치 (w1~w9)를 모든 위치에 적용
→ 위치가 변해도 패턴은 같은 방식으로 인식
```

---

## 1.3 왜 CNN이 Translation Invariant일 수 있는가?

### 이미지의 근본적 특성

**이미지는 절대 좌표계(absolute coordinate system)를 가집니다**

```
픽셀 (x, y)의 의미:
- (100, 150): 이미지의 좌상단에서 오른쪽 100, 아래 150 픽셀
- (200, 150): 이미지의 좌상단에서 오른쪽 200, 아래 150 픽셀

→ 좌표 자체가 절대적 위치 정보
```

### 자연 세계의 성질

**패턴은 위치와 무관하게 반복됩니다**

```
고양이의 특징 (whisker pattern):
- 왼쪽 이미지에 있든
- 오른쪽 이미지에 있든
- 가운데 이미지에 있든

→ 같은 고양이의 whisker

이건 물리 세계의 사실입니다. 고양이는 위치와 무관하게 고양이입니다.
```

---

## 1.4 Translation Invariance의 한계

### 문제: Pooling의 부정확성

```
Max Pooling:

원본:              약간 이동:
[1 2 3]           [2 3 4]
[4 5 6]     →     [5 6 7]
[7 8 9]           [8 9 10]

Max([1,2,4,5]) = 5    Max([2,3,5,6]) = 6

→ 결과가 정확히 같지는 않음 (약간의 robustness만 가능)
```

### 중요 통찰

**CNN의 translation invariance는 이론적으로는 완벽하지만, 
실제 pooling과 sub-sampling 때문에 근사적(approximate)입니다.**

---

# 제2부: GNN의 Permutation Invariance

## 2.1 그래프가 왜 다른가?

### 그래프에는 절대 좌표가 없다

```
이미지:                           그래프:
(0,0)───(1,0)                   Node A
 │       │                        ├─ Node B
(0,1)───(1,1)                    ├─ Node C
                                 └─ Node D

→ (x,y) 좌표가 절대값         → 노드 이름은 라벨일 뿐
→ 위치가 고정됨              → "위치" 개념이 없음
```

### 그래프의 구조적 특성

**그래프에서 중요한 것은 "누가 누와 연결되어 있는가"이지,
노드가 "어디에 있는가"가 아닙니다.**

```
같은 구조:

구조1:              구조2 (노드 순서만 다름):
  A                   C
 /|\                 /|\
B C D               B A D

수학적으로는 동일한 그래프입니다.
```

---

## 2.2 Permutation Invariance의 정의

**Permutation Invariance(순열 불변성)**란:

노드의 순서(라벨링)가 바뀌어도 같은 구조로 인식한다는 성질입니다.

### 수학적 표현

```
π: 순열(permutation)

f(G) = f(π(G))

의미: 그래프 G의 노드를 임의로 재정렬(π)해도 
      같은 결과 f를 반환
```

### 구체적 예시

```
Aggregation 함수:

neighbors = [h_B, h_C, h_D]

mean(neighbors) = (h_B + h_C + h_D) / 3

만약 순서를 바꾸면:
mean([h_D, h_B, h_C]) = (h_D + h_B + h_C) / 3

결과가 같다! ✓
```

---

## 2.3 GraphSAGE에서의 구현

### Mean Aggregator (가장 간단한 경우)

```python
def mean_aggregator(neighbor_features):
    """
    neighbor_features: 이웃 노드들의 특징 벡터 리스트
    반환: 이웃 특징의 평균
    """
    return neighbor_features.mean(axis=0)

# 예시
h_B = [1.0, 2.0, 3.0]
h_C = [2.0, 3.0, 4.0]
h_D = [3.0, 4.0, 5.0]

result1 = mean_aggregator([h_B, h_C, h_D])
# = [(1+2+3)/3, (2+3+4)/3, (3+4+5)/3]
# = [2.0, 3.0, 4.0]

result2 = mean_aggregator([h_D, h_B, h_C])  # 순서 다름
# = [(3+1+2)/3, (4+2+3)/3, (5+3+4)/3]
# = [2.0, 3.0, 4.0]

assert result1 == result2  # ✓ 같다!
```

### Max Aggregator

```python
def max_aggregator(neighbor_features):
    """
    이웃 특징 중 각 차원의 최댓값
    """
    return neighbor_features.max(axis=0)

# 예시
h_B = [1.0, 5.0, 3.0]
h_C = [2.0, 3.0, 4.0]
h_D = [3.0, 4.0, 2.0]

result1 = max_aggregator([h_B, h_C, h_D])
# = [max(1,2,3), max(5,3,4), max(3,4,2)]
# = [3.0, 5.0, 4.0]

result2 = max_aggregator([h_D, h_C, h_B])  # 순서 다름
# = [max(3,2,1), max(4,3,5), max(2,4,3)]
# = [3.0, 5.0, 4.0]

assert result1 == result2  # ✓ 같다!
```

### LSTM Aggregator (순서에 민감함!)

```python
# LSTM은 순서에 민감합니다 (시계열이기 때문)
# 해결책: 여러 순열을 시도하고 평균

def lstm_aggregator_permutation_invariant(neighbors, num_random=10):
    """
    LSTM aggregator를 permutation invariant하게 만들기
    """
    results = []
    
    for _ in range(num_random):
        # 이웃을 무작위로 섞기
        shuffled = random_shuffle(neighbors)
        # LSTM에 통과
        h_agg = lstm(shuffled)
        results.append(h_agg)
    
    # 여러 순열의 결과를 평균 → permutation invariant!
    return mean(results)
```

---

## 2.4 왜 GNN에 Permutation Invariance가 필요한가?

### 문제: 만약 순서에 민감했다면?

```python
# 나쁜 aggregator (순서 민감함)
def bad_weighted_aggregator(neighbors):
    # 첫 번째 이웃에 높은 가중치, 마지막에 낮은 가중치
    return 0.5 * neighbors[0] + 0.3 * neighbors[1] + 0.2 * neighbors[2]

# 같은 그래프인데 노드 순서만 다르면:
result1 = bad_weighted_aggregator([A, B, C])  # 0.5*A + 0.3*B + 0.2*C
result2 = bad_weighted_aggregator([B, A, C])  # 0.5*B + 0.3*A + 0.2*C

→ result1 ≠ result2  (같은 구조인데 다른 결과!)
→ 이건 말이 안 됨!
```

### 좋은 aggregator (순서 무관)

```python
def good_aggregator(neighbors):
    # 순서 무관: 합이나 평균
    return mean(neighbors)
    # 또는 return sum(neighbors)
    # 또는 return max(neighbors)

result1 = good_aggregator([A, B, C])
result2 = good_aggregator([B, A, C])

→ result1 == result2  (같은 구조 → 같은 결과!) ✓
```

---

# 제3부: CNN vs GNN 대비

## 3.1 근본 차이 표

| 특성 | CNN | GNN |
|------|-----|-----|
| **좌표계** | 절대 좌표 (x, y) | 없음 (상대 관계만) |
| **공간성** | 물리적 격자 구조 | 추상적 그래프 구조 |
| **불변성** | Translation Invariance | Permutation Invariance |
| **불변의 예** | 고양이가 왼쪽/오른쪽 모두 고양이 | 노드 순서가 바뀌어도 같은 구조 |
| **가중치 공유 방식** | 공간상 모든 위치에 적용 | 모든 노드에 동일 규칙 적용 |
| **연산** | Convolution (격자 위) | Message Passing (그래프 위) |
| **문제점** | 공간 정보 무시 불가 | 그래프 구조 정의가 중요 |

---

## 3.2 구체적 예시 비교

### 예시 1: 패턴 탐지

**CNN 관점**
```
이미지에서 "모서리(edge)" 탐지

원본:                 시프트 후:
┌─────────────┐       ┌─────────────┐
│ ░░░ ■■■ ░░░ │       │ ░░ ░░░ ■■■ │
│ ░░░ ■■■ ░░░ │  →    │ ░░ ░░░ ■■■ │
│ ░░░ ■■■ ░░░ │       │ ░░ ░░░ ■■■ │
└─────────────┘       └─────────────┘

"모서리" 패턴을 어디서든 찾음 ✓
```

**GNN 관점**
```
테마 네트워크에서 "중심 기업 + 주변 기업" 패턴 탐지

원본:                  노드 순서 변경:
Samsung                SK
  ├─ SK                  ├─ Samsung
  ├─ Intel              ├─ Intel
  └─ TSMC               └─ TSMC

같은 "중심 + 주변" 패턴을 같은 방식으로 인식 ✓
```

---

### 예시 2: 데이터 변형에 대한 강건성

**CNN: 픽셀 이동은 허용하지만 완전 변형 불가**
```
원본 고양이          90도 회전한 고양이        앞뒤 반전 고양이
[정상 인식]          [약간 인식 가능]          [거의 못 인식]

→ Translation은 불변이지만, 
   회전/반전은 별개의 불변성이 필요
```

**GNN: 노드 순서는 무시하지만 엣지 구조가 변하면 영향**
```
Original Network:
A─B, A─C, B─C (삼각형)

Permutation:
B─A, C─A, C─B (같은 삼각형)
→ 같은 결과 ✓

Edge change:
A─B, A─C, B─D (다른 구조)
→ 다른 결과 ✓ (당연함)
```

---

# 제4부: WWAI theme_graph에서의 적용

## 4.1 theme_graph_builder의 구조

```
theme_graph:
- 282 theme nodes (예: Semiconductor, AI, Green Energy)
- 2366 ticker nodes (개별 기업/종목)
- 관계: 테마 ─ 종목 (bipartite)

예시:
┌─────────────────┐
│  Semiconductor  │
├────────┬────────┤
│        │        │
Samsung SK    Intel
```

---

## 4.2 Permutation Invariance가 왜 중요한가?

### 같은 테마의 종목들

**상황:**
```
Semiconductor 테마 노드가 다음과 연결:
- Samsung
- SK Hynix
- Intel

GraphSAGE aggregator로 이 세 종목 특징을 모음
```

**문제: 종목 순서가 바뀐다면?**
```
Aggregation 순서1:
AGG([h_Samsung, h_SK, h_Intel])

Aggregation 순서2 (DB에서 다르게 조회됨):
AGG([h_Intel, h_Samsung, h_SK])

→ Mean aggregator를 썼다면?
   같은 결과! ✓ (순서 무관)

→ 만약 가중된 합을 썼다면?
   다른 결과... ✗ (큰 문제!)
```

---

## 4.3 구체적 WWAI 코드 패턴

### Good Practice ✓

```python
# GraphSAGE mean aggregator
def aggregate_ticker_features_for_theme(ticker_embeddings, theme_node):
    """
    theme_node 하나에 연결된 모든 ticker들의 embedding을 aggregate
    ticker 순서는 무관해야 함
    """
    # 이웃 tickers를 모아서 평균 계산
    neighbor_embeddings = [ticker_embeddings[tid] for tid in theme_node.tickers]
    
    # Mean aggregation: 순서 무관 ✓
    aggregated = np.mean(neighbor_embeddings, axis=0)
    
    return aggregated
```

### Bad Practice ✗

```python
# 절대 금지!
def bad_aggregate(ticker_embeddings, theme_node):
    """
    ticker 순서에 의존하는 aggregation → 위험!
    """
    tickers = theme_node.tickers  # [A, B, C] 또는 [C, A, B]?
    
    # 가중된 합: 순서 민감 ✗
    weights = np.array([0.5, 0.3, 0.2])
    embeddings = [ticker_embeddings[tid] for tid in tickers]
    
    result = np.dot(weights, embeddings)  # 순서에 따라 결과 달라짐!
    
    return result
```

---

## 4.4 실제 적용 체크리스트

### GraphSAGE를 WWAI에 올바르게 적용하기

```
□ Aggregator 선택
  ✓ Mean / Max / Sum (순서 무관)
  ✗ Position-dependent weighted sum (순서 민감)

□ 그래프 구조 정의
  ✓ 명확하고 재현가능한 엣지 규칙
  ✗ 임시방편이거나 일관성 없는 엣지

□ 배치 처리
  ✓ 이웃 수집 시 정렬 순서 무시
  ✗ 특정 순서를 가정하고 처리

□ 검증
  ✓ 같은 테마의 종목 임베딩이 consistent
  ✗ 종목 조회 순서에 따라 임베딩이 변함
```

---

# 제5부: 심화 개념

## 5.1 Translation vs Permutation: 언제 어느 것을 쓸까?

### Translation Invariance가 필요한 경우
```
- 이미지 인식
- 이미지 분할
- 비디오 분석 (시간축 + 공간축)
- 음성 신호 처리 (시간축의 이동)
```

### Permutation Invariance가 필요한 경우
```
- 그래프 신경망 (노드 순서 무관)
- Set 학습 (순서 없는 데이터)
- 포인트 클라우드 처리 (3D 점들의 순서 무관)
- Attention mechanism (쿼리-키 매칭은 순서 무관)
```

---

## 5.2 Equivariance vs Invariance

### 더 정교한 개념: Equivariance

```
Invariance (불변):
f(transform(x)) = f(x)
→ 입력이 변해도 출력이 같음

Equivariance (동변):
f(transform(x)) = transform(f(x))
→ 입력이 변하면 출력도 같은 방식으로 변함
```

### 예시

```
CNN convolution: equivariant to translation
f(I(·-Δ)) = f(I)(·-Δ)
→ 이미지를 이동하면 특징맵도 같은 방향으로 이동

GNN message passing: equivariant to permutation
f(permute(G)) = permute(f(G))
→ 노드를 재정렬하면 임베딩도 같게 재정렬됨
```

**GraphSAGE aggregation이 invariant가 아니라 equivariant:**
```
mean([h_B, h_C, h_D]) = h_agg
mean([h_D, h_B, h_C]) = h_agg  (같음! = invariant)

하지만 내부 임베딩은:
h_A_new = UPDATE(h_A, h_agg)
→ 이건 같은 노드 A에 항상 적용됨 (equivariance의 성질)
```

---

## 5.3 Heterophily와 Permutation Invariance의 충돌

### 문제: 비동질적 이웃

```
극극한 케이스:

노드 A의 이웃:
- B: 매우 유사한 특징
- C: 매우 다른 특징

Mean aggregator:
AGG([h_B, h_C]) = (h_B + h_C) / 2

→ 정보 손실! B와 C의 정보가 섞임
```

### 해결책: 주의가 필요

```
Heterophilic 그래프에서는:

1) GraphSAGE보다 attention-based GNN 검토
   (Attention은 각 이웃의 중요도를 배움)

2) 그래프 구조 재정의
   (정말 이웃으로 포함해야 하는가?)

3) 다른 aggregator 시도
   (Mean이 항상 최선은 아님)
```

---

# 제6부: WWAI에 대한 권고

## 6.1 theme_graph에서 주의할 점

### ✓ 좋은 설계

```python
class ThemeGraphSAGE:
    def __init__(self, theme_graph, ticker_features):
        self.graph = theme_graph
        self.ticker_features = ticker_features
        
    def sample_neighbors(self, theme_node, num_samples=10):
        """
        Same theme의 ticker들을 무작위로 샘플
        순서는 무시 (mean aggregator를 쓸 것이므로)
        """
        tickers = list(theme_node.tickers)
        sampled = random.sample(tickers, min(num_samples, len(tickers)))
        return sampled
    
    def aggregate(self, neighbor_embeddings):
        """
        Mean aggregation: 순서 무관
        """
        return np.mean(neighbor_embeddings, axis=0)
```

### ⚠️ 주의: Graph Construction

```
theme_graph의 엣지를 어떻게 정의할 것인가?

1) 분류 기반: 공식 분류체계 (GICS, KRX 섹터)
   → 명확하고 재현가능

2) 상관관계 기반: 수익률 상관계수 > 0.7
   → 시간에 따라 변함 (신중!)

3) 뉴스 공동 출현: "AI"가 함께 나오는 기업들
   → 동적이지만 noisy할 수 있음

→ 각각은 다른 "테마"를 만들 수 있음!
   선택이 결과를 크게 좌우함
```

---

## 6.2 LRS vs GraphSAGE: 혼동하지 말기

```
LRS (Long-Range Spillover):
→ "t 시점 수익률 변동이 t+k 시점에 도달"
→ 시간 차원의 인과 전파

GraphSAGE Embedding:
→ "t 시점의 네트워크 구조를 반영"
→ 정적 구조의 표현

둘은 근본적으로 다름!

LRS는 예측에 쓸 수 있지만,
GraphSAGE는 구조 학습에 씀
```

---

## 6.3 경제 해석의 주의

### ✓ 올바른 해석

```
"GraphSAGE 임베딩이 비슷한 두 종목은
 네트워크 관점에서 구조적으로 비슷하다"

→ 이건 데이터에서 관찰된 사실
```

### ✗ 잘못된 해석

```
"GraphSAGE 임베딩이 비슷하면 미래 수익률이 비슷하다"

→ 이건 증명되지 않은 주장!
   (구조 유사성 ≠ 수익률 동시성)
```

### 검증 필요

```python
# 올바른 검증 과정

# 1단계: GraphSAGE 임베딩 학습
embeddings = graphsage.train(theme_graph, ticker_features)

# 2단계: 임베딩 유사도 계산
similarity = cosine_similarity(embeddings)

# 3단계: 경제적 검증 (별도)
# - 수익률 상관관계 계산
# - Spillover effect 측정
# - Sector leadership 동시성 확인

# 4단계: 임베딩과 경제량의 관계 분석
# correlation(embedding_similarity, return_correlation) = ?
```

---

# 결론

## 한눈에 보기

| 개념 | CNN | GNN/GraphSAGE |
|------|-----|---|
| **세상의 성질** | 같은 패턴이 어디나 나타남 | 같은 관계 구조가 여러 곳에 나타남 |
| **불변성** | Translation (공간 이동) | Permutation (순서 변경) |
| **가정** | 절대 좌표계 존재 | 상대 관계만 중요 |
| **구현** | Weight sharing @ all locations | Weight sharing @ all nodes |
| **주의점** | 공간 정보는 중요 | 그래프 구조 정의가 중요 |

---

## WWAI theme_graph에서 실천 사항

1. **Mean/Max aggregator 사용** (순서 무관)
2. **그래프 엣지 명확하게 정의** (재현가능)
3. **LRS와 혼동하지 말기** (시간 차원 vs 정적 구조)
4. **경제 해석은 별도 검증** (구조 ≠ 예측력)
5. **Heterophily 확인** (종목들이 정말 homophilic한가?)

---

## 추가 학습 자료

### 개념
- Permutation invariant functions: DeepSets (Zaheer et al., 2017)
- Graph neural networks: Kipf & Welling (2016) GCN
- GraphSAGE: Hamilton et al. (2017)

### 심화
- Equivariance in GNNs: Weiler et al. (2021)
- Heterophily: Zhu et al. (2020), He et al. (2022)
- Set functions and Wasserstein distance
