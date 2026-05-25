# ESAA YB 수상작 리뷰 (4)

### [헬스케어 데이터 경진대회]

데이터

표본코호트 2.2 DB

: 의료급여수급권자의 진료내역 정보에서 추출한 표본 데이터

- RN_INDI
- SEX
- SGG
- GAIBJA_TYPE
- CTRB_Q10
- DSB_TYPE_CD
- G1E_OBJ_YN
- SMPL_TYPE_CD
- DSB_SVRT_CD_V2

.

.

.

코드 흐름

[사전 입력]

- 성별
- 지역 코드
- 장애 유형 코드
- 장애 심각도
- 강제 수준
- 주요 질병 코드

▼

주요 질병코드 input → 질병 분류

▼

증상 input (SICK_SYM1, SICK_SYM2)

```jsx
with open('classification.pkl', 'rb') as file:
    model = pickle.load(file)

def predict():
    if request.method == 'POST':
    # 세션에서 사용자 데이터 가져오기
    user data = session.get('useer data')
    if not user_data:
       return "User data not found in session."
       
    # 입력 변수 SICK_SYM1과 SICK_SYM2
    try:
        SICK_SYM1 = (request.form['SICK_SYM1'])
        SICK_SYM2 = (request.form['SICK_SYM2'])
    except ValueError:
        return "Invalid input for SICK_SYM1 or SICK_SYM2."
        
    # 예측 데이터 구성
    input_data = [
        (user_data['gender']),
        (user_data['region_code']),
        (user_data['disability_type']),
        (user_data['disability severity']),
        (user_data['economic_level'),
        (user_data['main_disease_code']),
        SICK_SYM1,
        SICK_SYM2
    ]
    
    # 모델 예측
    prediction = model.predict([input_data])
    
    # 예측 결과 반환
    return prediction
```

▼

[Model Classification]

- SEX EGG
- DSB_TYPE_CD
- DSB_SVRT_CD_V2
- CTRB_Q10
- SICK_SYM1
- SICK_SYM2
- MCEX_SICK_SYM

▼

Model Output - 병원종류

- 상급병원
- 종합병원
- 요양원

▼

병원 종류, 지역 코드 input

▼

병원명, 총 의사수 파악

새롭게 알게 된 내용 / 어려운 점 / 배울 점

- 맞춤형 추천 시스템을 만드는 방법에 대한 전체적 흐름, 해당 코드들에 대해 알아갈 수 있었다.
- python 객체 직렬화 모델 pickle을 활용하여, 입력 변수를 수집하고 예측 데이터를 구성한 후, 모델 예측 및 결과 반환하는 방법에 대해 새롭게 알게 되었다.
- 성별·지역·장애유형 등 범주형 변수를 모델 입력으로 쓰기 위해 인코딩(원-핫, 라벨 인코딩 등)과 결측치 처리 방식을 일관되게 설계해야 한다는 것을 다시금 배워갈 수 있었다.