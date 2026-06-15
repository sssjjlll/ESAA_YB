# ESAA YB 수상작 리뷰 (5)

### [ **제 4회 ETRI 휴먼이해 인공지능 논문경진대회 - 라이프로그 데이터를 활용한 수면 품질 및 상태 예측** ]

데이터

1. ETRI Lifelog Dataset

— 스마트폰 —

m_activity

- 걷기
- 뛰기
- 정지

m_usage_stats

- 앱 사용 시간

m_screen_status

- 화면 켜짐 여부

m_wifi

- 와이파이 정보

m_gps

- 위치

m_ble

- 블루투스 주변 기기

m_light

- 주변 밝기

— 스마트워치 —

w_hr

- 심박수

w_pedo

- 걸음 수
- 칼로리
- 이동거리

코드 흐름

1. Import Libaries

1. Data Labels
- Q1 : 수면 만족도 (0 / 1)
- Q2 : 취침 전 피로 (0 / 1)
- Q3: 취침 전 스트레스 (0 / 1)
- S1: 총 수면시간 (0~2등급)
- S2: 수면효율 (0~1)
- S3: 수면잠복기 (0~1)

→ 즉, 매일의 센서 데이터를 다음날 설문지표 예측으로 변경

1. Feature Engineering
- Temporal Normalization (시간 정렬)

→ 모든 센서를 동일한 시간축으로 맞춤

- Window Aggregation

→ 150분 단위로 묶음

- 평균
- 최대
- 최소
- 합

> 원래 센서 데이터를 180개 이상의 특징으로 변환
> 
- Heatmap 해석

→ 수면 예측에 중요한 영향을 주는 변수 파악.

→ 스마트폰 사용 패턴, 주변 환경, 심박수가 크게 영향을 줌을 확인할 수 있었음

1. CatBoost 학습

1) 전처리

1. 병렬 학습

→ LightGBM 학습 : 트리 모델이기에 비선형 관계를 잘 찾기 때문

→ TabNet 학습: Attention 사용하기 때문에 어떤 feature가 중요한지 자동 선택

1. 두 모델 결과 결합

ex) LightGBM (Q1=1일 확률 0.65), TabNet (Q1=1일 확률 0.55)일 때,

0.7 x 0.65 + 0.3 x 0.55

가중치 적용하여 최종 예측

```jsx
from lightgbm import LGBMClassifier
from pytorch_tabnet.tab_model import TabNetClassifier

# -------------------
# LightGBM
# -------------------
lgb_model = LGBMClassifier(
    random_state=42,
    n_estimators=1200,
    learning_rate=0.045,
    class_weight='balanced',
    force_col_wise=True
)

lgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[early_stopping(30)]
)

lgb_probs = lgb_model.predict_proba(X_test)

# -------------------
# TabNet
# -------------------

tabnet_model = TabNetClassifier(
    n_d=32,
    n_a=32,
    n_steps=5,
    gamma=1.5,
    lambda_sparse=1e-4,
    optimizer_params=dict(lr=2e-2),
    seed=42
)

tabnet_model.fit(
    X_train.values,
    y_train.values,
    eval_set=[(X_val.values, y_val.values)],
    eval_name=['valid'],
    eval_metric=['accuracy'],
    max_epochs=200,
    patience=30,
    batch_size=1024,
    virtual_batch_size=128
)

tabnet_probs = tabnet_model.predict_proba(X_test.values)

# -------------------
# Ensemble
# -------------------

test_pred += (
    0.3 * lgb_probs +
    0.7 * tabnet_probs
) / skf.n_splits

# -------------------
# Final prediction
# -------------------

final_preds[target] = np.argmax(test_pred, axis=1)
```

1. Evaluation
- TabNet: 44.37%
- CatBoost: 56.24%
- LightGBM: 55.81%
- LightGBM + Feature Engineering: 56.65%
- LightGBM + Ensemble: 57.2%
- TabBoost: 59:18%

→ 효과가 매우 컸음

새롭게 알게 된 내용 / 어려운 점 / 배울 점

#### 새롭게 알게 된 내용

1. 라이프로그 데이터는 단순한 센서 기록이 아니라, 스마트폰 사용 패턴, 심박수, 활동량, GPS, Wi-Fi 정보 등을 활용하여 수면의 질(Q1), 피로도(Q2), 스트레스(Q3)와 같은 개인의 상태를 예측할 수 있다는 점을 알게 되었다.
2.  LightGBM과 TabNet 같은 서로 다른 모델을 결합한 앙상블 기법이 예측 성능 향상에 효과적이라는 것을 이해하게 되었다.

#### 어려운 점

1. 분 단위로 수집되는 방대한 시계열 데이터를 어떻게 하루 단위의 특징(feature)으로 변환하는지 이해하는 과정이 가장 어려웠다. 특히 시간 정렬(Temporal Normalization), 윈도우 집계(Window Aggregation), Feature Engineering 과정과 TabNet의 Attention 메커니즘을 이해하는 데 시간이 필요했다.

#### 배울점

1. 좋은 예측 성능은 단순히 복잡한 모델을 사용하는 것보다 데이터 전처리와 Feature Engineering에 크게 좌우된다는 점을 배웠다. 
2.  여러 모델의 장점을 결합하는 앙상블 기법과 하이퍼파라미터 튜닝의 중요성을 알게 되었으며, 실제 데이터 분석에서는 데이터에 대한 충분한 이해가 모델 선택만큼 중요하다는 것을 배울 수 있었다.