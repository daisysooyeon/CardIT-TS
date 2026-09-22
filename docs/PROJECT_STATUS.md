# Berka 시계열 현금흐름 모델링 프로젝트 문서

최종 정리 기준: 2026-09-22

## 1. 프로젝트 목표

각 `account_id`에 대해 미래 기간의 inflow와 outflow를 별도로 예측하고, 다음과 같이 net flow를 계산한다.

```text
net_flow = inflow - outflow
```

최종적으로는 향후 지출과 최저잔액을 고려해 계좌별 투자 가능 금액을 산정하는 것이 목적이다.

현재까지는 다음 모델을 구현·비교했다.

1. Linear Regression
2. ARIMA
3. ARIMAX 탐색 구현
4. XGBoost
5. LightGBM
6. DeepAR

향후 주력 조건은 다음과 같이 변경한다.

| 항목 | 기존 실험 | 앞으로의 주력 기준 |
|---|---|---|
| 시간 단위 | monthly와 weekly 모두 | weekly 중심 |
| 모델 선택 기준 | horizon RMSE와 기간별 RMSE를 모두 확인 | net flow의 기간별 RMSE 평균 |
| 평가 대상 | inflow, outflow, net flow | net flow 우선, inflow/outflow 보조 확인 |
| 예측 방식 | recursive/direct 모두 | 모델별 지원 방식에 따라 수행하되 weekly를 우선 |

Monthly 결과는 폐기하지 않고 과거 비교 자료로 보존한다.

## 2. 데이터와 마스터테이블

Berka Dataset은 체코 은행 데이터로, 관측기간은 1993-01-01부터 1998-12-31이다. 현재 모델 입력은 account-period 단위의 parquet 마스터테이블이다.

```text
Berka Dataset/Berka Dataset/master_tables/berka_master_m.parquet
Berka Dataset/Berka Dataset/master_tables/berka_master_w.parquet
```

주별 데이터는 일요일 시작 기간을 사용한다. 테스트 구간은 주별로 1998-06-28부터 1998-12-27까지 27개 기간이다. 월별 테스트 구간은 1998년 하반기 6개월이다.

마스터테이블에는 이미 다음과 같은 값이 포함되어 있다.

- `inflow_amount`, `outflow_amount`, `net_flow`
- 거래금액과 거래횟수
- `birth_year`, `age`, `age_group`, `gender`
- `account_frequency`
- 거래 목적별 금액·횟수
- 대출 관련 상태 및 상환 정보
- 기존 lag/rolling/bagged 컬럼

다만 기본 LR/XGBoost/LightGBM 파이프라인은 parquet에 저장된 lag/rolling/bagged 컬럼을 그대로 읽지 않고, 학습 origin과 예측 history를 기준으로 다시 생성한다. 이 방식은 recursive 테스트에서 미래 실제값이 lag/rolling feature로 섞이는 것을 막기 위한 것이다.

## 3. Enriched 마스터테이블

다음 명령으로 원본 마스터테이블을 보존하면서 enriched 버전을 만든다.

```powershell
python enrich_master_features.py --frequency both
```

생성 파일:

```text
Berka Dataset/Berka Dataset/master_tables/berka_master_m_enriched.parquet
Berka Dataset/Berka Dataset/master_tables/berka_master_w_enriched.parquet
```

### 3.1 달력·공휴일 feature

1993~1998년 체코 공휴일은 외부 현재 연도 패키지에 의존하지 않고 `enrich_master_features.py` 안에 정확한 날짜를 명시했다.

월별:

- `holiday_count`
- `has_holiday`
- `month_sin`, `month_cos`
- `quarter_sin`, `quarter_cos`
- `scheduled_loan_repayment`

주별:

- `holiday_count`
- `has_holiday`
- `is_week_before_holiday`
- `is_week_after_holiday`
- `month_sin`, `month_cos`
- `quarter_sin`, `quarter_cos`
- `week_sin`, `week_cos`
- `has_month_start`, `has_month_end`
- `scheduled_loan_repayment`

### 3.2 예정 대출 상환액

`loan.payments`, `loan.date`, `loan.duration`을 사용해 계약별 미래 상환 스케줄을 생성한다.

- 대출 실행일이 12일 이전이면 해당 월 12일부터 시작
- 12일 이후면 다음 달 12일부터 시작
- `duration`만큼 월별 납부일 생성
- 월별 모델은 해당 월에 금액 반영
- 주별 모델은 12일이 포함된 일요일 시작 주에 금액 반영

예측시점 이후에 실행된 대출이 학습 feature에 들어가는 누수를 막기 위해, enriched forecast 공통 모듈에서 origin 기준 as-of masking을 적용한다. 테스트 기간의 달력 feature와 계약 스케줄은 미리 계산해 lookup한다.

## 4. 누수 방지 규칙

모든 모델의 테스트 구간에서는 테스트 실제 inflow/outflow를 미래 입력으로 사용하지 않는다.

### Recursive

```text
t 예측
→ t 예측값을 history에 추가
→ t+1 예측
```

이전 예측 오차가 누적될 수 있다.

### Direct

origin 하나에서 horizon별 모델을 이용해 전체 기간을 한 번에 예측한다. 테스트 기간의 실제값이나 테스트 기간에서 계산한 lag/rolling 값은 사용하지 않는다.

### 예측값 후처리

금액 예측은 음수가 되지 않도록 operational prediction에 대해 0 미만을 0으로 clip한다. 이는 실제 거래 데이터의 outlier 제거와는 다른 처리다.

## 5. 모델별 구현 요약

### 5.1 Linear Regression

파일: `linear_regression_forecast.py`

- inflow와 outflow를 별도 모델로 학습
- net flow는 예측 후 `inflow - outflow`로 계산
- recursive/direct 및 monthly/weekly 지원
- 수치 feature는 `StandardScaler`
- 범주형 feature는 one-hot encoding
- 계수 크기로 feature 영향도를 분석

### 5.2 ARIMA

파일: `arima_forecast.py`

- account별·flow별 ARIMA 모델
- recursive/direct 및 다양한 `(p,d,q)` order grid 실험
- 최소 history 부족, fitting 실패, 비유한 예측 등은 diagnostic에 기록
- 실패시 정의된 fallback을 사용하며, 빈 값을 평가에 넣지 않음
- 주력 비교에서는 non-enriched ARIMA 결과를 사용

ARIMAX 파일인 `arimax_forecast.py`는 별도 탐색 구현이다. 계산량이 크고 현재 표준 비교 결과의 주력 모델에는 포함하지 않는다.

### 5.3 XGBoost

파일: `xgboost_forecast.py`

- tree model이므로 수치 feature scaling 불필요
- `objective=reg:squarederror`
- 주요 grid: `max_depth`, `n_estimators`, `learning_rate`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha`, `gamma`
- feature importance는 gain 기준

대표 결과 경로:

```text
outputs/xgboost_regularization_sweep/
```

### 5.4 LightGBM

파일: `lightgbm_forecast.py`

- XGBoost와 동일한 데이터 누수 방지·평가 구조
- tree model이므로 scaling 불필요
- regularization/sampling sweep은 243개 grid
- 주요 grid: `num_leaves`, `min_child_samples`, `min_child_weight`, `reg_lambda`, `reg_alpha`, `min_split_gain`, `subsample`, `colsample_bytree`
- feature importance는 gain 기준
- CUDA-enabled LightGBM build가 있을 때만 `--device-type cuda` 사용

대표 결과 경로:

```text
outputs/lightgbm_regularization_sampling_sweep/
```

### 5.5 DeepAR

파일: `deepar_forecast.py`, `deepar_forecast_enriched.py`

- 현재는 LSTM 기반
- account별 별도 모델이 아니라 여러 account를 함께 학습하는 global model
- `account_id` embedding은 포함하지 않음
- account별 history와 scale은 사용하지만, account ID 자체는 neural feature가 아님
- Gaussian negative log-likelihood로 log 변환 target의 분포를 학습
- 대표 `prediction_inflow/outflow`는 현재 평균 예측값
- p10/p50/p90은 분포에서 계산할 수 있으나 enriched 결과 저장에서는 대표 평균값을 사용
- 계산 자원을 고려해 recursive만 구현

Enriched DeepAR은 알려진 미래 달력 feature, 계약성 대출 상환액, static feature를 함께 사용한다. 현재 잔차만 예측하는 구조는 아니다.

### 5.6 Enriched Random Forest

파일: `random_forest_forecast_enriched.py`

- enriched XGBoost/LightGBM과 동일한 공통 runner 사용
- `berka_master_m_enriched.parquet`, `berka_master_w_enriched.parquet` 사용
- inflow와 outflow를 별도 모델로 학습하고 net flow는 사후 계산
- monthly/weekly 및 recursive/direct 지원
- tree model이므로 numeric scaling 불필요
- `criterion="squared_error"`, `bootstrap=True`
- 기본 grid:
  - `n_estimators`: 300
  - `max_depth`: 8, 16
  - `min_samples_split`: 2, 10
  - `min_samples_leaf`: 1, 5
  - `max_features`: 0.7, 1.0
- feature importance는 XGBoost/LightGBM의 gain이 아니라 Random Forest의 impurity-based mean decrease in impurity(MDI)다.

기본 grid는 16개 조합이다. 실행 예시는 다음과 같다.

```powershell
python random_forest_forecast_enriched.py `
    --frequency weekly --method both `
    --output-dir outputs/random_forest_enriched `
    --n-jobs -1
```

특정 grid를 파일로 지정할 때 사용하는 JSON 형식:

```json
{
  "n_estimators": [300, 500],
  "max_depth": [8, 16],
  "min_samples_split": [2, 10],
  "min_samples_leaf": [1, 5],
  "max_features": [0.7, 1.0]
}
```

실행 결과에는 `predictions.parquet`, `metrics.csv`, `horizon_metrics.csv`, `feature_importance.csv`, `training_info.csv`, `model.joblib`가 저장된다.

### 5.7 모델별 feature importance 해석 주의

LR은 계수 크기, XGBoost/LightGBM은 gain, Random Forest는 MDI를 사용한다. 서로 값의 절대 크기를 직접 비교하지 말고, 각 모델 내부에서 feature 순위를 비교해야 한다.

## 6. 평가 지표

각 target에 대해 다음 지표를 저장한다.

- MAE
- RMSE
- WAPE
- Asymmetric MAE
- Asymmetric RMSE
- nMAE
- nRMSE

### Horizon RMSE

모든 테스트 기간의 account-level 예측을 한꺼번에 모아 계산한 RMSE다. 전체 horizon에서 큰 오차가 발생한 모델에 민감하다.

### 기간별 RMSE 평균

각 `horizon_step`별로 전체 account의 RMSE를 계산한 뒤 평균한다.

```text
period_mean_rmse = mean(rmse_step_1, ..., rmse_step_H)
```

앞으로의 모델 선택 기준은 weekly `net_flow`의 이 값이다. 따라서 horizon 합계 RMSE가 조금 낮더라도 기간별 RMSE 평균이 더 높으면 주력 모델로 선택하지 않는다.

## 7. 현재 비교·시각화 결과

모델 비교 스크립트:

```text
model_comparison_plots_both_rmse.py
```

이 스크립트는 LR 분석에서 사용한 동일한 10개 account를 사용하고, 모델별·frequency별로 다음 두 best run을 별도 선택한다.

- `horizon_rmse_best`
- `period_mean_rmse_best`

결과:

```text
outputs/model_comparison_both_rmse/
├─ horizon_rmse_best/
│  ├─ monthly_net_flow_accounts.png
│  ├─ weekly_net_flow_accounts.png
│  ├─ monthly/account_plots/account_*.png
│  └─ weekly/account_plots/account_*.png
├─ period_mean_rmse_best/
│  ├─ monthly_net_flow_accounts.png
│  ├─ weekly_net_flow_accounts.png
│  ├─ monthly/account_plots/account_*.png
│  └─ weekly/account_plots/account_*.png
└─ selected_best_runs.csv
```

특정 account만 다시 그릴 수 있다.

```powershell
python model_comparison_plots_both_rmse.py --account-ids 406 949 2043
```

## 8. 서버/GPU 이전 체크리스트

1. Python 3.11 가상환경 생성
2. `requirements.txt` 설치
3. `Berka Dataset` 폴더를 동일한 상대경로로 복사하거나 경로를 코드에 맞춤
4. CUDA가 필요한 경우 `torch.cuda.is_available()` 확인
5. LightGBM은 `device_type=cuda`가 실제로 활성화된 빌드인지 확인
6. 먼저 `--max-accounts 2`로 smoke test
7. smoke test 후 weekly 전체 account 실행
8. 결과의 `training_info.csv`, `metrics.csv`, `predictions.parquet`, `fit_diagnostics.csv` 확인

DeepAR GPU 확인:

```powershell
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

LightGBM CUDA는 Python 옵션만으로 CPU wheel이 GPU로 바뀌지 않는다. CUDA learner가 포함된 LightGBM을 별도로 빌드해야 한다.

## 9. Outlier 처리 현황

거래금액 outlier는 삭제하지 않았다.

- 대규모 거래도 원본 집계값에 포함
- winsorizing/IQR 기반 삭제 미적용
- master table에 outlier flag 없음
- rolling median은 smoothing feature이지 거래 삭제가 아님
- 예측값의 음수 clip만 적용

따라서 현재 평가 지표는 실제 대규모 거래를 포함한 금액 기준이다.

## 10. Git에 올릴 파일과 제외할 파일

Git에는 코드, 문서, requirements, 재현 가능한 grid 설정을 올린다.

올리지 않는 것:

- 원본 Berka CSV/parquet
- enriched parquet
- `outputs/`의 대량 예측·모델·그래프 파일
- `ts/` 가상환경
- `__pycache__/`

대용량 데이터와 결과는 별도의 스토리지나 Google Drive/서버 저장소로 관리하고, Git 문서에는 생성 방법과 경로만 기록한다.
