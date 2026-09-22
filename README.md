# Berka 시계열 현금흐름 모델링

Berka Dataset의 account-period 마스터테이블을 이용해 계좌별 미래 inflow/outflow를 예측하고, 다음과 같이 net flow를 계산하는 프로젝트다.

```text
net_flow = inflow_amount - outflow_amount
```

현재 운영 기준은 **weekly 단위**와 **기간별 RMSE 평균 기준 모델 선택**이다. Monthly 실험 결과는 비교·분석 기록으로 보존하지만, 이후 주력 실험에서는 기본적으로 사용하지 않는다.

자세한 구현 및 실험 이력은 [docs/PROJECT_STATUS.md](docs/PROJECT_STATUS.md)를 참고한다.

## 빠른 시작

PowerShell 기준:

```powershell
py -3.11 -m venv ts
.\ts\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

원본 데이터 위치:

```text
Berka Dataset/Berka Dataset/
```

기본 마스터테이블:

```text
Berka Dataset/Berka Dataset/master_tables/berka_master_m.parquet
Berka Dataset/Berka Dataset/master_tables/berka_master_w.parquet
```

Enriched 마스터테이블 생성:

```powershell
python enrich_master_features.py --frequency both
```

## 주력 평가 기준

각 모델·weekly forecast version에서 다음 값을 계산한다.

```text
period_mean_rmse = mean(RMSE(period_1), ..., RMSE(period_horizon))
```

앞으로는 `net_flow`의 `period_mean_rmse`가 가장 낮은 weekly 모델을 주력 모델로 선택한다. 단, MAE, WAPE, asymmetric MAE/RMSE, nMAE, nRMSE도 함께 저장해 투자 가능 금액 산정에서 과소예측 위험을 확인한다.

## 모델 실행 예시

주별 LR:

```powershell
python linear_regression_forecast.py --frequency weekly --method both --output-dir outputs/linear_regression_weekly
```

주별 XGBoost:

```powershell
python xgboost_forecast.py --frequency weekly --method both --output-dir outputs/xgboost_weekly --n-jobs -1
```

주별 LightGBM CPU:

```powershell
python lightgbm_forecast.py --frequency weekly --method both --device-type cpu --output-dir outputs/lightgbm_weekly --n-jobs -1
```

CUDA가 활성화된 LightGBM 빌드가 있는 서버에서는 `--device-type cuda`를 사용한다.

주별 DeepAR:

```powershell
python deepar_forecast.py --frequency weekly --method recursive --device cuda --output-dir outputs/deepar_weekly
```

Enriched DeepAR:

```powershell
python deepar_forecast_enriched.py --frequency weekly --method recursive --device cuda --output-dir outputs/deepar_enriched_weekly
```

## 모델 비교 그래프

기존 10개 account를 대상으로 모델별 best run을 선택해 비교하는 스크립트:

```powershell
python model_comparison_plots_both_rmse.py
```

특정 계좌만 비교:

```powershell
python model_comparison_plots_both_rmse.py --account-ids 406 949 2043
```

결과는 `outputs/model_comparison_both_rmse/`에 저장된다.

