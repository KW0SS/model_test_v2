# raw 데이터 기반 Group Split XGBoost 학습 결과

## 1. 실험 목적

이번 실험의 목적은 새로 수집한 `raw` 폴더의 JSON 데이터를 실제 모델 학습에 사용할 수 있는 형태로 변환한 뒤, `종목코드` 기준 Group Split 환경에서 XGBoost 모델의 성능을 확인하는 것이다. 기존에는 이미 만들어진 전처리 완료 CSV를 중심으로 모델을 학습했지만, 이번에는 `raw/healthy`, `raw/delisted`에 저장된 원본 재무제표 JSON을 다시 읽어 학습용 데이터셋을 새로 생성하였다.

특히 이번 실험에서는 단순히 점수를 높이는 것보다, 같은 기업의 데이터가 train, validation, test에 동시에 들어가지 않도록 하는 것을 중요하게 보았다. 그래서 `종목코드`를 기준으로 Group Split을 적용하였다. 이 방식은 행 단위 split보다 성능이 낮게 나올 수 있지만, 모델이 이미 본 기업과 비슷한 데이터를 다시 맞히는 상황을 줄일 수 있다.

최종 모델 선택 기준은 validation F1 score로 두었다. test set은 최종 확인용으로만 사용하였다. 따라서 test F1이 더 높은 후보가 있더라도, 그 후보를 test 결과만 보고 최종 모델로 선택하지는 않았다.

## 2. 사용 데이터

이번 학습에는 `raw` 폴더에 저장된 JSON 파일을 사용하였다. 별도의 API 호출은 새로 하지 않았고, 이미 받아둔 파일만 사용하였다.

- 입력 폴더: `raw/healthy`, `raw/delisted`
- 사용 보고기간: `ANNUAL`, `Q1`, `H1`, `Q3`
- 라벨 기준: `raw/healthy`는 `label=0`, `raw/delisted`는 `label=1`
- 생성된 학습 CSV: `C:\kwoss_C\model_test_v2\재무비율\raw_모델학습용_20260513_raw.csv`
- 생성된 전처리 완료 CSV: `C:\kwoss_C\model_test_v2\재무비율\raw_모델학습전처리완료_20260513_raw.csv`

데이터 변환 결과는 다음과 같다.

| 항목 | 값 |
| --- | ---: |
| raw JSON 로드 수 | 6,078 |
| 학습 행 수 | 6,077 |
| 스킵 파일 수 | 0 |
| 고유 종목코드 수 | 270 |
| label=0 행 수 | 3,229 |
| label=1 행 수 | 2,848 |

보고기간별 행 수는 다음과 같다.

| 보고기간 | 행 수 |
| --- | ---: |
| ANNUAL | 1,725 |
| H1 | 1,446 |
| Q1 | 1,456 |
| Q3 | 1,450 |

산업군별로는 `Communication Services`, `Consumer Discretionary`, `Industrials`, `Information Technology` 등이 포함되었다. 다만 `healthy` 데이터는 일부 산업군에 더 많이 분포하고, `delisted` 데이터는 더 다양한 산업군에 존재한다. 따라서 이번 결과는 산업군 분포 차이의 영향을 완전히 제거한 결과라고 보기는 어렵고, 이 점은 해석할 때 주의해야 한다.

## 3. raw 데이터 변환 방식

raw JSON 파일명은 `종목코드_연도_보고기간.json` 형태로 되어 있었다. 예를 들어 `036260_2020_Q1.json`이면 종목코드는 `036260`, 연도는 `2020`, 보고기간은 `Q1`로 해석하였다.

재무제표 JSON 안에는 계정명, 계정 ID, 당기 금액 등이 들어 있다. 기존 파이프라인에서 사용하던 계정 추출 로직을 재사용하여 자산, 부채, 자본, 매출액, 영업이익, 당기순이익, 현금및현금성자산, 차입금, 감가상각비 등 필요한 값을 찾았다. 이후 이 값들을 이용해 부채비율, 유동비율, 매출액순이익률, 총자본회전율 같은 재무비율을 계산하였다.

성장률 계열 변수는 같은 종목코드와 같은 보고기간의 전년도 값과 비교하였다. 예를 들어 `2020_Q1`의 매출액증가율은 `2019_Q1`의 매출액과 비교해서 계산하고, `2020_ANNUAL`은 `2019_ANNUAL`과 비교하였다. 이렇게 한 이유는 분기와 연간 재무제표의 기준 기간이 다르기 때문에, 서로 다른 보고기간끼리 비교하면 성장률 의미가 흐려질 수 있기 때문이다.

보고기간 자체는 문자열 feature로 넣지 않았다. 대신 다음과 같은 one-hot flag 형태의 숫자 feature로 변환하였다.

| feature | 의미 |
| --- | --- |
| `period_Q1` | 1분기 보고서이면 1, 아니면 0 |
| `period_H1` | 반기 보고서이면 1, 아니면 0 |
| `period_Q3` | 3분기 보고서이면 1, 아니면 0 |
| `period_ANNUAL` | 연간 보고서이면 1, 아니면 0 |

최종 feature 수는 총 64개였다. 이 안에는 재무비율 feature, 결측 여부를 표시하는 `_missing` feature, 보고기간 flag feature가 포함된다.

## 4. Group Split 결과

이번 실험에서는 `종목코드`를 기준으로 train, validation, test를 나누었다. 같은 종목코드가 서로 다른 split에 동시에 들어가면 데이터 누수 가능성이 있기 때문이다.

Group Split 결과는 다음과 같다.

| split | 행 수 | label=1 | label=0 | 고유 종목코드 수 |
| --- | ---: | ---: | ---: | ---: |
| train | 4,253 | 2,002 | 2,251 | 188 |
| validation | 968 | 458 | 510 | 41 |
| test | 856 | 388 | 468 | 41 |

split 간 종목코드 중복은 모두 0개였다.

| 비교 | 중복 종목코드 수 |
| --- | ---: |
| train-validation | 0 |
| train-test | 0 |
| validation-test | 0 |

따라서 이번 학습은 같은 기업의 서로 다른 연도 또는 분기 데이터가 train과 test에 동시에 들어가는 문제를 피한 상태에서 진행되었다.

## 5. 평가 지표

이번 실험에서는 `precision`, `recall`, `F1 score`, `ROC-AUC`, `PR-AUC`를 함께 확인하였다.

`precision`은 모델이 상폐 위험 기업이라고 예측한 것 중 실제 상폐기업이 얼마나 되는지를 의미한다. `recall`은 실제 상폐기업 중 모델이 얼마나 많이 찾아냈는지를 의미한다. `F1 score`는 precision과 recall의 균형을 보는 지표이다.

`ROC-AUC`는 모델이 label 0과 label 1을 전반적으로 얼마나 잘 구분하는지 보는 지표이다. `PR-AUC`는 precision-recall curve 아래 면적을 의미하며, 상폐기업처럼 관심 있는 클래스가 중요한 문제에서 같이 보는 것이 좋다. 이번 데이터는 label 0과 label 1의 차이가 아주 극단적이지는 않지만, 그래도 상폐기업 탐지가 핵심이기 때문에 PR-AUC를 함께 기록하였다.

## 6. Threshold 최적화 수행 여부

threshold 최적화는 수행하였다.

XGBoost는 각 행에 대해 label=1일 확률을 출력한다. 일반적으로는 확률이 0.5 이상이면 label=1로 분류하지만, 0.5가 항상 F1 score에 가장 좋은 기준은 아니다. 그래서 이번 실험에서는 validation set에서 threshold를 `0.05`부터 `0.95`까지 탐색하고, validation F1이 가장 높은 threshold를 선택하였다.

즉, 각 XGBoost 후보마다 다음 순서로 평가하였다.

1. train set으로 XGBoost를 학습한다.
2. validation set에서 label=1 확률을 예측한다.
3. 여러 threshold를 적용해 precision, recall, F1을 계산한다.
4. validation F1이 가장 높은 threshold를 선택한다.
5. 선택된 threshold를 test set에 그대로 적용한다.

주요 실험의 threshold는 다음과 같다.

| 실험 | 선택 threshold | validation F1 | test F1 |
| --- | ---: | ---: | ---: |
| baseline_default | 0.5200 | 0.7759 | 0.8027 |
| scale_grid1_03_0.75x | 0.5450 | 0.7945 | 0.7482 |
| feature_drop_high_missing | 0.5450 | 0.7945 | 0.7482 |
| feature_drop_low_importance | 0.6000 | 0.7764 | 0.7092 |

따라서 threshold 최적화는 아직 안 한 것이 아니라, 모든 실험 후보에 대해 이미 적용된 상태이다.

## 7. XGBoost 하이퍼파라미터 탐색

기본 XGBoost 모델을 먼저 학습한 뒤, 작은 범위의 하이퍼파라미터 탐색을 수행하였다. 너무 많은 조합을 돌리면 실험 관리가 어려워지므로, 이번에는 현실적으로 확인 가능한 범위에서만 탐색하였다.

1차 탐색에서는 다음 값을 비교하였다.

| 파라미터 | 탐색 값 |
| --- | --- |
| `max_depth` | 3, 4, 5 |
| `learning_rate` | 0.03, 0.05 |
| `n_estimators` | 200, 400 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |

이후 상위 후보에 대해 `scale_pos_weight`를 조정하였다. `scale_pos_weight`는 label=1과 label=0의 비중이 다를 때 XGBoost가 양성 클래스에 어느 정도 가중치를 둘지 조절하는 값이다. 이번 실험에서는 기본값의 `0.75배`, `1.0배`, `1.25배`를 비교하였다.

validation F1 기준 가장 좋은 설정은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| 실험명 | `scale_grid1_03_0.75x` |
| `max_depth` | 3 |
| `learning_rate` | 0.05 |
| `n_estimators` | 200 |
| `subsample` | 0.8 |
| `colsample_bytree` | 0.8 |
| `scale_pos_weight` | 0.8433 |
| 선택 threshold | 0.5450 |

이 설정은 baseline보다 validation F1이 높았다.

| 실험 | validation precision | validation recall | validation F1 | validation PR-AUC |
| --- | ---: | ---: | ---: | ---: |
| baseline_default | 0.6911 | 0.8843 | 0.7759 | 0.8210 |
| scale_grid1_03_0.75x | 0.7716 | 0.8188 | 0.7945 | 0.8267 |

baseline은 recall이 더 높았지만 precision이 낮았다. 반면 best validation 모델은 recall이 조금 낮아지는 대신 precision이 크게 좋아졌고, 그 결과 F1 score가 상승하였다.

## 8. Feature subset 및 제거 수행 여부

feature subset 및 제거 실험도 수행하였다. 다만 결과적으로 최종 모델에는 feature 제거를 적용하지 않았다.

이번에는 두 가지 방식으로 feature 제거를 시도하였다.

첫 번째는 `drop_high_missing`이다. 결측률이 60% 이상인 feature를 제거하려고 했다. 하지만 실제로 기준을 넘는 feature가 없어서 제거된 feature 수는 0개였다. 그래서 `drop_high_missing` 결과는 전체 feature를 사용한 best validation 모델과 동일하게 나왔다.

두 번째는 `drop_low_importance`이다. XGBoost feature importance를 기준으로 중요도가 0이거나 하위 20%에 해당하는 feature를 제거하였다. 이 실험에서는 64개 feature 중 28개가 제거되어 36개 feature만 사용하였다. 제거된 feature는 대부분 `_missing` flag 계열이었다.

feature 제거 결과는 다음과 같다.

| 실험 | feature 수 | 제거 feature 수 | validation F1 | test F1 | test PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| all_features best | 64 | 0 | 0.7945 | 0.7482 | 0.8874 |
| drop_high_missing | 64 | 0 | 0.7945 | 0.7482 | 0.8874 |
| drop_low_importance | 36 | 28 | 0.7764 | 0.7092 | 0.8887 |

`drop_low_importance`는 feature 수를 줄였지만 validation F1과 test F1이 모두 낮아졌다. test PR-AUC는 거의 유지되었지만, F1 score 관점에서는 좋지 않았다. 따라서 이번 최종 설정에서는 feature를 제거하지 않고 전체 feature를 유지하는 것이 더 적절하다고 판단하였다.

## 9. Validation 기준 상위 결과

validation F1 기준 상위 결과는 다음과 같다.

| 순위 | 실험명 | feature set | threshold | precision | recall | F1 | ROC-AUC | PR-AUC |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | scale_grid1_03_0.75x | all_features | 0.5450 | 0.7716 | 0.8188 | 0.7945 | 0.8531 | 0.8267 |
| 2 | feature_drop_high_missing | drop_high_missing | 0.5450 | 0.7716 | 0.8188 | 0.7945 | 0.8531 | 0.8267 |
| 3 | scale_grid1_01_1.25x | all_features | 0.6200 | 0.7684 | 0.8188 | 0.7928 | 0.8540 | 0.8308 |
| 4 | grid1_01 | all_features | 0.5600 | 0.7606 | 0.8253 | 0.7916 | 0.8525 | 0.8290 |
| 5 | grid1_02 | all_features | 0.6800 | 0.7947 | 0.7860 | 0.7903 | 0.8541 | 0.8310 |

1위와 2위가 같은 이유는 `drop_high_missing`에서 실제로 제거된 feature가 없었기 때문이다. 따라서 사실상 같은 모델 구조라고 볼 수 있다.

## 10. Test set에서 관찰된 결과

최종 모델은 validation F1 기준으로 선택했지만, test set에서 F1이 가장 높게 나온 후보도 따로 확인하였다. 이 값은 모델 선택 기준으로 사용하지 않고, 결과 해석을 위한 참고값으로만 사용한다.

test F1 기준 상위 결과는 다음과 같다.

| 순위 | 실험명 | threshold | precision | recall | F1 | ROC-AUC | PR-AUC |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | grid1_11 | 0.4950 | 0.8470 | 0.7706 | 0.8070 | 0.9125 | 0.9016 |
| 2 | grid1_08 | 0.5500 | 0.8596 | 0.7577 | 0.8055 | 0.9106 | 0.8982 |
| 3 | baseline_default | 0.5200 | 0.8501 | 0.7603 | 0.8027 | 0.9130 | 0.9012 |
| 4 | grid1_09 | 0.5800 | 0.8701 | 0.7423 | 0.8011 | 0.9123 | 0.9005 |
| 5 | grid1_12 | 0.5950 | 0.8653 | 0.7448 | 0.8006 | 0.9093 | 0.8990 |

여기서 주의할 점은 `grid1_11`의 test F1이 가장 높지만, validation F1 기준으로는 최종 모델보다 낮았다는 점이다. 실험 원칙상 test set을 보고 모델을 고르면 test set에 맞춘 선택이 될 수 있으므로, 최종 선택은 validation 기준을 유지하였다.

## 11. 최종 선택 결과

validation F1 기준 최종 선택 모델은 `scale_grid1_03_0.75x`이다.

| 항목 | 값 |
| --- | --- |
| feature set | all_features |
| feature 수 | 64 |
| threshold | 0.5450 |
| max_depth | 3 |
| learning_rate | 0.05 |
| n_estimators | 200 |
| scale_pos_weight | 0.8433 |

최종 모델의 성능은 다음과 같다.

| dataset | precision | recall | F1 | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| validation | 0.7716 | 0.8188 | 0.7945 | 0.8531 | 0.8267 |
| test | 0.8729 | 0.6546 | 0.7482 | 0.9041 | 0.8874 |

validation에서는 precision과 recall이 비교적 균형 있게 나왔다. test에서는 precision이 높고 recall이 낮아졌다. 즉, test set에서는 상폐기업이라고 예측한 경우에는 맞는 비율이 높았지만, 실제 상폐기업을 모두 많이 잡아내지는 못했다. 상폐기업 탐지 문제에서는 recall도 중요하므로, 이후 실험에서는 recall을 조금 더 확보하는 방향의 threshold 조정도 비교할 필요가 있다.

## 12. 이번 실험에서 확인한 점

이번 실험에서 threshold 최적화는 확실히 수행되었다. 기본 threshold 0.5를 그대로 사용하지 않고, validation F1이 가장 높은 threshold를 각 모델별로 선택하였다. 그 결과 모델마다 0.495, 0.520, 0.545, 0.600처럼 서로 다른 threshold가 선택되었다.

feature subset 및 제거도 수행되었다. 다만 high-missing 제거는 실제 제거 대상이 없어 결과가 동일했고, low-importance 제거는 feature 수를 줄였지만 F1이 낮아졌다. 따라서 이번 데이터에서는 feature를 줄이는 것이 성능 개선으로 바로 이어지지는 않았다.

하이퍼파라미터 탐색에서는 validation F1 기준으로 baseline보다 더 좋은 후보를 찾았다. baseline validation F1은 0.7759였고, best validation F1은 0.7945였다. 다만 test F1은 baseline이 0.8027로 더 높았다. 이 차이는 validation 기준으로 고른 모델이 test에서도 항상 최고가 되지는 않는다는 점을 보여준다.

## 13. 결론

이번 raw 기반 실험에서는 다음 작업을 모두 수행하였다.

- raw JSON을 학습용 CSV로 변환하였다.
- `종목코드` 기준 Group Split을 적용하였다.
- XGBoost baseline을 학습하였다.
- XGBoost 하이퍼파라미터 탐색을 수행하였다.
- validation 기준 threshold 최적화를 수행하였다.
- high-missing feature 제거와 low-importance feature 제거를 실험하였다.
- F1, ROC-AUC, PR-AUC를 함께 기록하였다.

최종적으로 validation F1 기준 best setting은 `scale_grid1_03_0.75x`였고, validation F1은 0.7945, test F1은 0.7482, test PR-AUC는 0.8874였다. 다만 test F1만 보면 `grid1_11`이 0.8070으로 가장 높았다. 하지만 test set을 기준으로 모델을 선택하면 평가가 흔들릴 수 있으므로, 공식 최종 모델은 validation F1 기준으로 선택하는 것이 더 적절하다.

이번 결과를 바탕으로 다음 실험에서는 validation 성능과 test 성능의 차이를 줄이는 방향이 필요하다. 예를 들어 Group Split seed를 고정한 상태에서 recall을 조금 더 중시하는 threshold 기준을 추가로 비교하거나, 산업군 분포 차이를 줄이는 split 방식도 검토할 수 있다.

## 14. 저장 파일

- raw 학습 CSV: `C:\kwoss_C\model_test_v2\재무비율\raw_모델학습용_20260513_raw.csv`
- raw 전처리 완료 CSV: `C:\kwoss_C\model_test_v2\재무비율\raw_모델학습전처리완료_20260513_raw.csv`
- 실험 metrics CSV: `C:\kwoss_C\model_test_v2\model\raw_xgboost_metrics_20260513_raw.csv`
- split summary CSV: `C:\kwoss_C\model_test_v2\model\raw_xgboost_split_summary_20260513_raw.csv`
- best model joblib: `C:\kwoss_C\model_test_v2\model\raw_xgboost_best_model_20260513_raw.joblib`
