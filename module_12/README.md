# ML-сервис кредитного скоринга

Проект реализует вариант 3 из задания: REST-сервис на классическом ML для скоринга. Сервис принимает параметры кредитной заявки, оценивает риск дефолта и возвращает бизнес-решение: одобрить или отказать.

## Постановка задачи

Цель: по признакам клиента и кредита предсказать риск дефолта `default_risk`.

Признаки:

- `age` - возраст клиента;
- `annual_income` - годовой доход;
- `loan_amount` - сумма кредита;
- `credit_score` - кредитный рейтинг;
- `employment_years` - стаж работы;
- `existing_debt` - текущая долговая нагрузка;
- `debt_to_income` - отношение долга к доходу;
- `loan_to_income` - отношение суммы кредита к доходу.

Внутренняя модель предсказывает риск дефолта:

- `default_risk = 1` означает высокий риск;
- `default_risk = 0` означает низкий риск.

API возвращает бизнес-класс решения:

- `class = 1` означает "одобрить";
- `class = 0` означает "отказать".

## Структура

- `app/main.py` - создание FastAPI-приложения, загрузка модели, логирование запросов.
- `app/api/routes.py` - эндпоинты `/health`, `/predict`, `/model-info`.
- `app/models/schemas.py` - Pydantic-схемы для валидации запроса и ответа.
- `app/utils/model_service.py` - генерация датасета, обучение, подбор параметров, загрузка артефактов, предсказание.
- `data/` - обучающий CSV-файл.
- `artifacts/` - сохраненная модель и метаданные.

## Данные и артефакты

Используется синтетический датасет. Он нужен только для учебной демонстрации и не подходит для реальных кредитных решений.

- `data/sample_customers.csv` - синтетические данные клиентов. Если файла нет, он создается автоматически при первом запуске.
- `artifacts/credit_risk_model.joblib` - лучшая модель после `GridSearchCV`.
- `artifacts/credit_risk_metadata.joblib` - метаданные модели: признаки, метрики, порог, лучшие параметры, тип датасета.

Если `artifacts/*.joblib` отсутствуют или метаданные устарели, модель обучается автоматически при старте сервиса.

## Обучение и качество модели

Обучение устроено так:

- train/test split: 75% / 25%;
- preprocessing: `StandardScaler`;
- модель: `RandomForestClassifier`;
- подбор гиперпараметров: `GridSearchCV`;
- `scoring="roc_auc"`;
- `cv=3`;
- `n_jobs=-1`;
- сетка содержит 16 комбинаций.

Подбираются параметры:

- `classifier__n_estimators`;
- `classifier__max_depth`;
- `classifier__min_samples_leaf`;
- `classifier__class_weight`.

Метрики `accuracy` и `roc_auc` считаются на test-части, выводятся в лог при первом обучении и сохраняются в `artifacts/credit_risk_metadata.joblib`.

## Запуск

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Документация API:

```text
http://127.0.0.1:8000/docs
```

Для Windows есть готовые скрипты:

```bat
install_deps.bat
run_rest.bat
curl_predict.bat
```

## Проверка статуса

```bash
curl http://127.0.0.1:8000/health
```

Ответ:

```json
{"status":"ok"}
```

## Информация о модели

```bash
curl http://127.0.0.1:8000/model-info
```

Пример ответа:

```json
{
  "feature_names": [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "existing_debt",
    "debt_to_income",
    "loan_to_income"
  ],
  "accuracy": 0.7657,
  "roc_auc": 0.8342,
  "threshold": 0.5,
  "best_params": {
    "classifier__class_weight": null,
    "classifier__max_depth": 5,
    "classifier__min_samples_leaf": 5,
    "classifier__n_estimators": 140
  },
  "dataset_type": "synthetic_credit_scoring"
}
```

## Примеры запросов

### 1. Хороший кредитный профиль

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 34,
    "annual_income": 82000,
    "loan_amount": 18000,
    "credit_score": 735,
    "employment_years": 6,
    "existing_debt": 12000
  }'
```

Пример ответа:

```json
{
  "class": 1,
  "decision": "одобрить",
  "model_target": "default_risk",
  "model_class_meaning": "internal model predicts default_risk; API class is approval decision",
  "probability_approved": 0.5982,
  "probability_risk": 0.4018,
  "decision_threshold": 0.5,
  "features": {
    "age": 34.0,
    "annual_income": 82000.0,
    "loan_amount": 18000.0,
    "credit_score": 735.0,
    "employment_years": 6.0,
    "existing_debt": 12000.0,
    "debt_to_income": 0.1463,
    "loan_to_income": 0.2195
  }
}
```

### 2. Рискованная заявка

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 48,
    "annual_income": 32000,
    "loan_amount": 95000,
    "credit_score": 420,
    "employment_years": 1,
    "existing_debt": 52000
  }'
```

Пример ответа:

```json
{
  "class": 0,
  "decision": "отказать",
  "model_target": "default_risk",
  "model_class_meaning": "internal model predicts default_risk; API class is approval decision",
  "probability_approved": 0.0825,
  "probability_risk": 0.9175,
  "decision_threshold": 0.5,
  "features": {
    "age": 48.0,
    "annual_income": 32000.0,
    "loan_amount": 95000.0,
    "credit_score": 420.0,
    "employment_years": 1.0,
    "existing_debt": 52000.0,
    "debt_to_income": 1.625,
    "loan_to_income": 2.9688
  }
}
```

### 3. Ошибка валидации

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 15,
    "annual_income": 82000,
    "loan_amount": 18000,
    "credit_score": 735,
    "employment_years": 6,
    "existing_debt": 12000
  }'
```

Пример ответа:

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "age"],
      "msg": "Input should be greater than or equal to 18",
      "input": 15,
      "ctx": {"ge": 18}
    }
  ]
}
```

## Валидация

Сервис проверяет:

- `age` от 18 до 100;
- `annual_income` от 0 до 1 000 000;
- `loan_amount` больше 0 и не больше 500 000;
- `credit_score` от 300 до 850;
- `employment_years` от 0 до 60;
- `existing_debt` от 0 до 1 000 000.

Некорректные запросы возвращают HTTP `422` с деталями Pydantic. Ошибки модели возвращают понятный HTTP-ответ.

## Ограничения

- Датасет синтетический, только для учебной демонстрации.
- Ответ модели нельзя использовать для реальных кредитных решений.
- Авторизации и постоянной базы данных нет.
- Переобучение запускается только при отсутствии артефактов или при устаревших метаданных.
