# Airfoil Self-Noise Prediction

Проект прогнозирует уровень шума аэродинамического профиля по параметрам потока и геометрии профиля.

## Структура проекта

- `task.ipynb` - анализ данных, EDA, feature engineering, обучение и оценка моделей.
- `task.md` - текст задания.
- `review_report.md` - текущий отчёт готовности проекта.
- `data.csv` - исходный датасет UCI Airfoil Self-Noise.
- `data/random_forest_model.pkl` - локальный артефакт обученной модели. Файл игнорируется git.
- `app/controller.py` - настройка Flask-приложения и маршрутов.
- `app/inferer.py` - загрузка модели и обработка API-запроса.
- `static/index.html` - HTML-форма для ручной проверки прогноза.
- `main.py` - точка запуска сервиса.
- `presentation.md` - структура презентации для защиты.
- `screencast.md` - файл для ссылки на скринкаст и сценария записи.

## Данные

Используется датасет UCI Airfoil Self-Noise:

- `frequency` - частота колебаний;
- `attack-angle` - угол атаки профиля;
- `chord-length` - длина хорды профиля;
- `free-stream-velocity` - скорость набегающего потока;
- `suction-side-displacement-thickness` - толщина вытеснения на стороне разрежения;
- `scaled-sound-pressure` - целевая переменная.

## Установка

```bash
pip install -r requirements.txt
```

## Запуск notebook / обучение модели

Открыть notebook:

```bash
jupyter notebook task.ipynb
```

Выполнить notebook из командной строки, если установлен Jupyter/nbconvert:

```bash
jupyter nbconvert --to notebook --execute --inplace task.ipynb
```

После выполнения notebook модель сохраняется в:

```text
data/random_forest_model.pkl
```

## Запуск сервиса

```bash
python main.py
```

После запуска HTML-форма доступна по адресу:

```text
http://127.0.0.1:8080/
```

## API

Эндпоинт:

```text
POST /api/inference
```

Пример запроса для Windows PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://127.0.0.1:8080/api/inference" `
  -Method Post `
  -ContentType "application/json" `
  -Body '{"frequency":1250,"attack-angle":0,"chord-length":0.2286,"free-stream-velocity":39.6,"suction-side-displacement-thickness":0.00253511}'
```

Пример запроса через `curl`:

```bash
curl -X POST http://127.0.0.1:8080/api/inference \
  -H "Content-Type: application/json" \
  -d '{"frequency":1250,"attack-angle":0,"chord-length":0.2286,"free-stream-velocity":39.6,"suction-side-displacement-thickness":0.00253511}'
```

Пример ответа:

```json
{
  "prediction": 129.27325000000033
}
```

## Итоговые метрики

Финальная модель: `RandomForestRegressor`.

Лучшие параметры по `GridSearchCV`:

```text
max_depth=None
min_samples_leaf=1
n_estimators=200
```

Метрики на тестовой выборке:

| Метрика | Значение |
| --- | ---: |
| MAE | 1.299809 |
| MSE | 3.287820 |
| RMSE | 1.813235 |
| R2 | 0.934373 |

## Ограничения

Модель обучена на датасете UCI Airfoil Self-Noise и предназначена для предварительной оценки в диапазонах, близких к исходным экспериментальным данным. Прогноз не заменяет физический эксперимент.
