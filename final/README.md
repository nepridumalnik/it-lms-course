# Airfoil Self-Noise Prediction

Проект прогнозирует уровень шума аэродинамического профиля по параметрам потока и геометрии профиля.

## Данные

Используется датасет UCI Airfoil Self-Noise:

- `frequency` - частота колебаний;
- `attack-angle` - угол атаки профиля;
- `chord-length` - длина хорды профиля;
- `free-stream-velocity` - скорость набегающего потока;
- `suction-side-displacement-thickness` - толщина вытеснения на стороне разрежения;
- `scaled-sound-pressure` - целевая переменная.

## Структура

- `task.ipynb` - ноутбук с анализом данных, обучением и оценкой модели;
- `data.csv` - исходный датасет;
- `data/random_forest_model.pkl` - сохранённая модель;
- `app/` - Flask-код для инференса;
- `static/index.html` - простая HTML-форма для ручной проверки;
- `main.py` - точка запуска сервиса.

## Установка

```bash
pip install -r requirements.txt
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

Пример запроса:

```bash
curl -X POST http://127.0.0.1:8080/api/inference ^
  -H "Content-Type: application/json" ^
  -d "{\"frequency\":1250,\"attack-angle\":0,\"chord-length\":0.2286,\"free-stream-velocity\":39.6,\"suction-side-displacement-thickness\":0.00253511}"
```

Пример ответа:

```json
{
  "prediction": 129.08567000000025
}
```

## Модель

Финальная модель загружается из `data/random_forest_model.pkl`. Обучение и сохранение модели выполняются в `task.ipynb`.
