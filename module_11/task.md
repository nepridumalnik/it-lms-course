Вот аккуратно оформленная версия задания в **Markdown**:

---

# 📌 Задание 11. Версионирование модели кластеризации текста

**Нужно сделать:** получить оценку

**Открыто:** 30 марта 2026, 00:21
**Срок сдачи:** 20 апреля 2026, 23:59

---

## 🧩 Краткое описание

Провести кластеризацию текста, предварительно создав эмбеддинги и уменьшив размерность.
Все этапы необходимо **версионировать с помощью MLflow на DAGsHub**.

---

## 🧠 Что такое DAGsHub

**DAGsHub** — это GitHub для задач Data Science и ML.
Позволяет:

* хранить код, модели и данные
* вести версионирование
* использовать MLflow в облаке
* управлять доступами

---

## 📤 Что сдавать

* Ссылка на **Colab ноутбук** с решением
* Ссылка на **открытый репозиторий DAGsHub (MLflow)**

---

## 🛠 Что использовать

* DAGsHub (репозиторий + MLflow)
* `sentence-transformers`
* `scikit-learn`
* `umap-learn`
* Метрики кластеризации:
  [https://neerc.ifmo.ru/wiki/index.php?title=Методы_оценки_качества_кластеризации](https://neerc.ifmo.ru/wiki/index.php?title=Методы_оценки_качества_кластеризации)

---

## ⚙️ Требования к коду

* Код оформлен в виде **функций**
* Есть **docstring / комментарии**
* Все параметры передаются через переменные (для MLflow логирования)
* Есть ссылка на **MLflow репозиторий**
* Имена:

  * моделей
  * переменных
  * run / experiment
    должны быть **информативными**

---

## 🔧 Пример параметров

```python
TSNE_PARAMS = {
    "n_components": 2,
    "perplexity": 30,
    "random_state": 0,
}

MAX_CLUSTER_COUNT = 6

def tsne_visualization(n_components, perplexity, ...):
    pass
```

---

## 🚀 Подпункты задания

### Подпункт 1. DAGsHub

* Зарегистрироваться
* Создать публичный репозиторий
* Проверить вкладку **Experiments → MLflow UI**

---

### Подпункт 2. Создать Colab ноутбук

Установить:

```bash
pip install mlflow sentence-transformers umap-learn scikit-learn
```

---

### Подпункт 3. Авторизация MLflow

```python
import mlflow
import os
from getpass import getpass

os.environ['MLFLOW_TRACKING_USERNAME'] = input('Enter your DAGsHub username: ')
os.environ['MLFLOW_TRACKING_PASSWORD'] = getpass('Enter your DAGsHub access token: ')
os.environ['MLFLOW_TRACKING_PROJECTNAME'] = input('Enter your DAGsHub project name: ')

mlflow.set_tracking_uri(
    f"https://dagshub.com/"
    + os.environ['MLFLOW_TRACKING_USERNAME']
    + "/"
    + os.environ['MLFLOW_TRACKING_PROJECTNAME']
    + ".mlflow"
)
```

---

### Подпункт 4. Данные

Использовать датасет:

👉 [https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset](https://scikit-learn.org/stable/datasets/real_world.html#newsgroups-dataset)

* Выбрать **8–10 тем**

---

### Подпункт 5. Очистка текста

Удалить:

* `\t`, `\n`
* URL
* email (`example@mail.com`)
* прочие шумы

---

### Подпункт 6. Эмбеддинги

* Использовать модель из `sentence-transformers`

---

### Подпункт 7. Снижение размерности

* `UMAP` или `t-SNE`

---

### Подпункт 8. Кластеризация

* Подобрать количество кластеров через:

  * **Silhouette score**
* Ориентир:

  * реальное число классов ±3

---

### Подпункт 9. MLflow

```python
mlflow.sklearn.autolog(log_models=True)

with mlflow.start_run(run_name="Clustering K-Means"):
    ...
```

---

## 📊 Метрики (обязательно)

### Внутренние:

* Silhouette Score
* Calinski-Harabasz Index
* Davies-Bouldin Index

### Внешние (>= 3):

* ARI (Adjusted Rand Index)
* NMI (Normalized Mutual Information)
* Accuracy / F1 / Purity (по желанию)

Логирование:

```python
mlflow.log_metric("silhouette", value)
```

---

## 📦 Версионирование модели

* `mlflow.log_model(...)`
* `mlflow.register_model(...)`

---

## ✅ Критерии оценки

### ✔️ Зачтено

* Все пункты выполнены
* Нет серьёзных ошибок
* Корректные термины и подход

### ❌ Не зачтено

* Не выполнены пункты задания
* Есть существенные ошибки
* Более 3 несущественных ошибок
* Поверхностное решение

---

## ⚠️ Критические требования (на практике)

Если коротко, без формальностей:

* ✔️ MLflow + DAGsHub работает
* ✔️ Есть эмбеддинги (sentence-transformers)
* ✔️ Есть снижение размерности
* ✔️ Есть кластеризация
* ✔️ Есть метрики (внутренние + внешние)
* ✔️ Всё залогировано

