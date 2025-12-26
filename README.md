# Рекомендательная система для продуктов Amazon

## Постановка задачи
Задача состоит в разработке рекомендательной системы, способной выдавать персонализированные рекомендации пользователям в реальном времени. Система должна учитывать отзывы пользователей на товары и использовать их для построения информативных эмбеддингов как для пользователей, так и для продуктов.

## Формат входных и выходных данных

- **Входные данные:**  
  Все комментарии конкретного пользователя на товары, с полями:
  - `reviewID` — Идентификатор отзыва
  - `nickname` — Псевдоним пользователя
  - `ProductName` — Название товара
  - `TextReview` — Текст отзыва
  - `Style` — Категория товара
  - `asin` — Уникальный идентификатор товара
  
- **Выходные данные:**  
  Информативные эмбеддинги для пользователей, собранные на основе их фидбека и комментариев, а также эмбеддинги для предоставляемых продуктов.

## Метрики

Основные метрики для оценки модели:
- **F1-score**
- **Accuracy**
- **loss**

## Валидация и тест

Для разделения выборки используется случайное разбиение: из исходного обучающего набора выделяется 10% юзеров для валидации с помощью функции `random_split` из `torch.utils.data.dataset`.

Для контроля ключевых метрик (F1-score, loss, Accuracy) будет использован **MLflow**.

## Датасеты

В качестве исходных данных используется [Amazon Review Data](https://nijianmo.github.io/amazonl) в разделе **musical instruments**. Этот датасет включает 231392 JSON объектов, каждый из которых представляет собой отзыв пользователя на товар. Каждый объект содержит следующие поля:
- `reviewID` — Идентификатор отзыва
- `nickname` — Псевдоним пользователя
- `ProductName` — Название товара
- `TextReview` — Текст отзыва
- `Style` — Категория товара

## Основная модель

Эмбеддинги пользователей и продуктов формируются с использованием модели **all-MiniLM-L6-v2**. Затем строятся рёбра в графе, где рёбра представляют эмбеддинги комментариев пользователей к продуктам. 

После этого выполняется **message passing** с использованием **attention** с применением одного из алгоритмов графовых нейронных сетей, таких как **GAT** (Graph Attention Network) или **TransformerConv**. Эти алгоритмы доступны в библиотеке **PyTorch Geometric** (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html).

Формула для финальной функции потерь (лосс-функции):
$$
\text{loss} = \frac{5}{2} \times (\text{cos\_sim}(\texttt{cust\_emb}, \texttt{prod\_emb}) + 1)
$$
где:
- `cust_emb` — эмбеддинг пользователя
- `prod_emb` — эмбеддинг продукта
- `cos_sim` — косинусное сходство между эмбеддингами пользователя и продукта

Цель модели — минимизировать лосс, что будет способствовать улучшению качества рекомендаций, основанных на комментариях пользователей.

- `asin` — Уникальный идентификатор товара

Этот датасет будет использован для построения рекомендательной системы и генерации персонализированных рекомендаций.



## Setup

Для настройки проекта выполните следующие шаги:

1. **Клонируйте репозиторий:**

`git clone https://github.com/FlexxofIvan/recommend-system.git`

2. **Перейдите в директорию проекта:**
   
`cd recommend-system`

3. **Установите зависимости с помощью uv:**

`uv sync`

Возможно придется прописать еще:

```bash
  uv add sentence-transformers
```

4. ***Установка и запуска pre-commit:**
```bash
uv run pre-commit install
uv run pre-commit run -a
```

## Train

```bash
 uv run -m rec_sys.train.train
```
Подгружает данные для обучения с гугл диска, обучает модель на них.

MlFlow:
```bash
  uv run mlflow ui --host 127.0.0.1 --port 8080
```

### Inference

Для инференса необходимо прописать в:

```bash
  config/infer_config.yaml 
```
путь до нужного чекпоинта.

Затем создать векторную базу данных для продуктов:
```bash
  uv run -m rec_sys.vector_database.create_vectorbase
```

Сам инференс:
```bash
  uv run -m rec_sys.inference.infer
```

Модель ожидает следующий формат данных:
```bash
inputs = {
    "user_features": None,
    "product_info_features": None,
    "product_name_features": None,
    "edge_index": None,
    "edge_attr": None,
}
```
получить нужные фичи можно, прогнав входные данные, приведеные к датафрейму (их вид):
```bash
{
  "reviewerID": "string_value",
  "asin": "string_value",
  "verified": true,
  "score": 4.5,
  "product_name": "string_value",
  "product_info": "string_value",
  "comment": "string_value"
}
```
через:
```bash
from rec_sys.dataset_modules.cols_data.vector_data_utils import vectorize_df
```

### Production Preparation

Экспортируем нужную нам модель:
```bash
  uv run python -m rec_sys.modules.model_export \
    ./checkpoints/<checkpoint name> \
    ./config/model/model_config.yaml \
    --output_path=./triton_utils/model_repository/graph_model/1/model.onnx
```
Для дальнейшего использования понадобится либо уже созданная модель и датабаза:
```bash
uv run dvc pull triton_utils/model_repository/graph_model/1/model.onnx.dvc
uv run dvc pull triton_utils/model_repository/graph_model/1/model.onnx.data.dvc
uv run dvc pull vector_index.dvc
```

либо надо создать базу данных нужной моделью, указав в infer_config нужный чекпоинт, и экспортировать эту же модель:
```bash
uv run -m rec_sys.vector_database.create_vectorbase

  uv run python -m rec_sys.modules.model_export \
    ./checkpoints/<checkpoint name> \
    ./config/model/model_config.yaml \
    --output_path=./triton_utils/model_repository/graph_model/1/model.onnx
```

### Launch 

```bash
  docker compose build --no-cache
  docker compose up
  docker ps
```

Сервер доступен по адресу https://0.0.0.0:8080

Если докер не запустился, то:
```bash
docker run --gpus all --rm \
  -p8000:8000 -p8001:8001 -p8002:8002 \
  -v $(pwd)/triton_utils/model_repository:/models \
  nvcr.io/nvidia/tritonserver:25.11-py3 \
  tritonserver --model-repository=/models
```

```bash
uv run uvicorn triton_utils.web_spp:app --reload --port 8080
```






