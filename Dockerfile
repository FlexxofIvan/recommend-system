FROM python:3.13.8-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    numpy \
    polars \
    pyyaml \
    sentence-transformers \
    tritonclient[grpc] \
    torch \
    torchvision


COPY . .

CMD ["uvicorn", "triton_utils.web_app:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
