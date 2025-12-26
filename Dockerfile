FROM python:3.13.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "triton_utils.web_spp:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]
