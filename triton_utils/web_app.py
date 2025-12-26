import logging
from typing import List

from fastapi import FastAPI
from starlette.responses import RedirectResponse

from triton_utils.schemas import Review
from triton_utils.triton import GraphInferenceClient

logger = logging.getLogger("uvicorn.error")

app = FastAPI()

data_cfg_path = "config/data/data_config.yaml"
vector_db_cfg_path = "config/vector_db/vector_db_config.yaml"
client = GraphInferenceClient(
    data_config_path=data_cfg_path, vector_db_config_path=vector_db_cfg_path
)


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


@app.post("/reviews/")
async def validate_reviews(reviews: List[Review]):
    recs = client.predict(reviews)
    return {"recommendations": recs}
