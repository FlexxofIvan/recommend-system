from pydantic import BaseModel


class Review(BaseModel):
    reviewerID: str
    asin: str
    verified: bool
    score: float
    product_name: str
    product_info: str
    comment: str
