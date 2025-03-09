from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/tests", tags=["tests"])

@router.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}


'''
/tests/search?keyword=apple&page=1&limit=10
'''
@router.get("/search")
async def search_items(
    keyword: str, 
    page: int = 1, 
    limit: int = 10
):
    return {"keyword": keyword, "pagination": {"page": page, "limit": limit}}


class ItemCreate(BaseModel):
    name: str
    description: str | None = None
    price: float
    tax: float | None = None
@router.post("/items")
async def create_item(item: ItemCreate):
    return {"item": item.dict(), "total_price": item.price * (1 + (item.tax or 0))}


