from fastapi import APIRouter

router = APIRouter()

@router.get("/kpis")
async def get_kpis():
    return {"message": "KPIs endpoint"}
