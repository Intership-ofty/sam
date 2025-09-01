from fastapi import APIRouter

router = APIRouter()

@router.get("/admin")
async def get_admin():
    return {"message": "Admin endpoint"}
