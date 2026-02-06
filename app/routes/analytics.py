from fastapi import APIRouter, Query
from app.analysis.department_analytics import (
    get_department_criterion_wise_distribution
)

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/department/criteria")
def department_criteria_analytics(
    dept: str = Query(..., description="Department name"),
    cls: str | None = Query(None, description="Class name (optional)")
):
    return get_department_criterion_wise_distribution(dept, cls)
