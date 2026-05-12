import asyncio
import time
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database import get_db
import crud_counts
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(tags=["Dashboard"])


# ── Helper: run a sync DB call in a thread so asyncio.gather can run them in parallel ──

async def run_count(func, db: Session) -> int:
    """Wraps a synchronous SQLAlchemy count function to be awaitable."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, func, db)


# ── 8 Individual Count Endpoints ──────────────────────────────────────────────

@router.get("/customers/count")
async def customers_count(db: Session = Depends(get_db)):
    logger.info("GET /customers/count")
    count = crud_counts.get_customers_count(db)
    logger.info(f"Response: customers={count}")
    return {"table": "customers", "count": count}


@router.get("/orders/count")
async def orders_count(db: Session = Depends(get_db)):
    logger.info("GET /orders/count")
    count = crud_counts.get_orders_count(db)
    logger.info(f"Response: orders={count}")
    return {"table": "orders", "count": count}


@router.get("/products/count")
async def products_count(db: Session = Depends(get_db)):
    logger.info("GET /products/count")
    count = crud_counts.get_products_count(db)
    logger.info(f"Response: products={count}")
    return {"table": "products", "count": count}


@router.get("/employees/count")
async def employees_count(db: Session = Depends(get_db)):
    logger.info("GET /employees/count")
    count = crud_counts.get_employees_count(db)
    logger.info(f"Response: employees={count}")
    return {"table": "employees", "count": count}


@router.get("/offices/count")
async def offices_count(db: Session = Depends(get_db)):
    logger.info("GET /offices/count")
    count = crud_counts.get_offices_count(db)
    logger.info(f"Response: offices={count}")
    return {"table": "offices", "count": count}


@router.get("/payments/count")
async def payments_count(db: Session = Depends(get_db)):
    logger.info("GET /payments/count")
    count = crud_counts.get_payments_count(db)
    logger.info(f"Response: payments={count}")
    return {"table": "payments", "count": count}


@router.get("/orderdetails/count")
async def orderdetails_count(db: Session = Depends(get_db)):
    logger.info("GET /orderdetails/count")
    count = crud_counts.get_orderdetails_count(db)
    logger.info(f"Response: orderdetails={count}")
    return {"table": "orderdetails", "count": count}


@router.get("/productlines/count")
async def productlines_count(db: Session = Depends(get_db)):
    logger.info("GET /productlines/count")
    count = crud_counts.get_productlines_count(db)
    logger.info(f"Response: productlines={count}")
    return {"table": "productlines", "count": count}


# ── Aggregated Concurrent Endpoint ────────────────────────────────────────────

@router.get("/overall_counts")
async def overall_counts(db: Session = Depends(get_db)):
    logger.info("GET /overall_counts — starting all 8 concurrent DB queries")
    start = time.perf_counter()

    # All 8 queries launched simultaneously with asyncio.gather
    (
        customers,
        orders,
        products,
        employees,
        offices,
        payments,
        orderdetails,
        productlines,
    ) = await asyncio.gather(
        run_count(crud_counts.get_customers_count, db),
        run_count(crud_counts.get_orders_count, db),
        run_count(crud_counts.get_products_count, db),
        run_count(crud_counts.get_employees_count, db),
        run_count(crud_counts.get_offices_count, db),
        run_count(crud_counts.get_payments_count, db),
        run_count(crud_counts.get_orderdetails_count, db),
        run_count(crud_counts.get_productlines_count, db),
    )

    elapsed = round(time.perf_counter() - start, 4)
    logger.info(f"asyncio.gather() completed in {elapsed}s")

    result = {
        "customers": customers,
        "orders": orders,
        "products": products,
        "employees": employees,
        "offices": offices,
        "payments": payments,
        "orderdetails": orderdetails,
        "productlines": productlines,
        "_response_time_seconds": elapsed,
    }

    logger.info(f"GET /overall_counts — response: {result}")
    return result
