from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
import crud, schemas
from database import get_db
from logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/customers", tags=["Customers"])


@router.get("/", response_model=List[schemas.CustomerOut])
def list_customers(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    logger.info(f"GET /customers — skip={skip}, limit={limit}")
    customers = crud.get_customers(db, skip=skip, limit=limit)
    logger.info(f"Returning {len(customers)} customers.")
    return customers


@router.get("/{customer_number}", response_model=schemas.CustomerOut)
def get_customer(customer_number: int, db: Session = Depends(get_db)):
    logger.info(f"GET /customers/{customer_number}")
    customer = crud.get_customer(db, customer_number)
    if not customer:
        logger.warning(f"Customer {customer_number} not found — returning 404")
        raise HTTPException(status_code=404, detail=f"Customer {customer_number} not found")
    return customer


@router.post("/", response_model=schemas.CustomerOut, status_code=201)
def create_customer(customer: schemas.CustomerCreate, db: Session = Depends(get_db)):
    logger.info(f"POST /customers — creating: {customer.customerName}")
    existing = crud.get_customer(db, customer.customerNumber)
    if existing:
        logger.warning(f"Customer {customer.customerNumber} already exists")
        raise HTTPException(status_code=400, detail="Customer already exists")
    return crud.create_customer(db, customer)


@router.put("/{customer_number}", response_model=schemas.CustomerOut)
def update_customer(customer_number: int, updates: schemas.CustomerUpdate, db: Session = Depends(get_db)):
    logger.info(f"PUT /customers/{customer_number}")
    updated = crud.update_customer(db, customer_number, updates)
    if not updated:
        logger.warning(f"Customer {customer_number} not found for update — returning 404")
        raise HTTPException(status_code=404, detail=f"Customer {customer_number} not found")
    return updated


@router.delete("/{customer_number}", response_model=schemas.CustomerOut)
def delete_customer(customer_number: int, db: Session = Depends(get_db)):
    logger.info(f"DELETE /customers/{customer_number}")
    deleted = crud.delete_customer(db, customer_number)
    if not deleted:
        logger.warning(f"Customer {customer_number} not found for delete — returning 404")
        raise HTTPException(status_code=404, detail=f"Customer {customer_number} not found")
    return deleted


@router.get("/{customer_number}/orders", response_model=List[schemas.OrderOut])
def get_customer_orders(customer_number: int, db: Session = Depends(get_db)):
    logger.info(f"GET /customers/{customer_number}/orders")
    customer = crud.get_customer(db, customer_number)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_number} not found")
    orders = crud.get_customer_orders(db, customer_number)
    return orders


@router.get("/{customer_number}/payments", response_model=List[schemas.PaymentOut])
def get_customer_payments(customer_number: int, db: Session = Depends(get_db)):
    logger.info(f"GET /customers/{customer_number}/payments")
    customer = crud.get_customer(db, customer_number)
    if not customer:
        raise HTTPException(status_code=404, detail=f"Customer {customer_number} not found")
    payments = crud.get_customer_payments(db, customer_number)
    return payments
