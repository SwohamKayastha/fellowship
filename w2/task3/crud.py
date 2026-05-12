from sqlalchemy.orm import Session
import models, schemas
from logger import get_logger

logger = get_logger(__name__)

def get_customers(db: Session, skip: int = 0, limit: int = 100):
    logger.info(f"Fetching customers: skip={skip}, limit={limit}")
    customers = db.query(models.Customer).offset(skip).limit(limit).all()
    logger.info(f"Fetched {len(customers)} customers.")
    return customers


def get_customer(db: Session, customer_number: int):
    logger.info(f"Fetching customer with ID: {customer_number}")
    customer = db.query(models.Customer).filter(
        models.Customer.customerNumber == customer_number
    ).first()
    if not customer:
        logger.warning(f"Customer not found: ID {customer_number}")
    return customer


def create_customer(db: Session, customer: schemas.CustomerCreate):
    logger.info(f"Creating customer: {customer.customerName}")
    db_customer = models.Customer(**customer.model_dump())
    db.add(db_customer)
    db.commit()
    db.refresh(db_customer)
    logger.info(f"Customer created: ID {db_customer.customerNumber}")
    return db_customer


def update_customer(db: Session, customer_number: int, updates: schemas.CustomerUpdate):
    logger.info(f"Updating customer: ID {customer_number}")
    db_customer = get_customer(db, customer_number)
    if not db_customer:
        return None
    for field, value in updates.model_dump(exclude_unset=True).items():
        setattr(db_customer, field, value)
    db.commit()
    db.refresh(db_customer)
    logger.info(f"Customer updated: ID {customer_number}")
    return db_customer


def delete_customer(db: Session, customer_number: int):
    logger.info(f"Deleting customer: ID {customer_number}")
    db_customer = get_customer(db, customer_number)
    if not db_customer:
        return None
    db.delete(db_customer)
    db.commit()
    logger.info(f"Customer deleted: ID {customer_number}")
    return db_customer


def get_customer_orders(db: Session, customer_number: int):
    logger.info(f"Fetching orders for customer: ID {customer_number}")
    orders = db.query(models.Order).filter(
        models.Order.customerNumber == customer_number
    ).all()
    logger.info(f"Found {len(orders)} orders for customer {customer_number}")
    return orders


def get_customer_payments(db: Session, customer_number: int):
    logger.info(f"Fetching payments for customer: ID {customer_number}")
    payments = db.query(models.Payment).filter(
        models.Payment.customerNumber == customer_number
    ).all()
    logger.info(f"Found {len(payments)} payments for customer {customer_number}")
    return payments
