from sqlalchemy.orm import Session
import models
from logger import get_logger

logger = get_logger(__name__)


def get_customers_count(db: Session) -> int:
    logger.info("DB query: COUNT customers")
    count = db.query(models.Customer).count()
    logger.info(f"customers count = {count}")
    return count


def get_orders_count(db: Session) -> int:
    logger.info("DB query: COUNT orders")
    count = db.query(models.Order).count()
    logger.info(f"orders count = {count}")
    return count


def get_products_count(db: Session) -> int:
    logger.info("DB query: COUNT products")
    count = db.query(models.Product).count()
    logger.info(f"products count = {count}")
    return count


def get_employees_count(db: Session) -> int:
    logger.info("DB query: COUNT employees")
    count = db.query(models.Employee).count()
    logger.info(f"employees count = {count}")
    return count


def get_offices_count(db: Session) -> int:
    logger.info("DB query: COUNT offices")
    count = db.query(models.Office).count()
    logger.info(f"offices count = {count}")
    return count


def get_payments_count(db: Session) -> int:
    logger.info("DB query: COUNT payments")
    count = db.query(models.Payment).count()
    logger.info(f"payments count = {count}")
    return count


def get_orderdetails_count(db: Session) -> int:
    logger.info("DB query: COUNT orderdetails")
    count = db.query(models.OrderDetail).count()
    logger.info(f"orderdetails count = {count}")
    return count


def get_productlines_count(db: Session) -> int:
    logger.info("DB query: COUNT productlines")
    count = db.query(models.ProductLine).count()
    logger.info(f"productlines count = {count}")
    return count
