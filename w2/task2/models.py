from sqlalchemy import Column, Integer, String, Numeric, Date, Text, SmallInteger, LargeBinary, ForeignKey
from sqlalchemy.orm import relationship
from database import Base


class ProductLine(Base):
    __tablename__ = "productlines"

    productLine = Column("productLine", String(50), primary_key=True)
    textDescription = Column("textDescription", String(4000), nullable=True)
    htmlDescription = Column("htmlDescription", Text, nullable=True)
    image = Column("image", LargeBinary, nullable=True)

    products = relationship("Product", back_populates="product_line_rel")


class Product(Base):
    __tablename__ = "products"

    productCode = Column("productCode", String(15), primary_key=True)
    productName = Column("productName", String(70), nullable=False)
    productLine = Column("productLine", String(50), ForeignKey("productlines.productLine"), nullable=False)
    productScale = Column("productScale", String(10), nullable=False)
    productVendor = Column("productVendor", String(50), nullable=False)
    productDescription = Column("productDescription", Text, nullable=False)
    quantityInStock = Column("quantityInStock", Integer, nullable=False)
    buyPrice = Column("buyPrice", Numeric(10, 2), nullable=False)
    MSRP = Column("MSRP", Numeric(10, 2), nullable=False)

    product_line_rel = relationship("ProductLine", back_populates="products")


class Office(Base):
    __tablename__ = "offices"

    officeCode = Column("officeCode", String(10), primary_key=True)
    city = Column("city", String(50), nullable=False)
    phone = Column("phone", String(50), nullable=False)
    addressLine1 = Column("addressLine1", String(50), nullable=False)
    addressLine2 = Column("addressLine2", String(50), nullable=True)
    state = Column("state", String(50), nullable=True)
    country = Column("country", String(50), nullable=False)
    postalCode = Column("postalCode", String(15), nullable=False)
    territory = Column("territory", String(10), nullable=False)

    employees = relationship("Employee", back_populates="office")


class Employee(Base):
    __tablename__ = "employees"

    employeeNumber = Column("employeeNumber", Integer, primary_key=True)
    lastName = Column("lastName", String(50), nullable=False)
    firstName = Column("firstName", String(50), nullable=False)
    extension = Column("extension", String(10), nullable=False)
    email = Column("email", String(100), nullable=False)
    officeCode = Column("officeCode", String(10), ForeignKey("offices.officeCode"), nullable=False)
    reportsTo = Column("reportsTo", Integer, ForeignKey("employees.employeeNumber"), nullable=True)
    jobTitle = Column("jobTitle", String(50), nullable=False)

    office = relationship("Office", back_populates="employees")


class Customer(Base):
    __tablename__ = "customers"

    customerNumber = Column("customerNumber", Integer, primary_key=True)
    customerName = Column("customerName", String(50), nullable=False)
    contactLastName = Column("contactLastName", String(50), nullable=False)
    contactFirstName = Column("contactFirstName", String(50), nullable=False)
    phone = Column("phone", String(50), nullable=False)
    addressLine1 = Column("addressLine1", String(50), nullable=False)
    addressLine2 = Column("addressLine2", String(50), nullable=True)
    city = Column("city", String(50), nullable=False)
    state = Column("state", String(50), nullable=True)
    postalCode = Column("postalCode", String(15), nullable=True)
    country = Column("country", String(50), nullable=False)
    salesRepEmployeeNumber = Column("salesRepEmployeeNumber", Integer,
                                    ForeignKey("employees.employeeNumber"), nullable=True)
    creditLimit = Column("creditLimit", Numeric(10, 2), nullable=True)

    orders = relationship("Order", back_populates="customer")
    payments = relationship("Payment", back_populates="customer")


class Payment(Base):
    __tablename__ = "payments"

    customerNumber = Column("customerNumber", Integer, ForeignKey("customers.customerNumber"), primary_key=True)
    checkNumber = Column("checkNumber", String(50), primary_key=True)
    paymentDate = Column("paymentDate", Date, nullable=False)
    amount = Column("amount", Numeric(10, 2), nullable=False)

    customer = relationship("Customer", back_populates="payments")


class Order(Base):
    __tablename__ = "orders"

    orderNumber = Column("orderNumber", Integer, primary_key=True)
    orderDate = Column("orderDate", Date, nullable=False)
    requiredDate = Column("requiredDate", Date, nullable=False)
    shippedDate = Column("shippedDate", Date, nullable=True)
    status = Column("status", String(15), nullable=False)
    comments = Column("comments", Text, nullable=True)
    customerNumber = Column("customerNumber", Integer, ForeignKey("customers.customerNumber"), nullable=False)

    customer = relationship("Customer", back_populates="orders")
    order_details = relationship("OrderDetail", back_populates="order")


class OrderDetail(Base):
    __tablename__ = "orderdetails"

    orderNumber = Column("orderNumber", Integer, ForeignKey("orders.orderNumber"), primary_key=True)
    productCode = Column("productCode", String(15), ForeignKey("products.productCode"), primary_key=True)
    quantityOrdered = Column("quantityOrdered", Integer, nullable=False)
    priceEach = Column("priceEach", Numeric(10, 2), nullable=False)
    orderLineNumber = Column("orderLineNumber", SmallInteger, nullable=False)

    order = relationship("Order", back_populates="order_details")
