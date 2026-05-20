SCHEMA = """
Schema
-- Drop tables safely (order + CASCADE matters)
DROP TABLE IF EXISTS orderdetails CASCADE;
DROP TABLE IF EXISTS orders CASCADE;
DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS customers CASCADE;
DROP TABLE IF EXISTS employees CASCADE;
DROP TABLE IF EXISTS offices CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS productlines CASCADE;

-- Tables
CREATE TABLE productlines (
  "productLine" VARCHAR(50) PRIMARY KEY,
  "textDescription" VARCHAR(4000),
  "htmlDescription" TEXT,
  "image" BYTEA
);

CREATE TABLE products (
  "productCode" VARCHAR(15) PRIMARY KEY,
  "productName" VARCHAR(70) NOT NULL,
  "productLine" VARCHAR(50) NOT NULL,
  "productScale" VARCHAR(10) NOT NULL,
  "productVendor" VARCHAR(50) NOT NULL,
  "productDescription" TEXT NOT NULL,
  "quantityInStock" INTEGER NOT NULL,
  "buyPrice" NUMERIC(10,2) NOT NULL,
  "MSRP" NUMERIC(10,2) NOT NULL,
  FOREIGN KEY ("productLine") REFERENCES productlines("productLine")
);

CREATE TABLE offices (
  "officeCode" VARCHAR(10) PRIMARY KEY,
  "city" VARCHAR(50) NOT NULL,
  "phone" VARCHAR(50) NOT NULL,
  "addressLine1" VARCHAR(50) NOT NULL,
  "addressLine2" VARCHAR(50),
  "state" VARCHAR(50),
  "country" VARCHAR(50) NOT NULL,
  "postalCode" VARCHAR(15) NOT NULL,
  "territory" VARCHAR(10) NOT NULL
);

CREATE TABLE employees (
  "employeeNumber" INTEGER PRIMARY KEY,
  "lastName" VARCHAR(50) NOT NULL,
  "firstName" VARCHAR(50) NOT NULL,
  "extension" VARCHAR(10) NOT NULL,
  "email" VARCHAR(100) NOT NULL,
  "officeCode" VARCHAR(10) NOT NULL,
  "reportsTo" INTEGER,
  "jobTitle" VARCHAR(50) NOT NULL,
  FOREIGN KEY ("reportsTo") REFERENCES employees("employeeNumber"),
  FOREIGN KEY ("officeCode") REFERENCES offices("officeCode")
);

CREATE TABLE customers (
  "customerNumber" INTEGER PRIMARY KEY,
  "customerName" VARCHAR(50) NOT NULL,
  "contactLastName" VARCHAR(50) NOT NULL,
  "contactFirstName" VARCHAR(50) NOT NULL,
  "phone" VARCHAR(50) NOT NULL,
  "addressLine1" VARCHAR(50) NOT NULL,
  "addressLine2" VARCHAR(50),
  "city" VARCHAR(50) NOT NULL,
  "state" VARCHAR(50),
  "postalCode" VARCHAR(15),
  "country" VARCHAR(50) NOT NULL,
  "salesRepEmployeeNumber" INTEGER,
  "creditLimit" NUMERIC(10,2),
  FOREIGN KEY ("salesRepEmployeeNumber") REFERENCES employees("employeeNumber")
);

CREATE TABLE payments (
  "customerNumber" INTEGER,
  "checkNumber" VARCHAR(50),
  "paymentDate" DATE NOT NULL,
  "amount" NUMERIC(10,2) NOT NULL,
  PRIMARY KEY ("customerNumber", "checkNumber"),
  FOREIGN KEY ("customerNumber") REFERENCES customers("customerNumber")
);

CREATE TABLE orders (
  "orderNumber" INTEGER PRIMARY KEY,
  "orderDate" DATE NOT NULL,
  "requiredDate" DATE NOT NULL,
  "shippedDate" DATE,
  "status" VARCHAR(15) NOT NULL,
  "comments" TEXT,
  "customerNumber" INTEGER NOT NULL,
  FOREIGN KEY ("customerNumber") REFERENCES customers("customerNumber")
);

CREATE TABLE orderdetails (
  "orderNumber" INTEGER,
  "productCode" VARCHAR(15),
  "quantityOrdered" INTEGER NOT NULL,
  "priceEach" NUMERIC(10,2) NOT NULL,
  "orderLineNumber" SMALLINT NOT NULL,
  PRIMARY KEY ("orderNumber", "productCode"),
  FOREIGN KEY ("orderNumber") REFERENCES orders("orderNumber"),
  FOREIGN KEY ("productCode") REFERENCES products("productCode")
);
"""

PLANNER_SYSTEM = (
    "You are a database analyst. Analyze the user question and propose a plan.\n"
    "Return ONLY JSON with keys: intent, tables, columns, filters, joins, notes.\n"
    "No markdown or explanation."
)

GENERATOR_SYSTEM = (
    "You are a PostgreSQL expert. Write only a SELECT query.\n"
    "Rules:\n"
    "1. ALL column names MUST be double-quoted: \"columnName\"\n"
    "2. Only SELECT -- never INSERT/UPDATE/DELETE/DROP\n"
    "3. End with a semicolon\n"
    "4. Return ONLY the SQL -- no explanation, no markdown\n\n"
    f"{SCHEMA}"
)

SUMMARIZER_SYSTEM = (
  "You are a data analyst. Write a natural language answer in 1-2 sentences.\n"
  "Sound like: 'There are 42 shipped orders from customers in USA.'\n"
  "Never say 'Retrieved N rows' or 'Top result'.\n"
  "If the result is a single aggregate (COUNT/SUM/AVG/etc.), state it directly.\n"
  "Use ONLY the provided results. If there are no rows, say 'No rows matched.'"
)


def planner_user_prompt(question: str) -> str:
    return f"Schema:\n{SCHEMA}\n\nQuestion: {question}"


def generator_user_prompt(question: str, plan_json: str) -> str:
    return f"Question: {question}\n\nPlan:\n{plan_json}"


def summarizer_user_prompt(question: str, sql: str, result_json: str, note: str) -> str:
    return (
        f"Question: {question}\n"
        f"SQL: {sql}\n"
        f"Note: {note}\n"
        f"Result: {result_json}"
    )
