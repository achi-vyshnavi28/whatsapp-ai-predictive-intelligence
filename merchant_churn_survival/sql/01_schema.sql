-- ============================================================================
-- smb-ai-operations-intelligence
-- PostgreSQL schema for the shared base data used by both churn_risk/ and
-- demand_forecasting/. Source: Online Retail II (UCI Machine Learning
-- Repository / Kaggle, real transaction data from a UK-based online
-- gift/homeware wholesaler, Dec 2009-Dec 2011).
--
-- Framing: this dataset's "Customer ID" is disproportionately a repeat
-- WHOLESALE buyer (many purchase in bulk quantities, resembling small
-- businesses restocking inventory, not one-off consumers) -- used here as
-- a structural analogue for recurring SMB merchants on a platform, the
-- same way the sibling repos use Olist as a stand-in for merchant
-- acquisition/operations.
-- ============================================================================

DROP TABLE IF EXISTS transactions CASCADE;
DROP TABLE IF EXISTS products CASCADE;
DROP TABLE IF EXISTS customers CASCADE;

CREATE TABLE customers (
    customer_id  INTEGER PRIMARY KEY,
    country      VARCHAR(60)
);

-- description is the most frequently occurring non-null description for
-- that stock code (the raw data has minor spelling drift for a small
-- number of codes across 2 years of manual entry -- a real, documented
-- data-quality quirk, not an error in this schema).
CREATE TABLE products (
    stock_code   VARCHAR(20) PRIMARY KEY,
    description  VARCHAR(200)
);

-- Grain: one row per invoice line item. customer_id is intentionally
-- nullable with NO enforced NOT NULL -- 243,007 of 1,067,371 real rows
-- (22.8%) have no customer ID at all (guest/unregistered purchases), a
-- genuine data-quality characteristic of this export, documented rather
-- than silently dropped.
CREATE TABLE transactions (
    invoice          VARCHAR(20),
    stock_code       VARCHAR(20) REFERENCES products(stock_code),
    customer_id      INTEGER REFERENCES customers(customer_id),
    country          VARCHAR(60),
    invoice_date     TIMESTAMP,
    quantity         INTEGER,
    price            NUMERIC(10,3),
    is_cancellation  BOOLEAN
);

CREATE INDEX idx_transactions_customer ON transactions(customer_id);
CREATE INDEX idx_transactions_date ON transactions(invoice_date);
CREATE INDEX idx_transactions_stock ON transactions(stock_code);
CREATE INDEX idx_transactions_invoice ON transactions(invoice);
