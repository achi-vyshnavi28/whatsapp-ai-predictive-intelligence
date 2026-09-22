-- ============================================================================
-- smb-ai-operations-intelligence
-- Analytical SQL: multi-table joins, subqueries, CTEs, window functions
-- Run against smb_ai_operations (see sql/01_schema.sql + python/load_data.py)
-- Real data: Online Retail II (UCI/Kaggle), Dec 2009-Dec 2011.
-- ============================================================================


-- ----------------------------------------------------------------------------
-- Q1. Monthly active customers and revenue trend (CTE + date_trunc + window
--     function for month-over-month change).
-- ----------------------------------------------------------------------------
WITH monthly AS (
    SELECT
        DATE_TRUNC('month', invoice_date)::date AS month,
        COUNT(DISTINCT customer_id) AS active_customers,
        SUM(quantity * price) FILTER (WHERE NOT is_cancellation) AS revenue
    FROM transactions
    WHERE customer_id IS NOT NULL
    GROUP BY 1
)
SELECT
    month, active_customers, ROUND(revenue::numeric, 2) AS revenue,
    ROUND(100.0 * (revenue - LAG(revenue) OVER (ORDER BY month)) /
          NULLIF(LAG(revenue) OVER (ORDER BY month), 0), 1) AS revenue_mom_change_pct
FROM monthly
ORDER BY month;


-- ----------------------------------------------------------------------------
-- Q2. Cohort retention: for each customer's first-purchase month, what % of
--     that cohort is still active N months later (classic cohort analysis,
--     CTE + window functions).
-- ----------------------------------------------------------------------------
WITH first_purchase AS (
    SELECT customer_id, DATE_TRUNC('month', MIN(invoice_date))::date AS cohort_month
    FROM transactions
    WHERE customer_id IS NOT NULL AND NOT is_cancellation
    GROUP BY customer_id
),
activity AS (
    SELECT DISTINCT customer_id, DATE_TRUNC('month', invoice_date)::date AS active_month
    FROM transactions
    WHERE customer_id IS NOT NULL AND NOT is_cancellation
),
cohort_activity AS (
    SELECT fp.cohort_month, a.active_month,
           (EXTRACT(YEAR FROM a.active_month) - EXTRACT(YEAR FROM fp.cohort_month)) * 12 +
           (EXTRACT(MONTH FROM a.active_month) - EXTRACT(MONTH FROM fp.cohort_month)) AS month_number,
           a.customer_id
    FROM activity a
    JOIN first_purchase fp ON fp.customer_id = a.customer_id
),
cohort_size AS (
    SELECT cohort_month, COUNT(DISTINCT customer_id) AS cohort_customers
    FROM first_purchase GROUP BY cohort_month
)
SELECT ca.cohort_month, ca.month_number,
       COUNT(DISTINCT ca.customer_id) AS active_customers,
       cs.cohort_customers,
       ROUND(100.0 * COUNT(DISTINCT ca.customer_id) / cs.cohort_customers, 1) AS retention_pct
FROM cohort_activity ca
JOIN cohort_size cs ON cs.cohort_month = ca.cohort_month
WHERE ca.month_number BETWEEN 0 AND 6
GROUP BY ca.cohort_month, ca.month_number, cs.cohort_customers
ORDER BY ca.cohort_month, ca.month_number;


-- ----------------------------------------------------------------------------
-- Q3. RFM base metrics per customer (Recency / Frequency / Monetary) --
--     CTE feeding the churn-risk model's feature set.
-- ----------------------------------------------------------------------------
WITH dataset_end AS (SELECT MAX(invoice_date) AS max_date FROM transactions),
rfm AS (
    SELECT t.customer_id,
           EXTRACT(DAY FROM (d.max_date - MAX(t.invoice_date))) AS recency_days,
           COUNT(DISTINCT t.invoice) AS frequency,
           ROUND(SUM(t.quantity * t.price)::numeric, 2) AS monetary
    FROM transactions t
    CROSS JOIN dataset_end d
    WHERE t.customer_id IS NOT NULL AND NOT t.is_cancellation
    GROUP BY t.customer_id, d.max_date
)
SELECT customer_id, recency_days, frequency, monetary,
       NTILE(4) OVER (ORDER BY recency_days DESC) AS recency_quartile,
       NTILE(4) OVER (ORDER BY frequency) AS frequency_quartile,
       NTILE(4) OVER (ORDER BY monetary) AS monetary_quartile
FROM rfm
ORDER BY monetary DESC
LIMIT 20;


-- ----------------------------------------------------------------------------
-- Q4. Churn flag: a customer is "churned" if their last purchase was more
--     than 90 days before the dataset's last recorded date (subquery +
--     CASE aggregation).
-- ----------------------------------------------------------------------------
WITH last_purchase AS (
    SELECT customer_id, MAX(invoice_date) AS last_order
    FROM transactions
    WHERE customer_id IS NOT NULL AND NOT is_cancellation
    GROUP BY customer_id
),
labeled AS (
    SELECT customer_id,
           CASE WHEN (SELECT MAX(invoice_date) FROM transactions) - last_order > INTERVAL '90 days'
                THEN 'churned' ELSE 'active' END AS status
    FROM last_purchase
)
SELECT status, COUNT(*) AS customers,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 1) AS pct
FROM labeled
GROUP BY status;


-- ----------------------------------------------------------------------------
-- Q5. Top products by revenue with running cumulative share (window
--     function) -- feeds the demand-forecasting product-selection step.
-- ----------------------------------------------------------------------------
WITH product_revenue AS (
    SELECT t.stock_code, p.description, SUM(t.quantity * t.price) AS revenue
    FROM transactions t
    JOIN products p ON p.stock_code = t.stock_code
    WHERE NOT t.is_cancellation
    GROUP BY t.stock_code, p.description
)
SELECT stock_code, description, ROUND(revenue::numeric, 2) AS revenue,
       ROUND(100.0 * SUM(revenue) OVER (ORDER BY revenue DESC) /
             SUM(revenue) OVER (), 2) AS cumulative_pct
FROM product_revenue
ORDER BY revenue DESC
LIMIT 15;


-- ----------------------------------------------------------------------------
-- Q6. Monthly demand (units sold) for the top 5 products by volume -- the
--     direct input to the demand-forecasting time series model.
-- ----------------------------------------------------------------------------
WITH top_products AS (
    SELECT stock_code FROM transactions
    WHERE NOT is_cancellation
    GROUP BY stock_code ORDER BY SUM(quantity) DESC LIMIT 5
)
SELECT t.stock_code, DATE_TRUNC('month', t.invoice_date)::date AS month,
       SUM(t.quantity) AS units_sold
FROM transactions t
JOIN top_products tp ON tp.stock_code = t.stock_code
WHERE NOT t.is_cancellation
GROUP BY t.stock_code, DATE_TRUNC('month', t.invoice_date)
ORDER BY t.stock_code, month;


-- ----------------------------------------------------------------------------
-- Q7. Revenue by country with rank (window function) -- multi-table not
--     needed here (country is denormalized onto transactions), still a
--     real window-function ranking query.
-- ----------------------------------------------------------------------------
SELECT country, COUNT(DISTINCT customer_id) AS customers,
       ROUND(SUM(quantity * price)::numeric, 2) AS revenue,
       RANK() OVER (ORDER BY SUM(quantity * price) DESC) AS revenue_rank
FROM transactions
WHERE NOT is_cancellation
GROUP BY country
ORDER BY revenue_rank
LIMIT 10;


-- ----------------------------------------------------------------------------
-- Q8. Anomaly: customers whose revenue dropped sharply month over month
--     (LAG window function) -- an early-warning churn-risk signal, not
--     just a post-hoc "they left" label.
-- ----------------------------------------------------------------------------
WITH monthly_customer_revenue AS (
    SELECT customer_id, DATE_TRUNC('month', invoice_date)::date AS month,
           SUM(quantity * price) AS revenue
    FROM transactions
    WHERE customer_id IS NOT NULL AND NOT is_cancellation
    GROUP BY customer_id, DATE_TRUNC('month', invoice_date)
),
with_change AS (
    SELECT customer_id, month, revenue,
           LAG(revenue) OVER (PARTITION BY customer_id ORDER BY month) AS prev_revenue
    FROM monthly_customer_revenue
)
SELECT customer_id, month, ROUND(revenue::numeric, 2) AS revenue,
       ROUND(prev_revenue::numeric, 2) AS prev_revenue,
       ROUND(100.0 * (revenue - prev_revenue) / prev_revenue, 1) AS mom_change_pct
FROM with_change
WHERE prev_revenue > 500 AND revenue < prev_revenue * 0.2
ORDER BY mom_change_pct ASC
LIMIT 20;


-- ----------------------------------------------------------------------------
-- Q9. Subquery: customers whose average order value is above the overall
--     average order value (high-value accounts worth prioritizing in a
--     churn-prevention outreach).
-- ----------------------------------------------------------------------------
WITH order_values AS (
    SELECT invoice, customer_id, SUM(quantity * price) AS order_value
    FROM transactions
    WHERE customer_id IS NOT NULL AND NOT is_cancellation
    GROUP BY invoice, customer_id
)
SELECT customer_id, COUNT(*) AS orders, ROUND(AVG(order_value)::numeric, 2) AS avg_order_value
FROM order_values
GROUP BY customer_id
HAVING AVG(order_value) > (SELECT AVG(order_value) FROM order_values)
ORDER BY avg_order_value DESC
LIMIT 15;
