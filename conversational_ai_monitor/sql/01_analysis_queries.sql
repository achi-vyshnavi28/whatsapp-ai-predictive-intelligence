-- ============================================================================
-- conversational_ai_monitor -- analytical SQL against the structured
-- `conversations` table (loaded via python/etl_to_structured_table.py from
-- Firestore-shaped documents). Joins/CTEs/window functions/subqueries.
-- ============================================================================

-- Q1. Accuracy and avg confidence by batch, with a window function flagging
--     batches that dropped >5pp below the running baseline of prior batches.
WITH batch_stats AS (
    SELECT batch, COUNT(*) AS n,
           ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS accuracy_pct,
           ROUND(AVG(confidence)::numeric, 3) AS avg_confidence
    FROM conversations GROUP BY batch
),
with_baseline AS (
    SELECT *, AVG(accuracy_pct) OVER (ORDER BY batch ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS running_baseline
    FROM batch_stats
)
SELECT batch, n, accuracy_pct, avg_confidence, ROUND(running_baseline, 1) AS running_baseline,
       (running_baseline - accuracy_pct) > 5 AS drift_alert
FROM with_baseline ORDER BY batch;

-- Q2. Per-intent precision proxy: of predictions FOR each intent, what
--     share were actually correct (subquery + CASE aggregation).
SELECT predicted_intent,
       COUNT(*) AS predictions,
       SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) AS correct,
       ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS precision_pct
FROM conversations
GROUP BY predicted_intent
HAVING COUNT(*) >= 5
ORDER BY precision_pct ASC;

-- Q3. Top confusion pairs (actual vs predicted where wrong) -- join the
--     table to itself is not needed since both columns live on one row.
SELECT true_intent, predicted_intent, COUNT(*) AS occurrences
FROM conversations
WHERE NOT is_correct AND true_intent <> predicted_intent
GROUP BY true_intent, predicted_intent
ORDER BY occurrences DESC
LIMIT 10;

-- Q4. Category-level accuracy with rank (window function).
SELECT category, COUNT(*) AS n,
       ROUND(100.0 * SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS accuracy_pct,
       RANK() OVER (ORDER BY SUM(CASE WHEN is_correct THEN 1 ELSE 0 END)::float / COUNT(*) ASC) AS worst_rank
FROM conversations
GROUP BY category
ORDER BY worst_rank;

-- Q5. Low-confidence correct predictions vs high-confidence wrong ones --
--     the two calibration failure modes that matter operationally
--     (subquery-driven anomaly flags).
SELECT
    (SELECT COUNT(*) FROM conversations WHERE confidence < 0.4 AND is_correct) AS under_confident_but_right,
    (SELECT COUNT(*) FROM conversations WHERE confidence > 0.8 AND NOT is_correct) AS over_confident_but_wrong;
