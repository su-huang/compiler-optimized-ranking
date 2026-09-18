-- Input: the `pairs` table produced by build_pairs.sql.
--
-- Output: a `user_splits` table with columns (user_id, split), where split is one of
-- 'train' / 'val' / 'test', assigned per user_id (not per pair) so a user's reviews
-- never span more than one split. Each user_id is hashed to a deterministic [0, 1)
-- bucket, then thresholded against $train_frac / $val_frac.

WITH hashed_users AS (
    SELECT DISTINCT
        user_id,
        hash(user_id) / (pow(2, 64) - 1) AS bucket
    FROM pairs
)
SELECT
    user_id,
    CASE
        WHEN bucket < $train_frac THEN 'train'
        WHEN bucket < $train_frac + $val_frac THEN 'val'
        ELSE 'test'
    END AS split
FROM hashed_users

