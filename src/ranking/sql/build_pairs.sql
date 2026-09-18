-- Input: a `reviews` relation with at least (review_id, user_id, stars, text) columns,
-- registered by register_raw_reviews() in build_pairs.py.
--
-- Output: a `pairs` table with columns (user_id, text_a, text_b, label), where pairs
-- are formed within each user_id (self-join on user_id, review_id < review_id to
-- avoid duplicate/reversed pairs), ties (stars_a == stars_b) are dropped, label = 1
-- if stars_a > stars_b else 0, only users with >= $min_reviews_per_user reviews are
-- included, and at most $max_pairs_per_user pairs are kept per user.
--
-- Some Yelp users have thousands of reviews, so self-joining all of a user's reviews
-- can blow up to millions of rows for that user alone before QUALIFY caps it down to
-- $max_pairs_per_user. sampled_reviews pre-caps each eligible user to
-- $max_reviews_per_user reviews *before* the self-join so the join itself stays
-- bounded (at most $max_reviews_per_user choose 2 rows per user).

WITH eligible_users AS (
    SELECT user_id
    FROM reviews
    GROUP BY user_id
    HAVING COUNT(*) >= $min_reviews_per_user
),
sampled_reviews AS (
    SELECT r.review_id, r.user_id, r.stars, r.text
    FROM reviews r
    JOIN eligible_users u ON u.user_id = r.user_id
    QUALIFY row_number() OVER (PARTITION BY r.user_id ORDER BY random()) <= $max_reviews_per_user
)
SELECT
    a.user_id,
    a.text AS text_a,
    b.text AS text_b,
    IF (a.stars > b.stars, 1, 0) AS label
FROM sampled_reviews a
JOIN sampled_reviews b
    ON a.user_id = b.user_id
    AND a.review_id < b.review_id
WHERE a.stars != b.stars
QUALIFY row_number() OVER (PARTITION BY a.user_id ORDER BY random()) <= $max_pairs_per_user
