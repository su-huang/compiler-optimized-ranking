-- Input: a `reviews` relation with at least (review_id, user_id, stars, text) columns,
-- registered by register_raw_reviews() in build_pairs.py.
--
-- Output: a `sampled_reviews` table (review_id, user_id, stars, text), capped to
-- $max_reviews_per_user reviews per user among users with >= $min_reviews_per_user
-- reviews. Kept as its own table (rather than inlined into build_pairs.sql) so it can
-- be exported on its own for review-level eval (e.g. kendall_tau_by_user), and so the
-- downstream self-join in build_pairs.sql stays bounded regardless of power users with
-- thousands of reviews.

WITH eligible_users AS (
    SELECT user_id
    FROM reviews
    GROUP BY user_id
    HAVING COUNT(*) >= $min_reviews_per_user
)
SELECT r.review_id, r.user_id, r.stars, r.text
FROM reviews r
JOIN eligible_users u ON u.user_id = r.user_id
QUALIFY row_number() OVER (PARTITION BY r.user_id ORDER BY random()) <= $max_reviews_per_user
