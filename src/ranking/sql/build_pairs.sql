-- Input: the `sampled_reviews` table (review_id, user_id, stars, text) produced by
-- sample_reviews.sql.
--
-- Output: a `pairs` table with columns (user_id, text_a, text_b, label), where pairs
-- are formed within each user_id (self-join on user_id, review_id < review_id to
-- avoid duplicate/reversed pairs), ties (stars_a == stars_b) are dropped, label = 1
-- if stars_a > stars_b else 0, and at most $max_pairs_per_user pairs are kept per
-- user.

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
