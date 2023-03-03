CREATE TABLE IF NOT EXISTS batter_avg_historical AS
SELECT batter AS Batter
    , SUM(hit) AS Hit
    , SUM(atBat) AS atBat
    , (CASE WHEN SUM(atBat) > 0 THEN SUM(hit) / SUM(atBat) ELSE 0 END) AS Batting_Avg
FROM batter_counts
GROUP BY Batter
;

-- SELECT  Batter, Hit, atBat, Batting_Avg FROM batter_avg_historical;
-- SELECT COUNT(*) FROM batter_avg_historical;





CREATE TABLE IF NOT EXISTS batter_avg_annual AS
SELECT batter AS Batter
    , YEAR(game.local_date) AS For_Year
    , (CASE WHEN SUM(bc.atBat) > 0 THEN SUM(bc.Hit) / SUM(bc.atBat) ELSE 0 END) AS Batting_Avg
FROM batter_counts AS bc
    INNER JOIN game ON bc.game_id = game.game_id
GROUP BY Batter, For_Year
ORDER BY Batter, For_Year
;
-- SELECT Batter, For_Year, Batting_Avg FROM batter_avg_annual;
-- SELECT COUNT(*) FROM batter_avg_annual;



-- Create a temporary table to store batting data for each game
CREATE OR REPLACE TEMPORARY TABLE batter_avg_rolling_temp AS
SELECT game.game_id, bc.batter AS Batter, bc.Hit, bc.atBat, game.local_date
FROM batter_counts AS bc
    INNER JOIN game ON bc.game_id = game.game_id
;

CREATE INDEX Batter_id ON batter_avg_rolling_temp(Batter);
SELECT game_id, Batter, Hit, atBat, local_date FROM batter_avg_rolling_temp;
CREATE OR REPLACE TABLE batter_avg_rolling AS
SELECT bart1.batter
    , (CASE WHEN SUM(bart2.atBat) > 0 THEN SUM(bart2.Hit) / SUM(bart2.atBat) ELSE 0 END) AS Batting_Avg
    , bart1.game_id
    , DATE(bart1.local_date) AS local_date
    , DATE_SUB(bart1.local_date, INTERVAL 100 DAY) AS Date_since
FROM batter_avg_rolling_temp bart1
    INNER JOIN batter_avg_rolling_temp bart2 ON bart1.Batter = bart2.Batter
        AND bart2.local_date < bart1.local_date
        AND bart2.local_date > DATE_SUB(bart1.local_date, INTERVAL 100 DAY)
-- Where clause could be removed  for all players
-- WHERE bart1.batter = 435623
GROUP BY bart1.Batter, DATE(bart1.local_date)
ORDER BY bart1.Batter
;


-- SELECT batter, Batting_Avg, game_id, local_date, Date_since FROM batter_avg_rolling;
-- SELECT COUNT(*) FROM batter_avg_rolling;
