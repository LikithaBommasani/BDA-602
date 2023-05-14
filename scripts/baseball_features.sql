USE baseball;

CREATE OR REPLACE TABLE temp_team_pitching_counts AS
SELECT tp.game_id
    , tp.team_id
    , g.home_team_id
    , g.away_team_id
    , DATE(g.local_date) AS local_date
    , tp.atBat AS tp_atBat
    , tp.Hit AS tp_Hit
    , tp.Strikeout
    , tp.Walk
    , tp.Ground_Out
    , tp.Fly_Out
    , CASE WHEN b.away_runs < b.home_runs THEN 1 #noqa
            WHEN b.away_runs > b.home_runs THEN 0 #noqa
            ELSE 0 END AS HomeTeamWins   #noqa
FROM team_pitching_counts tp
    JOIN boxscore b ON b.game_id = tp.game_id
    JOIN game g ON g.game_id = tp.game_id
GROUP BY game_id, team_id
;



# SELECT  COUNT(*)   FROM temp_team_pitching_counts;
#
# SELECT  *   FROM temp_team_pitching_counts;

CREATE OR REPLACE TABLE temp_table AS
SELECT ttp.* #noqa
    , tb.Hit
    , tb.atBat
    , tb.Home_Run AS home_runs
FROM temp_team_pitching_counts ttp
    JOIN team_batting_counts tb ON ttp.game_id = tb.game_id
GROUP BY team_id, game_id
;

# SELECT  COUNT(*)   FROM temp_table;
#
# SELECT  *   FROM temp_table;


CREATE OR REPLACE TABLE temp_feature_table_1 AS
SELECT tt1.game_id
    , tt1.team_id
    , tt1.home_team_id
    , tt1.away_team_id
    , tt1.local_date
    , COALESCE(SUM(tt2.atBat), 0) AS atBat
    , COALESCE(SUM(tt2.Hit), 0) AS Hit
    , COALESCE(SUM(tt2.tp_atBat), 0) AS tp_atBat
    , COALESCE(SUM(tt2.tp_Hit), 0) AS tp_Hit
    , COALESCE(SUM(tt2.home_runs), 0) AS home_runs
    , COALESCE(SUM(tt2.Strikeout), 0) AS tp_Strikeout
    , COALESCE(SUM(tt2.Walk), 0) AS tp_Walk
    , COALESCE(SUM(tt2.Ground_Out), 0) AS tp_Ground_Out
    , COALESCE(SUM(tt2.Fly_Out), 0) AS tp_Fly_Out
    , tt1.HomeTeamWins
FROM temp_table tt1
    JOIN temp_table tt2 ON tt2.team_id = tt1.team_id
        AND tt2.local_date < tt1.local_date
        AND tt2.local_date >= DATE_ADD(tt1.local_date, INTERVAL - 100 DAY)
GROUP BY tt1.game_id, tt1.team_id
ORDER BY tt1.game_id, tt1.team_id
;

# UPDATE temp_feature_table_1 tf
# SET tf.HomeTeamWins
#     =       CASE HomeTeamWins
#                  WHEN 'H' THEN 1 #noqa
#                  WHEN 'A' THEN 0 #noqa
#             END #noqa
# ;

# SELECT  COUNT(*)   FROM temp_feature_table_1;
#
# SELECT  *   FROM temp_feature_table_1;


CREATE OR REPLACE TABLE feature_ratio_table AS
SELECT tf.game_id
    , tf.team_id
    , tf.away_team_id
    , tf.home_team_id
    , tf.local_date
    , ROUND(tfh.Hit / NULLIF(tfh.atBat, 0) / NULLIF(tfa.Hit / NULLIF(tfa.atBat, 0), 0), 2) AS Batting_Average_Ratio
    , ROUND(tfh.Hit / NULLIF(tfh.tp_Strikeout, 0) / NULLIF(tfa.Hit / NULLIF(tfa.tp_Strikeout, 0), 0), 2) AS Hit_per_Strikeout_Ratio
    , ROUND(tfh.tp_Strikeout / NULLIF(tfh.tp_Walk, 0) / NULLIF(tfa.tp_Strikeout / NULLIF(tfa.tp_Walk, 0), 0), 2) AS Strikeout_to_walk_Ratio
    , ROUND(tfh.tp_Ground_Out / NULLIF(tfh.tp_Fly_Out, 0) / NULLIF(tfa.tp_Ground_Out / NULLIF(tfa.tp_Fly_Out, 0), 0), 2) AS Groundout_to_Flyout_Ratio
    , ROUND(tfh.tp_Walk / NULLIF(tfh.atBat, 0) / NULLIF(tfa.tp_Walk / NULLIF(tfa.atBat, 0), 0), 2) AS Walks_per_atBat_Ratio
    , ROUND(tfh.tp_Strikeout / NULLIF(tfh.atBat, 0) / NULLIF(tfa.atBat / NULLIF(tfa.atBat, 0), 0), 2) AS Strikeout_per_atBat_Ratio
    , ROUND(tfh.home_runs / NULLIF(tfh.Hit, 0) / NULLIF(tfa.home_runs / NULLIF(tfa.Hit, 0), 0), 2) AS HR_H_Ratio
    , ROUND(tfh.atBat / NULLIF(tfh.home_runs, 0) / NULLIF(tfa.atBat / NULLIF(tfa.home_runs, 0), 0), 2) AS AB_HR_Ratio
    , tf.HomeTeamWins
FROM temp_feature_table_1 tf
    JOIN temp_feature_table_1 tfh ON tf.game_id = tfh.game_id AND tf.home_team_id = tfh.team_id
    JOIN temp_feature_table_1 tfa ON tf.game_id = tfa.game_id AND tf.away_team_id = tfa.team_id
GROUP BY tf.game_id, tf.home_team_id, tf.away_team_id
;

#  SELECT  COUNT(*)   FROM feature_ratio_table;
#
# SELECT  *   FROM feature_ratio_table
# ;
