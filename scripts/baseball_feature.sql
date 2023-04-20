Use baseball;

CREATE OR REPLACE TEMPORARY TABLE temp AS
SELECT
tbc.game_id,
tbc.team_id,
tbc.opponent_team_id,
tbc.updatedDate,
tbc.homeTeam,
tbc.awayTeam,
tbc.win,
tbc.finalScore,
tbc.Single ,
tpc.Single AS tp_Single,
tbc.Double,
tpc.Double AS tp_Double,
tbc.Triple ,
tpc.Triple AS tp_Triple,
tbc.atBat,
tpc.atBat AS tp_atBat,
tbc.Hit ,
tpc.Hit As tp_Hit,
tbc.plateApperance  ,
tpc.plateApperance As tp_plateApperance,
tbc.Hit_By_Pitch ,
tpc.Hit_By_Pitch As tp_Hit_By_Pitch,
tbc.Home_Run ,
tpc.Home_Run As tp_Home_Run,
tbc.Strikeout ,
tpc.Strikeout AS tp_Strikeout,
tbc.Walk ,
tpc.Walk As tp_Walk,
tbc.Ground_Out,
tpc.Ground_Out AS tp_GB,
tbc.Fly_Out,
tpc.Fly_Out AS tp_FB,
tbc.Sac_Fly ,
tpc.Sac_Fly AS tp_Sac_Fly,
pc.startingPitcher,
SUM(pc.startingInning)AS startingInnings,
SUM(pc.endingInning) AS endingingInnings,
b.away_runs,
b.winner_home_or_away
FROM team_batting_counts tbc
JOIN pitcher_counts pc ON tbc.game_id = pc.game_id AND tbc.team_id  = pc.team_id
JOIN team_pitching_counts tpc ON pc.game_id = tpc.game_id AND tpc.team_id = pc.team_id
JOIN boxscore b ON b.game_id = pc.game_id
GROUP BY pc.team_id , pc.game_id
ORDER BY pc.team_id , pc.game_id ;

UPDATE temp t
SET t.winner_home_or_away =
  CASE winner_home_or_away
    WHEN 'H' THEN 1
    WHEN 'A' THEN 0
  END;


SELECT * FROM temp t;


CREATE OR REPLACE TEMPORARY TABLE temp_features AS
SELECT t1.game_id,
t1.team_id,
t1.opponent_team_id,
t1.winner_home_or_away ,
COALESCE(SUM(t2.Hit), NULL) AS Hit,
COALESCE(SUM(t2.tp_Hit), NULL) AS tp_Hit,
COALESCE(SUM(t2.atBat), NULL) AS atBat,
COALESCE(SUM(t2.tp_atBat), NULL) AS tp_atBat,
COALESCE(SUM(t2.Home_Run), NULL) AS Home_Run,
IF(SUM(t2.tp_Home_Run) = 0, NULL, SUM(t2.tp_Home_Run)) AS tp_Home_Run,
COALESCE(SUM(t2.Strikeout), NULL) AS Strikeout,
COALESCE(SUM(t2.tp_Strikeout), NULL) AS tp_Strikeout,
COALESCE(SUM(t2.Single),0) AS B,
COALESCE(SUM(t2.tp_Single),0) AS tp_B,
COALESCE(SUM(t2.Double),0) AS 2B,
COALESCE(SUM(t2.tp_Double),0) AS tp_2B,
COALESCE(SUM(t2.Triple),0) AS 3B,
COALESCE(SUM(t2.tp_Triple),0) AS tp_3B,
COALESCE(SUM(t2.Single+2*t2.Double+3*t2.Triple+4*t2.Home_Run),0) AS TB,
COALESCE(SUM(t2.tp_Single+2*t2.tp_Double+3*t2.tp_Triple+4*t2.Home_Run),0) AS tp_TB,
COALESCE(AVG(t2.away_runs)) as avg_away_runs,
COALESCE(t2.endingingInnings - t2.startingInnings) AS Innings_pitched,
COALESCE(SUM(t2.Walk),0) AS BB,
COALESCE(SUM(t2.tp_Walk),0) AS tp_BB,
COALESCE(SUM(t2.Fly_Out),0) AS FB,
COALESCE(SUM(t2.tp_FB),0) AS tp_FB,
COALESCE(SUM(t2.Ground_Out),0) AS GB,
COALESCE(SUM(t2.tp_GB),0) AS tp_GB
FROM temp t1
JOIN game g1 ON t1.game_id = g1.game_id
JOIN temp t2 ON t1.team_id = t2.team_id
JOIN game g2 ON t2.game_id = g2.game_id AND  g1.local_date > g2.local_date
AND g2.local_date > g1.local_date - INTERVAL 100 DAY
GROUP BY t1.team_id, t1.game_id, g1.local_date
ORDER BY t1.team_id,g1.local_date;



SELECT * FROM temp_features tf;

CREATE OR REPLACE TEMPORARY TABLE  features AS
SELECT tf.game_id,
tf.team_id,
tf.opponent_team_id,
ROUND(tf.Hit / NULLIF(tf.atBat, 0) / (tf.tp_Hit / NULLIF(tf.tp_atBat, 0)), 2) AS Batting_Average_Ratio,
ROUND(tf.Home_Run / NULLIF(tf.Hit, 0) / (tf.tp_Home_Run / NULLIF(tf.tp_hit, 0)), 2) AS Home_runs_per_hit_Ratio,
ROUND(tf.Hit / NULLIF(tf.Strikeout, 0) / (tf.tp_Hit / NULLIF(tf.tp_Strikeout, 0)), 2) AS Hit_per_Strikeout_Ratio,
ROUND(tf.TB / NULLIF(tf.atBat, 0) / (tf.tp_TB / NULLIF(tf.tp_atBat, 0)), 2) AS SLG_Ratio,
ROUND(tf.TB / NULLIF(tf.tp_TB, 0), 2) AS TB_Ratio,
ROUND(tf.Strikeout / NULLIF(tf.BB, 0) / (tf.tp_Strikeout / NULLIF(tf.tp_BB, 0)), 2) AS Strikeout_to_walk_Ratio,
ROUND(tf.GB / NULLIF(tf.FB, 0) / (tf.tp_GB / NULLIF(tf.tp_FB, 0)), 2) AS Groundout_to_Airout_Ratio,
ROUND(9*(tf.BB / NULLIF(tf.Innings_pitched, 0)), 2) AS BB9,
ROUND((tf.tp_Home_Run + tf.tp_BB)  / NULLIF(tf.Innings_pitched, 0), 2) AS WHIP,
ROUND(9 * (tf.avg_away_runs) / NULLIF (tf.Innings_pitched,0), 2) AS ERA,
tf.winner_home_or_away AS response
FROM temp_features tf;

SELECT * FROM features;