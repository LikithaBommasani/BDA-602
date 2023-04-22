Use baseball;

CREATE OR REPLACE TABLE home_pitcher_stats as
SELECT
      g.local_date
    , pc.game_id
    , pc.team_id
    , pc.pitcher
    , pc.outsPlayed
    , pc.atBat
    , pc.Hit
    , pc.toBase
    , pc.Triple
    , pc.Double
    , pc.Flyout
    , pc.Forceout
    , pc.Groundout
    , pc.Hit_By_Pitch
    , pc.Home_Run
    , pc.Pop_Out
    , pc.Single
    , pc.Strikeout
    , pc.Walk
FROM pitcher_counts pc
    JOIN game g ON pc.game_id = g.game_id
WHERE homeTeam = 1;

SELECT * FROM home_pitcher_stats hp;

CREATE OR REPLACE TABLE away_pitcher_stats as
SELECT
      g.local_date
    , pc.game_id
    , pc.team_id
    , pc.pitcher
    , pc.outsPlayed
    , pc.atBat
    , pc.Hit
    , pc.toBase
    , pc.Triple
    , pc.Double
    , pc.Flyout
    , pc.Forceout
    , pc.Groundout
    , pc.Hit_By_Pitch
    , pc.Home_Run
    , pc.Pop_Out
    , pc.Single
    , pc.Strikeout
    , pc.Walk
FROM pitcher_counts pc
    JOIN game g ON pc.game_id = g.game_id
WHERE homeTeam = 0;

SELECT * FROM away_pitcher_stats ap;


CREATE INDEX idx_home_pitcher_stats_game_id ON home_pitcher_stats(game_id);
CREATE INDEX idx_away_pitcher_stats_game_id ON away_pitcher_stats(game_id);


CREATE OR REPLACE TABLE all_pitcher_stats AS
SELECT
      hp.game_id AS game_id
    , hp.team_id AS home_team_id
    , ap.team_id AS away_team_id
    , hp.local_date AS local_date
    , hp.pitcher AS home_pitcher
    , ap.pitcher AS away_pitcher
    , hp.outsPlayed AS home_outsPlayed
    , ap.outsPlayed AS away_outsPlayed
    , hp.atBat AS home_atBat
    , ap.atBat AS away_atBat
    , hp.Hit AS home_Hit
    , ap.Hit AS away_Hit
    , hp.toBase AS home_toBase
    , ap.toBase AS away_toBase
    , hp.Triple AS home_Triple
    , ap.Triple AS away_Triple
    , hp.Double AS home_Double
    , ap.Double AS away_Double
    , hp.Flyout AS home_Flyout
    , ap.Flyout AS away_Flyout
    , hp.Forceout AS home_Forceout
    , ap.Forceout AS away_Forceout
    , hp.Groundout AS home_Groundout
    , ap.Groundout AS away_Groundout
    , hp.Hit_By_Pitch AS home_Hit_By_Pitch
    , ap.Hit_By_Pitch AS away_Hit_By_Pitch
    , hp.Home_Run AS home_Home_Run
    , ap.Home_Run AS away_Home_Run
    , hp.Pop_Out AS home_Pop_Out
    , ap.Pop_Out AS away_Pop_Out
    , hp.Single AS home_Single
    , ap.Single AS away_Single
    , hp.Strikeout AS home_Strikeout
    , ap.Strikeout AS away_Strikeout
    , hp.Walk AS home_Walk
    , ap.Walk AS away_Walk
    , b.winner_home_or_away
    , b.home_runs
    , b.away_runs
FROM home_pitcher_stats hp
    JOIN away_pitcher_stats ap ON hp.game_id = ap.game_id
    JOIN boxscore b ON hp.game_id = b.game_id;


CREATE INDEX idx_all_pitcher_stats_game_id ON all_pitcher_stats(game_id);

SELECT * FROM all_pitcher_stats ps;


CREATE OR REPLACE TEMPORARY TABLE feature_sum AS
SELECT
      ps1.game_id
    , ps1.home_team_id
    , ps1.away_team_id
    , ps1.winner_home_or_away AS HomeTeamWins
    , COALESCE(SUM(ps1.home_Hit), NULL) AS home_Hit
    , COALESCE(SUM(ps1.away_Hit), NULL) AS away_Hit
    , COALESCE(SUM(ps1.home_atBat), NULL) AS home_atBat
    , COALESCE(SUM(ps1.away_atBat), NULL) AS away_atBat
    , COALESCE(SUM(ps1.home_runs), NULL) AS home_runs
    , COALESCE(SUM(ps1.away_runs), NULL) AS away_runs
    , COALESCE(SUM(ps1.home_Strikeout), NULL) AS home_Strikeout
    , COALESCE(SUM(ps1.away_Strikeout), NULL) AS away_Strikeout
    , COALESCE(SUM(ps1.home_Walk), NULL) AS home_Walk
    , COALESCE(SUM(ps1.away_Walk), NULL) AS away_Walk
    , COALESCE(SUM(ps1.home_Groundout), NULL) AS home_Groundout
    , COALESCE(SUM(ps1.away_Groundout), NULL) AS away_Groundout
    , COALESCE(SUM(ps1.home_Flyout), NULL) AS home_Flyout
    , COALESCE(SUM(ps1.away_Flyout), NULL) AS away_Flyout
    , COALESCE(SUM(ps1.home_outsPlayed), NULL) AS home_outsPlayed
    , COALESCE(SUM(ps1.away_outsPlayed), NULL) AS away_outsPlayed
    , COALESCE(SUM(ps1.home_Single + 2 * ps1.home_Double + 3 * ps1.home_Triple + 4 * ps1.home_runs),0) AS home_TB
    , COALESCE(SUM(ps1.away_Single + 2 * ps1.away_Double + 3 * ps1.away_Triple + 4 * ps1.away_runs),0) AS away_TB


FROM all_pitcher_stats ps1
GROUP BY ps1.home_team_id, ps1.away_team_id;

UPDATE feature_sum fs
SET fs.HomeTeamWins =
        CASE HomeTeamWins
          WHEN 'H' THEN 1
          WHEN 'A' THEN 0
        END
;

SELECT * FROM feature_sum fs;


CREATE OR REPLACE TABLE features_ratio AS
SELECT
    fs.game_id
    , fs.home_team_id
    , fs.away_team_id
    , ROUND(fs.home_Hit / NULLIF(fs.home_atBat, 0) / NULLIF(fs.away_Hit / NULLIF(fs.away_atBat, 0), 0), 2) AS Batting_Average_Ratio
    , ROUND(fs.home_runs / NULLIF(fs.home_Hit, 0) / NULLIF(fs.away_runs / NULLIF(fs.away_Hit, 0), 0), 2) AS Runs_per_hit_Ratio
    , ROUND(fs.home_Hit / NULLIF(fs.home_Strikeout, 0) / NULLIF(fs.away_Hit / NULLIF(fs.away_Strikeout, 0), 0), 2) AS Hit_per_Strikeout_Ratio
    , ROUND(fs.home_Strikeout / NULLIF(fs.home_Walk, 0) / NULLIF(fs.away_Strikeout / NULLIF(fs.away_Walk, 0), 0), 2) AS Strikeout_to_walk_Ratio
    , ROUND(fs.home_Groundout / NULLIF(fs.home_Flyout, 0) / NULLIF(fs.away_Groundout / NULLIF(fs.away_Walk, 0), 0), 2) AS Groundout_to_Flyout_Ratio
    , ROUND(fs.home_Walk / NULLIF(fs.home_atBat, 0) / NULLIF(fs.away_Walk / NULLIF(fs.away_atBat, 0), 0), 2) AS Walks_per_atBat_Ratio
    , ROUND(fs.home_Strikeout / NULLIF(fs.home_atBat, 0) / NULLIF(fs.away_Strikeout / NULLIF(fs.away_atBat, 0), 0), 2) AS Strikeout_per_atBat_Ratio
    , ROUND(fs.home_Strikeout / NULLIF(fs.home_outsPlayed, 0) / NULLIF(fs.away_Strikeout / NULLIF(fs.away_outsPlayed, 0), 0), 2) AS Strikeout_per_outsPlayed_Ratio
    , ROUND(fs.home_Hit / NULLIF(fs.home_outsPlayed, 0) / NULLIF(fs.away_Hit / NULLIF(fs.away_outsPlayed, 0), 0), 2) AS Hits_per_outsPlayed_Ratio
    , ROUND((fs.home_Walk + fs.home_Hit) / NULLIF(fs.home_outsPlayed, 0), 2) / ROUND(NULLIF((fs.away_Walk + fs.away_Hit) / NULLIF(fs.away_outsPlayed, 0), 0), 2) AS WHO_Ratio
    , ROUND(fs.home_TB / NULLIF(fs.away_TB, 0), 2) AS TB_Ratio
    , ROUND(fs.home_TB / NULLIF(fs.home_atBat, 0) / NULLIF(fs.away_TB / NULLIF(fs.away_atBat, 0), 0), 2) AS SLG_Ratio
    , fs.HomeTeamWins
FROM feature_sum fs
;
SELECT * FROM features_ratio;
