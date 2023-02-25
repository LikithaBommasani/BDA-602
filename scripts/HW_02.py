import sys

import pandas
import sqlalchemy


def main():
    db_user = "root"
    db_pass = "newrootpassword"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    # pragma: allowlist secret
    connect_string = (
        f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"
    )

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """DROP TABLE IF EXISTS batter_avg_historical;

CREATE TABLE batter_avg_historical AS
SELECT batter AS Batter,
    SUM(hit) AS Hit,
    SUM(atBat) AS atBat,
    (CASE WHEN SUM(atBat) > 0 THEN SUM(hit)/SUM(atBat) ELSE 0 END) AS Batting_Avg
FROM batter_counts
GROUP BY Batter;

SELECT * FROM batter_avg_historical;
SELECT COUNT(*) FROM batter_avg_historical;





DROP TABLE IF EXISTS batter_avg_annual;

CREATE TABLE batter_avg_annual AS
SELECT batter AS Batter,
      YEAR(game.local_date) AS For_Year,
       (CASE WHEN SUM(bc.atBat) > 0 THEN SUM(bc.Hit)/SUM(bc.atBat) ELSE 0 END) AS Batting_Avg
FROM batter_counts AS bc
INNER JOIN game ON bc.game_id = game.game_id
GROUP BY Batter, For_Year
ORDER BY Batter, For_Year;


SELECT * FROM batter_avg_annual;
SELECT COUNT(*) FROM batter_avg_annual;
    """
    df = pandas.read_sql_query(query, sql_engine)
    print(df.head())


if __name__ == "__main__":
    sys.exit(main())
