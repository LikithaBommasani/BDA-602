import sys

from pyspark import StorageLevel
from pyspark.ml import Transformer
from pyspark.sql import SparkSession

appName = "baseball"
master = "local"
database = "baseball"
user = "root"  # pragma: allowlist secret
password = "newrootpassword"  # pragma: allowlist secret
server = "localhost"
port = 3306
jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
jdbc_driver = "org.mariadb.jdbc.Driver"


class RollingAverageTransform(Transformer):
    def __init__(self, sparksession):
        self.sparksession = sparksession

    def _transform(self, df):
        df.createOrReplaceTempView("batter_avg_rolling_temp")
        df.persist(StorageLevel.MEMORY_ONLY)

        rolling_avg = self.sparksession.sql(
            """
         SELECT bart1.batter
    , (CASE WHEN SUM(bart2.atBat) > 0 THEN SUM(bart2.Hit) / SUM(bart2.atBat) ELSE 0 END) AS Batting_Avg
    , bart1.game_id
    , DATE(bart1.local_date) AS local_date
    , date_sub(DATE(bart1.local_date), 100) AS Date_since
FROM batter_avg_rolling_temp bart1
    INNER JOIN batter_avg_rolling_temp bart2 ON bart1.Batter = bart2.Batter
        AND bart2.local_date < bart1.local_date
        AND bart2.local_date > date_sub(DATE(bart1.local_date), 100)
GROUP BY bart1.Batter,bart1.game_id, DATE(bart1.local_date)
ORDER BY bart1.Batter

         """
        )
        rolling_avg.createOrReplaceTempView("rolling_avg")
        rolling_avg.persist(StorageLevel.MEMORY_ONLY)
        # rolling_avg.show(10)
        return rolling_avg


def load_data(sparksession, query):
    df = (
        sparksession.read.format("jdbc")
        .option("url", jdbc_url)
        .option("query", query)
        .option("user", user)
        .option("password", password)
        .option("driver", jdbc_driver)
        .load()
    )

    return df


def main():
    sparksession = SparkSession.builder.master("local[*]").getOrCreate()
    game_sql = "SELECT * FROM game"
    battercounts_sql = "SELECT * FROM batter_counts"
    game = load_data(sparksession, game_sql)
    game.createOrReplaceTempView("game")
    game.persist(StorageLevel.MEMORY_ONLY)
    batter_count = load_data(sparksession, battercounts_sql)
    batter_count.createOrReplaceTempView("batter_counts")
    batter_count.persist(StorageLevel.MEMORY_ONLY)
    df_temp_table = sparksession.sql(
        """ SELECT game.game_id, bc.batter AS Batter, bc.Hit, bc.atBat, game.local_date
    FROM batter_counts AS bc
        INNER JOIN game ON bc.game_id = game.game_id   """
    )
    roll_t = RollingAverageTransform(sparksession)
    result = roll_t.transform(df_temp_table)
    result.createOrReplaceTempView("batter_avg_rolling")
    result.persist(StorageLevel.MEMORY_ONLY)
    sparksession.sql("""SELECT * FROM batter_avg_rolling""").show()


if __name__ == "__main__":
    sys.exit(main())
