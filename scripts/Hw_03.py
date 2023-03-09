import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.ml import Transformer
from pyspark import keyword_only, StorageLevel
from pyspark.ml.pipeline import Transformer
from pyspark.ml.param.shared import HasInputCol, HasOutputCol, Param


class RollingAverageTransform(Transformer, HasInputCol, HasOutputCol):
    def __init__(self, window_size=5, inputCol="value", outputCol="rolling_avg", **kwargs):
        super(RollingAverageTransform, self).__init__()
        self.window_size = window_size
        self.setInputCol(inputCol)
        self.setOutputCol(outputCol)

    def _transform(self, data, spark):
        rolling_avg = rolling_average_calculation(spark, data)
        return rolling_avg


def intermediate_data():
    game_sql = """ \
            SELECT game_id, \
            local_date \
            FROM game"""

    battercounts_sql = """ \
            SELECT game.game_id, bc.batter AS Batter, bc.Hit, bc.atBat, game.local_date
            FROM batter_counts AS bc
            INNER JOIN game ON bc.game_id = game.game_id"""

    game = load_data(game_sql)  # getting the game table data
    batter_counts = load_data(battercounts_sql)  # Loading the batter_counts data

    intermediate_df = batter_counts.join(game, on="game_id")  # joining the tables

    return intermediate_df


def rolling_average_calculation(spark, data):
    batting_avg_rolling_sql = """ \
        SELECT bart1.batter
            , (CASE WHEN SUM(bart2.atBat) > 0 THEN SUM(bart2.Hit) / SUM(bart2.atBat) ELSE 0 END) AS Batting_Avg
            , bart1.game_id
            , DATE(bart1.local_date) AS local_date
            , DATE_SUB(bart1.local_date, INTERVAL 100 DAY) AS Date_since
        FROM batter_counts bart1
            INNER JOIN batter_counts bart2 ON bart1.Batter = bart2.Batter
                AND bart2.local_date < bart1.local_date
                AND bart2.local_date > DATE_SUB(bart1.local_date, INTERVAL 100 DAY)
        GROUP BY bart1.Batter, DATE(bart1.local_date), bart1.game_id
        ORDER BY bart1.Batter
    """

    data.createOrReplaceTempView("batter_counts")
    data.persist(storageLevel=StorageLevel.DISK_ONLY)
    rolling_avg_data = spark.sql(batting_avg_rolling_sql)

    return rolling_avg_data


def load_data(query):
    appName = "baseball"
    master = "local"
    # Create Spark session
    spark = SparkSession.builder \
        .appName(appName) \
        .master(master) \
        .getOrCreate()
    database = "baseball"
    user = "root"
    password = "newrootpassword"
    server = "localhost"
    port = 3306
    jdbc_url = f"jdbc:mysql://{server}:{port}/{database}?permitMysqlScheme"
    jdbc_driver = "org.mariadb.jdbc.Driver"

    # Create a data frame by reading data from MySQL via JDBC
    data_load: DataFrame = (
        spark.read.format("jdbc") \
            .option("url", jdbc_url) \
            .option("query", query) \
            .option("user", user) \
            .option("password", password) \
            .option("driver", jdbc_driver) \
            .load()
    )
    return data_load


appName = "baseball"
master = "local"


def main():
    intermediate_df = intermediate_data()
    spark = SparkSession.builder \
        .appName(appName) \
        .master(master) \
        .getOrCreate()
    obj = RollingAverageTransform(spark=spark)
    result = obj._transform(intermediate_df)
    data_load = load_data()
    data_load.show()


if __name__ == "__main__":
    sys.exit(main())
