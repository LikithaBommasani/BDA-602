import sys

import pandas
import sqlalchemy


def main():
    db_user = "root"
    db_pass = "newrootpassword"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = (
        f"mariadb+mariadbconnector://" f"{db_user}:{db_pass}@{db_host}/{db_database}"
    )  # pragma: allowlist secret

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
             SELECT * FROM features_ratio; """
    df = pandas.read_sql_query(query, sql_engine)
    print(df.head())
    R_Response = 'HomeTeamWins'
    ignore_columns = ['game_id', 'home_team_id', 'away_team_id']
    P_Predictors = [x for x in df.columns if x != R_Response and x not in ignore_columns]
    print(R_Response)
    print(P_Predictors)


if __name__ == "__main__":
    sys.exit(main())
