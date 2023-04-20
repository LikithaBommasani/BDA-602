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
             SELECT * FROM features ; """
    df = pandas.read_sql_query(query, sql_engine)
    print(df.head())


if __name__ == "__main__":
    sys.exit(main())
