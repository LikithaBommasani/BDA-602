#!/bin/sh

# Wait for MariaDB container to be ready
echo "Waiting for MariaDB container to be ready..."
while ! mysqladmin ping -h localhost -u root -pnewrootpassword --silent; do
    echo "MariaDB is unavailable - sleeping..."
    sleep 15
done
echo "MariaDB is up and running!"

# Define database credentials
#DB_HOST="localhost"
DB_HOST="mariadb"
DB_USER="root"
DB_PASSWORD="newrootpassword"  # pragma: allowlist secret
DB_NAME="baseball"

# Check if database exists, create it if not and load it with data
if ! mysql -h $DB_HOST -u $DB_USER -p$DB_PASSWORD -e "use $DB_NAME"; then
  echo "Creating baseball database..."
  mysql -h $DB_HOST -u $DB_USER -p$DB_PASSWORD -e "create database $DB_NAME;"
  echo "Loading baseball database with data..."
  mysql -u $DB_USER -p$DB_PASSWORD -h $DB_HOST --database=$DB_NAME < baseball.sql
else
  echo "Baseball database already exists."
fi

# Import features
echo "Importing features from /app/baseball_features.sql ..."
mysql -h $DB_HOST -u $DB_USER -p$DB_PASSWORD $DB_NAME < baseball_features.sql
echo "Features imported successfully."

# RUN python files

echo "run python"
python Hw_06.py
echo "Done Running python code !!!"