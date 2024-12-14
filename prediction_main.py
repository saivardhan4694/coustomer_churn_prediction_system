import pandas as pd
from sqlalchemy import create_engine

# Database connection details
host = "coustomer-database.cdwoe68s6i9h.us-east-1.rds.amazonaws.com"
port = "5432"
database = "customer_churn_data"
table_name = "customers"
user = "postgres"
password = "l2UfwUm1yUVusJVDEQVA"

# Path to the CSV file you downloaded
csv_file_path = r"C:\Users\user\Downloads\corrected_data_with_missing_values.csv"

# Create a database connection string
connection_string = f"postgresql://{user}:{password}@{host}:{port}/{database}"

# Create the database engine
engine = create_engine(connection_string)

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Upload the DataFrame to the PostgreSQL table
df.to_sql(table_name, engine, if_exists='append', index=False)

print("Data upload completed successfully.")
