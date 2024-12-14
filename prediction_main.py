import psycopg2

# AWS RDS connection details
DB_HOST = 'coustomer-database.cdwoe68s6i9h.us-east-1.rds.amazonaws.com'  # e.g., 'your-db-instance.xxxxxxx.us-east-1.rds.amazonaws.com'
DB_USER = 'postgres'          # Replace with your PostgreSQL username
DB_PASSWORD = 'l2UfwUm1yUVusJVDEQVA '      # Replace with your PostgreSQL password
DB_PORT = '5432'                   # Default PostgreSQL port
DB_NAME = 'customer_churn_data'          # The target database name

def list_tables():
    try:
        # Connect to the specified database
        connection = psycopg2.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT,
            dbname=DB_NAME
        )
        
        # Create a cursor object
        cursor = connection.cursor()
        
        # Execute the query to list all tables
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public';
        """)
        
        # Fetch all results
        tables = cursor.fetchall()
        
        # Print the list of tables
        print(f"Tables in the '{DB_NAME}' database:")
        for table in tables:
            print(table[0])
        
        # Close the cursor and connection
        cursor.close()
        connection.close()
    
    except Exception as e:
        print(f"Error: {e}")

# Call the function to list tables
list_tables()
