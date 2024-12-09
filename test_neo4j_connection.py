from neo4j import GraphDatabase

uri = "bolt://localhost:7687"
username = "neo4j"
password = "boopboop"  # Replace with your actual password

driver = GraphDatabase.driver(uri, auth=(username, password))

def test_connection():
    try:
        with driver.session() as session:
            result = session.run("RETURN 1 AS number")
            record = result.single()
            print("Connection successful:", record["number"])
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    test_connection()