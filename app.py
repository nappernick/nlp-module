import os
import openai
from flask import Flask, request, jsonify
import logging
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from neo4j import GraphDatabase

# MongoDB configuration
mongo_uri = os.getenv('MONGODB_URI', 'mongodb+srv://default_uri')  # Provide a default URI
mongo_client: MongoClient = MongoClient(mongo_uri)

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "boopboop"))

# Set up OpenAI API key
openai.api_key = os.getenv('OPENAI_API_KEY')

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
database = mongo_client["researcher"]
collection = database["researcher_one"]

# Flask app
app = Flask(__name__)

# Configuration
MAX_TOKENS = 150  # Adjust as necessary
TEMPERATURE = 0.5
NUM_THREADS = 4  # Number of threads for parallel requests

# Thread pool for multithreading
executor = ThreadPoolExecutor(max_workers=NUM_THREADS)

@app.route('/summarize', methods=['POST'])
def summarize():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    logger.info("Received request to summarize.")

    # Submit summarization task to the thread pool
    future = executor.submit(summarize_task, text)
    summary = future.result()

    # Return the result within the Flask context
    if isinstance(summary, dict) and 'error' in summary:
        return jsonify(summary), 500
    return jsonify({'summary': summary})

def summarize_task(text):
    """
    Task for processing summarization using the OpenAI API.
    Returns the summary or an error dictionary.
    """
    try:
        messages = [
            {
                'role': 'system',
                'content': 'You are a helpful assistant that summarizes texts concisely.',
            },
            {
                'role': 'user',
                'content': f'Please provide a concise summary of the following text:\n\n{text}',
            },
        ]

        response = openai.chat.completions.create(
            model='gpt-4o-mini',  # Use an available model
            messages=messages,
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
        )

        summary = response.choices[0].message.content.strip()
        logger.info(f"Generated summary: {summary}")

        # Convert the OpenAI response to a dictionary
        response_data = response

        # Save the summary and response data
        save_summary(text, summary, str(response_data))

        return summary

    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return {'error': 'Summarization failed'}

def save_summary(original_text, summary, response_data):
    """
    Save the summary to both Neo4j and MongoDB.
    """
    # Save to Neo4j
    try:
        with driver.session() as session:
            session.execute_write(create_summary_node, original_text, summary)
        logger.info("Summary saved successfully to Neo4j.")
    except Exception as e:
        logger.error("Error saving summary to Neo4j:")
        logger.error(e)

    # Save to MongoDB
    try:
        document = {
            'original_text': original_text,
            'summary': summary,
            'response_data': response_data,
        }
        collection.insert_one(document)
        logger.info("Summary and response data saved successfully to MongoDB.")
    except Exception as e:
        logger.error("Error saving summary to MongoDB:")
        logger.error(e)
        
def create_summary_node(tx, original_text, summary):
    """
    Create nodes and relationships in Neo4j.
    """
    tx.run(
        """
        MERGE (t:Text {content: $original_text})
        MERGE (s:Summary {content: $summary})
        MERGE (t)-[:HAS_SUMMARY]->(s)
        """,
        original_text=original_text,
        summary=summary,
    )


if __name__ == '__main__':
    try:
        logger.info("Starting Flask app...")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    except Exception as e:
        logger.critical(f"Application failed to start: {e}")
        
        