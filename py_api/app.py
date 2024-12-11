# app.py

from datetime import datetime
import json
import threading
import pika
from flask import Flask, request, jsonify
from typing import List, Dict
import logging
from concurrent.futures import ThreadPoolExecutor
from pymongo import MongoClient
from openai import OpenAI
from neo4j import GraphDatabase
from token_count import load_encoding, count_tokens_in_messages
import uuid
import os
from json_handler import parse_extraction_result

encoding = load_encoding('gpt-4o-mini')

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MongoDB configuration
mongo_uri = os.getenv("MONGODB_URI")
mongo_client = MongoClient(mongo_uri)
database = mongo_client["researcher"]
collection = database["researcher_one"]

# Verify MongoDB connection
try:
    mongo_client.admin.command('ping')
    logger.info("Connected to MongoDB successfully.")
except Exception as e:
    logger.critical("Could not connect to MongoDB:")
    logger.exception(e)
    raise

# Neo4j configuration
neo4j_uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.getenv('NEO4J_USER', 'neo4j')
neo4j_password = os.getenv('NEO4J_PASSWORD', 'boopboop')
driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))

# RabbitMQ connection parameters
rabbitmq_host = 'localhost'
rabbitmq_port = 5672  # Adjust as necessary
rabbitmq_username = os.getenv('RABBIT_USER')
rabbitmq_password = os.getenv('RABBIT_PASSWORD')

credentials = pika.PlainCredentials(rabbitmq_username, rabbitmq_password)
connection = pika.BlockingConnection(pika.ConnectionParameters(
    host=rabbitmq_host,
    port=rabbitmq_port,
    credentials=credentials
))
channel = connection.channel()

# Declare queues
channel.queue_declare(queue='summarize_queue', durable=True)
channel.queue_declare(queue='entity_extraction_queue', durable=True)

# Flask app
app = Flask(__name__)

# Configuration
MAX_TOKENS = 4000  # Adjust as necessary
TEMPERATURE = 0.5
NUM_THREADS = 4  # Number of threads for parallel requests

# Thread pool for multithreading
executor = ThreadPoolExecutor(max_workers=NUM_THREADS)


def on_request_summarize(ch, method, props, body):
    data = json.loads(body)
    text = data.get('text', '')
    logger.info("Received summarization request.")

    # Process summarization
    summary = summarize_task(text)

    response = summary

    # Send response back
    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties(
            correlation_id=props.correlation_id
        ),
        body=json.dumps(response)
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)

def on_request_extract_entities(ch, method, props, body):
    data = json.loads(body)
    text = data.get('text', '')
    logger.info("Received entity extraction request.")

    # Process entity extraction
    extraction_result = extract_entities_and_relations_task(text)

    response = extraction_result

    # Send response back
    ch.basic_publish(
        exchange='',
        routing_key=props.reply_to,
        properties=pika.BasicProperties(
            correlation_id=props.correlation_id
        ),
        body=json.dumps(response)
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)

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

        # Count tokens
        num_tokens = count_tokens_in_messages(messages, 'gpt-4o-mini')
        max_allowed_tokens = 8192  # Maximum tokens for gpt-4o-mini
        tokens_available = max_allowed_tokens - num_tokens

        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=messages,
            max_tokens=8192,  # Adjust max_tokens as needed
            temperature=0.6,
        )

        summary = response.choices[0].message.content.strip()
        logger.info("Generated summary.")

        return summary

    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        logger.exception(e)
        return {'error': 'Summarization failed'}

def split_text_into_chunks(text, max_tokens_per_chunk):
    enc = load_encoding('gpt-4o-mini')
    tokens = enc.encode(text)
    chunks = []
    start = 0

    while start < len(tokens):
        end = start + max_tokens_per_chunk
        chunk_tokens = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens)
        chunks.append(chunk_text)
        start = end

    return chunks

def summarize_combined_text(text):
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

    response = client.chat.completions.create(
        model='gpt-4o-mini',  # Adjust with the correct model name
        messages=messages,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
    )

    final_summary = response.choices[0].message.content.strip()
    logger.info(f"Generated final summary: {final_summary}")
    return final_summary

def save_summary_to_neo4j(original_text, summary):
    try:
        with driver.session() as session:
            session.write_transaction(create_summary_node, original_text, summary)
        logger.info("Summary saved successfully to Neo4j.")
    except Exception as e:
        logger.error("Error saving summary to Neo4j:")
        logger.exception(e)

def create_summary_node(tx, original_text, summary):
    try:
        tx.run(
            """
            MERGE (t:Text {content: $original_text})
            MERGE (s:Summary {content: $summary})
            MERGE (t)-[:HAS_SUMMARY]->(s)
            """,
            original_text=original_text,
            summary=summary,
        )
    except Exception as e:
        logger.error("Error during Neo4j transaction:")
        logger.exception(e)

@app.route('/extract_entities_and_relations', methods=['POST'])
def extract_entities_and_relations():
    data = request.get_json()
    text = data.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400

    logger.info("Received request to extract entities and relations.")

    # Submit extraction task to the thread pool
    future = executor.submit(extract_entities_and_relations_task, text)
    result = future.result()

    if isinstance(result, dict) and 'error' in result:
        return jsonify(result), 500
    return jsonify(result)


def save_entities_and_relations(entities, relations, mongo_id):
    """
    Saves entities and relations to Neo4j, linking them to a MongoDB document.
    """
    try:
        with driver.session() as session:
            # Save entities
            for entity in entities:
                try:
                    if 'name' in entity and 'type' in entity and 'id' in entity:
                        session.write_transaction(
                            create_entity_node,
                            name=entity['name'],
                            entity_type=entity['type'],
                            entity_id=entity['id'],
                            mongo_id=mongo_id
                        )
                        logger.info(f"Entity saved: {entity['name']} ({entity['id']})")
                    else:
                        logger.error(f"Entity missing 'name', 'type', or 'id': {entity}")
                except Exception as e:
                    logger.error(f"Failed to save entity: {entity}")
                    logger.exception(e)
                    # Continue with next entity

            # Save relations
            for relation in relations:
                try:
                    session.write_transaction(create_relation, relation, mongo_id)
                    logger.info(f"Relation saved: {relation['source']} -> {relation['target']} ({relation['type']})")
                except Exception as e:
                    logger.error(f"Failed to save relation: {relation}")
                    logger.exception(e)
                    # Continue with next relation

        logger.info("Entities and relations saved successfully to Neo4j.")
    except Exception as e:
        logger.error("Error saving entities and relations to Neo4j:")
        logger.exception(e)


def create_entity_node(tx, name, entity_type, entity_id, mongo_id):
    try:
        tx.run(
            """
            MERGE (e:Entity {id: $id})
            ON CREATE SET
                e.name = $name,
                e.type = $type,
                e.createdAt = datetime(),
                e.mongo_id = $mongo_id
            ON MATCH SET
                e.updatedAt = datetime()
            """,
            id=entity_id,
            name=name,
            type=entity_type,
            mongo_id=mongo_id
        )
    except Exception as e:
        logger.error(f"Error creating entity node for {name}: {e}")
        raise



def create_relation(tx, relation, mongo_id):
    try:
        query = f"""
        MATCH (source:Entity {{id: $source_id}})
        MATCH (target:Entity {{id: $target_id}})
        MERGE (source)-[r:{relation['type']} {{id: $id}}]->(target)
        ON CREATE SET
            r.createdAt = datetime(),
            r.mongo_id = $mongo_id
        ON MATCH SET
            r.updatedAt = datetime()
        """
        tx.run(
            query,
            source_id=relation['source_id'],
            target_id=relation['target_id'],
            id=relation['id'],
            mongo_id=mongo_id
        )
    except Exception as e:
        logger.error(f"Error creating relation {relation['type']} between {relation['source']} and {relation['target']}: {e}")
        raise



def save_entities_to_mongo(data: Dict):
    try:
        logger.debug(f"Preparing to save data to MongoDB: {data}")

        required_fields = ['original_text', 'summary', 'entities', 'relations']
        for field in required_fields:
            if field not in data:
                logger.error(f"Missing required field: {field}")
                return

        # Insert the document
        result = collection.insert_one(data)
        mongo_id = str(result.inserted_id)
        logger.info(f"Inserted document with _id: {mongo_id} into MongoDB.")

    except Exception as e:
        logger.error("Error saving data to MongoDB:")
        logger.exception(e)

def save_entities_to_neo4j(entities: List[Dict], mongo_id: str):
    """
    Saves a list of entities to Neo4j, linking them to a MongoDB document.
    """
    try:
        with driver.session() as session:
            for entity in entities:
                if 'name' in entity and 'type' in entity:
                    session.write_transaction(create_entity_node, entity['name'], entity['type'], mongo_id)
                else:
                    logger.error(f"Entity missing 'name' or 'type': {entity}")
        logger.info("Entities saved successfully to Neo4j.")
    except Exception as e:
        logger.error("Error saving entities to Neo4j:")
        logger.exception(e)

def start_consuming():
    channel.basic_qos(prefetch_count=1)

    channel.basic_consume(
        queue='summarize_queue',
        on_message_callback=on_request_summarize,
    )

    channel.basic_consume(
        queue='entity_extraction_queue',
        on_message_callback=on_request_extract_entities,
    )

    logger.info("Awaiting RPC requests.")
    channel.start_consuming()

def save_data(original_text: str, summary: str, entities: List[Dict], relations: List[Dict], response_data: Dict):
    """
    Saves comprehensive data to both MongoDB and Neo4j.

    Args:
        original_text (str): The original text being processed.
        summary (str): The summary of the original text.
        entities (List[Dict]): Extracted entities.
        relations (List[Dict]): Extracted relations.
        response_data (Dict): Additional response data from OpenAI.
    """
    try:
        # Prepare the data dictionary
        data = {
            'original_text': original_text,
            'summary': summary,
            'entities': entities,
            'relations': relations,
            'response_data': response_data,
            'timestamp': datetime.utcnow()
        }

        # Insert into MongoDB
        logger.info("Attempting to insert document into MongoDB.")
        mongo_result = collection.insert_one(data)
        mongo_id = str(mongo_result.inserted_id)
        logger.info(f"Inserted document with _id: {mongo_id} into MongoDB.")

        # Save entities and relations to Neo4j with MongoDB _id
        save_entities_and_relations(entities, relations, mongo_id)

    except Exception as e:
        logger.error("Error saving comprehensive data:")
        logger.exception(e)


def extract_entities_task(user_prompt_text):
    try:
        text = user_prompt_text
        content = f'Extract entities from the following text:\n{text}\n\nProvide the entities in JSON format.'

        # OpenAI API call
        response = client.chat.completions.create(
            model='gpt-4o-mini',  # Ensure this is the correct model name
            messages=[{
                'role': 'user',
                'content': content
            }],
            max_tokens=4000,
            temperature=0.5,
        )

        entities_json = response.choices[0].message.content.strip()
        logger.debug(f"Entities JSON Response: {entities_json}")
        logger.info(f"Entities extracted: {entities_json}")

        # Parse the extraction result
        entities, _, _ = parse_extraction_result(entities_json)
        logger.debug(f"Parsed entities: {entities}")

        # Save entities to MongoDB and Neo4j
        save_data(
            original_text=text,
            summary='Entity extraction completed.',
            entities=entities,
            relations=[],  # No relations in this task
            response_data={'extraction_response': entities_json}
        )

        return entities

    except Exception as e:
        logger.error(f"Error during entity extraction: {e}")
        logger.exception(e)
        return {'error': 'Entity extraction failed'}


def extract_entities_and_relations_task(text):
    try:
        # Define maximum number of tokens per chunk based on model capacity
        max_tokens_per_chunk = 1500  # Adjust based on model's max tokens

        # Split text into chunks
        chunks = split_text_into_chunks(text, max_tokens_per_chunk)
        logger.info(f"Text split into {len(chunks)} chunks for entity and relation extraction.")

        all_entities = []
        all_relations = []
        name_to_id = {}

        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx + 1}/{len(chunks)} for extraction.")

            messages = [
                {
                    'role': 'system',
                    'content': 'You are an AI assistant that extracts entities and relationships from text and returns them in JSON format.',
                },
                {
                    'role': 'user',
                    'content': f'''
Extract entities and relationships from the following text:

"{chunk}"

Format the response as a JSON object with "entities" and "relations" keys.

Each item in the "entities" array should be an object with the following keys:

- "name": the name of the entity
- "description": a brief description of the entity
- "type": type of the entity, e.g., Person, Location, Organization

Each item in the "relations" array should be an object with the following keys:

- "source": the "name" of the source entity
- "target": the "name" of the target entity
- "type": the type of relationship

Ensure that the JSON is strictly valid: use double quotes for strings, no trailing commas, and no comments.
Do not include any markdown formatting or code block delimiters.
Provide only the JSON object without additional text.
''',
                },
            ]

            response = client.chat.completions.create(
                model='gpt-4o-mini',  # Adjust the model if needed
                messages=messages,
                max_tokens=8192,
                temperature=0.3,
            )

            extraction_result = response.choices[0].message.content.strip()
            logger.debug(f'Extraction result: {extraction_result}')

            # Parse the extraction result
            entities, relations, _ = parse_extraction_result(extraction_result)

            # Process entities
            for entity in entities:
                name = entity['name']
                if name not in name_to_id:
                    entity_id = entity['id']
                    name_to_id[name] = entity_id
                    all_entities.append(entity)
                else:
                    # Optional: update the existing entity if needed
                    pass

            # Process relations
            for relation in relations:
                source_name = relation['source']
                target_name = relation['target']

                # Check and add source entity if missing
                if source_name not in name_to_id:
                    entity_id = str(uuid.uuid4())
                    new_entity = {
                        'id': entity_id,
                        'name': source_name,
                        'type': 'Unknown',
                        'description': ''
                    }
                    all_entities.append(new_entity)
                    name_to_id[source_name] = entity_id

                # Check and add target entity if missing
                if target_name not in name_to_id:
                    entity_id = str(uuid.uuid4())
                    new_entity = {
                        'id': entity_id,
                        'name': target_name,
                        'type': 'Unknown',
                        'description': ''
                    }
                    all_entities.append(new_entity)
                    name_to_id[target_name] = entity_id

                # Assign IDs
                relation['source_id'] = name_to_id[source_name]
                relation['target_id'] = name_to_id[target_name]
                relation['id'] = relation.get('id', str(uuid.uuid4()))
                all_relations.append(relation)

        # After processing all chunks
        response_data = {
            'extraction_details': 'Extraction completed successfully.'
        }

        # Save comprehensive data
        save_data(
            original_text=text,
            summary='Entity and relation extraction completed.',
            entities=all_entities,
            relations=all_relations,
            response_data=response_data
        )

        return {'entities': all_entities, 'relations': all_relations, 'response_data': response_data}

    except Exception as e:
        logger.error(f"Error during entity and relation extraction: {e}")
        logger.exception(e)
        return {'error': 'Entity and relation extraction failed'}


if __name__ == '__main__':
    try:
        # Start RabbitMQ consumers in a separate daemon thread
        threading.Thread(target=start_consuming, daemon=True).start()

        logger.info("Starting Flask app...")
        app.run(host='0.0.0.0', port=6660, threaded=True)
    except Exception as e:
        logger.critical(f"Application failed to start: {e}")