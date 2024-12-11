# py_api/json_handler.py

from logger import logger  # Use relative import
import tolerantjson as tjson
import re
from typing import Tuple, List, Dict
import uuid  # Import uuid for generating unique IDs if needed

def parse_extraction_result(result_text) -> Tuple[List[Dict], List[Dict], Dict]:
    try:
        # Use regex to extract JSON content between ```json and ```
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', result_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            logger.debug("Extracted JSON from code block.")
        else:
            # If no code block is found, attempt to parse the entire text
            json_str = result_text.strip()
            logger.debug("No JSON code block found. Using entire response for parsing.")

        logger.debug(f"Cleaned extraction result: {json_str}")

        # Use tolerantjson to parse the potentially malformed JSON
        data = tjson.tolerate(json_str)  # Corrected function call

        logger.debug(f"Parsed JSON data: {data}")

        # Ensure data is a dictionary
        if not isinstance(data, dict):
            logger.error("Extraction result is not a JSON object.")
            return [], [], {}

        # Extract entities and relations
        entities = data.get('entities', [])
        relations = data.get('relations', [])

        # Ensure entities is a list
        if not isinstance(entities, list):
            logger.error("Entities is not a list.")
            entities = []

        # Ensure relations is a list
        if not isinstance(relations, list):
            logger.error("Relations is not a list.")
            relations = []

        # Process entities
        all_entities = []
        for entity in entities:
            # Each entity should have 'name' and 'type'
            if isinstance(entity, dict) and 'name' in entity and 'type' in entity:
                entity_id = entity.get('id', str(uuid.uuid4()))
                entity_clean = {
                    'id': entity_id,
                    'name': entity['name'],
                    'type': entity['type'],
                    'description': entity.get('description', '')
                }
                all_entities.append(entity_clean)
            else:
                logger.error(f"Invalid entity format: {entity}")

        # Process relations
        all_relations = []
        for relation in relations:
            # Assuming relations have 'source', 'target', and 'type'
            if isinstance(relation, dict) and 'source' in relation and 'target' in relation and 'type' in relation:
                relation_id = relation.get('id', str(uuid.uuid4()))
                relation_clean = {
                    'id': relation_id,
                    'source': relation['source'],
                    'target': relation['target'],
                    'type': relation['type']
                }
                all_relations.append(relation_clean)
            else:
                logger.error(f"Invalid relation format: {relation}")

        # Return the extracted entities and relations
        return all_entities, all_relations, {}

    except Exception as e:
        logger.error(f"Error parsing JSON with tolerantjson: {e}")
        logger.error(f"Result text: {result_text}")
        return [], [], {}