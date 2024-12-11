from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from neo4j import GraphDatabase
from thefuzz import process
import string
import os
import logging
import spacy

nlp = spacy.load("en_core_web_sm")
# Suppress the tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load models
model_name = "Babelscape/rebel-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Neo4j connection
URI = "bolt://localhost:7687"
USERNAME = os.getenv('NEO4J_USER')
PASSWORD = os.getenv('NEO4J_PASSWORD')

driver = GraphDatabase.driver(URI, auth=(USERNAME, PASSWORD))


def extract_triplets_spacy(text):
    doc = nlp(text)
    triplets = []
    for sent in doc.sents:
        subjects = [tok for tok in sent if (tok.dep_ == "nsubj" or tok.dep_ == "nsubjpass") and tok.head.dep_ == "ROOT"]
        verbs = [tok for tok in sent if tok.dep_ == "ROOT"]
        objects = [tok for tok in sent if tok.dep_ in ("dobj", "pobj") and tok.head in verbs]
        for subj in subjects:
            for verb in verbs:
                for obj in objects:
                    if verb == subj.head and (obj.head == verb or obj.head.head == verb):
                        triplets.append({
                            'subject': subj.text,
                            'relation': verb.lemma_,
                            'object': obj.text
                        })
    return triplets
          
def split_text(text, max_chunk_tokens=400, overlap_tokens=50):
    tokens = tokenizer.tokenize(text)
    total_tokens = len(tokens)
    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + max_chunk_tokens, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.convert_tokens_to_string(chunk_tokens)
        chunks.append(chunk_text)
        if end == total_tokens:
            break
        start = end - overlap_tokens  # Overlap for context

    return chunks

def normalize_entity_name(name):
    name = name.lower()
    name = name.translate(str.maketrans('', '', string.punctuation))
    name = ' '.join(name.split())  # Ensures single spaces
    return name

def find_matching_entity(name, existing_entities):
    normalized_name = normalize_entity_name(name)
    if normalized_name in existing_entities:
        return normalized_name
    else:
        # Enhanced fuzzy matching with partial ratios
        match = process.extractOne(normalized_name, existing_entities, scorer=process.fuzz.WRatio)
        if match and match[1] >= 90:  # Adjust threshold as needed
            return match[0]
        else:
            return None


def extract_triplets(text):
    triplets = []
    relation = ''
    text = text.strip()
    current = 'x'
    subject, relation, object_ = '', '', ''
    
    for token in text.replace("<s>", "").replace("<pad>", "").replace("</s>", "").split():
        if token == "<triplet>":
            current = 't'
            if relation != '':
                triplets.append({
                    'subject': subject.strip(),
                    'relation': relation.strip(),
                    'object': object_.strip()
                })
                relation = ''
            subject = ''
        elif token == "<subj>":
            current = 's'
            if relation != '':
                triplets.append({
                    'subject': subject.strip(),
                    'relation': relation.strip(),
                    'object': object_.strip()
                })
            object_ = ''
        elif token == "<obj>":
            current = 'o'
            relation = ''
        else:
            if current == 't':
                subject += ' ' + token
            elif current == 's':
                object_ += ' ' + token
            elif current == 'o':
                relation += ' ' + token
    if subject != '' and relation != '' and object_ != '':
        triplets.append({
            'subject': subject.strip(),
            'relation': relation.strip(),
            'object': object_.strip()
        })
    return triplets

def get_relation_triplets(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    output_ids = model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        max_length=512,        # Increase max_length to allow longer outputs
        num_beams=5,
        no_repeat_ngram_size=2, # Avoid repetition
        early_stopping=False,   # Let the model generate more content
    )
    decoded_output = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
    triplets = extract_triplets(decoded_output)
    return triplets



def insert_triplet(tx, subject, relation, object_, existing_entities):
    # Normalize entity names
    normalized_subject = normalize_entity_name(subject)
    normalized_object = normalize_entity_name(object_)

    # Check if the normalized entities already exist
    subject_match = find_matching_entity(normalized_subject, existing_entities)
    object_match = find_matching_entity(normalized_object, existing_entities)

    # Use existing entity names if matches are found
    if subject_match:
        subject_node = subject_match
    else:
        subject_node = subject
        # Add to existing_entities
        existing_entities.add(normalized_subject)
        # Create or merge the subject node with normalized_name
        tx.run("""
            MERGE (e:Entity {normalized_name: $normalized_name})
            ON CREATE SET e.name = $name
            """,
            name=subject_node, normalized_name=normalized_subject)

    if object_match:
        object_node = object_match
    else:
        object_node = object_
        existing_entities.add(normalized_object)
        tx.run("""
            MERGE (e:Entity {normalized_name: $normalized_name})
            ON CREATE SET e.name = $name
            """,
            name=object_node, normalized_name=normalized_object)

    # Normalize relation
    normalized_relation = relation.lower().strip()

    # Create the relationship
    tx.run("""
        MATCH (a:Entity {normalized_name: $subject_normalized})
        MATCH (b:Entity {normalized_name: $object_normalized})
        MERGE (a)-[r:RELATION {type: $relation_normalized}]->(b)
        """,
        subject_normalized=normalized_subject,
        object_normalized=normalized_object,
        relation_normalized=normalized_relation
    )

def get_existing_entities(tx):
    result = tx.run("MATCH (e:Entity) RETURN e.normalized_name AS name")
    return {record["name"] for record in result if record["name"]}

def get_texts(tx):
    result = tx.run("MATCH (e:Entity) RETURN e.content AS content")
    return [record["content"] for record in result if record["content"]]



# Fetch existing entities before processing texts
with driver.session() as session:
    existing_entities = session.execute_read(get_existing_entities)

# Ensure existing_entities is a set for faster lookups
if not isinstance(existing_entities, set):
    existing_entities = set(existing_entities)

# Fetch texts from the database
with driver.session() as session:
    texts = session.execute_read(get_texts)
    
def create_concurrent_relations(tx, entities_in_text):
    entity_list = list(entities_in_text)
    for i in range(len(entity_list)):
        for j in range(i+1, len(entity_list)):
            entity_a = entity_list[i]
            entity_b = entity_list[j]
            tx.run("""
                MATCH (a:Entity {normalized_name: $entity_a})
                MATCH (b:Entity {normalized_name: $entity_b})
                MERGE (a)-[:CO_OCCURS_WITH]->(b)
                """,
                entity_a=entity_a, entity_b=entity_b)



def get_relation_triplets_combined(text):
    # Extract triplets using the transformer model
    transformer_triplets = get_relation_triplets(text)
    # Extract triplets using SpaCy
    spacy_triplets = extract_triplets_spacy(text)
    # Combine and deduplicate triplets
    combined_triplets = transformer_triplets + spacy_triplets
    # Remove duplicates
    unique_triplets = [dict(t) for t in {tuple(sorted(d.items())) for d in combined_triplets}]
    return unique_triplets

max_input_length = tokenizer.model_max_length  # Typically 1024 tokens
max_chunk_length = max_input_length - 2  # Leave room for special tokens
overlap_length = 50  # Overlap for context

# Process each text
for idx, text in enumerate(texts):
    logging.info(f"Processing text {idx+1}/{len(texts)}")
    if not text.strip():
        logging.warning("Empty text encountered. Skipping.")
        continue

    # Split text into chunks
    try:
        chunks = split_text(text)
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        continue

    for chunk in chunks:
        try:
            triplets = get_relation_triplets_combined(chunk)
        except Exception as e:
            logging.error(f"Error extracting triplets: {e}")
            continue
        if not triplets:
            logging.info("No triplets extracted for this chunk.")
            continue
        logging.info(f"Extracted {len(triplets)} Triplets:")
        entities_in_text = set()
        for triplet in triplets:
            logging.info(triplet)
            try:
                with driver.session() as session:
                    session.execute_write(
                        insert_triplet,
                        triplet['subject'],
                        triplet['relation'],
                        triplet['object'],
                        existing_entities  # Pass the existing_entities set
                    )
                # Update existing_entities after insertion
                subject_norm = normalize_entity_name(triplet['subject'])
                object_norm = normalize_entity_name(triplet['object'])
                existing_entities.add(subject_norm)
                existing_entities.add(object_norm)
                entities_in_text.add(subject_norm)
                entities_in_text.add(object_norm)
            except Exception as e:
                logging.error(f"Error inserting triplet into Neo4j: {e}")
                continue
        # Create co-occurrence relationships
        try:
            with driver.session() as session:
                session.execute_write(
                    create_concurrent_relations,
                    entities_in_text
                )
        except Exception as e:
            logging.error(f"Error creating co-occurrence relations: {e}")
            continue


# Process each text
for idx, text in enumerate(texts):
    logging.info(f"Processing text {idx+1}/{len(texts)}")
    if not text.strip():
        logging.warning("Empty text encountered. Skipping.")
        continue
    
    # Split text into chunks
    try:
        chunks = split_text(text)
    except Exception as e:
        logging.error(f"Error splitting text: {e}")
        continue

    for chunk in chunks:
        try:
            triplets = get_relation_triplets(chunk)
        except Exception as e:
            logging.error(f"Error extracting triplets: {e}")
            continue
        if not triplets:
            logging.info("No triplets extracted for this chunk.")
            continue
        logging.info(f"Extracted {len(triplets)} Triplets:")
        for triplet in triplets:
            logging.info(triplet)
            try:
                with driver.session() as session:
                    session.execute_write(
                        insert_triplet,
                        triplet['subject'],
                        triplet['relation'],
                        triplet['object'],
                        existing_entities  # Pass the existing_entities set
                    )
                # Update existing_entities after insertion
                existing_entities.add(normalize_entity_name(triplet['subject']))
                existing_entities.add(normalize_entity_name(triplet['object']))
            except Exception as e:
                logging.error(f"Error inserting triplet into Neo4j: {e}")
                continue



def split_text(text, max_chunk_length, overlap_length):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(tokens)
    chunks = []
    start = 0

    while start < total_tokens:
        end = min(start + max_chunk_length, total_tokens)
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if end == total_tokens:
            break
        start = end - overlap_length  # Overlap for context

    return chunks

def get_relation_triplets_combined(text):
    # Extract triplets using the transformer model
    triplets = get_relation_triplets(text)
    # Extract triplets using SpaCy
    spacy_triplets = extract_triplets_spacy(text)
    triplets.extend(spacy_triplets)
    return triplets

def get_relation_triplets(text):
    encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=tokenizer.model_max_length)
    output_ids = model.generate(
        input_ids=encoded_input['input_ids'],
        attention_mask=encoded_input['attention_mask'],
        max_length=256,        # Adjust as needed for output length
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=False,
    )
    decoded_output = tokenizer.batch_decode(output_ids, skip_special_tokens=False)[0]
    triplets = extract_triplets(decoded_output)
    return triplets

def create_concurrent_relations(tx, entities_in_text):
    entity_list = list(entities_in_text)
    for i in range(len(entity_list)):
        for j in range(i+1, len(entity_list)):
            entity_a = entity_list[i]
            entity_b = entity_list[j]
            tx.run("""
                MATCH (a:Entity {normalized_name: $entity_a})
                MATCH (b:Entity {normalized_name: $entity_b})
                MERGE (a)-[:CO_OCCURS_WITH]->(b)
                """,
                entity_a=entity_a, entity_b=entity_b)

def get_orphan_nodes(tx):
    result = tx.run("MATCH (n) WHERE NOT (n)--() RETURN n")
    return [record["n"] for record in result]

def merge_orphan_node(tx, orphan_node_id, existing_node_norm_name):
    # Merge the orphan node with the existing node
    tx.run("""
        MATCH (orphan)
        WHERE id(orphan) = $orphan_node_id
        MATCH (existing:Entity {normalized_name: $existing_node_norm_name})
        // Merge properties from orphan to existing node
        SET existing += orphan
        // Delete the orphan node
        DELETE orphan
        """,
        orphan_node_id=orphan_node_id,
        existing_node_norm_name=existing_node_norm_name
    )

def create_relationship_with_existing(tx, orphan_node_id, existing_node_norm_name):
    # Create a relationship between the orphan node and the existing node
    tx.run("""
        MATCH (orphan)
        WHERE id(orphan) = $orphan_node_id
        MATCH (existing:Entity {normalized_name: $existing_node_norm_name})
        MERGE (orphan)-[:RELATED_TO]->(existing)
        """,
        orphan_node_id=orphan_node_id,
        existing_node_norm_name=existing_node_norm_name
    )

# Fetch existing entities before processing orphan nodes
with driver.session() as session:
    existing_entities = session.execute_read(get_existing_entities)
if not isinstance(existing_entities, set):
    existing_entities = set(existing_entities)

# Fetch and process orphan nodes
with driver.session() as session:
    orphan_nodes = session.execute_read(get_orphan_nodes)

for node in orphan_nodes:
    # Inspect properties of the node
    properties = dict(node.items())
    # Implement logic to process and connect the node
    # For example, if the node has a 'name' property:
    name = properties.get('name')
    if name:
        # Normalize and attempt to match with existing entities
        normalized_name = normalize_entity_name(name)
        match = find_matching_entity(normalized_name, existing_entities)
        if match:
            # Option 1: Merge the orphan node with the matched existing node
            with driver.session() as session:
                session.execute_write(
                    merge_orphan_node,
                    orphan_node_id=node.id,
                    existing_node_norm_name=match
                )
            # Update existing_entities set
            existing_entities.add(normalized_name)
        else:
            # Option 2: Create a relationship between the orphan node and the existing node
            # For this example, let's assume we want to relate similar entities
            similar_match = process.extractOne(normalized_name, existing_entities, scorer=process.fuzz.partial_ratio)
            if similar_match and similar_match[1] >= 80:
                with driver.session() as session:
                    session.execute_write(
                        create_relationship_with_existing,
                        orphan_node_id=node.id,
                        existing_node_norm_name=similar_match[0]
                    )
                # Update existing_entities set
                existing_entities.add(normalized_name)
            else:
                # No match found, add to existing_entities
                existing_entities.add(normalized_name)
    else:
        # Handle nodes without a 'name' property if necessary
        logging.warning(f"Orphan node with id {node.id} has no 'name' property.")

# # Example text
# text = """The Battle of Mutina was fought on April 21, 43 BC between the forces of Mark Antony and the forces of the consuls Hirtius and Pansa, along with Octavian. The battle resulted in a victory for the consuls and Octavian."""

# # Get triplets
# triplets = get_relation_triplets(text)
# print("Extracted Triplets:")
# for triplet in triplets:
#     print(triplet)

# # Insert into Neo4j with normalization
# for triplet in triplets:
#     with driver.session() as session:
#         session.execute_write(
#             insert_triplet,
#             triplet['subject'],
#             triplet['relation'],
#             triplet['object'],
#             existing_entities  # Pass existing_entities
#         )
#     # Update existing_entities after insertion
#     existing_entities.add(normalize_entity_name(triplet['subject']))
#     existing_entities.add(normalize_entity_name(triplet['object']))