import tiktoken
import openai
from openai import OpenAI
import json
import os

client = OpenAI(
  api_key=os.getenv('OPENAI_API_KEY')
)

def load_encoding(model_name: str):
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        print(f"Model {model_name} not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    return encoding

def count_tokens_in_string(text: str, encoding) -> int:
    return len(encoding.encode(text))

def count_tokens_in_messages(messages: list, model_name: str) -> int:
    encoding = load_encoding(model_name)
    tokens_per_message = 3
    tokens_per_name = 1

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == 'name':
                num_tokens += tokens_per_name
    num_tokens += 3  # For assistant reply
    return num_tokens


def num_tokens_from_messages(messages, model="gpt-4o-mini"):
    """Return the number of tokens used by a list of messages for a specific model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model {model} not recognized. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    
    if model in {"gpt-4o", "gpt-4o-mini"}:
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"Token counting not implemented for model {model}.")

    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # Assistant reply tokens
    return num_tokens

def num_tokens_for_functions_and_messages(functions, messages, model="gpt-4o-mini"):
    """Return the number of tokens used by messages and functions for a specific model."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print(f"Warning: model {model} not recognized. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    
    if model in {"gpt-4o", "gpt-4o-mini"}:
        tokens_per_function = 3
    else:
        raise NotImplementedError(f"Token counting not implemented for model {model}.")

    # Count tokens for messages
    num_tokens = num_tokens_from_messages(messages, model=model)
    
    # Count tokens for functions
    for function in functions:
        num_tokens += tokens_per_function
        num_tokens += len(encoding.encode(function["name"]))
        num_tokens += len(encoding.encode(function.get("description", "")))
        parameters = function.get("parameters", {})
        parameters_str = json.dumps(parameters)
        num_tokens += len(encoding.encode(parameters_str))
    
    return num_tokens

# Example usage
functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit of temperature to return",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["location"],
        },
    }
]

example_messages = [
    {"role": "system", "content": "You are a helpful assistant that provides weather information."},
    {"role": "user", "content": "What's the weather like in New York City?"},
]

# Estimate tokens
estimated_tokens = num_tokens_for_functions_and_messages(functions, example_messages, model="gpt-4o-mini")
print(f"Estimated tokens: {estimated_tokens}")

# Make API request
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=example_messages,
    functions=functions,
    max_tokens=50,
    temperature=0,
)

api_usage = response.usage.prompt_tokens
print(f"API reported tokens: {api_usage}")