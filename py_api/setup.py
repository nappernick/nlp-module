# setup.py

from setuptools import setup, find_packages

setup(
    name='py_api',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'tolerantjson',
        'uuid',
        'neo4j',
        'pika',
        'mcp',
        'tiktoken',
        'openai',
        'python-dotenv',
        'pymongo',
        'dnspython',
        'bitsandbytes',
        'optimum',
        'onnxruntime',
        'onnxruntime-tools',
        'deepspeed',
        'accelerate',
        'sentencepiece',
        'torch',
        'transformers',
        'axios',
        'daemon',
        'requests',
        'bs4',
        'flask',
        'pika',
        'pytest-mock',
        'tolerantjson',
        'pytest',
        'pytest-cov',
        'pytest-builtin-types',
        'logger'
    ],
)
