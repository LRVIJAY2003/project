# Core and Flask packages
Flask==2.3.3
Flask-Cors==4.0.0
Flask-RESTful==0.3.10
python-dotenv==1.0.0
gunicorn==21.2.0

# Data processing
beautifulsoup4==4.12.2
lxml==4.9.3
html2text==2020.1.16
jsonschema==4.19.0
markdown==3.4.4

# API and HTTP
requests==2.31.0
requests-cache==1.1.0
urllib3==2.0.5
PyJWT==2.8.0

# ML and Vector Database - order matters
numpy==1.26.3
scipy==1.12.0
scikit-learn==1.3.2
pandas==2.1.4
torch==2.2.0
transformers==4.30.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4

# Text processing
spacy==3.7.2
nltk==3.8.1
rank-bm25==0.2.2

# Google AI and Gemini - critical conflict resolution
protobuf==3.20.0
googleapis-common-protos==1.56.4
google-auth==2.16.0
google-api-core==2.11.0
google-api-python-client==2.80.0
google-cloud-aiplatform==1.25.0
google-auth-oauthlib==1.0.0

# Caching and optimization
redis==5.0.0
cachetools==5.3.1

# Utilities
tqdm==4.66.1
pydantic==2.5.3
tenacity==8.2.3
loguru==0.7.2
python-slugify==8.0.1
