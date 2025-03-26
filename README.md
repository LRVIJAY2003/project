# Core Flask
Flask==2.0.3
Flask-Cors==3.0.10
Flask-RESTful==0.3.9
python-dotenv==0.20.0
gunicorn==20.1.0

# Data processing - simple versions
beautifulsoup4==4.11.1
lxml==4.9.1
html2text==2020.1.16
jsonschema==4.15.0
markdown==3.3.7

# HTTP
requests==2.28.1
requests-cache==0.9.7
urllib3==1.26.12
PyJWT==2.4.0

# ML with binary wheels (no compilation)
numpy==1.23.4
scipy==1.9.3
scikit-learn==1.1.3
pandas==1.5.1

# Text processing (minimal versions)
nltk==3.7
rank-bm25==0.2.2

# Simplified Google packages
protobuf==3.19.4
google-api-python-client==2.65.0
google-auth==2.12.0
google-auth-oauthlib==0.5.3

# Utilities with minimal dependencies
tqdm==4.64.1
pydantic==1.10.2
loguru==0.6.0
tenacity==8.1.0

# Pre-built ML packages (optional - install separately if needed)
# torch
# transformers
# sentence-transformers
# faiss-cpu