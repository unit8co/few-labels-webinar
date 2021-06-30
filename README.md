# Demo: learning with few labeled data and many unlabeled
Learning with few labels

Inspired from GAM paper

# How to run this
Get elasticsearch: https://www.elastic.co/guide/en/elasticsearch/reference/current/install-elasticsearch.html
Run elasticsearch in a separate terminal. If you have mem issues run it like (for example with 1G mem):
```
ES_JAVA_OPTS="-Xms1g -Xmx1g" ./bin/elasticsearch
```

Install requirements:
```
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements
```

Run:
```
python -m src.train_agree_add_loop
```

# Info
Formatted with black!
```
pip install black
black -l 80 -t py38 src/*
```

Contributions from:
* Mostafa Ajallooeian, mostafa.ajallooeian@unit8.co

License: have fun and share!
