# Demo: learning with few labeled data and many unlabeled
Demo for learning with few labels webinar.
The idea is inspired from (but not the same) 
[Graph Agreement Models](https://proceedings.neurips.cc/paper/2019/file/4772c1b987f1f6d8c9d4ef0f3b764f7a-Paper.pdf).


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

To train MLM:
```
python -m src.train_agree_add_loop --only_mlm 1
```

To train on top default base model:
```
python -m src.train_agree_add_loop
```

To train on top of your trained MLM:
```
python -m src.train_agree_add_loop --base_model_name models/nlu_evaluation_data/mlm/epoch_SOMENUMBER
```

To run with fasttext as classifier for computing the baseline:
```
python -m src.train_agree_add_loop --use_fasttext 1
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
