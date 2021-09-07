# AI-Log-Analyzer

AI-Log-Analyzer is an open source toolkit based on deep-learning, for unstructured log anomaly detection.

## Components

### Anomaly Detection: The core of the project.
1. **Log Parsing:** Logs are structured using the [drain3 tool](https://github.com/IBM/Drain3)
2. **Training:** An unsupervised LSTM model is trained to learn the normal workflow of a system.
3. **Anomaly Detection:** If the model has been trained, it can predict anomalies in log sequences.

You can read the papers about [deeplog](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf) and [loganomaly](https://www.ijcai.org/proceedings/2019/0658.pdf) for further information.

### Database
The module provide wrappers to help the user to saves logs in a database:
- sqlite3
- mongodb (You must have a proper installation of mongodb to use it, and the pymongo module)

### Visualisation
I plan to add a way to visualize the content of the database.

## Installation
```
git clone ...
cd AI-Log-Analyzer
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you have a compatible gpu you can install CUDA. Training a neural network on gpu is way faster than cpu.

At this point you can only use the "deeplog" model. "loganomaly" model use word2Vec to convert logs into vectors. To do this you have to download the file "cc.en.300.vec", wich contains the semantic representation of each english words.

```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip cc.en.300.vec.gz
```

Then run in a python interpreter:
```python
from ailoganalyzer.dataset.dbm_vec import install_vectors
install_vectors("cc.en.300.vec")
```

## Quick start

### Train the model

```python
from ailoganalyzer.anomalyDetection.LSTM import DeepLog

model = DeepLog(prefix_file="test") # initialization of the model
# The attribute "prefix_file" is used to save the model in a file

with open("your_log_file.log", "r") as f:
    for line in f:
        line = line.strip() # remove the ending "\n".
        # It is recommended to extract headers such as timestamp, ID, hostname,
        # severity... to improve the performance of the model
        model.add_train_log(line)

lstm.train() # train the model
```

### Detect Anomaly
```python
from ailoganalyzer.anomalyDetection.LSTM import DeepLog

model = DeepLog(prefix_file="test") # initialization of the model

with open("your_log_file.log", "r") as f:
    for line in f:
        line = line.strip()
        model.predict(line) # return True if abnormal
```
