# AI-Log-Analyzer

AI-Log-Analyzer is an open source toolkit, user friendly, based on deep-learning, for unstructured log anomaly detection.

## Components

### Anomaly Detection: The core of the project.
1. **Log Parsing:** Logs are structured using the [drain3 tool](https://github.com/IBM/Drain3)
2. **Training:** An unsupervised LSTM model is trained to learn the normal workflow of a system.
3. **Anomaly Detection:** If the model has been trained, it can predict anomalies in log sequences.

You can read the papers about [deeplog](https://www.cs.utah.edu/~lifeifei/papers/deeplog.pdf) and [loganomaly](https://www.ijcai.org/proceedings/2019/0658.pdf) for further information.

## Installation
```
git clone ...
cd AI-Log-Analyzer
pip3 install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you have a compatible gpu you can install CUDA. Training a neural network on gpu is way faster than cpu.

Unlike DeepLog, LogAnomaly convert log into semantic vectors. To use it, you need to download a dictionnary that map words into vectors. Bellow the instruction to download word2vec for English:

```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.vec.gz
gunzip cc.en.300.vec.gz
```

Then run in a python interpreter:
```python
from ailoganalyzer.dataset.dbm_vec import install_vectors
install_vectors("cc.en.300.vec", "en_vec")
```

To ensure the dictionnary is installed:
```python
with open("en_vec") as d:
    print("hello" in d)
    print(d["hello"])
```

## Quick start

### Train the model

```python
from ailoganalyzer.dataset import LogFileDataset
from ailoganalyzer.model import DeepLog, LogAnomaly
from torch.utils.data import DataLoader
import lightning as L

log_file = "path/to/your/logfile.log"
dataset = LogFileDataset(log_file, semantic_vector="en_vec", seq_label=True)

train_dataloader = DataLoader(train_dataset, batch_size=100)

model = LogAnomaly(dataset.get_num_classes(), optimizer_fun="adam")

trainer = L.Trainer(max_epochs=100)
trainer.fit(model=model, train_dataloaders=train_dataloader)
```
