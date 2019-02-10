Abstractive Summarization
======================

The repository is a reproduction of ACL 2017 paper *[Get To The Point: Summarization with Pointer-Generator Networks](https://arxiv.org/abs/1704.04368)*.

It heavily replied on the repository [here](https://github.com/becxer/pointer-generator). 


# Data preparation
I use the [code](https://github.com/becxer/pointer-generator) to process CNN/DailyMail dataset. 

# Model
Regarding model, i reconstruct the code using Tensorflow. It support (1) Pointer-Generator and (2) Coverage Mechanism

please see model.py for more details

# Train & Validation & decode
Training validation and decode can be done concurrently.

please see run.py for more details.

# Requirement
It support python3. 

# Hyperparameter
please see config.py for more details.

# Evaluation
ROUGE score, standard metric.
I use the [code](https://github.com/becxer/pointer-generator).






