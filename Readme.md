# Gradio App is running on the following url
Running on public URL: https://acb84f1afb5817dedd.gradio.live

This share link expires in 72 hours.
* O means the word doesn’t correspond to any entity.
* B-PER/I-PER means the word corresponds to the beginning of/is inside a person entity.
* B-ORG/I-ORG means the word corresponds to the beginning of/is inside an organization entity.
* B-LOC/I-LOC means the word corresponds to the beginning of/is inside a location entity.
* B-MISC/I-MISC means the word corresponds to the beginning of/is inside a miscellaneous entity.
# Fine tuning Bert
# Model is performing Named Entity Recognition
**BERT- Bidirectional Encoder Representations from Transformers**
BERT architecture consists of several Transformer encoders stacked together. Each Transformer encoder encapsulates two sub-layers: a self-attention layer and a feed-forward layer.
***Named entity recognition (NER):***
Find the entities (such as persons, locations, or organizations) in a sentence. This can be formulated as attributing a label to each token by having one class per entity and one class for “no entity.”
***Part-of-speech tagging (POS):***
Mark each word in a sentence as corresponding to a particular part of speech (such as noun, verb, adjective, etc.).
* O means the word doesn’t correspond to any entity.
* B-PER/I-PER means the word corresponds to the beginning of/is inside a person entity.
* B-ORG/I-ORG means the word corresponds to the beginning of/is inside an organization entity.
* B-LOC/I-LOC means the word corresponds to the beginning of/is inside a location entity.
* B-MISC/I-MISC means the word corresponds to the beginning of/is inside a miscellaneous entity.


##  The entire code for the model and also for launching the gradio interface is in a single notebook (Fine_tuning_Bert)
 

 Also it takes a large time to train the model

 # Named Entity Recognition with Fine-Tuned BERT Model

This project demonstrates how to perform Named Entity Recognition (NER) using a fine-tuned BERT model. NER is a natural language processing task that involves identifying entities (such as names of persons, organizations, locations, etc.) in a text.

## Table of Contents

- [Overview](#overview)
- [Setup](#setup)
- [Usage](#usage)
- [Gradio Interface](#gradio-interface)


## Overview

This project uses the Hugging Face Transformers library to fine-tune a BERT model on the CoNLL 2003 NER dataset. The fine-tuned model is then used to perform NER on input text, identifying entities and their corresponding labels. The project includes the following components:

- Data loading and preprocessing using the `datasets` library.
- Tokenization and alignment of labels using the `BertTokenizerFast` class.
- Fine-tuning the BERT model for token classification using the `Trainer` class.
- Creating an NER pipeline for inference using the fine-tuned model.
- Launching a Gradio interface for interactive NER.

## Setup

To run this project, you need to have Python 3.6 or later installed. You can install the required packages using the following command:

## Usage

To utilize this project for Named Entity Recognition (NER) using a fine-tuned BERT model, follow these steps:

1. **Load and Preprocess the Dataset**: Begin by loading the CoNLL 2003 NER dataset using the `datasets` library. This dataset contains labeled entities for training, validation, and testing.

2. **Tokenization and Alignment**: Tokenize the input text and align the corresponding labels with the tokenized inputs. This alignment ensures accurate matching of labels during training and evaluation.

3. **Fine-Tuning the BERT Model**: Fine-tune a pre-trained BERT model for token classification by employing the `Trainer` class from the `transformers` library. Define the necessary training arguments, data collator, and evaluation metrics.

4. **Create an NER Pipeline**: Instantiate an NER pipeline for inference purposes using the fine-tuned BERT model and the tokenizer. The pipeline will automatically process input text and extract named entities.

5. **Perform Named Entity Recognition**: Utilize the created NER pipeline to conduct named entity recognition on sample text. The pipeline will extract entities such as persons, organizations, locations, and more.

## Gradio Interface
The project includes a Gradio interface that allows you to interactively perform NER using the fine-tuned BERT model. The interface allows you to input text and view the identified entities along with their labels.


# Link for the Dataset used -
[Link text](https://huggingface.co/datasets/conll2003/viewer/conll2003/train)
