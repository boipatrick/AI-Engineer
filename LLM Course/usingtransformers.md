## Intro
Transformers library was created to provide a unified API for loading, training, and saving any Transformer model. Its main features are:
The 🤗 Transformers library is a popular open-source toolkit for working with state-of-the-art natural language processing models. It supports a wide range of tasks, including text classification, question answering, and text generation.

1. Ease of use
2. Flexibility
3. Simplicity

## Behind the Pipeline

Tokenization
Model
Postprocessing- convert logits into probabilities

- **Tokenization**: Converts text inputs into tokens (numbers) the model can process.
- **Model**: Runs the tokens through the model to get predictions.
- **Postprocessing**: Converts raw model outputs (logits) into probabilities.

```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier([
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
])
```

## Tokenization

Tokenization is the process of converting text into numbers. The tokenizer:

- Splits input into tokens (words, subwords, symbols)
- Maps each token to an integer
- Adds additional inputs required by the model

Preprocessing must match the model's pretraining. Use `AutoTokenizer.from_pretrained()` to fetch the correct tokenizer:

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
```

Transformer models accept tensors as input.

## Going Through the Model

Download the pretrained model similarly to the tokenizer. Each input yields a high-dimensional vector representing its contextual meaning.

## High-Dimensional Vectors

Transformer outputs are typically 3D tensors:

- Batch size: Number of sequences processed at once
- Sequence length: Length of each input sequence
- Hidden size: Dimensionality of each input's vector

## Model Heads

Model heads project hidden states onto task-specific dimensions, often using linear layers. The embeddings layer converts input IDs to vectors, and attention layers refine these representations.

Common architectures include:

- Model (hidden states)
- ForCausalLM
- ForMaskedLM
- ForMultipleChoice
- ForQuestionAnswering
- ForSequenceClassification
- ForTokenClassification

## Postprocessing

Model outputs (logits) are raw scores. Use a Softmax layer to convert logits to probabilities.

## Models
Automodel class is quite handy when you want to instantiate any model from a check point

## Creating a Transformer

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("bert-base-cased")
```

the from_pretrained() method will download and cache the model data from the Hugging Face Hub.

the checkpoint name corresponds to a specific model architecture and weights. in this case a BERT model with a basic architecture (12 layers, 768 hidden size, 12 attention heads) and cased inputs (meaning that the uppercase/lowercase distinction is important) 

The automodel class and its associates are wrappers designed to fetch the appropriate model architecture for a given checkpoint. 

It's an "auto" class meaning it will guess the appropriate model architecture for you and instantiate the correct model class.

 However, if you know the type of model you want to use, you can use the class that defines its architecture directly:

```python

from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

## Loading and Saving
saving a model is as simple as saving a tokenizer. models actually have the same save_pretrained() method, which saves the model's weights and architecture configuration

```python
model.save_pretrained("directory_on_my_computer")
```
Thus will save two files to your disk

ls directory_on_my_computer

config.json model.safetensors

Looking at the contents of config.json file. we'll see the necessary attributes needed to build the model architecture. it also contains some metadata, such as where the checkpoint originated and what transformers version you were using when you last saved the checkpoint. 

The pytorch_model.safetensors file is known as the state dictionary; it contains all your model’s weights. The two files work together: the configuration file is needed to know about the model architecture, while the model weights are the parameters of the model.

To reuse a saved model, use the from_pretrained() method again:


```python
from transformers import AutoModel

model = AutoModel.from_pretrained("directory_on_my_computer")

```

## Encoding text
```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

encoded_input = tokenizer("Hello, I'm a single sentence!")
print(encoded_input)
```

output:
```python
{'input_ids': [101, 8667, 117, 1000, 1045, 1005, 1049, 2235, 17662, 12172, 1012, 102], 
 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
```

input ids: numerical representations of your tokens
token_type_ids: these tell the model which part of the input is Sentence A and which is Sentence B
attention_mask: this indicates which tokens should be attended to and which should not. 

We can decode the input IDs to get back the original text:

```python
tokenizer.decode(encoded_input["input_ids"])
```
"[CLS] Hello, I'm a single sentence! [SEP]"

[CLS] and [SEP] are special token added by the tokenizer. Not all models need special tokens: they're utilized when a model was pretrained with them, in which case the tokenizer needs to add them as that model expects these tokens 


You can encode multiple sentences at once, either by batching them together (we’ll discuss this soon) or by passing a list:

## Padding Inputs
If we ask the tokenizer to pad the inputs, it will make all sentences the same length by adding a special padding token to the sentences that are shorter than the longest one:

Note that the padding tokens have been encoded into input IDs with ID 0, and they have an attention mask value of 0 as well. This is because those padding tokens shouldn’t be analyzed by the model: they’re not part of the actual sentence.

```python
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"], padding=True, return_tensors="pt"
)
print(encoded_input)
```

## Truncating Inputs
The tensors might get too big to be processed by the model. For instance, BERT was only pretrained with sequences up to 512 tokens, so it cannot process longer sequences. If you have sequences longer than the model can handle, you’ll need to truncate them with the truncation parameter:


```python
encoded_input = tokenizer(
    "This is a very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very very long sentence.",
    truncation=True,
)
print(encoded_input["input_ids"])
```
By combining the padding and truncation arguments, you can make sure your tensors have the exact size you need:

```python
encoded_input = tokenizer(
    ["How are you?", "I'm fine, thank you!"],
    padding=True,
    truncation=True,
    max_length=5,
    return_tensors="pt",
)
print(encoded_input)
```

## Adding Special Tokens
Important to BERT and derived models. 
These tokens are added to better represent the sentence boundaries, such as the beginning of a sentence ([CLS]) or separator between sentences ([SEP]).

Day 8
##  Tokenizers
Tokenizers are one of the core components of the NLP pipeline. They translate text into data that can be processed by the model. 
They are so many ways in which we can do this the goal is to find the most meaningful representation- that is, the one that makes the most sense to the model. 

## Tokenization algorithms
### Word-Based
Generally very easy to set up and use with only a few rules, and it often yields decent results.

One way to reduce the amount of unknown tokens is to go one level deeper, using a character-based tokenizer.

### Character-based
Split text into characters rather than words. Benefits include:
1. The vocabulary is muvh smaller
2. There are much fewer out-of-vocabulary(unknown) tokens, since every word cabn be built from characters

each character doesn’t mean a lot on its own, whereas that is the case with words.

This differs according to the language

We'll end up with a very large amount of tokens to be processed by our model: whereas a word would only be a single token with a word-based tokenizer, it can easily turn into 10 or more tokens when converted into characters.

### Subword tokenization

![Subword Tokenization](tokenizers.PNG)

Subword tokenization algorithms rely on the principle that frequently used words should not be split into smaller subwords, but rare words should be decomposed into meaningful subwords.

### And more!
There are many more techniques out there. To name a few:
Byte-level BPE, as used in GPT-2
WordPiece, as used in BERT
SentencePiece or Unigram, as used in several multilingual models

### Loading and Saving
Its based on two models: from_pretrained() and save_pretrained(). These methods will load or save the algorithm used by the tokenizer(a bit like the architecture of the model) as well as its vocabulary.

### Encoding
Translating text to numbers is known as encoding. Its done in a two step process: the tokenization, followed by the conversion to input IDs

As we’ve seen, the first step is to split the text into words (or parts of words, punctuation symbols, etc.), usually called tokens. 
There are multiple rules that can govern that process, which is why we need to instantiate the tokenizer using the name of the model, to make sure we use the same rules that were used when the model was pretrained.

The second step is to convert those tokens into numbers, so we can build a tenso out of them and feed them to the model. To do this, the tokenizer has a vocabulary, which is the part we download when we instantiate it with the from_pretrained()method.

### Tokenizers
The tokenization process is done by the tokenize() method of the tokenizer:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)

print(tokens)
```

the output is a list of strings.
This tokenizer is a subword tokenizer from the results, it splits the words until it obtains tokens that can be represented by its vocabulary.

### From tokens to input IDs
The conversion to input IDs is handled by the convert_tokens_to_ids() tokenizer method:


```python
ids = tokenizer.convert_tokens_to_ids(tokens)

print(ids)
```

These outputs, once converted to the appropriate framework tensor, can then be used as inputs to a model as seen earlier in this chapter.

### Decoding
This is the inverse: from vocabulary indices, we want to get a string. This can be done with the decode()method

```python
decoded_string = tokenizer.decode([7993, 170, 11303, 1200, 2443, 1110, 3014])
print(decoded_string)
```
This method not only converts the indices back to tokens, but also groups together the tokens that were part of the same words to produce a readable sentence.


### Handlin multiple sequences

In the previous section, we explored the simplest of use cases: doing inference on a single sequence of a small length.
However, some questions emerge already:

1. How do we handle multiple sequences?
2. How do we handle multiple sequences of different lengths?
3. Are vocabulary indices the only inputs that allow a model to work well?
4. Is there such a thing as too long a sequence?

The Transformer API solves this
Batching is the act of sending multiple sentences through the model, all at once. If you only have one sentence, you can just build a batch with a sinle sequence

It allows the model to work when you feed it multiple sentences.When submitting two or some sentences of different lengths we usually pad the inputs. 

Padding enables us to make our tensors have a rectangular shape. 
Padding makes sure all our sentences have the same length by adding a special word called the padding token to the sentences with fewer values

The attention mask is used to tell thw attention layers to ignore the padding tokens. 

## Attention Masks
Attention masks are tensors with the exact same shape as the input IDs tensor, filled with 0s and 1s:1s indicate ther tokens suld not attended


## longer Sequences
With TF Models there is a limit to the lengths of the sequences we can pass the models. Most models handle sequences of up to 512 or 1024 tokens, and will crash when asked to process longer sequences.

There are two solutions to this problem:
1. Use a model with a longer supported sequence length
2. Truncate your sequences(Otherwise, we recommend you truncate your sequences by specifying the max_sequence_length parameter:)

## Putting It All Together

The Transformer API handles all steps. Tokenize and prepare inputs:

```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
sequence = "I've been waiting for a HuggingFace course my whole life."
model_inputs = tokenizer(sequence)
```

## From tokenizer to model
Now that we’ve seen all the individual steps the tokenizer object uses when applied on texts, let’s see one final time how it can handle multiple sequences (padding!), very long sequences (truncation!), and multiple types of tensors with its main API:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
```

### Recap

The basic building blocks of a Transformer model
Learned what makes up a tokenization pipeline
Saw how to use a Transformer model in practice
Learned how to leverage a tokenizer to convert text to tensors that are understandable by the model
Setup a tokenizer and a model together to get from text to predictions
Learned the limitations of input IDs and attention masks
Played around with versatile and configurable tokenizer models

# Optimized Inference Deployment
Here we will look at advanced frameworks for optimizing deployments: Text Generation Inference(TGI), vLLM, and llama.cp. 

These applications are used in production environments to serve LLMs to users. 

## Framework Selection Guide
TGI, vLLM and llama.cpp serve similar purposes but have distinct characteristics depending on the use case. Next we will look at the key differeces between them, focusing on performance and integration.

### Memory Management and Performance
TGI- designed to be stable and predictable in production, using  fixed sequence lengths to keep memory usage consistent. 
It manages memory using Flash Attention 2 and continous batching techniques.
This means it can process attention 2 and calculations very efficiently and keep the GPU busy by constantly feeding it work.
The system can move parts of the model between CPU and GPU when needed, which helps handle larger models.

vLLM takes a different approach by using PagedAttention.
vLLM splits the model's memory into smaller blocks.
 
With this it can handle different-sized requests more flexibly and doesn't waste memory space.

Its particularly good at sharing memory betwen different requests and reduces memory fragmentation, which makes the whole system more efficient.


llama.cpp is a highly optimized C/C++ implementation originally designed for LLaMa models on consumer hardware. 

It focuses on CPU efficiency with optional GPU acceleration and is ideal for resource-constrained environments.

It uses quantization techniwues to reduce  model size and memory requirements while maintaining good performance.

It implements optimized kernels for various CPU architectures and supports basic KV cache management for efficient token generation.

## Deployment and Integration
TGI excels in enterprise-level deployment with its production-ready features.
Comes with built-in Kubernetes support and includes everything you need for running in production, like monitoring via Prometheus, automatic scaling, and comprehensive safety features.

System also includes enterprise-grade logging and various protective measures like content filtering and rate limiting tokeep your deployment secure and stable. 

vLLM takes a more flexible, developer-friendly approach to deployment
built with python at its core and can easily replace OpenAI's API in your existing apps.

The framework focuses on delivering raw perfomace well with Ray for managing clusters, making it a great choice when you need high performance and adaptibility.

llama.cpp prioritizes simplicity and portability.
Its server implementation is lightweight and can run on a wide range of hardware, from powerful srvers to consumer laptops and even some high end mobile devices
With minimal dependancies and C/C++ core, it's easy to deploy in environments where installing python frameworks would be challenging. 
The server provides an OpenAI-compatible API while maintaining a much smaller resource footprint than other solutions.
