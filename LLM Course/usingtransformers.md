## Intro
Transormers library was created with the goal to provide a single API through which any TF moel can be loaded, trained and saved. 
The library's main features include:
1. Ease of use
2. Flexibility
3. Simplicity

## Behind the Pipeline
What happens inside the pipeline() Function

Tokenization
Model
Postprocessing- convert logits into probabilities


```python
from transformers import pipeline

classifier = pipeline("sentiment-analysis")
classifier(
    [
        "I've been waiting for a HuggingFace course my whole life.",
        "I hate this so much!",
    ]
)
```

## Tokenization
first step is to convert the text inputs into numbers that the model can make sense of. To do this we use a tokenizer, which will be responsible for:
splitting the input into words, subwords, or symbols that are called tokens
mapping each token to an integer
Adding additional inputs that may be useful to the model

All this preprocessing needs to be done in exactly the same way as when the model was pretrained, so we first need to download that information from the Model Hub. To do this, we use the AutoTokenizer class and its from_pretrained() method. Using the checkpoint name of our model, it will automatically fetch the data associated with the model’s tokenizer and cache it (so it’s only downloaded the first time you run the code below).

Since the default checkpoint of the sentiment-analysis pipeline is distilbert-base-uncased-finetuned-sst-2-english (you can see its model card here), we run the following:


```python
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

```

Once we have the tokenizer, we can directly pass our sentence to it and we'll get back a dictionary that's ready to feed to our model.
Transformer models only accept tensors as input. 

## Going through the model
we download our pretrained model same way we did with our tokenizer.
For each model input we'll retrieve a high-dimensional vector representing the contextual understanding of that input by the transformer model. 

## A high-dimensional vector
vector output by the transformer module is usually large. It has 3 dimensions

Batch size no of sequences pocessed at a time
sequence lenth- length of the numerical representation of the sequence
Hidden size: the vector dimension of each model input

You can access the elements by attributes or by key or even by index

## Model Heads: Making sense out of numbers
The model heads take the high-dimensional vector of hidden states as input and project them onto a different dimension. They are usually composed of one or a few linear layers:
The embeddings layer converts each input ID in the tokenized input into a vector that represents the associated token. 
Subsequent layers manipulate those vectors using the attention mechanism to produce the final representation of the sentences

There are many different architectures available in 🤗 Transformers, with each one designed around tackling a specific task. Here is a non-exhaustive list:

*Model (retrieve the hidden states)
*ForCausalLM
*ForMaskedLM
*ForMultipleChoice
*ForQuestionAnswering
*ForSequenceClassification
*ForTokenClassification
and others 🤗

## Postprocessing the output
logits, the raw, unnormalized scores outputted by the last layer of the model. To be converted to probabilities they need to go via Softmax layer

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