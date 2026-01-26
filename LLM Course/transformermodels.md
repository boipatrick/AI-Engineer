# Large Language Models (LLMs) Course Notes

---

## Understanding NLP and LLMs

### What is NLP?

**Natural Language Processing (NLP)** is a much broader field focused on enabling computers to understand and generate human language. It encompasses many techniques and tasks such as sentiment analysis, named entity recognition, and machine translation.

NLP is a field of linguistics and machine learning focused on understanding everything related to human language. Its task is not to understand single words individually, but to be able to understand the **context** of those words.

### Common NLP Tasks

- **Classifying whole sentences** – Determine if an email is spam, or if a sentence is grammatically correct
- **Classifying each word in a sentence** – Identify named entities (e.g., person, location, organization)
- **Generating text content** – Fill in the blanks or complete prompts
- **Extracting an answer from text** – Answer questions based on provided information
- **Generating a new sentence from input text** – Translate text into another language, summarize content

> NLP also tackles complex challenges in **speech recognition** and **computer vision**, such as generating a transcript of an audio sample or a description of an image.

---

### What are LLMs?

**Large Language Models (LLMs)** are a powerful subset of NLP models characterized by their massive size, extensive training data, and ability to perform a wide range of tasks with minimal task-specific training. Good examples include ChatGPT.

---

## The Rise of LLMs

An LLM is an AI model trained on massive amounts of text data that can understand and generate human-like text, recognize patterns in language, and perform a wide variety of language tasks without task-specific training.

### Key Characteristics

1. **Scale** – Billions of parameters
2. **General Capabilities** – Versatile across many tasks
3. **In-Context Learning** – Learn from examples in the prompt
4. **Emergent Abilities** – Unexpected capabilities that arise at scale

### LLM Limitations

| Limitation | Description |
|------------|-------------|
| Hallucinations | Generating false or fabricated information |
| Lack of True Understanding | Pattern matching without genuine comprehension |
| Bias | Reflecting biases present in training data |
| Context Windows | Limited amount of text that can be processed at once |
| Computational Resources | Requires significant hardware to run |

---

## Transformers

### What They Can Do

Transformers are used to solve all kinds of tasks across different modalities, including:
- NLP
- Computer vision
- Audio processing

### Transformers Library

The **Transformers Library** provides the functionality to create and use shared models (e.g., Google AI). The **Model Hub** contains millions of pretrained models that anyone can download and use.

### The Pipeline Function

The `pipeline()` function returns an end-to-end object that performs an NLP task on one or several texts. It is the most basic object in the Transformers library.

It connects a model with its necessary **preprocessing** and **postprocessing** steps, allowing us to directly input any text and get an intelligible answer.

#### Steps When Passing Text to a Pipeline

1. Text is **preprocessed** into a format the model can understand
2. The preprocessed inputs are **passed to the model**
3. The predictions of the model are **post-processed** so we can make sense of them

---

## Available Pipelines

The `pipeline()` function supports multiple modalities, allowing you to work with text, images, audio, and even multimodal tasks.

### Text Pipelines

| Pipeline | Description |
|----------|-------------|
| `text-generation` | Generate text from a prompt |
| `text-classification` | Classify text into predefined categories |
| `summarization` | Create a shorter version of text while preserving key information |
| `translation` | Translate text from one language to another |
| `zero-shot-classification` | Classify text without prior training on specific labels |
| `feature-extraction` | Extract vector representations of text |

### Image Pipelines

| Pipeline | Description |
|----------|-------------|
| `image-to-text` | Generate text descriptions of images |
| `image-classification` | Identify objects in an image |
| `object-detection` | Locate and identify objects in images |

### Audio Pipelines

| Pipeline | Description |
|----------|-------------|
| `automatic-speech-recognition` | Convert speech to text |
| `audio-classification` | Classify audio into categories |
| `text-to-speech` | Convert text to spoken audio |

### Multimodal Pipelines

| Pipeline | Description |
|----------|-------------|
| `image-text-to-text` | Respond to an image based on a text prompt |

---

## Inference Providers

Inference providers allow you to test all the models directly through your browser.

---

## Specific NLP Tasks

### Mask Filling

Fills in the blanks in a given text.

### Named Entity Recognition (NER)

NER is a task where the model has to find parts of the input text that correspond to entities such as **locations**, **persons**, or **organizations**. Give it text and it will seek out the entities you specified.

### Question Answering

This pipeline answers questions using information from a given context. The pipeline works by **extracting** information from the provided context—it does not generate the answer.

### Summarization

The task of reducing a text into a shorter version while keeping all (or most) of the important aspects referenced in the text.

### Translation

Converts text from one language to another.

---

## Combining Data from Multiple Sources

Transformer models can combine and process data from multiple sources. This comes in handy when you need to:

1. Search across multiple databases or repositories
2. Consolidate information from different formats (text, images, audio)
3. Create a unified view of related information

**Example Use Case:** Build a system that:
- Searches for information across databases in multiple modalities (e.g., text and image)
- Combines results from different sources into a single coherent response
- Presents the most relevant information from documents and metadata


### Day 3
Transformers are Language Models
GPT,BERT T5 are all examples of language models- Trained on large amounts of raw text in a self-supervised fashio n

SS learning is a type of training in which the objective is automatically computed from the inputs of the model. 
Humans are not needed to label data

The model develops a statistical understanding of the language it has been trained on. (less useful for specific practical tasks)

Due to this the pretrained model then goes through transferlearning or finetuning. During this process, the model is fine tuned in a supervised way i.e human-annotated labels on a given task. E.g predicting the next word in a sentence having read the n previous words
Casual language modeling- coz the output depends on the past and present inputs but not the future ones. 

Masked language modelling- the model predicts a masked word in the sentence 


### Transformers are big models

General strategy to achieve better performance is by increasing te model's size as well as the amount of data they are pretrained on. 


## Day 4 
Training a model, especially a large one, requires a large amount of ata. This becomes costly in time and resources. This can even translates to environmental impact. 


When it comes to Environmental impact
1. Type of Energy
2. Training Time
3. Hardware you use 

Other elements to consider
1. Fine tuning
2. Using pretrained models when they are available
3. Starting with smaller experiments and debugging
4. Random search
5. Doing a literature review 

Sharing language models is paramount: sharing the trained weights and buidling on top of already trained weights reduces the overall compute cost and carbon footprint of the community. 

Tools to evaluate Carbon footprintof your models' training: CodeCarbon and ML CO2 Impact


### Transfer Learning
 pretraining is the act of training a model from scratch: the weights are randomly initialized, and the training starts without any prior knowledge. 
  It is done on very large amounts of data. Requires large corpus of data, and training can take up to several weeks. 

  Fine tuning on the otherhand is the training done after a model has been pretrained. Here you first acquire a pretrained language model, then perform additional training with a dataset specific to your task.

  Fine tuning has lower time, data, financial, and environmental costs. Its also quicker and easier to iterate over different fine-tuning schemes, as the training is less constraining than a full pretraining. 


  ### General Transformer Architecture
  Model is composed of two blocks the encoder and decoder
  The encoder receives an input and builds a representation of it. The model is optimized to acquire understanding from the input
  Decoder: it uses encoder's representation(features) along with other inputs to generate a target sequence. Model is optimized for generating outputs. 

  Each of these can be used independently depending on the task.
  1. Encoder-only models: Good for tasks that require understanding of the input such as sentence classification
  2. Decoder-only models: text generation
  3. Encoder-decoder models: good for generative tasks that require an input such as translation or summarization


## Attention layers
special layers used to build transformer models.
This layer will tell the model to pay specific attention to certain words in the sentence you passed it.

## The Original Architecture
The TF architecture was originally designed for translation. 
During training the encoder receives inputs(sentences) in a certain language, while the decoder receives the same sentences in the desired target language.

### How Encoder and Decoder Attention Differs

**Encoder Attention (Sees Everything):**
- The encoder can look at ALL words in a sentence at once—both before AND after any given word
- Why? Because to translate "I love cats" → "J'aime les chats", understanding the full sentence helps translate each word correctly
- Think of it like reading the whole sentence first before trying to understand any part

**Decoder Attention (Sees Only the Past):**
- The decoder works step-by-step, like writing one word at a time
- It can only see words it has ALREADY generated, not future words
- Example: When predicting the 4th word, it can only look at words 1, 2, and 3

**Why This Restriction on the Decoder?**
Imagine you're translating and trying to predict word #2. If you could already see word #2, that's cheating! The task would be too easy—just copy the answer.

**The Training Trick:**
- During training, we actually give the decoder the WHOLE target sentence at once (for speed)
- BUT we use a "mask" to hide future words, so it still can't peek ahead
- This way, the model learns to predict without cheating, but training goes much faster than generating word-by-word

**Simple Analogy:**
- **Encoder** = Reading a full question before answering
- **Decoder** = Writing your answer one word at a time, only seeing what you've written so far

## Architectures Vs Checkpoints

Architecture: skeleton of the model- definition of each layer and each operation that happens within the model

Checkpoints: weights that will be loaded in a given architecture

Model : umbrella term that isn't as precise as architecture or checkpoint but it can mean both. 


---

# How Transformers Solve Tasks

Different transformer models are designed for different tasks. Here's a quick overview:

| Model | Type | Best For |
|-------|------|----------|
| BERT | Encoder-only | Text classification, NER, Q&A |
| GPT-2 | Decoder-only | Text generation |
| BART, T5 | Encoder-Decoder | Summarization, Translation |
| ViT | Vision Transformer | Image classification |
| Whisper | Encoder-Decoder | Speech recognition |
| DETR | Vision Transformer | Object detection |

> **Prerequisite:** Understanding encoders, decoders, and attention (covered above) will help you grasp how these models work!

---

## Transformer Models for Language

Language models learn statistical patterns between words/tokens in text. They're pretrained on massive text data, then fine-tuned for specific tasks.

### Two Main Training Approaches

| Approach | Used By | How It Works |
|----------|---------|--------------|
| **Masked Language Modeling (MLM)** | BERT (encoder) | Randomly hide words, predict them using context from BOTH sides |
| **Causal Language Modeling (CLM)** | GPT (decoder) | Predict the next word using only PREVIOUS words |

### Three Types of Language Models

1. **Encoder-only (BERT):** Sees context from both directions → great for *understanding* text (classification, Q&A)
2. **Decoder-only (GPT, Llama):** Only sees previous words → great for *generating* text
3. **Encoder-Decoder (T5, BART):** Encoder understands input, decoder generates output → great for *transforming* text (translation, summarization)

---

## Text Generation (GPT-2)

GPT-2 is a **decoder-only** model. Here's how it generates text:

**Step-by-step process:**
1. **Tokenize** the input using Byte Pair Encoding (BPE)
2. **Add position info** so the model knows word order
3. **Pass through decoder blocks** with masked self-attention
4. **Predict the next word** using a language modeling head

**Key Point - Masked Self-Attention:**
- GPT-2 can ONLY look at words to the LEFT (previous words)
- Future words have their attention scores set to 0
- This is different from BERT's [MASK] token—it's hiding future positions, not random words

**Training objective:** Predict the next word in a sequence (causal language modeling)

---

## Text Classification (BERT)

BERT is an **encoder-only** model that sees words from BOTH directions.

**How BERT processes text:**
1. **Tokenize** using WordPiece
2. Add special tokens: `[CLS]` at the start, `[SEP]` between sentences
3. Add **segment embeddings** to distinguish sentence pairs
4. Pass through encoder layers
5. Use the `[CLS]` token output for classification

**BERT's Two Pretraining Tasks:**
1. **Masked Language Modeling:** Hide ~15% of words, predict them
2. **Next Sentence Prediction:** Given two sentences, predict if B follows A

**For classification:** Add a simple classification head on top that converts the `[CLS]` output into class probabilities.

---

## Token Classification (BERT for NER)

Same as text classification, but instead of using just `[CLS]`, we use the output of EVERY token to classify each word (e.g., is this word a Person? Location? Organization?).

---

## Question Answering (BERT)

For Q&A, BERT finds the answer WITHIN the given text (extractive, not generative).

**How it works:**
- Add a span classification head
- Predict TWO positions: where the answer STARTS and where it ENDS
- Extract that text span as the answer

> 💡 **Notice:** Once BERT is pretrained, you just swap the "head" on top for different tasks!

---

## Summarization (BART)

BART is an **encoder-decoder** model—perfect for transforming long text into short summaries.

**How BART works:**
1. **Encoder** (like BERT) processes the input text
2. **Pretraining trick:** Corrupt the input (e.g., replace spans with `[MASK]`), then train the decoder to reconstruct it
3. **Decoder** predicts the original/target text token by token

**Text Infilling:** Replace multiple words with a SINGLE `[MASK]` → model must learn how many words are missing!

---

## Translation (BART/T5)

Translation = Sequence-to-sequence task → use encoder-decoder models.

**BART's Translation Approach:**
- Add a NEW randomly initialized encoder for the source language
- This encoder's output feeds into the pretrained BART encoder
- The decoder generates the target language

**mBART:** Multilingual version trained on many languages for better translation.

---

## Beyond Text: Other Modalities

Transformers aren't just for text! They work on:
-  **Audio/Speech** (Whisper)
-  **Images** (ViT, ConvNeXT)
-  **Video**

---

## Speech Recognition (Whisper)

Whisper is an **encoder-decoder** model trained on 680,000 hours of audio!

**How it works:**
1. Convert audio → **log-Mel spectrogram** (visual representation of sound)
2. **Encoder** processes the spectrogram
3. **Decoder** generates text tokens one by one

**Why Whisper is special:**
- Trained on MASSIVE diverse data from the internet
- Works "zero-shot" on many languages without fine-tuning
- Uses special tokens to switch between tasks (transcription, translation, language ID)

---

## Image Classification (ViT)

**Vision Transformer (ViT)** treats images like sentences!

**The key insight:** Split image into patches, treat each patch like a word token.

**How ViT processes images:**
1. **Split** image into 16x16 pixel patches (224x224 image → 196 patches)
2. **Flatten** each patch into a vector (patch embedding)
3. Add a learnable `[CLS]` token at the start (just like BERT!)
4. Add **position embeddings** (model doesn't know patch order otherwise)
5. Pass through Transformer encoder
6. Use `[CLS]` output for classification

**The parallel with BERT:**
| BERT | ViT |
|------|-----|
| Words → Tokens | Image Patches → Patch Embeddings |
| [CLS] for classification | [CLS] for classification |
| Position embeddings | Position embeddings |

> Transformers can process ANY sequence—whether it's words, image patches, or audio frames!


## How transformers Solve tasks
Here we'll look at the three main architectural variants of Transformer models and undersand when to use each.
 

 ### Encoder Models(BERT, DistilBERT,ModernBERT)
 only use the encoder of a tf model
 at each stage the attention layer can access all the words in the initial sentence.
 often characterized as having bi-directional attention and are often called aut-encoding moodels.
 pretraining contains deceit such as masking words and tasking the model with finding or reconstructing the initial sentence.

 Encoder models are suited for sentence classification, named entity recognition and extractive question answering(these are all tasks that require an understanding of the full sentence)

### Decoder Models (GPT2)
little less of performance
self-attention-masked self attention
Have access to one context either right or left

Used in wide variety of tasks especially text generation
Having left context they become good at text generation


Examples Huggin Face SmolLM series, deepseek's v3 and google's Gemma series.

### Modern Large Lnuae Models(LLMs)

Most llms use the decoder-only architecture.
They are typically trained in two phases
1. Pretraining: the model learns to predict the next token on vast amounts of text data

2. Instruction tuning: the model is fine-tuned to follow instructions and generate helpful responses. 

This approach has led to models that can understand and generate human-like text across a wide rang of topics and tasks.

### Key capabilities of modern LLMs
1. Text generation
2.  Summarization
3. Translation
4. Question answering
5. Code generation
6. Reasoning
7. Few-shot learning- classifying text after seeing just 2-3 examples


# Sequence to Sequence models (T5)(BART)
Use both parts of the tf architecture

at each stage, the attention layers of the encoder can access all the words in the initial sentence, whereas the attention layers of the decoder can only access the words positioned before a given word, 

Best suited for tasks revolving around generating new sentences depending on a given input, such as summarization, translation or generative question answering. 

Sequence to sequence models excel at tasks that require transforming one form of text into another while preserving the meaning
E.g Machine translation, Text Summarization, Data to text generation, grammar correction,Quenstion answering(based on text)

Representatives include BART, mBART, Marian and T5.


## Choosing the right architecture
Text classification (sentiment, topic)	Encoder	BERT, RoBERTa.
Text generation (creative writing)	Decoder	GPT, LLaMA.
Translation	Encoder-Decoder	T5, BART.
Summarization	Encoder-Decoder	BART, T5.
Named entity recognition	Encoder	BERT, RoBERTa.
Question answering (extractive)	Encoder	BERT, RoBERTa.
Question answering (generative)	Encoder-Decoder or Decoder	T5, GPT.
Conversational AI	Decoder	GPT, LLaMA.

When in doubt ask yourself:
1. what kind of understanding does my task need(Bidirectional or unidirectional)
2. Are you generating new text or analyzing existing text
3. Do you need to transform one sequence into another. 

## The Evolution of LLMs
Attention Mechanisms
LSH attention
Local attention
Axial positional encodings

## Day 6
### Inference with LLMs
Inference is the process of using a trained LLM to generate human-like text from a given input prompt.
Models predict and generate the next token in a sequence, one word at a time(Sequential Generation)

LLMs leverage probabilities from billions of parameters to generate coherent and contextually relevant text.

The attention mechanism is what gives LLMs their ability to understand context and generate coherent responses. When predicting the next word, not every word in a sentence carries equal weight. 

The attention mechanism is the key to LLMs being able to generate text that is both coherent and context-aware. It sets modern LLMs apart from previous generations of language models

## Context Length and Attention Span 
CL referes to the max no. of token(words or parts of words) that LLM can process at once. Think of it as the size f the models' working memory.

Limitiations
1. Model's architecture and size
2. Available computational resources
3. The complexity of the input and desired output. 

## The Two-Phase Inference Process
phase 1: Prefill
Phase 2: decode

### Prefill Phase
where all the initial ingredients are processed and made ready
It involves three key steps:
1. Tokenization: converting the input text into tokens
2. Embedding Conversion: Trransforming these tokens into numericcal representations that capture their meaning
3. Initial Processing: Running these embeddings through the model's neural networks to creading of the context. 

This phase is computationally intensive think of it as reading and understanding an entire paragraph before starting to write a response. 

## The Decode Phase
where the actual text generation takes place
the model generates one token at a time in what we call an autoreressive process(where each token depends on all previous tokens)

It involves several key steps  that happen for each new token
1. Attention Computation: Looking back at all previous tokens to understand the context
2. Pronbability Calculation: Determining the likelihood of each possible next token.
3. Token Selection: Choosing the next token based on these probabilities.
4. Continuation Check: Deciding whether to continue or stop generation\

PHASE is memory-intensive because the model needs to keep track of all previously generated tokens and their relationships. 

### Sampling Strategies


## Understanding Token Selection: From Probabilities to Token Choices
When the model needs to choose the next token, it starts with raw probabilities (called logits) for every word in its vocabulary. But how do we turn these probabilities into actual choices? Let’s break down the process:

1. Raw Logits: Think of these as the model’s initial gut feelings about each possible next word
2. Temperature Control: Like a creativity dial - higher settings (>1.0) make choices more random and creative, lower settings (<1.0) make them more focused and deterministic
3. Top-p (Nucleus) Sampling: Instead of considering all possible words, we only look at the most likely ones that add up to our chosen probability threshold (e.g., top 90%)
4. Top-k Filtering: An alternative approach where we only consider the k most likely next words

## Managing Repetition :Keeping Output Fresh
To avoid LLMS from repeting themselves we use two types  of penalties:
1. Presence Penalty: AA fixed penalty applied to any token that has appeared before, regardless of how often. This helps prevent the model from reusing the same words.

Frequency Penalty: A scaling penalty that increases based on how often a token has been used. The more a word appears, the less likely it is to be chosen again.

## Controlling Generation Length: Setting Boundaries
We need ways to control how much text our LLM generates.We can control generation length in several ways:
1. Token limits: setting minimum and max tokens
2. Stop Sequence: Defining specific patterns that signal the end of generation
3. End-of-sequence Detection: Letting the model naturally conclude its response.

## Beam search: Looking ahead for Better Coherence
beam searches takes a more holistic approach. Instead of committing to a single choice at each step, it explores mutliple possible paths simultaneously-like a chess player thinking several moves ahead. 

Here’s how it works:

1. At each step, maintain multiple candidate sequences (typically 5-10)
2. For each candidate, compute probabilities for the next token
3. Keep only the most promising combinations of sequences and next tokens
4. Continue this process until reaching the desired length or stop condition
5. Select the sequence with the highest overall probability

This approach often produces more coherent and grammatically correct text, though it requires more computational resources than simpler methods.

### Practical Challenges and Optimization
When working with LLMs four critical metrics will shape your implementation decisions:
1. Time to First Token(TTFT): How quickly can you get the first response?
2. Time Per Output Token(TPOT): How fast can you generate subsequent tokens
3. Throughput: How many requests can you handle simultaneously? This affects scaling and cost efficiency.
4. vram Usage: How much GPU memory do you need? 

## The Context Length Challenge
The biggest challenges in inference  is managing context lengths effectively. Longer contexts provide more information but come with substantial costs:
1. Memory Usage: Grows quadratically with context lenght
2. Processing speed: Decreases linearly with longer contexts
3. Resource Allocation: Requires careful balancing of VRAM usage.

## The KV Cache Optimization
To address these challenges, one of the most powerful optimizations is KV (Key-Value) caching. This technique significantly improves inference speed by storing and reusing intermediate calculations. This optimization:

1. Reduces repeated calculations
2. Improves generation speed
3. Makes long-context generation practical
4. The trade-off is additional memory usage, but the performance benefits usually far outweigh this cost.

## Conclusion
Understanding LLM inference is crucial for effectively deploying and optimizing these powerful models. We’ve covered the key components

The fundamental role of attention and context
The two-phase inference process
Various sampling strategies for controlling generation
Practical challenges and optimizations
By mastering these concepts, you’ll be better equipped to build applications that leverage LLMs effectively and efficiently.

Remember that the field of LLM inference is rapidly evolving, with new techniques and optimizations emerging regularly. Stay curious and keep experimenting with different approaches to find what works best for your specific use cases.

## Bias and Limitations
Whether you want to use a pretrained model or a ine-tuuned one it is important to know that all these have limitations. 

The biggest of these is that , to enable pretraining on large amounts of data, researchers often scrape all the content they can find, taking the best as well as the worst of what is available on the internet.

When you use these tools, you therefore need to keep in the back of your mind that the original model you are using could very easily generate sexist, racist, or homophobic content. Fine-tuning the model on your data won’t make this intrinsic bias disappear.
