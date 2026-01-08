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