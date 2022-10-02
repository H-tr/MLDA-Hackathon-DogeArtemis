- [MLDA-Hackathon-DogeArtemis](#mlda-hackathon-dogeartemis)
  - [Introduction](#introduction)
  - [Dependency](#dependency)
  - [File structure](#file-structure)
  - [Usage](#usage)
    - [model](#model)
    - [QAsystem](#qasystem)
    - [VITS](#vits)
    - [whisper_step](#whisper_step)

# MLDA-Hackathon-DogeArtemis

## Introduction
Service provision by companies and the government in SG is shifting towards higher efficiency powered by automation.
However, assistance for managing problems from these services - customer/user support, is not.
Customer support in SG is time and effort consuming; thus, transformation is necessary.

Text-based automated support agent: user inputs general commands; the chatbot sends relevant responses.
Voice-based automated support agent:
- The agent requests the user to input option numbers using the keypad; sends relevant responses.
- The agent requests the user to input the command via voice; sends relevant responses.

Live agent support: a human agent connects with the user; provides the necessary support.
*Common limitation: regardless of the support each user demands, everyone should go through the identical process until the automated agent correctly labels the type of user support necessary. This is time and effort consuming to both the user and the service provider.

Eliminating the “waiting time” users have to spend to reach support; while providing the support relevant to their sentiment.

Our solution mainly targets the field of voice-based automated support.
The automated agent immediately asks for the user’s inquiry; while the user speaks out to the agent, the algorithm analyses the sentiment behind the user’s voice and judges urgency.
Depending on user sentiment, a relevant support will be provided.
If the algorithm judges the user would be willing to handle with an automated support agent, relevant support will be provided by an agent powered by a natural language generator model.
However, if the user is in a negative sentiment, direct connection to a human support agent will be given.

## Dependency
- Python=3.9.13
- ffmpeg
- rust

## File structure

```
+-- models
+-- QAsystem
+-- VITS
+-- whisper_step
```

## Usage

### model
### QAsystem
To use the Q&A system, first import the class
```python
import QAsystem from QAsystem
```
Then create an instance for the class
```python
query = "What is Covid-19"
system = QAsystem()
answer = system.generate_answer(query)
```
Put the query which is in string type into the `generate_answer` function, we can get the result of this system.
### VITS
### whisper_step
### Audio Sentiment Classification
To make a sentiment classification of the input audio, access the code files in the directory /models/model-predictions.
Import the relevant module:
```python
from model_predictions import return_predictions
```
Then configure the respective parameters:
<br>```model_path```: FILE PATH TO THE .pth MODEL.
<br>```audio_path```: FILE PATH TO THE .wav AUDIO FILE.
<br>```sample_rate```: ```int``` VALUE; HOW MANY TIMES PER SECOND A SOUND IS SAMPLED; DEFAULT ```22050```.
<br>```num_samples```: ```int``` VALUE; MAXIMUM AUDIO LENGTH, NUMBER OF SAMPLES ALLOWED; DEFAULT ```22050```.

<br>Input the parameters; then execute the ```return_predictions()``` function.
```python
prediction = return_predictions(model_path, audio_path, sample_rate, num_samples)
```
