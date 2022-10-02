### VITS module instruction
To use the VITS text-to-speech(TTS) conversion, import the VITS class:
```
from VITS.inference import VITS_TTS_converter
```
Create an instance of the class:
```
vits = VITS_TTS_converter(model='ljs', device='cpu')
```
where ```model``` could be ```ljs``` (single speaker) or ```vctk```(multiple speaker)

To run inference:
```
audio = vits.infer(sentence:str, sid=None)
```
where ```sentence``` is the text you would like to convert to speech, ```sid``` is the ID of the speaker you would like to use (only support for ```vctk```model, which has 109 speakers)
the output ```audio``` is an 1-D ```numpy.ndarray```