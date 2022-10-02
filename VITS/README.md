### VITS module instruction
# Prerequisites
1. download pretrained weights [here](https://drive.google.com/drive/folders/1ksarh-cJf3F5eKJjLVWY0X1j1qsQqiS2)
models weights should be put under directory ```/VITS```
2. Install espeak, for Linux OS, run ```apt-get install espeak```;
For Windows or Mac OS, please refer to respective installation instruction
3. Build Monotonic Alignment Search:
```
cd monotonic_align
python setup.py build_ext --inplace
```

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
