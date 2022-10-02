import nltk
import whisper


def trans_audio_sentence(path):
    model = whisper.load_model("base")
    result = model.transcribe(path)
    paragraph = result["text"]
    paragraph = paragraph.lower()
    sen_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sen_tokenizer.tokenize(paragraph)
    return sentences


if __name__ == '__main__':
    sentences = trans_audio_sentence('../audio.mp3')

    print(sentences)
