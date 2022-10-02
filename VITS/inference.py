import matplotlib.pyplot as plt
import IPython.display as ipd

import os
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from data_utils import TextAudioLoader, TextAudioCollate, TextAudioSpeakerLoader, TextAudioSpeakerCollate
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence

from scipy.io.wavfile import write

dir="C:/Program Files (x86)/eSpeak/command_line/espeak.exe"
os.environ['PHONEMIZER_ESPEAK_PATH']=dir

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

class VITS_TTS_converter():
    def __init__(self, model='ljs', device='cpu'):
        self.model = model
        self.device = device
        self.hps = utils.get_hparams_from_file(f"./configs/{model}_base.json")
        self.net_g = SynthesizerTrn(
                len(symbols),
                self.hps.data.filter_length // 2 + 1,
                self.hps.train.segment_size // self.hps.data.hop_length,
                n_speakers=self.hps.data.n_speakers,
                **self.hps.model).to(device)
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(f"pretrained_{model}.pth", self.net_g, None)

    def infer(self, sentence: str, sid=None):
        stn_tst = get_text(sentence, self.hps)
        with torch.no_grad():
            x_tst = stn_tst.to(self.device).unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(self.device)
            sid = torch.LongTensor([sid]).to(self.device) if self.model=='vctk' else None
            audio = self.net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0, 0].data.cpu().float().numpy()
        return audio

if __name__ == "__main__":
    # #%%---------------LJ Speech------------------
    # hps = utils.get_hparams_from_file("./configs/ljs_base.json")
    #
    # net_g = SynthesizerTrn(
    #     len(symbols),
    #     hps.data.filter_length // 2 + 1,
    #     hps.train.segment_size // hps.data.hop_length,
    #     **hps.model).cuda()
    # _ = net_g.eval()
    #
    # _ = utils.load_checkpoint("./pretrained_ljs.pth", net_g, None)
    #
    # stn_tst = get_text("VITS is Awesome!", hps)
    # with torch.no_grad():
    #     x_tst = stn_tst.cuda().unsqueeze(0)
    #     x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
    #     audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    # ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))

    #%%-------------VCTK------------------------
    hps = utils.get_hparams_from_file("./configs/vctk_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).cuda()
    _ = net_g.eval()

    _ = utils.load_checkpoint("pretrained_vctk.pth", net_g, None)

    stn_tst = get_text("VITS is Awesome!", hps)
    with torch.no_grad():
        x_tst = stn_tst.cuda().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cuda()
        sid = torch.LongTensor([4]).cuda()
        audio = net_g.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
    ipd.display(ipd.Audio(audio, rate=hps.data.sampling_rate, normalize=False))




