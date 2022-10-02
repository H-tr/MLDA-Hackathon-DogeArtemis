class VoiceDataset(Dataset):
    
    def __init__(self, audio_dir, transformation, target_sample_rate, num_samples):
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
    
    # returns the number of audio samples
    def __len__(self):
        return len(self.annotations)
    
    # fetch the audio and its label; return the processed mel spectrogram and its label
    def __getitem__(self, index):
        audio_sample_path = self.audio_dir
        signal, sample_rate = torchaudio.load(audio_sample_path)
        
        signal = self._resample(signal, sample_rate)
        signal = self._mix_down(signal)
        
        signal = self._cut(signal)
        signal = self._rightpad(signal)
        
        signal = self.transformation(signal)
        
        return signal
    
    # resample the audio
    def _resample(self, signal, sample_rate):
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, self.target_sample_rate)
            signal = resampler(signal)
            
        return signal
    
    # merge channels if the audio is stereo
    def _mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
            
        return signal
    
    # cut the audio if it has more sample than the target sample rate
    def _cut(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
            
        return signal
    
    # zero pad the audio if it is shorter than the target sample rate
    def _rightpad(self, signal):
        signal_length = signal.shape[1]
        if signal_length < self.num_samples:
            missing_samples = self.num_samples - signal_length
            padding = (0, missing_samples)
            signal = torch.nn.functional.pad(signal, padding)
            
        return signal