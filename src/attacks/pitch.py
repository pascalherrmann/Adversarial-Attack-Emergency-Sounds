from attacks.attack import Attack

from torchaudio import functional as AF
import torch.nn.functional as F
import torchaudio
import math
import torch
import librosa
import numpy as np

class PitchAttack(Attack):

    def attackSample(self, x, y, num_iter=1, lower=-1, upper=1):
        n_steps_search_range = torch.arange(lower, upper, (upper-lower)/num_iter)
        losses = []
        stretched_inputs = []
        
        with torch.no_grad():
            x_original = x['audio']
            
            for n_steps in n_steps_search_range:
                # we expect full batch to have the same sample rate here
                x['audio'] = self.pitch_shift(x_original.clone(), sr=x['sample_rate'][0], n_steps=n_steps)
                stretched_inputs.append(x['audio'])
                losses.append(F.nll_loss(self.model(x), y, reduction='none'))
                 
        best_rates = torch.stack(losses).argmax(0)

        for x_i in range(len(best_rates)): # here is potential to further vectorize 
            x['audio'][x_i] = stretched_inputs[best_rates[x_i]][x_i].clamp(-1, 1)
        return x
    
    def pitch_shift(self, batch, sr, n_steps, bins_per_octave=12):
        # https://librosa.github.io/librosa/_modules/librosa/effects.html#pitch_shift
        assert bins_per_octave > 1 or not np.issubdtype(type(bins_per_octave), np.integer)

        rate = 2.0 ** (-float(n_steps) / bins_per_octave)

        # Stretch in time, then resample
        batch_shifted = torchaudio.transforms.Resample(float(sr)/rate, sr)(self.time_stretch(batch, rate))

        # Crop to the same dimension as the input
        return self.pad_sample(batch_shifted, batch.size(1))
    
    def time_stretch(self, batch, speedup_rate, device="cuda"):
        if speedup_rate == 1:
            return batch

        n_fft = torch.tensor(2048)  # windowsize
        hop_length = torch.floor(n_fft / 4.0).int().item()

        # time stretch
        stft = torch.stft(batch, n_fft.item(), hop_length=hop_length)
        
        phase_advance = torch.linspace(0, math.pi * hop_length, stft.shape[1])[..., None].to(device)
        # time stretch via phase_vocoder (not differentiable):
        vocoded = AF.phase_vocoder(stft, rate=speedup_rate, phase_advance=phase_advance) 
        istft = AF.istft(vocoded, n_fft.item(), hop_length=hop_length).squeeze()

        return istft
    
    # this method could be further vectorized
    def pad_sample(self, batch, max_length):
        if batch.size(1) < max_length:
            # padding
            pad_l = int((max_length - batch.shape[1])/2)
            pad_r = max_length - (pad_l + batch.shape[1])
            return F.pad(batch, (pad_l, pad_r))
        else:
            # chopping of
            low = int((batch.shape[1] - max_length)/2)
            return batch[:,low:low+max_length]