from attacks.attack import Attack

from torchaudio import functional as AF
import torch.nn.functional as F
import torchaudio
import math
import torch

class PitchAttack(Attack):

    def attackSample(self, x, y, num_iter=1, lower=1, upper=5):
        n_steps_search_range = torch.arange(lower, upper, (upper-lower)/num_iter)
        losses = []
        stretched_inputs = []
        
        with torch.no_grad():
            for n_steps in n_steps_search_range:
                stretched = self.pitch_shift(x[0].squeeze(), sr=x[1], n_steps=n_steps)
                stretched = stretched.unsqueeze(0)
                stretched_inputs.append(stretched)
                losses.append(F.nll_loss(self.model([stretched, x[1]]), y))
        best_rate = torch.stack(losses).argmax().item()
        return stretched_inputs[best_rate].clamp(-1,1), x[1]

    def pitch_shift(self, sample, sr, n_steps, bins_per_octave=12): 
        # https://librosa.github.io/librosa/_modules/librosa/effects.html#pitch_shift
        assert bins_per_octave >= 1
        rate = 2.0 ** (-float(n_steps) / bins_per_octave)

        # Stretch in time, then resample, compare librosa
        resample = torchaudio.transforms.Resample(float(sr.cpu())/rate, sr.cpu()).cpu()
        y_shift = resample(self.time_stretch(sample, rate).cpu()).cuda() # not diff'able
        
        # back to original size
        max_length = sample.shape[0]
        low = int((y_shift.shape[0] - max_length)/2)
        return y_shift[low:low+max_length]

    """
        Time stretching *without* padding
        
        Note this function is **not** differentiable:
        through speedup frames get deleted/inserted, 
        which makes the function not differentiable)
    """
    def time_stretch(self, sample, speedup_rate):
        if speedup_rate == 1:
            return sample

        n_fft = torch.tensor(2048)  # windowsize
        hop_length = torch.floor(n_fft / 4.0).int().item()

        # time stretch
        stft = torch.stft(sample, n_fft.item(), hop_length=hop_length).unsqueeze(0)
        phase_advance = torch.linspace(0, math.pi * hop_length, stft.shape[1])[..., None].cuda()
        # time stretch via phase_vocoder (not differentiable):
        vocoded = AF.phase_vocoder(stft, rate=speedup_rate, phase_advance=phase_advance) 
        return AF.istft(vocoded, n_fft.item(), hop_length=hop_length).squeeze()