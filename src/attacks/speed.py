from attacks.attack import Attack

from torchaudio import functional as AF
import torch.nn.functional as F
import torch

class TimeStretchAttack(Attack):
    
    """
        lower: lower bound for streching rate 
        upper: upper bound for streching rate

        stretching_rate > 1 means speedup
        stretching_rate > 1 means slowdown
    """
    def attackSample(self, x, y, epsilon=0, num_iter=1, lower=0.9, upper=1.1):
        rate_search_range = torch.arange(lower, upper, float(1)/num_iter)
        losses = []
        stretched_inputs = []

        with torch.no_grad():
            for rate in rate_search_range:
                stretched = self.time_stretch(x.squeeze().cpu(), rate)
                stretched = stretched.cuda().unsqueeze(0)
            stretched_inputs.append(stretched)
            losses.append(F.nll_loss(self.model(stretched), y))
        best_rate = torch.stack(losses).argmax().item()
        return stretched_inputs[best_rate]

    """
        Time stretching with padding
        
        Note this function is **not** differentiable:
        through speedup frames get deleted/inserted, 
        which makes the function not differentiable)
    """
    def time_stretch(self, sample, speedup_rate):
        if speedup_rate == 1:
            return sample

        n_fft = torch.tensor(2048)  # windowsize
        hop_length = torch.floor(n_fft / 4)

        # time stretch
        stft = torch.stft(sample, n_fft, hop_length=hop_length, window=window).unsqueeze(0)
        phase_advance = torch.linspace(0, math.pi * hop_length, stft.shape[1])[..., None]
        # time stretch via phase_vocoder (not differentiable):
        vocoded = AF.phase_vocoder(stft, rate=speedup_rate, phase_advance=phase_advance) 
        istft = AF.istft(vocoded, n_fft, hop_length=hop_length, window=window).squeeze()

        # padding
        max_length = sample.shape[0]
        if speedup_rate > 1:
            # faster means output is smaller -> padding
            pad_l = int((max_length - istft.shape[0])/2)
            pad_r = max_length - (pad_l + istft.shape[0])
            return F.pad(istft, (pad_l, pad_r))
        else:
            # slower means longer -> chopping of
            low = int((istft.shape[0] - max_length)/2)
            return istft[low:low+max_length]