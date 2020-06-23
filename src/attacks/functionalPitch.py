from attacks.attack import Attack

import torch.nn.functional as F
import torch

class FunctionalVolumeAttack(Attack):

    '''
        - Epsilon step in gradient direction (step size defined by norm)
        - Fully vectorized 
        - For num_iter=1 and norm='inf': Degrades to Fast Gradient Sign Method (FGSM)
    '''
    def attackSample(self, x, y, epsilonVolume=0, epsilonPGD=0, num_iter=1,
                        norm='inf', loss_fn=F.nll_loss, lower=0.2, upper=2):
        
        batch_size = x['audio'].size(0)
        clip_length = x['audio'].size(1)
        
        a = torch.ones(batch_size).unsqueeze(1).to(self.device)
        delta = torch.zeros(batch_size, clip_length).to(self.device)
        
        for _ in range(num_iter):
           
            a.requires_grad_()
            delta.requires_grad_()
            
            tmp_audio = x['audio']
            x['audio'] = (a * x['audio'] + delta).clamp(-1, 1)
            
            loss = loss_fn(self.model(x), y)
            self.model.zero_grad()
            loss.backward()
            
            x['audio'] = tmp_audio

            if norm == "inf":
                a = a + epsilonVolume * a.grad.sign()
                a = a.clamp(lower, upper).detach()
                
                delta = delta + epsilonPGD * delta.grad.sign()
                delta = delta.detach()
            else:
                a = a + epsilonVolume * a.grad/a.grad.norm(p=foat(norm), dim=[1]).unsqueeze(1)
                a = a.clamp(lower, upper).detach()
                delta = delta + epsilonPGD * delta.grad/delta.grad.norm(p=foat(norm), dim=[1]).unsqueeze(1)
                delta = delta.detach()
            
        x['audio'] = (a * x['audio'] + delta).clamp(-1, 1)
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