from attacks.attack import Attack

from torchaudio import functional as AF
import torchaudio
import torch.nn.functional as F
import math
import torch

class FunctionalTimeStretchAttack(Attack):
    
    """
        lower: lower bound for streching rate 
        upper: upper bound for streching rate

        stretching_rate > 1 means speedup
        stretching_rate < 1 means slowdown
    """
    def attackSample(self, x, y, num_iter_stretch=1, lower_stretch=0.9, upper_stretch=1.1,
                    epsilon=0, num_iter=1, norm='inf', loss_fn=F.nll_loss, lower=0.2, upper=2):
        
        x = self.time_stretch_attack(x, y, num_iter_stretch, lower_stretch, upper_stretch)
        x = self.noise_attack(x, y, epsilon, num_iter, norm, loss_fn, lower, upper)

        return x
        
    def noise_attack(self, x, y, epsilon=0, num_iter=1,
                        norm='inf', loss_fn=F.nll_loss, lower=0.2, upper=2):
        
        batch_size = x['audio'].size(0)
        clip_length = x['audio'].size(1)
        
        delta = torch.zeros(batch_size, clip_length).to(self.device)
        
        for _ in range(num_iter):
           
            delta.requires_grad_()
            
            tmp_audio = x['audio']
            x['audio'] = (x['audio'] + delta).clamp(-1, 1)
            
            loss = loss_fn(self.model(x), y)
            self.model.zero_grad()
            loss.backward()
            
            x['audio'] = tmp_audio

            if norm == "inf":
                delta = delta + epsilon * delta.grad.sign()
                delta = delta.detach()
            else:
                delta = delta + epsilon * delta.grad/delta.grad.norm(p=foat(norm), dim=[1]).unsqueeze(1)
                delta = delta.detach()
            
        x['audio'] = (x['audio'] + delta).clamp(-1, 1)
        return x

    def time_stretch_attack(self, x, y, num_iter=1, lower=0.9, upper=1.1):    

        rate_search_range = torch.arange(lower, upper, (upper-lower)/num_iter)
        losses = []
        stretched_inputs = []

        with torch.no_grad():
            x_original = x['audio']
            
            for rate in rate_search_range:
                x['audio'] = self.time_stretch(x_original.clone(), rate)
                stretched_inputs.append(x['audio'])
                losses.append(F.nll_loss(self.model(x), y, reduction='none'))
                 
        best_rates = torch.stack(losses).argmax(0)

        for x_i in range(len(best_rates)): # here is potential to further vectorize 
            x['audio'][x_i] = stretched_inputs[best_rates[x_i]][x_i].clamp(-1, 1)
        
        return x

    """
        Time stretching with padding
        
        Note this function is **not** differentiable:
        through speedup frames get deleted/inserted, 
        which makes the function not differentiable)
    """
    def time_stretch(self, batch, speedup_rate):
        if speedup_rate == 1:
            return batch
        
        n_fft = torch.tensor(2048)  # windowsize
        hop_length = torch.floor(n_fft / 4.0).int().item()

        # time stretch
        stft = torch.stft(batch, n_fft.item(), hop_length=hop_length)
        
        #phase_advance = torch.linspace(0, math.pi * hop_length, stft.shape[1])[..., None].to(self.device)
        # time stretch via phase_vocoder (not differentiable):
        #vocoded = AF.phase_vocoder(stft, rate=speedup_rate, phase_advance=phase_advance) 
        vocoded = torchaudio.TimeStretch()(stft, speedup_rate)
        istft = AF.istft(vocoded, n_fft.item(), hop_length=hop_length).squeeze()
        if batch.size(0) == 1:
            istft = istft.unsqueeze(0)
            
        return  self.pad_sample(istft, batch.size(1), speedup_rate)
    
    def pad_sample(self, istft, max_length, speedup_rate):
        if speedup_rate > 1:
            # faster means output is smaller -> padding
            pad_l = int((max_length - istft.shape[1])/2)
            pad_r = max_length - (pad_l + istft.shape[1])
            return F.pad(istft, (pad_l, pad_r))
        else:
            try:
            # slower means longer -> chopping of
                low = int((istft.shape[1] - max_length)/2)
                return istft[:,low:low+max_length]
            except IndexError:
                print(max_length)
                print(istft.shape)