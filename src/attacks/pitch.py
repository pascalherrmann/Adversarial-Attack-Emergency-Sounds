from attacks.attack import Attack

from torchaudio import functional as AF
import torch.nn.functional as F
import torchaudio
import math
import torch
import librosa

class PitchAttack(Attack):

    def attackSample(self, x, y, num_iter=1, lower=-1, upper=1):
        raise Exception("Not vectorized - TODO")
        n_steps_search_range = torch.arange(lower, upper, (upper-lower)/num_iter)
        losses = []
        stretched_inputs = []
        
        with torch.no_grad():
            for n_steps in n_steps_search_range:
                stretched = librosa.effects.pitch_shift(x['audio'].squeeze().cpu().numpy(),
                                                        sr=x['sample_rate'], n_steps=n_steps)
                stretched = torch.tensor(stretched).unsqueeze(0).cuda()
                stretched_inputs.append(stretched)
                x_pert = {'audio': stretched, 'sample_rate': x['sample_rate']}
                losses.append(F.nll_loss(self.model(x_pert), y))
        best_rate = torch.stack(losses).argmax().item()

        x_pert = {'audio': stretched_inputs[best_rate].clamp(-1, 1)}
        x_pert['sample_rate'] = x['sample_rate']
        return x_pert