from attacks.attack import Attack

import torch
import torch.nn.functional as F

class ProjectedGradientDescent(Attack):
    
    '''
        - Epsilon step in gradient direction (step size defined by norm)
        - Fully vectorized 
        - For num_iter=1 and norm='inf': Degrades to Fast Gradient Sign Method (FGSM)
    '''

    def attackSample(self, x, y, epsilon=0, num_iter=1, norm='inf', loss_fn=F.nll_loss):
        if epsilon == 0:
            return x

        for _ in range(num_iter):
            x['audio'].requires_grad_()

            loss = loss_fn(self.model(x), y)
            self.model.zero_grad()
            loss.backward()

            if norm == "inf":
                x['audio'] = x['audio'] + epsilon * x['audio'].grad.sign()
            else:
                normed_grad = x['audio'].grad.norm(p=float(norm), dim=[1]).unsqueeze(1)
                x['audio'] = x['audio'] + epsilon * x['audio'].grad/normed_grad

            # projection in case epsilon is too large
            x['audio'] = x['audio'].clamp(-1, 1).detach()
        return x
    


class PGD_Real(Attack):

    #
    # based on https://gist.github.com/oscarknagg/45b187c236c6262b1c4bbe2d0920ded6
    #
    
    def attackSample(self, x, y, epsilon, norm, step_size=0.2, step_norm="inf", loss_fn=F.nll_loss, num_iter=1,
                               clamp=(0,1), y_target=None):
        x_dict = x
        x = x["audio"]
        """Performs the projected gradient descent attack on a batch of images."""
        x_adv = x.clone().detach().requires_grad_(True).to(x.device)
        targeted = y_target is not None
        num_channels = x.shape[1]

        for i in range(num_iter):
            _x_adv = x_adv.clone().detach().requires_grad_(True)

            prediction = self.model({"audio":_x_adv})
            loss = loss_fn(prediction, y_target if targeted else y)
            loss.backward()

            with torch.no_grad():
                # Force the gradient step to be a fixed size in a certain norm
                if step_norm == 'inf':
                    gradients = _x_adv.grad.sign() * step_size
                else:
                    
                    gradients = _x_adv.grad * step_size / _x_adv.grad.view(_x_adv.shape[0], -1)\
                        .norm(int(step_norm), dim=-1)\
                        .view(_x_adv.shape[0], 1)

                if targeted:
                    # Targeted: Gradient descent with on the loss of the (incorrect) target label
                    # w.r.t. the image data
                    x_adv -= gradients
                else:
                    # Untargeted: Gradient ascent on the loss of the correct label w.r.t.
                    # the model parameters
                    x_adv += gradients

            # Project back into l_norm ball and correct range
            if norm == 'inf':
                # Workaround as PyTorch doesn't have elementwise clip
                x_adv = torch.max(torch.min(x_adv, x + epsilon), x - epsilon)
            else:
                delta = x_adv - x

                # Assume x and x_adv are batched tensors where the first dimension is
                # a batch dimension
                mask = delta.view(delta.shape[0], -1).norm(int(norm), dim=1) <= epsilon

                scaling_factor = delta.view(delta.shape[0], -1).norm(int(norm), dim=1)
                scaling_factor[mask] = epsilon

                # .view() assumes batched images as a 4D Tensor
                delta *= epsilon / scaling_factor.view(_x_adv.shape[0], 1)

                x_adv = x + delta

            x_adv = x_adv.clamp(*clamp)
        x_dict["audio"] = x_adv.detach()
        #print(x_dict)
        return x_dict