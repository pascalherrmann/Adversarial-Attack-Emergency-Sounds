import Attack

class FGSM(Attack):
    
    def attackSample(self, model, x, y, epsilon, num_iter=1):
        
        x.requires_grad = True
        
        loss = F.nll_loss(model(x), y)
        model.zero_grad()
        loss.backward()
        
        perturbed_sample = x + epsilon * x.grad.data.sign()
        return perturbed_sample

if __name__ == '__main__':
    pass