'''

This is the code is of the original paper from Cohen et al. available at https://github.com/locuslab/smoothing/blob/master/code/core.py

'''


import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


class SmoothClassifier(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return SmoothClassifier.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]
        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return SmoothClassifier.ABSTAIN
        else:
            return top2[0]

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size
                
                batch = {}
                batch["audio"] = x["audio"].repeat((this_batch_size, 1))
                noise = torch.randn_like(batch["audio"], device='cuda') * self.sigma
                batch["audio"] = x["audio"] + noise
                predictions = self.base_classifier(batch).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts

    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]
    
    def forward(self, inputs):
        noise = torch.randn_like(inputs) * self.sigma
        return self.base_classifier((inputs + noise).clamp(-1, 1))



    
import os
from datasets.datasethandler import DatasetHandler
import torch
from attacks.cert.smoothing import SmoothClassifier
from torch.utils.data import Dataset, DataLoader
from tqdm.autonotebook import tqdm
from utils.RobustnessExperiment import load_module

def evaluate_randomized_smoothing(model_tuples, sigma = 0.5, alpha = 0.05, cert_batch_size = 100, num_samples_1 = 100, num_samples_2 = 1000, subset = None):
    
    datasetHandler = DatasetHandler()
    
    print("data-subset:", subset)
    print("Alpha", alpha)
    print("Sigma", sigma)
    print("num_samples_1", num_samples_1)
    print("num_samples_2", num_samples_2)
    
    for (model_path, model_class) in model_tuples:
        
        # load model
        model = load_module(model_path, model_class)
        model.prepare_data()
        datasetHandler.load(model, 'training')
        datasetHandler.load(model, 'validation')
        # set: not needed, only for training. because now, we have baseclassifier that does noise!!
        model.cuda()

        # create smooth classifier
        smooth_model = SmoothClassifier(base_classifier=model, sigma=sigma, num_classes=2)
        
        # prepare data
        test_dataset = model.dataset["validation"]
        if subset:
            test_dataset  = torch.utils.data.Subset(test_dataset, range(0, subset))
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        abstains = 0
        false_predictions = 0
        correct_certified = 0
        radii = []

        for batch in tqdm(iter(test_loader), total=len(test_dataset)):
            x, y = batch, batch["label"]
            x["audio"] = x["audio"].cuda()
            y = y

            top_class, radius = smooth_model.certify(x = x, n0 = num_samples_1, n = num_samples_2, 
                                               alpha = alpha, batch_size = cert_batch_size)

            if top_class == -1:
                abstains += 1
                radii.append(0)
            elif top_class == y:
                correct_certified += 1
                radii.append(radius)
            elif top_class != y:
                false_predictions += 1
                radii.append(0)

        avg_radius = torch.tensor(radii).mean().item()
        
        _, file_name = os.path.split(model_path)
        print("\n" + "="*60 + "\n", file_name)
        print("avg_radius", avg_radius)
        print("correct_certified", correct_certified)