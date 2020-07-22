# Project 4: Adversarial Examples for Emergency Vehicle Detection

In recent years, the robustness of machine learning models has been addressed in many research works [1, 2, 3, 4, 5, 6, 7]. Robust models are crucial for many safety-critical applications, where small changes in data should not affect the prediction of machine learning models. One safety-critical task is the detection of other vehicles in autonomous driving, and detecting emergency vehicles is often considered as one of the key challenges concerning safety guarantees on the road. While various approaches exist to detect emergency vehicles in traffic situations [8], no works address the robustness of sound-wave based emergency vehicle detection systems so far.

## Main Contributions

In this project we (1) compile a dataset for emergency vehicle classification, (2) implement state-of-the-art end-to-end machine learning approaches for audio classification and fine-tune them on our emergency vehicle detection dataset, (3) develop adversarial attacks against end-to-end audio classification models, (4) evaluate the robustness of our models, and (5) improve their robustness by deploying various adversarial training techniques. The following list gives a more detailed overview about our contributions:

- Dataset construction
- Implementation and fine-tuning of 4 end-to-end models 
- Development of adversarial attacks 
    - Standard noise attacks 
    - Sound property attacks
    - Functional attacks 
- Robustness analysis using adversarial attacks 
- Robustness enhancement via adversarial training

To the best of our knowledge, we are the first to describe sound property attacks and functional attacks against end-to-end sound classifiers.

## Project Structure

```
├── notebooks
│   ├── datasets
│   ├── evaluations
│   ├── utils
│   ├── attack_exploration.ipynb
│   ├── attacks-usage.ipynb
│   ├── experiments.ipynb
│   ├── randomized_smoothing.ipynb
│   └── wiki_plots.ipynb
└── src
    ├── datasets
    ├── classification
    ├── attacks
    ├── utils
    └── config.py
```

## Contact

For questions and feedback please contact:

- Pascal Herrmann (pascal.herrmann@tum.de)
- Yan Scholten (yan.scholten@tum.de)

## References

[1] Goodfellow, Ian J., Jonathon Shlens, and Christian Szegedy. "Explaining and harnessing adversarial examples." arXiv preprint arXiv:1412.6572 (2014).

[2] Szegedy, Christian, et al. "Intriguing properties of neural networks." arXiv preprint arXiv:1312.6199 (2013).

[3] Madry, Aleksander, et al. "Towards deep learning models resistant to adversarial attacks." arXiv preprint arXiv:1706.06083 (2017).

[4] Huang, Sandy, et al. "Adversarial attacks on neural network policies." arXiv preprint arXiv:1702.02284 (2017).

[5] Kurakin, Alexey, et al. "Adversarial attacks and defences competition." The NIPS'17 Competition: Building Intelligent Systems. Springer, Cham, 2018. 195-231.

[6] Yakura, Hiromu, and Jun Sakuma. "Robust audio adversarial example for a physical attack." arXiv preprint arXiv:1810.11793 (2018).

[7] Subramanian, Vinod, et al. "Adversarial attacks in sound event classification." arXiv preprint arXiv:1907.02477 (2019).

[8] Ebizuka, Yuki, Shin Kato, and Makoto Itami. "Detecting approach of emergency vehicles using siren sound processing." 2019 IEEE Intelligent Transportation Systems Conference (ITSC). IEEE, 2019
