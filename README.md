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

TODO: 
```
!tree .
```

## Contact

For questions and feedback please contact:

- Pascal Herrmann (pascal.herrmann@tum.de)
- Yan Scholten (yan.scholten@tum.de)
