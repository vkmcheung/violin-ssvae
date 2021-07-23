# Generating violin fingerings with semi-supervision using VAEs
For every note they play, violinists need to decide on which string to play the note, which position to place their hands, and with which finger to press. However, these decisions are often non-trivial as they depend on factors such as stylistic nuances and playability. Although existing models can generate fingerings by training over annotated datasets, such data are rare. We overcome this problem with a varational encoder(VAE)-based model that allows training via semi-supervision. Our proposed model generates high-quality violin fingerings that matches the state-of-the-art with only half the amount of training data required.

Please read our paper "Semi-supervised Violin Fingering Generation Using Variational Autoencoders" presented at ISMIR 2021 for more details.

Here you can find the code to our model and supplementary information to our paper.

## Requirements
- python >= 3.7
- tensorflow/keras >= 2.4.1
- numpy 1.17?
