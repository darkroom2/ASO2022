# ASO2022

Multi-View Deep Correlation

#### First steps

1. Prepare dataset

I downloaded `ModelNet10` from link below. It contains circa 5k models for 10 different object classes. For it to be
usable for our problem, the models have to be converted to png images that contain different views of the same objects.
To obtain the renders with different camera positions I used the `f3d` program. `F3D` flatpack package include all
needed libraries to support `.off` files that the `ModelNet10` contains.
After running the converter, we end up with ~10k images for two view angles.

2. Extract the features from images

TODO

3. Construct the deep corelation classifier 

TODO

#### Loose notes from the lecturer

- Obiekt dla jednej instancji posiada N widoków
- Zbudować najprostszy klasyfikator, bazujący na korelacji głębokiej (deep corelation)
- Wchodzą 2 wejścia, wyznacza się korelację pomiedzy jednym widokiem a drugim (mapuje się jedną przestrzeń cech na
  drugą)
- Wejście -> cechy -> korelacja jednych cech na drugie i z drugich na pierwsze

https://github.com/SAMY-ER/Multi-View-Image-Classification
https://towardsdatascience.com/multi-view-image-classification-427c69720f30
https://modelnet.cs.princeton.edu/#
