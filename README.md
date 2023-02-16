# AnimeCharactersHairColorTransfer

Our Model is trained by their StarGAN pytorch implementation https://github.com/yunjey/stargan

## Run Our Model

1. Install requirements.txt
  ~~~
  pip install -r requirements.txt
  ~~~
<br>
2. Put your input image in /inputs/ folder<br>
  ![484](https://user-images.githubusercontent.com/49235533/219293677-b4d1ae76-4241-4b57-bd59-095ed139e45f.JPG)

3. Run
  ~~~
  python inference_model.py
  ~~~
## Train your own model
Please used the dataset we provided in /dataset/ folder and the clone StarGAN pytorch implementation https://github.com/yunjey/stargan. Follow the Training instruction in their readme.
