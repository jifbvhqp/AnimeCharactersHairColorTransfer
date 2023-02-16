# AnimeCharactersHairColorTransfer

We use their StarGAN pytorch implementation to train our model https://github.com/yunjey/stargan.<br>
This project uses the super resolution model ESRGAN pretrained by https://github.com/xinntao/ESRGAN.

## Run Our Model
1. Install requirements.txt
  ~~~
  pip install -r requirements.txt
  ~~~

2. Put your input image in /inputs/ folder<br>
  ![484](https://user-images.githubusercontent.com/49235533/219293677-b4d1ae76-4241-4b57-bd59-095ed139e45f.JPG)

3. Run
  ~~~
  python inference_model.py
  ~~~
## Result
Check result in folder /results/. The first column represent the input(origin) image, the other column represent the style transfered image.<br>
![4908410984](https://user-images.githubusercontent.com/49235533/219300562-36ca7135-4d4e-4104-b151-3db80fecb1d4.JPG)



## Train your own model
Please used the dataset we provided in /dataset/ folder and clone their StarGAN pytorch implementation https://github.com/yunjey/stargan. Follow the Training instruction in their readme.
