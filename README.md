# keras-digits-gan
Implementation of Wasserstein GAN for handwritten digits in Keras

To run training: ``` python3 train.py ``` 

Apply -h flag for help with arguments. 

Available args:
   * --batch_size=
   * --num_iterations=
   * --z_size=
   * --d_iters=
   * --log_dir=
   * --save_dir=

To run Tensorboard: ``` tensorboard --logdir=./logs ```

Links:
   * https://myurasov.github.io/2017/09/24/wasserstein-gan-keras.html?r#wasserstein-gan
   * https://arxiv.org/pdf/1701.07875.pdf

