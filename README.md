# PyTorch implementation of [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781)

To train the model on MovieLens-1M (with default hyper-parameters):
~~~bash
python3 main.py
~~~

This implementation supports training with different hyperparameters. Refer to [opts.py](../blob/master/opts.py) for more details. An example varying the attention stack (b=2), length of sequence (n=50) and hidden dimension (d=50):
~~~bash
python3 main.py --b=2 --n=50 --d=50 --n_epochs=10 --lr=0.001
~~~

