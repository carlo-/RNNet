# RNNet
Python implementation of a recurrent neural network.

Training is performed exclusively on CPU, and the gradient descent can work either with AdaGrad or with RMSProp.\
The source includes examples trained with [J.K. Rowling's Goblet of Fire](https://web.cs.hacettepe.edu.tr/~aykut/classes/spring2018/cmp784/goblet_book.txt) and [~32k tweets from @realDonaldTrump](https://github.com/bpb27/trump_tweet_data_archive).

## Technical details
- **Initialization:** He
- **Activation function:** Hyperbolic tangent
- **Cost function:** Sum of cross-entropy losses
- **Output:** Softmax of the scores at the final layer
- **Optimizer:** AdaGrad or RMSProp (or custom)

## Network initialization
The `RNNet` object must be initialized with two arguments, the dimentionality of the hidden state (m), and the number of total unique characters in the text.
```Python
book = dataset.load_goblet_of_fire()
net = RNNet(m=100, K=book.K)
```

## Training example
```Python
book = dataset.load_goblet_of_fire()

np.random.seed(42)
net = RNNet(m=100, K=book.K)

# optimizer = RNNet.AdaGrad(net, eta=0.1)
optimizer = RNNet.RMSProp(net, eta=0.001, gamma=0.9)

config = {
    'epochs': 10,
    'output_folder': '../out',
    'optimizer': optimizer,
    'sequence_length': 25,
    'record_interval': 1_000,
    'test_length': 200
}

res = net.train(book, config)
```

## License
This project is released under the MIT license. See `LICENSE` for more information.
