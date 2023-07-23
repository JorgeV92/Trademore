
# Trademore

Trademore takes a CSV file of historical stock prices as input, where each row is assumed to be a record of stock price information, and applies time-series prediction models on it to forecast future prices. Under the hood, it leverages sequence-based prediction models from recurrent networks (RNN, LSTM, GRU) to the Transformer (as seen in GPT).

This is not meant to be too heavyweight library with a billion switches and knobs. It is one hackable file, and is mostly intended for educational purposes. [PyTorch](https://pytorch.org) is the only requirement.

Current implementation follows a few key papers:

- RNN, following [Mikolov et al. 2010](https://www.fit.vutbr.cz/research/groups/speech/publi/2010/mikolov_interspeech2010_IS100722.pdf)
- LSTM, following [Graves et al. 2014](https://arxiv.org/abs/1308.0850)
- GRU, following [Kyunghyun Cho et al. 2014](https://arxiv.org/abs/1409.1259)
- Transformer, following [Vaswani et al. 2017](https://arxiv.org/abs/1706.03762)

### Usage

The included stock_data.csv dataset, as an example, contains historical stock price data for Apple. It looks like:

```
Date	Open	High	Low	Close	Adj Close	Volume
1980-12-12	0.128348	0.128906	0.128348	0.128348	0.099584	469033600
1980-12-15	0.122210	0.122210	0.121652	0.121652	0.094388	175884800
1980-12-16	0.113281	0.113281	0.112723	0.112723	0.087461	105728000
...
```

Let's point the script at it:

```bash
$ python trademore.py -i stock_data.csv -o stock_model

```

Training progress and logs and model will all be saved to the working directory stock_model. The default model is a super tiny 200K param transformer; many more training configurations are available - see the argparse and read the code. Training does not require any special hardware, but if you have a GPU then training will be faster. As training progresses the script will print some samples throughout. However, if you'd like to forecast manually, you can use the --forecast-only flag, e.g. in a separate terminal do:

```bash
$ python trademore.py -i stock_data.csv -o stock_model --forecast-only
```

This will load the best model so far and print more samples on demand. Remember that predicting stock prices is notoriously difficult, and model predictions should not be used for actual trading without further validation.

Have fun!

### License

MIT
