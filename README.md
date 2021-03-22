# Pytorch Implmentation of Video-and-Language Future Event Predicition

This is the unofficial implementation of [What is More Likely to Happen Next? Video-and-Language Future Event Prediction](https://arxiv.org/abs/2010.07999)

## Environment

* Numpy
* Pytorch
* Transformers

## Configuration

Edit config.json to set configurations

## Training

```
python main.py config.json saved_model_name.pt
```

## Performance

|                           | Test set (paper) | Val set (this repo) |
| ------------------------- | ---------------- | ------------------- |
| video + future            | 59.03            | 59.27               |
| dialogue + future         | 66.63            | 68.49               |
| video + dialogue + future | 67.46            | 68.76               |
