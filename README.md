# Video-and-Language Future Event Predicition in PyTorch

This is an unofficial implementation of [What is More Likely to Happen Next? Video-and-Language Future Event Prediction](https://arxiv.org/abs/2010.07999)

## Environment

* Numpy
* PyTorch
* Transformers

## Configuration

Edit config.json to set configurations

| Key                         | value                                                        |
| --------------------------- | ------------------------------------------------------------ |
| `vid_feature`               | Video feature in 1 fps, each in a .npy file                  |
| `subtitles`                 | A preprocessed dictionary-like subtitles json file,<br />with video as subtitles and subtitles as value,<br />provided in `data/vlep_subtitles.json` |
| `train_anno`                | Annotations in the training split                            |
| `val_anno`                  | Annotations in the validation split                          |
| `test_anno`                 | Annotations in the test split                                |
| `train_dialogue`            | Whether dialogues are used.<br />This only affects data loading, and does not change the model architecture. |
| `nhead`                     | The number of heads for all transformer-like modules         |
| `base_epochs` and `base_lr` | The number of training epochs and learning rate before fine-tuning RoBERT |
| `fine_epochs` and `fine_lr` | The number of training epochs and learning rate to fine-tune RoBERTa |
| `grad_accum`                | The number of iterations for gradient accumulation           |

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
