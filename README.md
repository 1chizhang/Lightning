
### Installation
In a virtual environment follow the steps below (verified on Ubuntu):
```bash
git clone https://github.com/1chizhang/Lightning
cd Lightning
conda create -n Lightning python=3.11
pip install -U pip
pip install -e .
pip install lightning
pip install tensorboard
```


### Training
All the configurations regarding dataloader, training strategy, and etc should be set in the `lit_config.py` followed by the command:
```bash
python lit_train.py --comment "ema ddp compile full precision TCM-L MLIC-Train-100k"
```

### Resume Training
To resume training from a checkpoint, set the `ckpt_path` in `lit_config.py`:
```python
class Trainer:
    ckpt_path = "ckpt/tcm-lambda=0.013-beta=None/lightning_logs/version_0/checkpoints/epoch=2-loss=1.6389-last.ckpt"
```
Then run the same training command. The checkpoint contains model weights, optimizer state, scheduler state, and EMA weights (if enabled).


### Evaluation
To evaluate a saved checkpoint of a model, `compressai.utils.eval` is used. An example to test the rate-distoriton perfomance of a SwinT-ChARM checkpoint:

```bash
python -m compressai.utils.eval_model checkpoint /home/yichi/Project/dataset/kodak  -a tcm --cuda -v -p /home/yichi/Project/Lightning/ckpt/tcm-lambda=0.013-beta=None/lightning_logs/Flickr30k/version_2/checkpoints/epoch=263-loss=1.6261-last.ckpt

```



## Code Structure
This unofficial PyTorch implementation follows the [CompressAI](https://github.com/InterDigitalInc/CompressAI) code structure and then is wrapped by the [Lightning](https://github.com/Lightning-AI/lightning) framework. Tensorflow implementation of [SwinT-ChARM](https://github.com/Nikolai10/SwinT-ChARM) is used as the reference.

The design paradigm of [CompressAI](https://github.com/InterDigitalInc/CompressAI) is closely followed which results to modifications/additions in the following directories. [Lightning](https://github.com/Lightning-AI/lightning)-based python files are also shown below:
```
|---compressai
|    |---losses
|    |    ├───rate_distortion.py       rate-disortion loss
|    |---layers
|    |    ├───swin.py                  blocks needed by TBTC models
|    |---models
|    |    ├───qualcomm.py              TBTC models
|    |---zoo
|         ├───image.py                 model creation based on config
|
├───lit_config.py                      configuration file
├───lit_data.py                        lighting data-module   
├───lit_model.py                       lightning module
├───lit_train.py                       main script to start/resume training
```
