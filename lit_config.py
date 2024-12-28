class Config:

    class Model:
        model = 'tcm'  # choose one from compressai.zoo.image_models
        quality = 'L'
        lmbda = 0.013
        beta = None

    class Data:
        dataset_directory = '/home/yichi/Project/dataset'

        train_batch_size = 8
        valid_batch_size = 1

        train_patch_size = (256, 256)
        valid_patch_size = (256, 256)

        num_workers = 16
        pin_memory = True
        persistent_workers = True

    class Trainer:
        train_batch_size = 8
        accelerator = 'gpu'
        devices = [1,0]  # list of devices to train on
        num_epochs = 200
        seed = 1234
        net_lr = 0.0001*len(devices)*train_batch_size/16

        validation_cadence = 1  # [epochs]
        gradient_clip_norm = 2.0  # necessary for gradient-exploding-free training

        ckpt_path ="/home/yichi/Project/Lightning/ckpt/tcm-lambda=0.013-beta=None/lightning_logs/version_0/checkpoints/epoch=2-loss=1.6389-loss_ema=0.0000-last.ckpt"  # 'path/to/a/checkpoint'
        log_cadence = 25  # [steps] Note: global_steps=accumulated total number of calling step() for any optimizer
                          # but this value gets compared to the number of training_steps which is counted only once for
                          # each usage of training_step
        
        strategy = 'auto'  # "ddp_find_unused_parameters_true" or "auto"
        float32_matmul_precision = 'high'
        # precision = '16-true'
        compile = True
        ema = True
        
        
