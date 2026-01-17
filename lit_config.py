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
        accelerator = 'gpu'
        devices = [1,0]  # list of devices to train on
        num_epochs = 320
        seed = 1234
        train_batch_size = 8  # Must match Data.train_batch_size for correct LR scaling
        net_lr = 0.0001*len(devices)*train_batch_size/16  # Linear LR scaling: base_lr * (global_bs / reference_bs)

        validation_cadence = 1  # [epochs]
        gradient_clip_norm = 5.0  # necessary for gradient-exploding-free training

        # ckpt_path = "path/to/checkpoint.ckpt"  # Uncomment and set path to resume training
        log_cadence = 25  # [steps] Note: global_steps=accumulated total number of calling step() for any optimizer
                          # but this value gets compared to the number of training_steps which is counted only once for
                          # each usage of training_step
        
        strategy = 'auto'  # "ddp_find_unused_parameters_true" or "auto"
        float32_matmul_precision = 'high'
        # precision = '16-true' #enable mixed precision training, not tested, do not use without validation
        compile = True
        ema = True
        
        
