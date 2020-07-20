def config():
    cfg = {
        # raw csv data
        'train_csv_path': ...,
        'test_csv_path': ...,
        # images path
        'train_img_root': ...,
        'test_img_root': ...,
        
        # parameters
        'random_state': 123,
        'test_split': 0.1,
        'image_size': 800,
        'batch_size':4,
        'test_batch_size':2,
        'num_classes': ..., # K + 1 clasess and other background 
        'num_epochs': 200, 
        'logdir': ...,
        'device': None,
        'verbose': True,
        'check': False,
        'class_name': ..., # K class names
        'lr': 0.001,
    }
    return cfg