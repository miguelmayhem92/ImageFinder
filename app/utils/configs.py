class configs():
    ## configs for csv treatment
    proj_name = 'ImageFinder'
    raw_data_interview = 'data-interview.csv'
    clean_data_interview = 'data-interview-clean.csv'
    csv_file = 'extract_data/'
    ## configs for dataset preparation
    n_train_images = 200
    seed_data = 123
    num_samples = 190
    seed = 42
    ## ML configs
    show_top = 3
    model_ckpt = 'google/vit-base-patch16-224-in21k'
    run_id = 'fcf2d0d7273247e8a95e9db9ef58e133'
    image_path = '/tmp_image/'
    embedding_db_path = '/embedding_db/'
