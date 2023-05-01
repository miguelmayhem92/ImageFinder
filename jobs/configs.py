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
    model_ckpt = 'google/vit-base-patch16-224-in21k'
    run_id = 'baa131b56b21444094a8e589940b1ac0'
    image_path = 'tmp_image/'
    embedding_db_path = 'embedding_db/'
