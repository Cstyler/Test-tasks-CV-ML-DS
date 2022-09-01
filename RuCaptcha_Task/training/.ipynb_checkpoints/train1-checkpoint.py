import psutil

from grcnn.processing import filter_bad_samples
from grcnn.training import keras, train
from grcnn.utils import DATASET_DIR, find_best_model_epoch, find_model_path


def main():
    batch_size = 64
    model_num = 12
    initial_epoch = 0
    width, height = 64, 32
    max_text_len = 17
    n_classes = 10
    epochs = 200
    grcl_niter = 3
    grcl_fsize = 3
    lstm_units = 512
    train_augment_prob = .5
    lr = .0001
    optimizer = keras.optimizers.Adam(lr)
    pretrain_model_num = 11
    pretrain_model_path = find_model_path(pretrain_model_num, find_best_model_epoch(pretrain_model_num))
    img_dir = f'images_train_padding_zero_size{width}x{height}'
    train_df_name = f'train_set_len{max_text_len}'
    val_df_name = f'val_set_len{max_text_len}'
    train(train_df_name, val_df_name, img_dir,
          model_num, initial_epoch, epochs, train_augment_prob,
          batch_size, pretrain_model_path, optimizer, width, height, max_text_len, n_classes,
          grcl_niter, grcl_fsize, lstm_units)

    df_name = f'train_set_len{max_text_len}'
    save_df_name = f'train_set_len{max_text_len}_filtered1'
    model_num = 12
    batch_size = 128
    image_dir = f'images_train_padding_zero_size{width}x{height}'
    epoch = find_best_model_epoch(model_num)
    filter_bad_samples(DATASET_DIR, df_name, save_df_name, model_num, epoch, max_text_len, batch_size, image_dir)

    batch_size = 64
    model_num = 13
    initial_epoch = 0
    width, height = 64, 32
    max_text_len = 17
    n_classes = 10
    # loss_weights = (1., .02)
    epochs = 200
    grcl_niter = 3
    grcl_fsize = 3
    lstm_units = 512
    train_augment_prob = .5
    lr = .0001
    optimizer = keras.optimizers.Adam(lr)
    pretrain_model_num = 12
    pretrain_model_path = find_model_path(pretrain_model_num, find_best_model_epoch(pretrain_model_num))
    img_dir = f'images_train_padding_zero_size{width}x{height}'
    train_df_name = f'train_set_len{max_text_len}_filtered1'
    val_df_name = f'val_set_len{max_text_len}'
    train(train_df_name, val_df_name, img_dir,
          model_num, initial_epoch, epochs, train_augment_prob,
          batch_size, pretrain_model_path, optimizer, width, height, max_text_len, n_classes,
          grcl_niter, grcl_fsize, lstm_units)

def kill_process(process_id):
    try:
        p = psutil.Process(process_id)
    except psutil.NoSuchProcess:
        return
    p.kill()


if __name__ == '__main__':
    main()
