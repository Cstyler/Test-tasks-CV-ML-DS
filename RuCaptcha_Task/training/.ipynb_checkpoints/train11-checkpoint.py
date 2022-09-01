import keras.optimizers as keras_optimizers

from grcnn.processing import filter_bad_samples
from grcnn.training import train
from grcnn.utils import DATASET_DIR, find_best_model_epoch, find_model_path


def main():
    width, height = 64, 32
    max_text_len = 17
    n_classes = 10
    loss_weights = (1., .005)
    grcl_niter = 5
    grcl_fsize = 3
    lstm_units = 512
    img_dir = f'images_train_size{width}x{height}'

    batch_size = 48
    model_num = 28
    initial_epoch = 0
    epochs = 200
    train_augment_prob = .1
    lr = .5
    optimizer = keras_optimizers.Adadelta(lr, rho=.9)
    pretrain_model_num = 26
    pretrain_model_path = find_model_path(pretrain_model_num, find_best_model_epoch(pretrain_model_num))
    train_df_name = f'train_set_relabel_len{max_text_len}'
    val_df_name = f'val_set_len{max_text_len}'
    train(train_df_name, val_df_name, img_dir,
          model_num, initial_epoch, epochs, train_augment_prob,
          batch_size, pretrain_model_path, optimizer, width, height, max_text_len, n_classes,
          grcl_niter, grcl_fsize, lstm_units, loss_weights)

    batch_size = 48
    model_num = model_num + 1
    initial_epoch = 0
    epochs = 200
    train_augment_prob = .5
    lr = .1
    optimizer = keras_optimizers.Adadelta(lr, rho=.9)
    pretrain_model_num = 26
    pretrain_model_path = find_model_path(pretrain_model_num, find_best_model_epoch(pretrain_model_num))
    train_df_name = f'train_set_relabel_len{max_text_len}'
    val_df_name = f'val_set_len{max_text_len}'
    train(train_df_name, val_df_name, img_dir,
          model_num, initial_epoch, epochs, train_augment_prob,
          batch_size, pretrain_model_path, optimizer, width, height, max_text_len, n_classes,
          grcl_niter, grcl_fsize, lstm_units)

    batch_size = 48
    model_num = model_num + 1
    grcl_niter = 3
    initial_epoch = 0
    epochs = 200
    train_augment_prob = .5
    lr = .1
    optimizer = keras_optimizers.Adadelta(lr, rho=.9)
    pretrain_model_num = 22
    pretrain_model_path = find_model_path(pretrain_model_num, find_best_model_epoch(pretrain_model_num))
    train_df_name = f'train_set_relabel_len{max_text_len}'
    val_df_name = f'val_set_len{max_text_len}'
    train(train_df_name, val_df_name, img_dir,
          model_num, initial_epoch, epochs, train_augment_prob,
          batch_size, pretrain_model_path, optimizer, width, height, max_text_len, n_classes,
          grcl_niter, grcl_fsize, lstm_units, loss_weights)


if __name__ == '__main__':
    main()
