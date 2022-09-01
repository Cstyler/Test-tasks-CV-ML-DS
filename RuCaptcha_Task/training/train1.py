import tensorflow.keras.optimizers as keras_optimizers

from grcnn.training import train


def main():
    width, height = 100, 32
    max_text_len = 10
    n_classes = 10
    loss_weights = (1., .5)
    grcl_niter = 5
    grcl_fsize = 3
    lstm_units = 512
    img_file = f'imgs_train_size{width}x{height}.npy'

    batch_size = 100
    model_num = 9
    initial_epoch = 0
    epochs = 200
    train_augment_prob = .1
    lr = .5
    optimizer = keras_optimizers.Adadelta(lr, rho=.9)
    pretrain_model_path = None
    train_df_name = f'train_set'
    val_df_name = f'val_set'
    print(f"Model {model_num}")
    train(train_df_name, val_df_name, img_file,
          model_num, initial_epoch, epochs, train_augment_prob,
          batch_size, pretrain_model_path, optimizer, width, height, max_text_len, n_classes,
          grcl_niter, grcl_fsize, lstm_units, loss_weights)


if __name__ == '__main__':
    main()
