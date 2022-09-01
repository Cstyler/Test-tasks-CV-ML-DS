from typing import List

from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import (CSVLogger, Callback, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau, TensorBoard)

# from pylibs.keras.callbacks import CyclicLR
from .utils import LOGS_DIR, MODELS_DIR, TENSORBOARD_DIR


class CustomTensorBoard(TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs.update(lr=K.eval(self.model.optimizer.lr))
        super().on_epoch_end(epoch, logs)


def get_callbacks(model_num: int, base_lr, max_lr, step_size) -> List[Callback]:
    csv_filename = str(LOGS_DIR / f'model{model_num}.csv')
    csvLoggerCallback = CSVLogger(csv_filename, append=True)
    earlyStoppingCallback = EarlyStopping(monitor='loss', min_delta=0,
                                          patience=50, verbose=1)
    tensorboard_folder = TENSORBOARD_DIR / f'run{model_num}'
    tensorboard_folder.mkdir(parents=True, exist_ok=True)
    tensorboard_callback = CustomTensorBoard(str(tensorboard_folder), write_graph=True, write_images=True)

    reduceLRCallback = ReduceLROnPlateau(monitor='loss', factor=.5,
                                         patience=10, verbose=1,
                                         min_lr=1e-8, min_delta=0, cooldown=3)
    # cyclicLR = CyclicLR(base_lr, max_lr, step_size, 'triangular2')
    model_dir = MODELS_DIR / f'model{model_num}'
    model_dir.mkdir(exist_ok=True, parents=True)
    model_file_path = model_dir / 'best_model.hdf5'
    modelCheckpointCallback = ModelCheckpoint(monitor='val_f1-score', filepath=str(model_file_path),
                                              verbose=0, save_best_only=True, mode='max')
    callbacks = [modelCheckpointCallback,
                 csvLoggerCallback,
                 earlyStoppingCallback,
                 tensorboard_callback,
                 reduceLRCallback]
    return callbacks
