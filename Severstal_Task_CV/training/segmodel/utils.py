from pathlib import Path

from tensorflow.keras.models import Model

from .model_lib import Unet

LOGS_DIR = Path(f"./checkpoints/logs")
MODELS_DIR = Path(f'./checkpoints')
TENSORBOARD_DIR = Path(f'../tensorboard/training')
DATASET_DIR = Path(f'../dataset')
ARR_FILE_FORMAT = '%s.npy'
for d in (MODELS_DIR, LOGS_DIR, TENSORBOARD_DIR, DATASET_DIR):
    d.mkdir(exist_ok=True, parents=True)


def find_model_path(model_num: int) -> str:
    model_dir = MODELS_DIR / f'model{model_num}'
    try:
        path, *_ = model_dir.glob(f'best_model.hdf5')
        return str(path)
    except ValueError:
        raise ValueError(f"Model not found. Model dir: {model_dir}")


def find_and_load_model(model_num: int, n_classes, debug: bool = False):
    model_file_path = find_model_path(model_num)
    model: Model = Unet('resnext101', classes=n_classes,
                           activation='softmax',
                           encoder_weights=None, weights=model_file_path)
    if debug:
        model.summary()
    return model
