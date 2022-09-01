import ujson as json

from .types import Pathlike


def write_json(path: Pathlike, _dict: dict):
    open_mode = 'w'
    with open(path, open_mode) as _file:
        json.dump(_dict, _file, ensure_ascii=False)


def read_json(path: Pathlike) -> dict:
    open_mode = 'r'
    with open(path, open_mode) as _file:
        return json.load(_file)

def read_json_str(s: str) -> dict:
    return json.loads(s)