import os
import pickle


def save_obj(obj, name, path=""):
    """
    Pickles a given object and stores it under a given name and path.
    :param obj: The object to pickle.
    :param name: The desired filename: "<name>.pkl".
    :param path: (optional) The path to store the object.
    :return: None.
    """
    with open(os.path.join(path, name + '.pkl'), 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name, path=""):
    """
    Unpickles an object using filename and path.
    :param name: The filename to unpickle: "<name>.pkl".
    :param path: (optional) The path where the pickled object is stored.
    :return: The unpickled object.
    """
    if name.endswith('pkl'):
        with open(os.path.join(path, name ), 'rb') as f:
            return pickle.load(f)
    else:
        with open(os.path.join(path, name + '.pkl'), 'rb') as f:
            return pickle.load(f)
