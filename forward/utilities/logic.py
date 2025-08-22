import os


def get_repo_root() -> str:
    """ Get root directory of the repository.

        Returns
        -------
        str: Root directory of the repository.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


def get_config_path() -> str:
    """ Get configuration directory of the repository.

        Returns
        -------
        str: Configuration directory of the repository.
    """
    return os.path.join(get_repo_root(), "config")
