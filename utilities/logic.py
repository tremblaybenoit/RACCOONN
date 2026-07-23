from pathlib import Path


def get_repo_root() -> str:
    """ Get root directory of the repository.

        Returns
        -------
        str: Root directory of the repository.
    """
    return str(Path(__file__).parent.parent.resolve())


def get_config_path() -> str:
    """ Get configuration directory of the repository.

        Returns
        -------
        str: Configuration directory of the repository.
    """
    return str(Path(get_repo_root()) / "config")


def get_resolved_path(path: str, dir: str | None = None) -> str:
    """ Resolve relative paths to absolute; leave absolute paths unchanged.

        Parameters
        ----------
        path: str. The path to resolve.
        dir: str or None. The base directory to resolve relative paths against.

        Returns
        -------
        str: The resolved absolute path.
    """

    # Convert string to Path instance
    p = Path(path)

    # If the path is absolute, return it as-is
    if p.is_absolute():
        return str(p)

    # If no base directory is specified, resolve relative to cwd
    if dir is None:
        return str(p.resolve())

    # Otherwise, resolve relative to the specified base directory
    return str((Path(dir) / path).resolve())
