from omegaconf import OmegaConf, DictConfig, ListConfig


def filter_keys(config: DictConfig, keys: list[str] | str | None = None) -> DictConfig | ListConfig:
    """Filter config to keep only selected keys.

    Parameters
    ----------
    config : DictConfig
        Configuration to filter.
    keys : list[str] | str | None
        Keys to keep. Single string, list of strings, or None (keep all).

    Returns
    -------
    DictConfig | ListConfig
        Filtered config containing only selected keys.
    """
    if not keys:
        return config

    if isinstance(keys, str):
        keys_list = [keys]
    else:
        keys_list = list(keys) if keys else []

    if not keys_list:
        return config

    # Create filtered config with only selected keys
    filtered = OmegaConf.create({})
    for k in keys_list:
        if k in config:
            filtered[k] = config[k]
    
    return filtered


def resolver_extract(config: DictConfig, keys: str | list | None = None) -> DictConfig | ListConfig:
    """Resolver for filtering config keys.
    
    Used in YAML: ${extract:config_dict,key1,key2,key3}
    
    Parameters
    ----------
    config : DictConfig
        Configuration to filter.
    keys : str | list | None
        Comma-separated keys as string (e.g., "prof,meta,surf")
        or list of keys (e.g., [prof, meta, surf]).
        If None or empty, returns config unchanged.
    
    Returns
    -------
    DictConfig
        Filtered configuration.
    """
    # Handle both string and list inputs, including ListConfig from OmegaConf
    if isinstance(keys, (list, tuple, ListConfig)):
        keys_list = list(keys)
    elif isinstance(keys, str):
        keys_list = [k.strip() for k in keys.split(',')]
    else:
        keys_list = None
    
    return filter_keys(config, keys_list)


def resolver_extract_nested(config: DictConfig, keys: str | list | None = None, subkey: str | None = None) -> DictConfig:
    """Resolver for filtering keys and extracting a nested subkey from each.
    
    Supports two forms:
    1. Extract from all keys: ${extract_nested:config, subkey}
    2. Extract from selected keys: ${extract_nested:config, key1,key2, subkey}
    
    Parameters
    ----------
    config : DictConfig
        Configuration to filter and extract from.
    keys : str | list | None
        When used as 2-arg form, this is the subkey.
        When used as 3+ arg form, this is comma-separated keys or a list.
    subkey : str | None
        Nested subkey to extract from each key (e.g., "transformations").
        Only provided in 3+ arg form.
    
    Returns
    -------
    DictConfig
        Filtered configuration with only the extracted subkey from each key.
    """
    # Auto-detect 2-argument form: ${extract_nested:config, subkey}
    if subkey is None and isinstance(keys, str):
        subkey = keys
        keys = None
    
    # Parse keys
    if isinstance(keys, (list, tuple, ListConfig)):
        keys = list(keys)
    elif isinstance(keys, str):
        keys = [k.strip() for k in keys.split(',') if k.strip()]
    else:
        keys = None
    
    # Filter to selected keys
    filtered = OmegaConf.create({})
    for k in config.keys():
        if keys is None or k in keys:
            filtered[k] = config[k]
    
    # Extract subkey from each item
    extracted = OmegaConf.create({})
    for k in filtered.keys():
        if subkey and subkey in filtered[k]:
            extracted[k] = filtered[k][subkey]
    
    return extracted


def register_custom_resolvers() -> None:
    """Register all custom resolvers with OmegaConf.
    
    Call this function early in your main script before Hydra resolves configs.
    """
    OmegaConf.register_new_resolver(
        "extract",
        resolver_extract,
        replace=True
    )
    
    OmegaConf.register_new_resolver(
        "extract_nested",
        resolver_extract_nested,
        replace=True
    )


# Auto-register on import
register_custom_resolvers()

