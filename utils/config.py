import os


def _parse_simple_yaml(text):
    """Parse a minimal YAML subset for the datasets config."""
    data = {}
    current_section = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue

        if not line.startswith(" "):
            if line.endswith(":"):
                current_section = line[:-1].strip()
                data[current_section] = {}
            else:
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
            continue

        if current_section is None:
            raise ValueError("Invalid YAML: indented line without section")

        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError("Invalid YAML entry: missing ':'")
        key, value = stripped.split(":", 1)
        data[current_section][key.strip()] = value.strip()

    return data


def load_yaml(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()

    try:
        import yaml
    except ImportError:
        return _parse_simple_yaml(text)

    return yaml.safe_load(text)


def load_train_config(path):
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("Train config must be a mapping")
    return data
