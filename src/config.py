"""YAML loading with scientific-notation floats.

PyYAML follows YAML 1.1, where `1.9e7` or `1e19` (no exponent sign) load as
*strings*. This loader resolves them as floats, as YAML 1.2 does.
"""

import re

import yaml


class _Loader(yaml.SafeLoader):
    pass


_Loader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(r"""^(?:[-+]?(?:[0-9][0-9_]*)(?:\.[0-9_]*)?(?:[eE][-+]?[0-9]+)?
                  |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
                  |[-+]?\.(?:inf|Inf|INF)
                  |\.(?:nan|NaN|NAN))$""", re.X),
    list("-+0123456789."),
)


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.load(f, Loader=_Loader)
