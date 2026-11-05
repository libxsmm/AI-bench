import importlib.util
import os
import pathlib
import re
import sys
import types


# From importlib docs:
# https://docs.python.org/3/library/importlib.html#importing-a-source-file-directly
def import_from_path(
    module_name: str, file_path: str | os.PathLike
) -> types.ModuleType:
    """Import a module directly from a source file.
    Args:
        module_name: Name of the module
        file_path: Absolute path to a Python file
    Returns:
        Loaded module
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def import_kernel_from_path(file_path: str | os.PathLike) -> types.ModuleType:
    """Import a kernel file as a module of its own.

    Every load gets a name not in ``sys.modules``: a kernel's module has to stay
    reachable by name, as Helion finds module globals through ``sys.modules``
    when a kernel compiles, possibly long after other kernels were loaded.
    The name comes from the file's stem and, when that stem was loaded before,
    from the load order.

    Args:
        file_path: Absolute path to a Python file
    Returns:
        Loaded module
    """
    stem = re.sub(r"\W", "_", pathlib.Path(file_path).stem)
    name = base = f"kernel_model_{stem}"
    copy = 0
    while name in sys.modules:
        copy += 1
        name = f"{base}_{copy}"
    return import_from_path(name, file_path)
