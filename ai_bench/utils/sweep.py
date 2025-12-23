"""Sweep configuration generator for benchmark variants.

Generates cross-product of parameter combinations from sweep configs.
"""

import itertools
from pathlib import Path

import yaml


def load_sweep_config(sweep_path: Path | str) -> dict:
    """Load sweep configuration from YAML file.

    Args:
        sweep_path: Path to sweep config file

    Returns:
        Sweep configuration dict
    """
    with open(sweep_path) as f:
        return yaml.safe_load(f)


def generate_variants(
    kernel_name: str,
    sweep_config: dict,
    base_variant: dict | None = None,
) -> list[dict]:
    """Generate all variant combinations from sweep config.

    Args:
        kernel_name: Name of the kernel (e.g., "1_Conv2D_ReLU_BiasAdd")
        sweep_config: Sweep configuration dict
        base_variant: Optional base variant to extend (dims are merged, not replaced)

    Returns:
        List of variant dicts with all parameter combinations

    Example:
        >>> config = {
        ...     "1_Conv2D_ReLU_BiasAdd": {
        ...         "batch_size": [64, 128],
        ...         "in_channels": [32, 64],
        ...         "_linked": {"width": "height"},
        ...         "_fixed": {"kernel_size": 3},
        ...     }
        ... }
        >>> variants = generate_variants("1_Conv2D_ReLU_BiasAdd", config)
        >>> len(variants)  # 2 x 2 = 4 combinations
        4
    """
    if kernel_name not in sweep_config:
        return []

    kernel_config = sweep_config[kernel_name]

    # Separate special keys from sweep parameters
    linked = kernel_config.pop("_linked", {})
    fixed = kernel_config.pop("_fixed", {})

    # Get sweep parameters (lists of values)
    sweep_params = {
        k: v
        for k, v in kernel_config.items()
        if isinstance(v, list) and not k.startswith("_")
    }

    if not sweep_params:
        return []

    # Generate cross-product of all sweep parameters
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[k] for k in param_names]

    variants = []
    for combination in itertools.product(*param_values):
        dims = dict(zip(param_names, combination))

        # Apply linked parameters (e.g., width = height)
        for target, source in linked.items():
            if source in dims:
                dims[target] = dims[source]

        # Apply fixed parameters
        dims.update(fixed)

        # Normalize keys to uppercase for dims
        dims_upper = {k.upper(): v for k, v in dims.items()}

        # Start with base variant if provided
        if base_variant:
            variant = base_variant.copy()
            # Merge dims: base dims first, then override with sweep dims
            base_dims = base_variant.get("dims", {})
            merged_dims = {**base_dims, **dims_upper}
            variant["dims"] = merged_dims
        else:
            variant = {"dims": dims_upper}

        variants.append(variant)

    return variants


def generate_all_variants(
    sweep_config: dict,
    base_specs: dict | None = None,
) -> dict[str, list[dict]]:
    """Generate variants for all kernels in sweep config.

    Args:
        sweep_config: Sweep configuration dict
        base_specs: Optional dict of base specs per kernel

    Returns:
        Dict mapping kernel names to lists of variants
    """
    all_variants = {}

    for kernel_name in sweep_config:
        if kernel_name.startswith("_"):
            continue

        base_variant = None
        if base_specs and kernel_name in base_specs:
            base_variant = base_specs[kernel_name]

        variants = generate_variants(kernel_name, sweep_config.copy(), base_variant)
        if variants:
            all_variants[kernel_name] = variants

    return all_variants


def merge_sweep_into_spec(
    spec: dict,
    kernel_name: str,
    sweep_config: dict,
    target_key: str = "bench-gpu",
) -> dict:
    """Merge sweep-generated variants into a spec dict.

    Args:
        spec: Original spec dict
        kernel_name: Name of the kernel
        sweep_config: Sweep configuration
        target_key: Target key in spec (e.g., "bench-gpu", "bench-cpu")

    Returns:
        Updated spec dict with generated variants
    """
    if kernel_name not in sweep_config:
        return spec

    # Get base variant template from existing spec
    base_variant = None
    if target_key in spec and spec[target_key]:
        base_variant = spec[target_key][0].copy()
        # Remove dims as they'll be replaced
        base_variant.pop("dims", None)

    # Generate variants
    kernel_config = sweep_config[kernel_name].copy()

    # Extract special keys
    linked = kernel_config.pop("_linked", {})
    fixed = kernel_config.pop("_fixed", {})

    # Get sweep parameters
    sweep_params = {
        k: v
        for k, v in kernel_config.items()
        if isinstance(v, list) and not k.startswith("_")
    }

    if not sweep_params:
        return spec

    # Generate cross-product
    param_names = list(sweep_params.keys())
    param_values = [sweep_params[k] for k in param_names]

    variants = []
    for combination in itertools.product(*param_values):
        dims = dict(zip(param_names, combination))

        # Apply linked parameters
        for target, source in linked.items():
            source_lower = source.lower()
            if source_lower in dims:
                dims[target.lower()] = dims[source_lower]

        # Apply fixed parameters
        for k, v in fixed.items():
            dims[k.lower()] = v

        # Normalize keys to uppercase
        dims_upper = {k.upper(): v for k, v in dims.items()}

        variant = {"dims": dims_upper}

        # Merge with base variant
        if base_variant:
            variant = {**base_variant, "dims": dims_upper}
            # Ensure params is set
            if "params" not in variant and "params" in spec.get(target_key, [{}])[0]:
                variant["params"] = spec[target_key][0]["params"]

        variants.append(variant)

    # Update spec
    spec[target_key] = variants
    return spec


def count_variants(sweep_config: dict) -> dict[str, int]:
    """Count number of variants per kernel.

    Args:
        sweep_config: Sweep configuration dict

    Returns:
        Dict mapping kernel names to variant counts
    """
    counts = {}

    for kernel_name, config in sweep_config.items():
        if kernel_name.startswith("_"):
            continue

        # Get sweep parameters (exclude special keys)
        sweep_params = {
            k: v
            for k, v in config.items()
            if isinstance(v, list) and not k.startswith("_")
        }

        if sweep_params:
            count = 1
            for values in sweep_params.values():
                count *= len(values)
            counts[kernel_name] = count

    return counts
