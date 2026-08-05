# Problem Spec YAML Reference

A quick reference for hand-writing problem spec files (for examples, see files in
`problems/specs/KernelBench/level*/*.yaml`).

## Annotated template

A generic skeleton showing what every part of a spec file is for:

```yaml
# Declare every input tensor the kernel needs, keyed by the name used in each
# variant's "params" below.
inputs:
  INPUT_NAME:                 # arbitrary name, referenced by "params"
    shape: [DIM_A, DIM_B]     # dim names, sized per-variant in "dims"
    dtype: inherit            # a torch dtype (e.g. float32), or "inherit"
                              # to reuse the variant's own "dtype"
    range: [0, DIM_B]         # optional: value range for int/bool inputs
    inits: [symmetric]        # optional: init transforms, applied in order

# Initialization values.
# Optional: Values passed positionally to the kernel's constructor.
# Every name here must also appear in each variant's "dims" below.
inits:
  - dim: DIM_B

# A "variant category" - a named group of configurations to run. Category
# names are free-form; "ci"/"simple-cpu"/"bench-cpu"/"bench-gpu" are the
# conventional ones selected via --variant/--ci/--bench.
CATEGORY_NAME:
  - params: [INPUT_NAME]          # which "inputs" to build and pass in, in order
    dtype: float32                # dtype applied to the model/inputs
    dims:                         # concrete size for every dim name used above
      DIM_A: 128
      DIM_B: 64
    flop: "2*DIM_A*DIM_B"         # optional: FLOP count or formula over "dims"
    mem_bytes: "4*DIM_A*DIM_B"    # optional: memory traffic or formula
    rtol: 1.0e-03                 # optional: correctness - relative tolerance override
    atol: 1.0e-05                 # optional: correctness - absolute tolerance override
    memory_format: channels_last  # optional: tensor memory format
```

Every spec file has this shape: an `inputs` map, an optional `inits` list,
and one or more **variant categories** (`ci`, `bench-cpu`, `bench-gpu`,
`simple-cpu`, ...) each holding a list of variants to run.

## Top-level keys

| Key | Required | Type | Description |
|---|---|---|---|
| `inputs` | yes | mapping | Named tensor inputs fed to the kernel. See [Inputs](#inputs). |
| `inits` | no | list | Dimension names passed *positionally* to the kernel's constructor, in order. See [Inits](#inits). |
| `<variant category>` | at least one | list | A named group of variants, e.g. `ci`, `simple-cpu`, `bench-cpu`, `bench-gpu`. Category names are free-form (any name works; the ones above are the conventional ones used across the repo and picked via `--variant`/`SpecKey`). See [Variants](#variants). |
| `name` | no | string | Informational only. Not read by any runner. |
| `description` | no | string | Informational only. Not read by any runner. |

## Inputs

`inputs` maps an input name (used in a variant's `params`) to its spec:

```yaml
inputs:
  X:
    shape: [BATCH, IN_FEAT]
    dtype: inherit
  IDX:
    shape: [BATCH]
    dtype: int64
    range: [0, IN_FEAT]
  A:
    shape: [N, N]
    dtype: inherit
    inits: [symmetric]
```

| Field | Required | Type | Description |
|---|---|---|---|
| `shape` | yes | list of dim names | Names of dims (declared in a variant's `dims`) composing the tensor's shape. |
| `dtype` | yes | string | A torch dtype name (`float32`, `float16`, `bfloat16`, `int64`, `bool`, ...), or `inherit` to reuse the variant's own `dtype`. |
| `range` | no | `[low, high]` | Value range for integer/bool inputs. Each bound is a number or a dim name. |
| `inits` | no | list of strings | Initialization transforms applied in order (see [Input init transforms](#input-init-transforms)). |

### Input init transforms

Values for `inputs.<name>.inits`:

| Value | Effect |
|---|---|
| `scale` | Multiply by a random scalar. |
| `softmax` | Apply softmax over the last dim. |
| `abs` | Take absolute value. |
| `normalize` | L2-normalize over the last dim. |
| `symmetric` | Symmetrize a square 2D tensor: `(A + A.T) / 2`. |
| `triu` | Keep upper triangle (2D only). |
| `tril` | Keep lower triangle (2D only). |
| `transpose` | Transpose a 2D tensor. |
| `uniform` | Fill with `Uniform(-1, 1)`. |
| `rademacher` | Fill with random ±1 values. |

## Inits

`inits` (top level, not per-input) is a list of dimension names passed
positionally to the kernel's `Model.__init__`, in declaration order:

```yaml
inits:
  - dim: IN_FEATURES
  - dim: OUT_FEATURES
```

Each entry is a mapping with a single `dim` key. Every dim named here must
be present in every variant's `dims` (see below). Omit `inits` entirely, or
use `inits: []`, if the kernel constructor takes no arguments.

## Variants

Each variant category (`ci`, `bench-cpu`, `bench-gpu`, `simple-cpu`, ...) is
a list of variant entries - one per concrete shape/dtype configuration to
run:

```yaml
bench-gpu:
  - params: [A, B]
    dtype: float16
    dims:
      N: 2048
    flop: "2*N*N*N"
    mem_bytes: "4*N*N"
    rtol: 1.0e-03
    atol: 1.0e-05
    memory_format: channels_last
```

| Field | Required | Type | Description |
|---|---|---|---|
| `params` | yes | list of input names | Which declared `inputs` to build and pass to the kernel, in order. |
| `dtype` | no* | string | Torch dtype name applied to the model/inputs. Required unless every input using it has an explicit (non-`inherit`) `dtype` of its own. |
| `dims` | yes | mapping | Concrete size for every dim name used by this variant's inputs/inits. Values are numbers, booleans, or lists of numbers (for shape-valued dims like `BIAS_SHAPE: [32, 1, 1]`). May also be a list of such mappings to expand the variant across several shapes (see [Expanding dims](#expanding-dims)). |
| `flop` | no | number or formula string | FLOP count, or a formula over `dims` (e.g. `"2*BATCH*N*N"`). Estimated automatically if omitted (PyTorch backend only). |
| `mem_bytes` | no | number or formula string | Memory traffic in bytes, or a formula over `dims`. Estimated automatically if omitted (PyTorch backend only). |
| `rtol` | no | number | Relative tolerance override for correctness checks. Default: `1e-2`. |
| `atol` | no | number | Absolute tolerance override for correctness checks. Default: `1e-5`. |
| `memory_format` | no | string | A torch memory format name (`channels_last`, `channels_last_3d`, `contiguous_format`, `preserve_format`, ...). |

\* `dtype` is often to minimize inputs list together with `inherit` and ensures that Model's parameters (e.g., constant weights in `nn.Linear`) are alinged with inputs.

### Formula fields (`flop`, `mem_bytes`)

Formulas are plain arithmetic strings using `dims` names as variables, e.g.:

```yaml
flop: "2*M*N*K"
mem_bytes: "(M*K + K*N + M*N) * 4" # f32
```

List-valued dims can be indexed, which is useful for parameterized shapes such as MLP layer lists:

```yaml
flop: "2*BATCH*(IN_SIZE*LAYER_SIZES[0] + LAYER_SIZES[0]*LAYER_SIZES[1])"
```

More complex logic can be expressed with built-in helpers, e.g.:

```yaml
flop: "2*BATCH*adjacent_prod_sum(IN_SIZE, LAYER_SIZES, OUT_SIZE)"
```

Supported operators: `+ - * / **`.  
List indexing is supported.  
No arbitrary function calls.

#### Built-in helpers:
| Helper | Description |
|---|---|
| `adjacent_prod_sum(...)` | Flatten scalar and list arguments into one sequence, then sum products of adjacent values. |


### Expanding dims

To run one variant across several shapes without repeating the whole entry,
set `dims` to a list of mappings. Each mapping is expanded into its own
variant, and per-variant formula fields (`flop`, `mem_bytes`) are evaluated
against each option:

```yaml
bench-gpu:
  - params: [A, B]
    dtype: float16
    dims:
      - N: 1024
      - N: 2048
      - N: 4096
    flop: "2*N*N*N"
    rtol: 1.0e-03
    atol: 1.0e-05
```

This is equivalent to writing three separate variants that share the same
`params`, `dtype`, `flop`, and tolerances but differ in `N`.

## Tips

- `dims` must cover every dim name referenced by the `inputs.*.shape`/`range`
  used by that variant's `params`, plus every dim named in top-level `inits`.
- `params` must only reference names declared under `inputs`.
- If any input uses `dtype: inherit`, the variant must set its own `dtype`.
- Custom variant category names beyond standardized keys e.g., `ci` or `bench-gpu`, are allowed
