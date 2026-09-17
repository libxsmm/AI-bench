import cuda.tile as ct


def _softplus(value, threshold=20.0):
    return ct.where(
        value > threshold,
        value,
        ct.log(1.0 + ct.exp(value)),
    )


def tanh(value):
    inv_ln2 = 1.4426950408889634
    z = 2.0 * value
    e = ct.exp2((-z) * inv_ln2)
    sig = 1.0 / (1.0 + e)
    return 2.0 * sig - 1.0


def mish(value):
    return value * tanh(_softplus(value))


def gelu(value):
    return 0.5 * value * (
        1.0
        + ct.tanh(
            0.7978845608028654
            * (value + 0.044715 * value * value * value)
        )
    )
