import triton
import triton.language as tl


@triton.jit
def _softplus(x, THRESHOLD: tl.constexpr = 20.0):
    return tl.where(x > THRESHOLD, x, tl.math.log(1.0 + tl.exp(x)))


@triton.jit
def tanh(x):
    # tanh(x) = 2*sigmoid(2x) - 1
    # sigmoid(z) = 1/(1 + exp2(-z * log2(e)))
    inv_ln2: tl.constexpr = 1.4426950408889634
    z = 2.0 * x
    e = tl.math.exp2((-z) * inv_ln2)
    sig = 1.0 / (1.0 + e)
    return 2.0 * sig - 1.0


@triton.jit
def mish(x):
    return x * tanh(_softplus(x))


@triton.jit
def gelu(x):
    return 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))
