# notes

This file is my note on the MLP project.

## back propagation

The `backprobagation` is noted as $(\sigma_i)_{i \in n}$. The last value is originally $\sigma_3$, with
$$
\sigma_3 = a_3 - y_{enc}
$$
$a_i$ are the outputs. $a_3$ is the final one.

Then we also need the input $z$ to calculate previous $\sigma$.
$$
\sigma_2 = f(\sigma_3, z_2)
$$
The order of calculation:

```txt
a1, z2, a2, z3, a3, z4, a4
```

