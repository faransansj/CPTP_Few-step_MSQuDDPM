#!/usr/bin/env python3
"""Optional deterministic parity check against the official TensorCircuit gate order."""
import json

import torch

from msquddpm.reverse_model import ReverseMSQuDDPM

try:
    import tensorcircuit as tc
except ImportError as error:
    raise SystemExit("Run with: uv run --locked --with tensorcircuit==0.11.0 python scripts/check_tensorcircuit_parity.py") from error


tc.set_backend("pytorch")
tc.set_dtype("complex128")
total, depth = 3, 4
params = torch.linspace(-0.2, 0.49, 2 * total * depth, dtype=torch.float64)
model = ReverseMSQuDDPM(1, n_ancilla=2, depth=depth, ancilla="zero", seed=1)
with torch.no_grad():
    model.theta[0].copy_(params.reshape(depth, total, 2))

circuit = tc.Circuit(total)
for layer in range(depth):
    for qubit in range(total):
        offset = 2 * layer * total + 2 * qubit
        circuit.rx(qubit, theta=params[offset])
        circuit.ry(qubit, theta=params[offset + 1])
    for left in range(0, total - 1, 2):
        circuit.cz(left, left + 1)
    for left in range(1, total - 1, 2):
        circuit.cz(left, left + 1)

max_abs_error = float((model._unitary(1) - circuit.matrix()).abs().max().detach())
assert max_abs_error < 1e-12, max_abs_error
print(json.dumps({"max_abs_error": max_abs_error, "tolerance": 1e-12}))
