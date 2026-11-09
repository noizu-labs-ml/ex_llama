# Threat Model Summary

**Library, not a service.** No network listener, no HTTP client, no credentials
or endpoints in the tree. Crown jewels: BEAM process integrity (a NIF crash takes
the VM down) and model/head-file integrity (drives AI output).

## Trust Boundaries
Operator-supplied file paths → C++ parsers (`jev_head.h` + llama.cpp) → BEAM.
Model files are untrusted; everything after load is trusted by the host.

## Register
- T-001 High — malformed GGUF/head crashes the NIF/VM. Partial (head validated; llama.cpp is upstream).
- T-002 Medium — untrusted model/head tampers with decisions. Partial (head sha256 recorded, no signing).
- T-003 Medium — unbounded max_tokens/prompt DoS. Open (caller-controlled).
- T-004 Low — classifier fan-out / option limits. Mitigated (2..256, per-model mutex).
- T-005 Low — process-dictionary session state visible in-process. Accepted.
- T-006 Medium — llama.cpp submodule / Hex supply chain. Partial (pinned v0.6.0, mix.lock).

## Residual Risk
Untrusted model files accepted by design (validate/sign before load); caller owns
resource sizing; keeping the library credential-free and egress-free is the
property to preserve.
