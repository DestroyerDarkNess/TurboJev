# Native ABI test dependency

`nlohmann/json.hpp` is the unmodified JSON for Modern C++ 3.12.0 header
from llama.cpp revision `b29c606e28a01b1bc8c1351026a0fa6e616bf6c4`,
path `vendor/nlohmann/json.hpp`. It is licensed under MIT; see
`nlohmann/LICENSE.MIT` and the copyright notice in the header.

This header lets the complete production shim compile in the ABI regression
gate on a clean checkout. The adjacent llama stubs intentionally reject grammar
generation and cannot establish model accuracy. Real generation is checked
separately by installed-package tests.
