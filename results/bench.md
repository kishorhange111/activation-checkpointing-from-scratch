GPT with 16.9M parameters: d = 256, 16 blocks, vocabulary 8192, batch 8 × 256 tokens, fp32, 4 CPU threads. Chain: x_0 → embed → 16 blocks → head + loss, L = 18. Memory is activation memory (parameters excluded).

| schedule | predicted peak MB | measured peak MB | Δ RSS peak MB | forwards | predicted s | measured s |
|---|---|---|---|---|---|---|
| store all | 611.88 | 611.88 | 743.6 | 18 | 1.957 | 1.905 |
| optimal, same compute as store all | 611.88 | 611.88 | 743.0 | 18 | 1.957 | 1.905 |
| per layer | 106.99 | 106.99 | 239.1 | 35 | 2.496 | 2.453 |
| optimal, same compute as per layer | 88.11 | 88.11 | 220.0 | 35 | 2.496 | 2.450 |
| √L segments | 176.67 | 176.67 | 276.9 | 33 | 2.432 | 2.378 |
| optimal, same compute as √L segments | 147.03 | 147.03 | 278.7 | 33 | 2.431 | 2.388 |
| optimal within 73.4 MB | 73.43 | 73.43 | 205.4 | 50 | 3.000 | 2.908 |
| optimal within 133.3 MB | 132.25 | 132.25 | 263.9 | 33 | 2.452 | 2.347 |
| optimal within 193.1 MB | 191.17 | 191.17 | 322.5 | 31 | 2.374 | 2.322 |
| optimal within 252.9 MB | 252.18 | 252.18 | 383.9 | 29 | 2.303 | 2.249 |
| optimal within 312.7 MB | 311.10 | 311.10 | 442.8 | 28 | 2.261 | 2.218 |
| optimal within 372.6 MB | 359.43 | 359.43 | 490.6 | 26 | 2.203 | 2.194 |
| optimal within 432.4 MB | 422.55 | 422.55 | 554.0 | 24 | 2.140 | 2.118 |
| optimal within 492.2 MB | 485.66 | 485.66 | 616.7 | 22 | 2.078 | 2.055 |
| optimal within 552.1 MB | 548.77 | 548.77 | 680.3 | 20 | 2.017 | 2.017 |
| optimal within 611.9 MB | 611.88 | 611.88 | 743.4 | 18 | 1.957 | 1.946 |

| layer | x MB | tape MB | forward ms | taped forward ms | backward ms |
|---|---|---|---|---|---|
| 1: Embed | 2.10 | 0.00 | 1.1 | 1.3 | 3.0 |
| 2: Block | 2.10 | 31.56 | 42.0 | 31.3 | 60.3 |
| 17: Block | 2.10 | 31.56 | 33.1 | 44.3 | 71.4 |
| 18: Head | 0.00 | 69.22 | 65.3 | 106.3 | 120.8 |
