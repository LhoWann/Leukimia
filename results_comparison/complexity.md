# Kompleksitas Model & Latensi

Input 224×224, FP32, device **NVIDIA GeForce RTX 3050 Laptop GPU**. Latensi = warmup 10 + rata-rata 50 iterasi (batch=1) / 12 iterasi (batch=32), CUDA-synced. Bobot acak (tidak memengaruhi params/FLOPs/latensi).

| Model | Params (M) | FLOPs (G) | GMACs | Latensi b1 (ms/img) | Latensi b32 (ms/img) | Throughput b32 (img/s) |
| ----- | :--------: | :-------: | :---: | :-----------------: | :------------------: | :--------------------: |
| ConvNeXtV2-Tiny (Ours, no MHA) | 27.9 | 8.91 | 4.45 | 13.80 | 5.17 | 193 |
| CoAtNet-0 (baseline) | 26.7 | 8.81 | 4.41 | 15.89 | 4.30 | 233 |

> FLOPs via `torch.utils.flop_counter.FlopCounterMode` (total_flops = MACs). GMACs = FLOPs/2. Latensi spesifik-hardware; pakai untuk perbandingan relatif antar-model pada perangkat sama, bukan angka absolut universal.