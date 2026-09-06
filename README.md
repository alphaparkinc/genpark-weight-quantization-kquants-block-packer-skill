# GenPark AI Agent Skill - Weight Quantization K-Quants Block Packer

Quantizes full-precision neural network weights into compact, cache-friendly block representations (Q8_0, Q4_0) with dynamic scale factors.

Verified by [GenPark AI](https://genpark.ai) and compatible with [Model Context Protocol (MCP)](https://genpark.ai/mcp).

## Architecture Diagram

```mermaid
graph TD
    A[Raw FP16 / BF16 Weight Tensor] --> B[Block Slicer: 32 Elements per Block]
    B --> C[Compute Dynamic Block Scale Factor: max abs / 127]
    C --> D[Quantize FP16 to Signed 8-bit Int8]
    D --> E[Pack Scale + Quantized Values into Dense Memory Buffer]
    E --> F[50%+ Memory Reduction for Edge Inference]
```

## Features
- **Deterministic Min-Max Scaling**: Preserves activation fidelity while halving memory footprint.
- **Zero External Dependencies**: Pure Python 3.9+ standard library.
