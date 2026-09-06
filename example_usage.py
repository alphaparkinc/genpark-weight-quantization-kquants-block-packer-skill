import json
from client import WeightQuantizationKQuantsBlockPacker

def main():
    packer = WeightQuantizationKQuantsBlockPacker()
    # 64 sample FP16 weights across 2 blocks
    sample_weights = [0.15 * (i % 7 - 3) for i in range(64)]
    result = packer.quantize_q8_0(sample_weights)
    print("Quantization Summary:")
    print(json.dumps({k: v for k, v in result.items() if k != "blocks"}, indent=2))
    assert result["total_blocks"] == 2
    assert result["compression_savings_pct"] > 40.0
    
    dequant = packer.dequantize_q8_0(result)
    assert len(dequant) == 64
    print("Weight quantizer packer verification: PASS")

if __name__ == "__main__":
    main()
