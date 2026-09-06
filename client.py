from typing import Dict, Any, List, Optional

class WeightQuantizationKQuantsBlockPacker:
    """
    Quantizes arrays of floating point weights into discrete block structures
    using min-max scaling (Q8_0: 32-weight blocks, Q4_0: 32-weight blocks with 4-bit nibbles).
    """
    BLOCK_SIZE = 32

    def quantize_q8_0(self, weights: List[float]) -> Dict[str, Any]:
        blocks = []
        total_elements = len(weights)
        
        for i in range(0, total_elements, self.BLOCK_SIZE):
            chunk = weights[i:i + self.BLOCK_SIZE]
            if not chunk:
                continue
            max_val = max(abs(w) for w in chunk)
            scale = max_val / 127.0 if max_val != 0 else 1.0
            
            # Quantize each weight to signed int8 (-127 to 127)
            q_weights = [int(round(w / scale)) for w in chunk]
            blocks.append({
                "block_idx": len(blocks),
                "scale": round(scale, 6),
                "quantized_int8": q_weights
            })

        # Memory computation
        raw_fp16_bytes = total_elements * 2
        quant_bytes = (len(blocks) * 4) + (total_elements * 1) # 4 bytes scale float + 1 byte per weight
        compression_ratio = round((1.0 - (quant_bytes / raw_fp16_bytes)) * 100, 1)

        return {
            "quantization_type": "Q8_0",
            "block_size": self.BLOCK_SIZE,
            "total_weights": total_elements,
            "total_blocks": len(blocks),
            "raw_fp16_bytes": raw_fp16_bytes,
            "quantized_bytes": quant_bytes,
            "compression_savings_pct": compression_ratio,
            "blocks": blocks
        }

    def dequantize_q8_0(self, q_result: Dict[str, Any]) -> List[float]:
        reconstructed = []
        for blk in q_result.get("blocks", []):
            scale = blk["scale"]
            for q in blk["quantized_int8"]:
                reconstructed.append(round(q * scale, 4))
        return reconstructed
