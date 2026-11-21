import os
import sys
import onnx
from onnxruntime.transformers import float16


MODELS_DIR = os.path.join(
    os.environ.get("LOCALAPPDATA", os.path.expanduser("~")),
    "karaoke-engine",
    "models",
)

MODEL_FILES = [
    "UVR_MDXNET_KARA_2.onnx",
    "UVR-MDX-NET-Inst_HQ_3.onnx",
]


def convert_model(fp32_path: str) -> str:
    fp16_path = fp32_path[:-5] + ".fp16.onnx" if fp32_path.endswith(".onnx") else fp32_path + ".fp16.onnx"
    tmp_path = fp16_path + ".tmp"

    print(f"Converting {os.path.basename(fp32_path)} -> {os.path.basename(fp16_path)}...")
    model = onnx.load(fp32_path)
    fp16_model = float16.convert_float_to_float16(model, keep_io_types=False)
    onnx.save(fp16_model, tmp_path)

    # Validate generated ONNX graph structure before committing atomically
    onnx.checker.check_model(tmp_path)
    size_bytes = os.path.getsize(tmp_path)
    if size_bytes < 1_000_000:
        raise RuntimeError(f"Converted FP16 model unexpectedly small ({size_bytes} bytes): {tmp_path}")

    os.replace(tmp_path, fp16_path)
    fp32_mb = os.path.getsize(fp32_path) / (1024.0 * 1024.0)
    fp16_mb = size_bytes / (1024.0 * 1024.0)
    print(f"Ready: {fp16_path} ({fp32_mb:.1f} MB -> {fp16_mb:.1f} MB)")
    return fp16_path


def main() -> int:
    if not os.path.isdir(MODELS_DIR):
        print(f"Models directory not found: {MODELS_DIR}", file=sys.stderr)
        return 1

    converted = 0
    for name in MODEL_FILES:
        src = os.path.join(MODELS_DIR, name)
        if not os.path.isfile(src):
            print(f"Skipping missing model: {src}")
            continue
        convert_model(src)
        converted += 1

    if converted == 0:
        print("No models found to convert.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
