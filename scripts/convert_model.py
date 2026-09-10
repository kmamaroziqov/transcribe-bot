"""Convert the official local checkpoint to FP16 without contacting the Hub."""
import json
from pathlib import Path
import sys
import tempfile

from ctranslate2.converters import TransformersConverter
from transformers import WhisperTokenizerFast

class RubaiConverter(TransformersConverter):
    def load_model(self, model_class, model_name_or_path, **kwargs):
        # CTranslate2 4.8 uses dtype; pinned Transformers 4.55 uses torch_dtype.
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
        return super().load_model(model_class, model_name_or_path, **kwargs)


SOURCE_REVISION = "af4601176140501f8ffbced420f6e2e23c9bb071"


def main():
    source, output = map(Path, sys.argv[1:3])
    if not (source / "model.safetensors").is_file():
        raise SystemExit("Download the model first: python scripts/download_model.py")
    manifest = output / "rubai-manifest.json"
    if manifest.exists() and (output / "model.bin").exists():
        existing = json.loads(manifest.read_text())
        if existing.get("revision") == SOURCE_REVISION:
            print("Verified revision is already converted.")
            return
    # Never overwrite a finished model without an explicit operator action.
    if (output / "model.bin").exists():
        raise SystemExit("Output contains a model without the expected manifest; choose an empty output directory.")
    print("Converting Rubai STT v2 to FP16 on CPU...", flush=True)
    converter = RubaiConverter(
        str(source), copy_files=["preprocessor_config.json"], load_as_float16=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    # /output is a bind mount: convert in a child directory, never remove the mount.
    with tempfile.TemporaryDirectory(prefix=".convert-", dir=output) as temporary:
        stage = Path(temporary) / "model"
        converter.convert(str(stage), quantization="float16")
        # Complete and validate metadata before publishing any output files.
        tokenizer = WhisperTokenizerFast.from_pretrained(str(source), local_files_only=True)
        tokenizer.backend_tokenizer.save(str(stage / "tokenizer.json"))
        config = json.loads((stage / "config.json").read_text())
        generation = json.loads((source / "generation_config.json").read_text())
        config["alignment_heads"] = generation["alignment_heads"]
        (stage / "config.json").write_text(json.dumps(config, indent=2))
        converted_manifest = {
            "repository": "islomov/rubaistt_v2_medium",
            "revision": SOURCE_REVISION,
            "quantization": "float16",
            "ctranslate2": "4.8.2",
        }
        for file in stage.iterdir():
            file.replace(output / file.name)
        manifest.write_text(json.dumps(converted_manifest, indent=2))
    print("Conversion complete.", flush=True)


if __name__ == "__main__":
    main()
