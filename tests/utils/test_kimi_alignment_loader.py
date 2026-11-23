"""CPU regression checks for the real Kimi model-loader dispatch.

Extract only the loader definition so accelerator initialization is not needed
to verify package versus checkpoint-code selection.
"""
import ast
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch


class TestKimiAlignmentLoader(unittest.TestCase):
    def test_package_model_selection_and_legacy_fallback(self):
        source = Path(__file__).resolve().parents[2] / "verl/workers/engine/fsdp/transformer_impl.py"
        definition = next(
            node for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "_load_hf_language_model"
        )
        namespace = {"os": os}
        exec(compile(ast.Module(body=[definition], type_ignores=[]), str(source), "exec"), namespace)
        load = namespace["_load_hf_language_model"]

        class PackageModel:
            @classmethod
            def _from_config(cls, *args, **kwargs):
                return "package_meta", args, kwargs

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return "package_weights", args, kwargs

        class AutoModel:
            @classmethod
            def from_config(cls, *args, **kwargs):
                return "auto_meta", args, kwargs

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return "auto_weights", args, kwargs

        config = types.SimpleNamespace(
            hf_config=types.SimpleNamespace(model_type="kimi_k31"),
            local_path="/checkpoint",
            trust_remote_code=True,
        )
        module = types.ModuleType("fsdp_turbo.models.kimi.kimi_k31.dsa")
        module.KimiK31ForConditionalGeneration = PackageModel
        with patch.dict(sys.modules, {module.__name__: module}):
            for flag, owner in [("0", "auto"), ("1", "package")]:
                with patch.dict(os.environ, {"KIMI_ALIGNMENT_RECURRENT_KDA": flag}):
                    for pretrained, suffix in [(False, "meta"), (True, "weights")]:
                        with self.subTest(flag=flag, pretrained=pretrained):
                            result, args, kwargs = load(AutoModel, config, "bf16", pretrained)
                            self.assertEqual(result, owner + "_" + suffix)
                            self.assertEqual(kwargs["torch_dtype"], "bf16")
                            if owner == "package":
                                self.assertNotIn("trust_remote_code", kwargs)
            with patch.dict(os.environ, {"KIMI_ALIGNMENT_RECURRENT_KDA": "1"}):
                config.hf_config.model_type = "other"
                self.assertEqual(load(AutoModel, config, "bf16", True)[0], "auto_weights")


if __name__ == "__main__":
    unittest.main()
