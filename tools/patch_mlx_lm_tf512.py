"""Make mlx-lm (<= 0.31.3, and current git main) import under transformers >= 5.12.

transformers 5.12 changed AutoTokenizer.register to require a config *class*
as the key (it reads key.__module__); mlx-lm passes the string
"NewlineTokenizer", so `import mlx_lm` dies with
AttributeError: 'str' object has no attribute '__module__'.

This rewrites the register call in the installed mlx_lm/tokenizer_utils.py to
register a dummy PretrainedConfig subclass instead. NewlineTokenizer stays
discoverable by name (tokenizer_config.json's "tokenizer_class"), so models
that use it keep working. Compatible with transformers 5.10 too.

Usage:  python tools/patch_mlx_lm_tf512.py [path-to-python]
        (defaults to the interpreter running this script)
"""

import pathlib
import subprocess
import sys

OLD = 'AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)'
NEW = '''try:
    AutoTokenizer.register("NewlineTokenizer", fast_tokenizer_class=NewlineTokenizer)
except AttributeError:
    # transformers >= 5.12 requires a config class as the registry key
    from transformers import PretrainedConfig

    class _NewlineTokenizerConfig(PretrainedConfig):
        model_type = "newline_tokenizer"

    AutoTokenizer.register(_NewlineTokenizerConfig, fast_tokenizer_class=NewlineTokenizer)'''


def main():
    py = sys.argv[1] if len(sys.argv) > 1 else sys.executable
    out = subprocess.run(
        [py, "-c", "import mlx_lm.tokenizer_utils as t; print(t.__file__)"],
        capture_output=True, text=True,
    )
    if out.returncode == 0:
        target = pathlib.Path(out.stdout.strip())
    else:
        # import itself fails (that's the bug) — locate the file without importing
        out = subprocess.run(
            [py, "-c",
             "import importlib.util; "
             "print(importlib.util.find_spec('mlx_lm.tokenizer_utils').origin)"],
            capture_output=True, text=True, check=True,
        )
        target = pathlib.Path(out.stdout.strip())

    src = target.read_text()
    if NEW.splitlines()[1].strip() in src and "except AttributeError" in src:
        print(f"already patched: {target}")
        return
    if OLD not in src:
        sys.exit(f"expected register call not found in {target}; mlx-lm changed?")
    target.write_text(src.replace(OLD, NEW))
    print(f"patched: {target}")

    check = subprocess.run([py, "-c", "import mlx_lm; print(mlx_lm.__version__)"],
                           capture_output=True, text=True)
    if check.returncode != 0:
        sys.exit(f"patch applied but import still fails:\n{check.stderr}")
    print(f"import OK (mlx_lm {check.stdout.strip()})")


if __name__ == "__main__":
    main()
