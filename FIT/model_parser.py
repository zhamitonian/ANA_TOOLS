"""
Model string preprocessor for GenericFit.

Resolves composite operation calls (PROD, SUM, RSUM, FCONV, NCONV) into
intermediate workspace PDFs and substitutes their names back in place.
Everything else (yields, fractions, coefficients, operators) is RooFit
factory syntax and is passed through unchanged.

The model string must have a top-level operation as its root:
    "SUM(nsig[100,0,1000]*sig, nbkg[50,0,500]*bkg)"
    "SUM(nsig[...]*FCONV(x, bw, gauss), nbkg[...]*PROD(bkg1, bkg2))"

version : 2.1
author : zheng wang
date : 2026-03-26
"""

import re
from typing import List, Tuple
import ROOT

_KNOWN_OPS = {'PROD', 'SUM', 'RSUM', 'FCONV', 'NCONV'}


class ModelParser:
    """Resolves nested op calls in a model string into workspace PDFs."""

    def __init__(self, workspace: ROOT.RooWorkspace):
        self.workspace = workspace
        self.counter = 0

    def _get_unique_name(self, prefix: str = "tmp") -> str:
        self.counter += 1
        return f"{prefix}_{self.counter}"

    def _split_args(self, args_str: str) -> List[str]:
        """Split comma-separated args respecting nested parens."""
        args, depth, current = [], 0, ""
        for ch in args_str:
            if ch == '(':
                depth += 1
                current += ch
            elif ch == ')':
                depth -= 1
                current += ch
            elif ch == ',' and depth == 0:
                args.append(current)
                current = ""
            else:
                current += ch
        if current.strip():
            args.append(current)
        return args

    def _resolve(self, expr: str) -> str:
        """Find the first op call, resolve its args recursively, create the
        intermediate PDF, substitute the name back, then repeat."""
        pattern = r'\b(' + '|'.join(_KNOWN_OPS) + r')\s*\('
        m = re.search(pattern, expr, flags=re.IGNORECASE)
        if not m:
            return expr

        op = m.group(1).upper()
        paren_start = expr.index('(', m.start())

        depth, end = 0, paren_start
        for i, ch in enumerate(expr[paren_start:], paren_start):
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    end = i
                    break

        args = self._split_args(expr[paren_start + 1:end])
        resolved_args = [self._resolve(a.strip()) for a in args]

        tmp_name = self._get_unique_name(op.lower())
        factory_str = f"{op}::{tmp_name}({', '.join(resolved_args)})"
        self.workspace.factory(factory_str)
        print(f"[ModelParser] {factory_str}")

        return self._resolve(expr[:m.start()] + tmp_name + expr[end + 1:])

    def parse_model(self, model_str: str, final_pdf_name: str = None) -> List[str]:
        """
        Resolve all op calls in *model_str*.

        Returns:
            (pdf_name, yield_names) where pdf_name is the name of the
            top-level PDF created in the workspace, and yield_names are
            identifiers with [...] followed by * in the original string.

        Args:
            final_pdf_name: Optional explicit name for the top-level PDF.
                If provided, model_str must be a top-level operation call
                (SUM/PROD/RSUM/FCONV/NCONV), and nested operations will be
                resolved while the final PDF is created using this name.
        """
        self.counter = 0
        yield_names = re.findall(r'\b([A-Za-z_]\w*)\[[\d.,\s+-]+\]\s*\*', model_str)
        if final_pdf_name is None:
            processed = self._resolve(model_str)
            return processed, yield_names

        expr = model_str.strip()
        m = re.match(r"^\s*([A-Za-z_]\w*)\s*\((.*)\)\s*$", expr, flags=re.IGNORECASE)
        if not m:
            raise ValueError(
                f"When final_pdf_name is set, model must be a top-level operation call, got: {model_str}"
            )

        op = m.group(1).upper()
        if op not in _KNOWN_OPS:
            raise ValueError(
                f"Unsupported top-level operation '{op}'. Supported: {list(_KNOWN_OPS)}"
            )

        args = self._split_args(m.group(2))
        resolved_args = [self._resolve(a.strip()) for a in args if a.strip()]

        factory_str = f"{op}::{final_pdf_name}({', '.join(resolved_args)})"
        self.workspace.factory(factory_str)
        print(f"[ModelParser] {factory_str}")

        return yield_names

"""
version log

version 2.0 : 2026-03-12
change to parse method, just handle the intermediate PDF creation and return the final model string

version 2.1 : 2026-03-26
support user provide final_pdf_name
"""