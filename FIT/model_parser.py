"""
Model string preprocessor for GenericFit.

Resolves composite operation calls (PROD, SUM, RSUM, FCONV, NCONV) into
intermediate workspace PDFs and substitutes their names back in place.
Everything else (yields, fractions, coefficients, operators) is RooFit
factory syntax and is passed through unchanged.

The model string must have a top-level operation as its root:
    "SUM(nsig[100,0,1000]*sig, nbkg[50,0,500]*bkg)"
    "SUM(nsig[...]*FCONV(x, bw, gauss), nbkg[...]*PROD(bkg1, bkg2))"

version : 2.0 
author : zheng wang
date : 2026-03-12
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
        m = re.search(pattern, expr)
        if not m:
            return expr

        op = m.group(1)
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

    def parse_model(self, model_str: str) -> Tuple[str, List[str]]:
        """
        Resolve all op calls in *model_str*.

        Returns:
            (pdf_name, yield_names) where pdf_name is the name of the
            top-level PDF created in the workspace, and yield_names are
            identifiers with [...] followed by * in the original string.
        """
        self.counter = 0
        yield_names = re.findall(r'\b([A-Za-z_]\w*)\[[\d.,\s+-]+\]\s*\*', model_str)
        processed = self._resolve(model_str)
        return processed, yield_names

"""
version log

version 2.0 : 2026-03-12
change to parse method, just handle the intermediate PDF creation and return the final model string
"""