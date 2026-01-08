"""
Model structure parser for GenericFit.

Parses complex model structures like:
- "nsig[100,0,1000] * pdf1 + nbkg[50,0,500] * pdf2"
- "nsig[100,0,1000] * PROD(pdf1, pdf2) + nbkg[50,0,500] * pdf3"
- "nsig[100,0,1000] * FCONV(var, pdf1, pdf2) + nbkg[50,0,500] * pdf3"
- "nsig[100,0,1000] * (pdf1 + pdf2) + nbkg[50,0,500] * PROD(pdf3, pdf4)"

The parser will:
1. Extract yield variables (nsig, nbkg, etc.)
2. Identify composite operations (PROD, FCONV, SUM, parentheses)
3. Automatically create intermediate PDFs in workspace
4. Convert to RooFit factory format
"""

import re
from typing import List, Tuple, Dict, Any
import ROOT


class ModelParser:
    """Parser for complex model structure strings."""
    
    def __init__(self, workspace: ROOT.RooWorkspace):
        """
        Initialize parser.
        
        Args:
            workspace: RooWorkspace where PDFs will be created
        """
        self.workspace = workspace
        self.counter = 0  # For generating unique intermediate PDF names
    
    def _get_unique_name(self, prefix: str = "tmp") -> str:
        """Generate unique name for intermediate PDF."""
        self.counter += 1
        return f"{prefix}_{self.counter}"
    
    def _parse_operation(self, expr: str, var_name: str = None) -> str:
        """
        Parse and build a single operation (PROD, FCONV, SUM).
        
        Args:
            expr: Expression like "PROD(pdf1, pdf2)" or "FCONV(var, pdf1, pdf2)"
            var_name: Variable name for FCONV operations
            
        Returns:
            Name of the created PDF
        """
        # Extract operation type and arguments
        match = re.match(r'(\w+)\((.*)\)', expr.strip())
        if not match:
            return expr.strip()
        
        op_type = match.group(1).upper()
        args_str = match.group(2)
        
        # Parse arguments
        args = []
        depth = 0
        current_arg = ""
        for char in args_str:
            if char == '(' :
                depth += 1
                current_arg += char
            elif char == ')':
                depth -= 1
                current_arg += char
            elif char == ',' and depth == 0:
                args.append(current_arg.strip())
                current_arg = ""
            else:
                current_arg += char
        if current_arg.strip():
            args.append(current_arg.strip())
        
        # Create PDF based on operation type
        pdf_name = self._get_unique_name(op_type.lower())
        
        if op_type == "PROD":
            # Product of PDFs
            # Recursively parse each argument in case they are also operations
            parsed_args = [self._parse_operation(arg, var_name) for arg in args]
            pdf_list = ", ".join(parsed_args)
            self.workspace.factory(f"PROD::{pdf_name}({pdf_list})")
            
        elif op_type == "FCONV":
            # Convolution: FCONV(var, pdf1, pdf2)
            if len(args) < 3:
                raise ValueError(f"FCONV requires at least 3 arguments: var, pdf1, pdf2")
            conv_var = args[0].strip()
            pdf1 = self._parse_operation(args[1], conv_var)
            pdf2 = self._parse_operation(args[2], conv_var)
            self.workspace.factory(f"FCONV::{pdf_name}({conv_var}, {pdf1}, {pdf2})")
            
        elif op_type == "SUM":
            # Sum of PDFs: SUM(coef*pdf1, pdf2) or SUM(pdf1, pdf2)
            parsed_args = [self._parse_operation(arg, var_name) for arg in args]
            # Check if first arg contains coefficient
            sum_str = ", ".join(parsed_args)
            self.workspace.factory(f"SUM::{pdf_name}({sum_str})")
            
        else:
            raise ValueError(f"Unknown operation type: {op_type}")
        
        return pdf_name
    
    def _parse_product_expression(self, expr: str, var_name: str = None) -> str:
        """
        Parse multiplication expression like "pdf1 * pdf2 * pdf3".
        
        Args:
            expr: Expression with * operators
            var_name: Variable name for convolution
            
        Returns:
            Name of the created PDF
        """
        # Split by * but respect parentheses and function calls
        parts = []
        depth = 0
        current_part = ""
        
        for char in expr:
            if char in '([':
                depth += 1
                current_part += char
            elif char in ')]':
                depth -= 1
                current_part += char
            elif char == '*' and depth == 0:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        if current_part.strip():
            parts.append(current_part.strip())
        
        if len(parts) == 1:
            # No multiplication, might be a single PDF or operation
            return self._parse_operation(parts[0], var_name)
        
        # Multiple parts - create PROD
        parsed_parts = [self._parse_operation(part, var_name) for part in parts]
        pdf_name = self._get_unique_name("prod")
        pdf_list = ", ".join(parsed_parts)
        self.workspace.factory(f"PROD::{pdf_name}({pdf_list})")
        
        return pdf_name
    
    def _parse_parentheses(self, expr: str, var_name: str = None) -> str:
        """
        Parse expression with parentheses like "(pdf1 + pdf2)".
        
        Args:
            expr: Expression with parentheses
            var_name: Variable name
            
        Returns:
            Name of the created PDF
        """
        expr = expr.strip()
        
        # Remove outer parentheses if present
        if expr.startswith('(') and expr.endswith(')'):
            # Check if these are the matching outer parentheses
            depth = 0
            for i, char in enumerate(expr):
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                    if depth == 0 and i == len(expr) - 1:
                        # These are matching outer parentheses
                        expr = expr[1:-1].strip()
                        break
        
        # Now parse the inner expression
        # Check if it contains + (sum operation)
        parts = []
        depth = 0
        current_part = ""
        
        for char in expr:
            if char in '([':
                depth += 1
                current_part += char
            elif char in ')]':
                depth -= 1
                current_part += char
            elif char == '+' and depth == 0:
                parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        if current_part.strip():
            parts.append(current_part.strip())
        
        if len(parts) == 1:
            # No sum, parse as product expression
            return self._parse_product_expression(parts[0], var_name)
        
        # Multiple parts - create SUM with equal fractions
        parsed_parts = [self._parse_product_expression(part, var_name) for part in parts]
        pdf_name = self._get_unique_name("sum")
        
        # Create SUM with equal coefficients
        n_parts = len(parsed_parts)
        if n_parts == 2:
            # Binary sum: SUM(frac*pdf1, pdf2)
            frac_name = self._get_unique_name("frac")
            self.workspace.factory(f"{frac_name}[0.5, 0, 1]")
            self.workspace.factory(f"SUM::{pdf_name}({frac_name}*{parsed_parts[0]}, {parsed_parts[1]})")
        else:
            # Multiple components - create nested SUMs
            current_pdf = parsed_parts[-1]
            for i in range(n_parts - 2, -1, -1):
                frac_name = self._get_unique_name("frac")
                new_pdf = self._get_unique_name("sum")
                self.workspace.factory(f"{frac_name}[0.5, 0, 1]")
                self.workspace.factory(f"SUM::{new_pdf}({frac_name}*{parsed_parts[i]}, {current_pdf})")
                current_pdf = new_pdf
            pdf_name = current_pdf
        
        return pdf_name
    
    def parse_model(self, model_str: str, var_name: str = None) -> Tuple[str, List[str]]:
        """
        Parse complete model structure and create all intermediate PDFs.
        
        Args:
            model_str: Model structure string like:
                "nsig[100,0,1000] * pdf1 + nbkg[50,0,500] * pdf2"
                "nsig[100,0,1000] * PROD(pdf1, pdf2) + nbkg[50,0,500] * pdf3"
                "nsig[100,0,1000] * (pdf1 + pdf2) + nbkg[50,0,500] * PROD(pdf3, pdf4)"
            var_name: Variable name for FCONV operations
            
        Returns:
            Tuple of (factory_string_for_model, list_of_yield_variables)
        """
        # Reset counter
        self.counter = 0
        
        # Extract and process each term (separated by + at top level)
        terms = []
        yields = []
        depth = 0
        current_term = ""
        
        for char in model_str:
            if char in '([':
                depth += 1
                current_term += char
            elif char in ')]':
                depth -= 1
                current_term += char
            elif char == '+' and depth == 0:
                terms.append(current_term.strip())
                current_term = ""
            else:
                current_term += char
        if current_term.strip():
            terms.append(current_term.strip())
        
        # Process each term
        model_components = []
        for term in terms:
            # Extract yield variable (e.g., "nsig[100,0,1000]")
            yield_match = re.match(r'(\w+)\[([\d\.,\s]+)\]\s*\*\s*(.*)', term)
            if yield_match:
                yield_var = yield_match.group(1)
                yield_params = yield_match.group(2)
                pdf_expr = yield_match.group(3)
                
                # Create yield variable in workspace
                self.workspace.factory(f"{yield_var}[{yield_params}]")
                yields.append(yield_var)
                
                # Parse PDF expression
                pdf_name = self._parse_parentheses(pdf_expr, var_name)
                
                # Add to model components
                model_components.append(f"{yield_var}*{pdf_name}")
            else:
                # No yield variable, just PDF expression
                pdf_name = self._parse_parentheses(term, var_name)
                model_components.append(pdf_name)
        
        # Combine into final model string
        model_factory_str = ", ".join(model_components)
        
        return model_factory_str, yields


def parse_and_build_model(workspace: ROOT.RooWorkspace, 
                          model_str: str, 
                          var_name: str = None) -> Tuple[ROOT.RooAbsPdf, List[str]]:
    """
    Convenience function to parse model structure and build the model PDF.
    
    Args:
        workspace: RooWorkspace containing PDFs
        model_str: Model structure string
        var_name: Variable name for convolution operations
        
    Returns:
        Tuple of (model_pdf, list_of_yield_variables)
        
    Example:
        >>> model_pdf, yields = parse_and_build_model(
        ...     workspace,
        ...     "nsig[100,0,1000] * PROD(sig1, sig2) + nbkg[50,0,500] * PROD(bkg1, bkg2)"
        ... )
    """
    parser = ModelParser(workspace)
    model_factory_str, yields = parser.parse_model(model_str, var_name)
    
    # Create final model
    workspace.factory(f"SUM::model({model_factory_str})")
    model = workspace.pdf("model")
    
    return model, yields
