import ROOT
import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any

class HistStyle:
    """
    Python class to encapsulate histogram drawing style parameters
    """
    def __init__(
        self,
        draw_option: str = "HIST",
        line_color: int = ROOT.kBlack,
        line_style: int = 1,
        line_width: int = 1,
        fill_style: int = 0,
        marker_style: int = 20,
        marker_size: float = 1.0,
        stack: bool = False,
        fill: bool = False
    ):
        """Constructor with defaults"""
        self.draw_option = draw_option
        self.line_color = line_color
        self.line_style = line_style
        self.line_width = line_width
        self.fill_style = fill_style
        self.marker_style = marker_style
        self.marker_size = marker_size
        self.stack = stack
        self.fill = fill
    
    @staticmethod
    def error_bars(color: int) -> 'HistStyle':
        """Create histogram style with error bars"""
        return HistStyle("E1", color, 1, 1, 0, 20, 1.0, False, False)
    
    @staticmethod
    def filled_hist(color: int, fill_style: int = 3001, stacked: bool = True) -> 'HistStyle':
        """Create filled histogram style"""
        return HistStyle("HIST", color, 1, 1, fill_style, 1, 1.0, stacked, True)
    
    @staticmethod
    def line_hist(color: int, line_style: int = 1, line_width: int = 2) -> 'HistStyle':
        """Create line histogram style"""
        return HistStyle("HIST", color, line_style, line_width, 0, 1, 1.0, False, False)
    
    @staticmethod
    def filled_line_hist(color: int, fill_style: int = 3001, line_width: int = 1) -> 'HistStyle':
        """Create non-stacked but filled histogram style"""
        return HistStyle("HIST", color, 1, line_width, fill_style, 1, 1.0, False, True)
    
    @staticmethod
    def points(color: int, marker_style: int = 20, marker_size: float = 1.0) -> 'HistStyle':
        """Create points histogram style"""
        return HistStyle("P", color, 1, 1, 0, marker_style, marker_size, False, False)
    
    def get_draw_option(self, same: bool = False) -> str:
        """Get the actual draw option string to use"""
        opt = self.draw_option
        if self.fill and not self.stack and "HIST" in self.draw_option:
            opt += " F"
        if same:
            opt += " SAME"
        return opt


def ensure_styles(num: int, styles: List[HistStyle] = None) -> List[HistStyle]:
    """
    Helper function to set default styles if not enough are provided
    """
    if styles is None:
        styles = []
    result = list(styles)  # Make a copy to avoid modifying the original
    
    # Define some default colors if we need to add more styles
    default_colors = [ROOT.kRed, ROOT.kBlue, ROOT.kGreen, ROOT.kMagenta, 
                     ROOT.kCyan, ROOT.kOrange, ROOT.kViolet, ROOT.kSpring, 
                     ROOT.kTeal, ROOT.kAzure]
    color_count = len(default_colors)
    
    # Add styles if needed
    while len(result) < num:
        idx = len(result)
        color = default_colors[idx % color_count]
        
        # Alternate between different style presets
        if idx == 0:
            result.append(HistStyle.error_bars(color))
        elif idx % 3 == 1:
            #result.append(HistStyle.line_hist(color, idx % 3 + 1))
            result.append(HistStyle.line_hist(color, idx % 3 ))
        else:
            result.append(HistStyle.filled_hist(color, 3001 + (idx % 10)))
    
    return result


def style_draw(
    hist_list: List[Union[ROOT.TH1F,ROOT.TH1D]],
    output_name: str,
    leg_texts: List[str] = None,
    styles: List[HistStyle] = None,
    show_stats: bool = False,
    log_y: bool = False,
    legend_position: int = 2,
    y_min: float = 0,
    y_max: float = -1,
    use_user_y_range: bool = False,
    pad: Optional[ROOT.TPad] = None,
    save: bool = True,
) -> Union[ROOT.TCanvas, ROOT.TPad]:
    """
    Advanced histogram drawing function with style objects.
    If pad is provided, draw into that pad instead of creating a new canvas.
    
    Parameters:
    -----------
    hist_list : List[ROOT.TH1F]
        List of histogram pointers
    output_name : str
        Output file name
    leg_texts : List[str], optional
        Vector of legend entries
    styles : List[HistStyle], optional
        Vector of HistStyle objects for each histogram
    show_stats : bool, optional
        Whether to show statistics boxes
    log_y : bool, optional
        Whether to use logarithmic Y scale
    legend_position : int, optional
        Position of the legend (0=left, 1=middle, 2=right)
    y_min : float, optional
        Minimum value for Y-axis (if use_user_y_range is True)
    y_max : float, optional
        Maximum value for Y-axis (if use_user_y_range is True)
    use_user_y_range : bool, optional
        Whether to use user-provided Y-axis range
    pad : ROOT.TPad, optional
        Existing TPad to draw into instead of creating a new canvas
    """
    if leg_texts is None:
        leg_texts = []
    
    if styles is None:
        styles = []
    
    # CRITICAL: Clone histograms IMMEDIATELY to avoid address issues
    # Create unique ID for this drawing session
    uid = f"{ROOT.TUUID().AsString()}"
    
    # Get number of histograms from the list length
    num = len(hist_list)
    
    # Clone all histograms first with unique names
    cloned_hists = []
    for i in range(num):
        if "RResultPtr" in str(type(hist_list[i])):
            h_temp = hist_list[i].GetValue()
        else:
            h_temp = hist_list[i]
        h_clone = h_temp.Clone(f"{h_temp.GetName()}_clone_{uid}_{i}")
        ROOT.SetOwnership(h_clone, False)
        cloned_hists.append(h_clone)
    
    # Work with cloned list from now on
    working_hists = cloned_hists
    
    ROOT.gStyle.SetLabelSize(0.04,"xyz")
    ROOT.gStyle.SetTitleSize(0.04,"xyz")
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetMarkerSize(0.5)
    ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetPadGridX(1)
    ROOT.gStyle.SetPadGridY(1)

    ROOT.gStyle.SetLegendFont(22)
    ROOT.gStyle.SetLegendBorderSize(0)
    ROOT.gStyle.SetLegendFillColor(0) 
    ROOT.gStyle.SetLabelFont(22,"XYZ")
    ROOT.gStyle.SetTitleFont(22,"XYZ")
    
    # Ensure we have enough styles for all histograms
    styles = ensure_styles(num, styles)
    
    # Create canvas
    if pad is not None:
        # Don't call UseCurrentStyle() for pad to preserve user's margin settings
        # else it would override user's margin settings, and use the gStyle settings
        #pad.UseCurrentStyle()
        pad.SetTickx(1)
        pad.SetTicky(1)
        pad.SetGridx(1)
        pad.SetGridy(1)
        pad.cd()
        
        c = pad.GetCanvas()
    else:
        c = ROOT.TCanvas("c", "", 1600, 1080)
        # Apply margin settings only for new canvas
        ROOT.gStyle.SetPadTopMargin(.1)
        ROOT.gStyle.SetPadLeftMargin(.14)
        ROOT.gStyle.SetPadRightMargin(.07)
        ROOT.gStyle.SetPadBottomMargin(.14)
        c.UseCurrentStyle()
        if log_y:
            c.SetLogy()

    if log_y and pad is not None:
        pad.SetLogy()

    # Configure stats display
    ROOT.gStyle.SetOptStat(1111 if show_stats else 0)
    ROOT.gStyle.SetStatX(0.93)
    ROOT.gStyle.SetStatY(0.9)

    
    # UNIQUE stack names to avoid collisions across calls
    hs_stacked = ROOT.THStack(f"hsStacked_{uid}", "")
    hs_unstacked = ROOT.THStack(f"hsUnstacked_{uid}", "")
    
    has_stack = False
    has_unstacked = False

    global_min = 1e9
    global_max = -1e9
    st = [None] * num

    # Apply styling to all cloned histograms
    for i in range(num):
        h = working_hists[i]
        
        # Apply styling
        h.SetLineColor(styles[i].line_color)
        h.SetLineStyle(styles[i].line_style)
        h.SetLineWidth(styles[i].line_width)
        h.SetMarkerStyle(styles[i].marker_style)
        h.SetMarkerSize(styles[i].marker_size)
        h.SetMarkerColor(styles[i].line_color)
        
        # Set Y-axis , you could give y axis unit in original title place
        title = working_hists[0].GetYaxis().GetTitle()

        # if is in [bracket], treat as unit, y title would be Entries/(XX unit)
        if title and title[0] == "[" and title[-1] == "]":
            y_unit = title[1:-1]

            bin_width = working_hists[0].GetBinWidth(1)
            if y_unit.strip() == "":
                y_unit = ""  # keep empty to avoid trailing space
            if y_unit == "MeV":
                title = f"Events/({bin_width * 1000:.2f} MeV)"
            elif y_unit:
                title = f"Events/({bin_width:.2f} {y_unit})"
            else:
                title = f"Events/({bin_width:.2f})"
        h.GetYaxis().SetTitle(title)
        
        # Apply fill style to all histograms that need it
        if styles[i].stack or styles[i].fill:
            h.SetFillColor(styles[i].line_color)
            h.SetFillStyle(styles[i].fill_style)
        
        # Add to appropriate THStack
        if styles[i].stack:
            hs_stacked.Add(h)
            has_stack = True
        elif styles[i].draw_option == "HIST":
            hs_unstacked.Add(h)
            has_unstacked = True
    
        # Calculate global Y range including error bars if applicable
        for b in range(1, h.GetNbinsX() + 1):
            cval = h.GetBinContent(b)
            cerr = h.GetBinError(b)
            
            # For histograms with error bars, consider the content ± error
            if "E" in styles[i].draw_option:
                global_min = min(global_min, cval - cerr)
                global_max = max(global_max, cval + cerr)
            else:
                global_min = min(global_min, cval)
                global_max = max(global_max, cval)

    if has_stack:
        global_max = max(hs_stacked.GetStack().Last().GetMaximum(), global_max)

    # Add some padding to the global range
    if log_y:
        global_min = max(global_min, 0.1)
        global_max *= 10
    else:
        global_min = 0 if global_min > 0 else global_min * 1.1
        global_max *= 1.1  # Add 10% padding
     
    # Use user-provided range if specified, otherwise use calculated global range
    final_min = y_min if use_user_y_range and y_max > y_min else global_min
    final_max = y_max if use_user_y_range and y_max > y_min else global_max
    
    # Draw the stacks and histograms in the right order
    if has_stack:
        hs_stacked.Draw("HIST")
        hs_stacked.GetXaxis().SetTitle(working_hists[0].GetXaxis().GetTitle())
        hs_stacked.GetYaxis().SetTitle(working_hists[0].GetYaxis().GetTitle())
        
        # Apply Y range
        hs_stacked.SetMinimum(final_min)
        hs_stacked.SetMaximum(final_max)
        
        # Draw unstacked histograms if any
        if has_unstacked:
            hs_unstacked.Draw("HIST NOSTACK SAME")
    elif has_unstacked:
        # Only unstacked histograms
        hs_unstacked.Draw("HIST NOSTACK")
        hs_unstacked.GetXaxis().SetTitle(working_hists[0].GetXaxis().GetTitle())
        hs_unstacked.GetYaxis().SetTitle(working_hists[0].GetYaxis().GetTitle())
        
        # Apply Y range
        hs_unstacked.SetMinimum(final_min)
        hs_unstacked.SetMaximum(final_max)
    elif working_hists:
        # If no stacks were used at all, draw first histogram directly
        working_hists[0].Draw(styles[0].get_draw_option())
        
        # Apply Y range
        working_hists[0].GetYaxis().SetRangeUser(final_min, final_max)
    
    # Draw histograms with special drawing options
    for i in range(num):
        if not styles[i].stack and styles[i].draw_option != "HIST":
            working_hists[i].Draw(styles[i].get_draw_option(True))  # Always draw with SAME
        
        # Configure stats boxes if enabled
        if show_stats and i > 0:
            ROOT.gStyle.SetStatTextColor(styles[i].line_color)
            c.Update()
            st[i] = working_hists[i].FindObject("stats")
            if st[i]:
                # Position stats boxes based on legend position
                if legend_position == 0:
                    x1, x2 = 0.14, 0.34  # left
                elif legend_position == 1:
                    x1, x2 = 0.43, 0.63  # middle
                else:
                    x1, x2 = 0.73, 0.93  # right
                
                st[i].SetX1NDC(x1)
                st[i].SetX2NDC(x2)
                st[i].SetY1NDC(0.9 - 0.16 * i - 0.16)
                st[i].SetY2NDC(0.9 - 0.16 * i)
    
    # Create and configure legend
    num_stats = num if show_stats else 0
    
    if legend_position == 0:  # left
        legend = ROOT.TLegend(0.14, 0.9 - 0.16 * num_stats - 0.05 * num, 0.34, 0.9 - 0.16 * num_stats)
    elif legend_position == 1:  # middle
        legend = ROOT.TLegend(0.43, 0.9 - 0.16 * num_stats - 0.05 * num, 0.63, 0.9 - 0.16 * num_stats)
    else:  # right (default)
        legend = ROOT.TLegend(0.73, 0.9 - 0.16 * num_stats - 0.05 * num, 0.93, 0.9 - 0.16 * num_stats)
    
    legend.SetFillStyle(0)
    legend.SetBorderSize(0)
    
    # Add legend entries if provided
    if leg_texts:
        for i in range(min(num, len(leg_texts))):
            if styles[i].stack or styles[i].fill:
                leg_style = "f"  # filled style
            elif "E" in styles[i].draw_option or "P" in styles[i].draw_option:
                leg_style = "lpe"  # line+point+error style
            else:
                leg_style = "l"  # line style

            # Use the histogram directly from our working list
            legend.AddEntry(working_hists[i], str(leg_texts[i]), str(leg_style))
        legend.Draw()
    
    # Store the working histograms in the canvas to prevent garbage collection
    # This is CRITICAL to keep the histogram objects alive
    if pad is not None:
        if not hasattr(pad, '_stored_hists'):
            pad._stored_hists = []
        pad._stored_hists.extend(working_hists)
        pad._stored_stacks = [hs_stacked, hs_unstacked]
        pad._stored_legend = legend
    else:
        if not hasattr(c, '_stored_hists'):
            c._stored_hists = []
        c._stored_hists.extend(working_hists)
        c._stored_stacks = [hs_stacked, hs_unstacked]
        c._stored_legend = legend
    
    c.SetName(f"canvas : {working_hists[0].GetTitle()}")
    c.Update()

    # Save output:
    # If drawing inside an external pad, user may want to control when to save.
    # We still honor output_name; it will save the whole canvas.
    if save and output_name:
        out_dir = os.path.dirname(output_name)
        if out_dir and not os.path.exists(out_dir):
            print("Warning: Output directory does not exist, exit.")
            return pad if pad is not None else c
            #os.makedirs(out_dir, exist_ok=True)
        if output_name.endswith(".root"):
            try:
                rf_out = ROOT.TFile.Open(output_name, "UPDATE")
                if not rf_out or rf_out.IsZombie():
                    rf_out = ROOT.TFile(output_name, "RECREATE")
                c.Write()
                rf_out.Close()
            except Exception as e:
                print(f"Error saving to ROOT file: {e}")
                c.Print(output_name)
        else:
            c.Print(output_name)

    return pad if pad is not None else c


def graph_draw(
    graph_list: List[Union[ROOT.TGraph, ROOT.TGraphErrors, ROOT.TGraphAsymmErrors]],
    output_name: str,
    leg_texts: List[str] = None,
    line_color: List[int] = None,
    line_style: List[int] = None,
    line_width: List[int] = None,
    marker_style: List[int] = None,
    marker_size: List[float] = None
) -> None:
    """
    Function to draw multiple TGraph objects with customizable styles
    """
    if leg_texts is None:
        leg_texts = []
    
    if line_color is None:
        line_color = [2, ROOT.kBlue, 1]
    
    if line_style is None:
        line_style = [1, 1, 1]
    
    if line_width is None:
        line_width = [1, 1, 1]
    
    if marker_style is None:
        marker_style = [20, 24,29]
    
    if marker_size is None:
        marker_size = [1.0, 1.0, 2.0]
       
    ROOT.gStyle.SetLabelSize(0.04,"xyz")
    ROOT.gStyle.SetPadTopMargin(.1)
    ROOT.gStyle.SetPadLeftMargin(.14)
    ROOT.gStyle.SetPadRightMargin(.07)
    ROOT.gStyle.SetPadBottomMargin(.14)
    ROOT.gStyle.SetTitleSize(0.04,"xyz")
    ROOT.gStyle.SetOptTitle(0)
    ROOT.gStyle.SetMarkerSize(0.5)
    ROOT.gStyle.SetLabelFont(22,"XYZ")
    ROOT.gStyle.SetTitleFont(22,"XYZ")
    ROOT.gStyle.SetCanvasDefH(1080)
    ROOT.gStyle.SetCanvasDefW(1600)
    ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
    ROOT.gStyle.SetPadTickX(1)
    ROOT.gStyle.SetPadTickY(1)
    ROOT.gStyle.SetPadGridX(1)
    ROOT.gStyle.SetPadGridY(1) 

    c = ROOT.TCanvas("c", "", 1600, 1080)
    
    num = len(graph_list)
    for i in range(num):
        graph_list[i].GetXaxis().SetTitle(graph_list[0].GetXaxis().GetTitle())
        graph_list[i].GetYaxis().SetTitle(graph_list[0].GetYaxis().GetTitle())
        graph_list[i].GetXaxis().CenterTitle()
        graph_list[i].GetYaxis().CenterTitle()
        graph_list[i].GetXaxis().SetTitleOffset(1.2)
        graph_list[i].GetYaxis().SetTitleOffset(1.2)
        
        graph_list[i].SetMarkerStyle(marker_style[i] if i < len(marker_style) else 20)
        graph_list[i].SetLineWidth(line_width[i] if i < len(line_width) else 1)
        graph_list[i].SetLineColor(line_color[i] if i < len(line_color) else ROOT.kBlack)
        graph_list[i].SetLineStyle(line_style[i] if i < len(line_style) else 1)
        graph_list[i].SetMarkerSize(marker_size[i] if i < len(marker_size) else 1.0)
        graph_list[i].SetMarkerColor(line_color[i] if i < len(line_color) else ROOT.kBlack)
        graph_list[i].SetTitle("")
        
        if i == 0:
            graph_list[i].Draw("AP")
        else:
            graph_list[i].Draw("P same")
    
    if leg_texts:
        # Create and position legend
        legend = ROOT.TLegend(0.73, 0.9 - 0.05 * num, 0.93, 0.9)
        legend.SetBorderSize(1)
        legend.SetFillColor(0)
        
        for i in range(min(num, len(leg_texts))):
            legend.AddEntry(graph_list[i], leg_texts[i], "PEL")
        
        legend.SetBorderSize(0)
        legend.SetFillStyle(0)
        legend.SetFillColor(0) 
        legend.Draw()

    c.Update()
    c.Print(output_name)


# For compatibility with Jupyter notebooks and interactive use
if __name__ == "__main__":
    print("AutoDraw Python module loaded.")
    print("Use help(function_name) for more information on specific functions.")
