import ROOT
import sys
from typing import Tuple, List, Optional, Dict
from .AutoDraw import style_draw, HistStyle
from math import sqrt
import re


class Brush:
    def __init__(self,output_dir):
        self.output_dir = output_dir
        pass

        
    def parse_topoana_file(self,filepath: str, top_n: int = 5) -> List[Dict]:
        """
        Parse the topoana output file and extract the top decay modes
        Args:
            filepath: Path to the topoana.txt file
            top_n: Number of top decay modes to return
        Returns:
            List of dictionaries with decay information
        """
        decay_modes = []
        parsing_decay_states = False
        
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not parsing_decay_states and "Decay initial-final states:" in line:
                    parsing_decay_states = True
                    continue
                    
                if parsing_decay_states and line.startswith("rowNo:"):
                    parts = re.split(r'\s+', line)
                    try:
                        dcyid = int(parts[3])
                        entries = int(parts[5])
                        next_line = next(f).strip()  # Read the decay string line
                        decay_string = next_line.strip()
                        
                        decay_modes.append({
                            "id": dcyid,
                            "entries": entries,
                            "decay": decay_string
                        })
                        
                        if len(decay_modes) >= top_n:
                            break
                    except (IndexError, ValueError) as e:
                        continue
        
        return decay_modes

    def plot_bkg(self, bkg_df: ROOT.RDataFrame, topoana_txt: str):
        """
        Plot the background distribution for the top decay modes and others.
        """
        top_decays = self.parse_topoana_file(topoana_txt)
        top_decays = top_decays[1:10]
        #top_decays = top_decays[::-1]
        
        filtered_dfs = []
        decay_names = []
        
        others_filter = ""
        for i, decay in enumerate(top_decays):
            filtered_df = bkg_df.Filter(f"iDcyIFSts == {decay['id']}")
            filtered_dfs.append(filtered_df)
            decay_names.append(f"{decay['decay']} ({decay['entries']})")
            
            # Build the "others" filter condition
            if i == 0:
                others_filter = f"iDcyIFSts != {decay['id']}"
            else:
                others_filter += f" && iDcyIFSts != {decay['id']}"
        
        others_df = bkg_df.Filter(others_filter)
        filtered_dfs.append(others_df)
        decay_names.append(f"Others")
        
        styles = [
            HistStyle.filled_hist(1, 3003),
            HistStyle.filled_hist(2, 3003),
            HistStyle.filled_hist(3, 3003),
            HistStyle.filled_hist(4, 3003),
            HistStyle.filled_hist(5, 3003),
            HistStyle.filled_hist(6, 3003),
            HistStyle.filled_hist(7, 3003),
            HistStyle.filled_hist(8, 3003),
            HistStyle.filled_hist(9, 3003),
            HistStyle.filled_hist(ROOT.kBlack, 3003)  
        ]

        
        # Plot invariant mass distributions
        mass_hists = []
        for i, df in enumerate(filtered_dfs):
            hist = df.Histo1D((f"m4K_{i}", ";//sqrt{s^{//prime}};", 100, 2, 4), "vpho_M")
            mass_hists.append(hist.GetValue())

        style_draw(mass_hists, self.output_dir + "bkg_top7_mass.png", decay_names, styles, y_min=0, y_max=150, use_user_y_range=1,legend_position=0)
        
        '''
        # Also plot m2Recoil distribution
        recoil_hists = []
        for i, df in enumerate(filtered_dfs):
            hist = df.Histo1D((f"m2Recoil_{i}", ";m^{2}(recoil) [GeV^{2}/c^{4}];Events", 100, -0.5, 1), "vpho_m2Recoil")
            recoil_hists.append(hist.GetValue())
        
        style_draw(recoil_hists, self.output_dir + "bkg_top7_m2recoil.png", decay_names, styles,y_min=0,y_max = 100,use_user_y_range=1)

        # m phi 
        phi_hists = []
        for i, df in enumerate(filtered_dfs):
            hist = df.Histo1D((f"m_phi_{i}", ";M_{#phi};", 100, 0.98, 1.1), "M_phi")
            phi_hists.append(hist.GetValue())
        style_draw(phi_hists, self.output_dir + "bkg_top7_m_phi.png", decay_names, styles, y_min=0, y_max=350, use_user_y_range=1)

        # isr E , theta
        isr_E_hists = []
        for i, df in enumerate(filtered_dfs):
            hist = df.Histo1D((f"isr_E_{i}", ";E_{ISR} [GeV];Events", 100, 4.7, 5.3), "isr_ee_cms_E")
            isr_E_hists.append(hist.GetValue())
        style_draw(isr_E_hists, self.output_dir + "bkg_top7_isr_E.png", decay_names, styles, y_min=0, y_max=200, use_user_y_range=1)

        isr_theta_hists = []
        for i, df in enumerate(filtered_dfs):
            df_with_cos = df.Define("cos_theta", "cos(isr_ee_cms_theta)")
            hist = df_with_cos.Histo1D((f"isr_theta_{i}", ";cos(#theta_{ISR}) ;Events", 100,-1, 1), "cos_theta")
            isr_theta_hists.append(hist.GetValue())
        style_draw(isr_theta_hists, self.output_dir + "bkg_top7_isr_theta.png", decay_names, styles, y_min=0, y_max=120, use_user_y_range=1)
        '''

def plot_data_vs_mc(trees, variable, nbins, xmin, xmax, cuts, output_path, 
                           title, leg_entries, styles, scales=[1.0, 1.0, 1.0, 1.0], 
                           hist_names=None):
    """
    Generic function to create and style histograms for comparing data, signal MC, and background MC
    
    Parameters:
    -----------
    trees: list of TTree
        List of ROOT trees [data, bkg, sig, (optional)sideband_data]
    variable: str
        Variable to plot from the trees
    nbins, xmin, xmax: int, float, float
        Histogram binning parameters
    cuts: list of str
        List of cut strings for each tree
    output_path: str
        Path to save the output plot
    title: str
        Histogram title/x-axis label
    leg_entries: list of str
        Legend entries
    styles: list of ROOT.HistStyle
        Styles for histograms
    scales: list of float
        Scale factors for histograms
    hist_names: list of str, optional
        Names for histograms, if None will be generated
    
    Returns:
    --------
    list of TH1F
        List of created histograms
    """
    if hist_names is None:
        hist_names = [f"h_{variable}_{i}" for i in range(len(trees))]
    
    # Create histograms
    histograms = []
    for i, tree in enumerate(trees):
        if tree is None:
            continue
        
        hist = ROOT.TH1F(hist_names[i], title, nbins, xmin, xmax)
            
        # Draw with cut if provided
        if cuts[i]:
            tree.Draw(f"{variable}>>{hist_names[i]}", cuts[i])
        else:
            tree.Draw(f"{variable}>>{hist_names[i]}")
        
        hist.Scale(scales[i])
            
        histograms.append(hist)
    
    # change the order of stacked histograms: signal bkg sideband
    if len(histograms) == 4 :
        histograms = [histograms[0]] + [histograms[3]] + histograms[1:3]
        leg_entries = [leg_entries[0]] + [leg_entries[3]] + leg_entries[1:3]
        styles = [styles[0]] + [styles[3]] + styles[1:3]
    
    # Style and save histograms
    #style_draw(histograms, output_path, leg_entries, styles,0,0,0,y_max=400,use_user_y_range=1)
    style_draw(histograms, output_path, leg_entries, styles,0,0,0)

    return histograms