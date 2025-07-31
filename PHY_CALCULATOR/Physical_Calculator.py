import ROOT 
import os
import sys
import numpy as np
import math
from typing import Dict, List, Tuple, Union, Optional

class PhysicsCalculator:
    def __init__(self, output_rootFile: str):
        """
        Initialize the PhysicsCalculator
        
        Parameters:
        -----------
        output_rootFile : str
            Path to ROOT output file where histograms will be saved
        """
        self.output_rootFile = output_rootFile
        self.nbin_tot = 0
        self.bins = np.array([])
        self.hist_model = None
        self.nbins = []
        self.bin_boundaries = []
        self.file_created = False

    def set_bins(self, bin_boundaries: List[float], 
                bin_width: Optional[List[float]] = None, 
                bin_num: Optional[List[int]] = None):
        """
        Set bin boundaries and widths
        
        Parameters:
        -----------
        bin_boundaries : List[float]
            List of bin boundary values
        bin_width : Optional[List[float]]
            List of bin widths for each region between boundaries
        bin_num : Optional[List[int]]
            List of number of bins for each region between boundaries
        """
        if bin_width is None and bin_num is None:
            raise ValueError("Either bin_width or bin_num must be provided.")
        if bin_width is not None and bin_num is not None:
            raise ValueError("Only one of bin_width or bin_num can be provided.")
        
        self.bin_boundaries = bin_boundaries

        if bin_width is not None:
            self.bin_width = bin_width
            self.nbins = [round((bin_boundaries[i + 1] - bin_boundaries[i]) / bin_width[i])
                         for i in range(len(bin_width))]
        else:
            self.nbins = bin_num
        self.nbin_tot = int(sum(self.nbins))
        print(self.nbins)
        
        # Create array of bin edges
        bins = []
        for i in range(len(bin_boundaries) - 1):
            bins.extend(np.linspace(bin_boundaries[i], bin_boundaries[i + 1], 
                                   int(self.nbins[i]), endpoint=False))
        bins.append(bin_boundaries[-1])
        for i in range(len(bins)):
            bins[i] = round(bins[i], 4)
        self.bins = np.array(bins)

        self.hist_model = ROOT.RDF.TH1DModel("hist", "", self.nbin_tot, self.bins)

    def saveHist(self, hist: Union[ROOT.TH1, ROOT.RDF.RResultPtr], name: Optional[str] = None):
        """
        Save histogram to ROOT file
        
        Parameters:
        -----------
        hist : Union[ROOT.TH1, ROOT.RDF.RResultPtr]
            Histogram to save
        name : Optional[str]
            Custom name for the histogram in memory. If None, the current histogram name is used.
        """
        # Decide file open mode based on whether it's the first save operation
        if not self.file_created:
            file = ROOT.TFile(self.output_rootFile, "RECREATE")
            self.file_created = True
        else:
            file = ROOT.TFile.Open(self.output_rootFile, "UPDATE")
        
        if not file or file.IsZombie():
            print(f"Error: Failed to open file {self.output_rootFile}")
            return
        
        is_result_ptr = hasattr(hist, 'GetValue')
        
        if name is not None:
            if is_result_ptr:
                hist_obj = hist.GetValue()
                hist_obj.SetName(name)
                hist_obj.Write()
            else:
                hist.SetName(name)
                hist.Write()
        else:
            if is_result_ptr:
                hist.GetValue().Write()
            else:
                hist.Write()
        
        file.Close()

    def calculateEfficiency(self, rootFile_config: Dict[str, Tuple[str, str]], 
                            process_func:Optional[callable] = None, 
                            truth_varDefine:Optional[callable] = None) -> ROOT.TH1F:
        """
        Calculate and save efficiency histogram from truth and reconstruction histograms
        
        Parameters:
        -----------
        rootFile_config: Dict["tree name": ("branch name", "path/to/file.root")]
            Dictionary with exactly two items: {"truth": (...), "reconstruction": (...)}
            Example: {"truth": ("mc_vpho_M", "path/to/file.root"), 
                      "ISRphiKK": ("M_phikk", "path/to/file.root")}
        Returns:
        --------
        ROOT.TH1F
            Efficiency histogram (reconstruction/truth)
        
        Raises:
        -------
        ValueError
            If root_files doesn't contain exactly two items
        """
        # Ensure there are exactly two items
        if len( rootFile_config) != 2:
            raise ValueError(f"root_files must contain exactly 2 items, but got {len(rootFile_config)}")
        
        # Extract the two keys (in a stable order)
        keys = list(rootFile_config.keys())
        truth_key = keys[0]  # First key is assumed to be truth
        reco_key = keys[1]   # Second key is assumed to be reconstruction

        df_truth = ROOT.RDataFrame(truth_key, rootFile_config[truth_key][1])
        df_reco = ROOT.RDataFrame(reco_key, rootFile_config[reco_key][1])
        
        if process_func is not None:
            df_reco = process_func(df_reco)
        if truth_varDefine is not None:
            df_truth = truth_varDefine(df_truth) 

        # Create histograms for both entries
        truth_hist = df_truth.Histo1D(self.hist_model, rootFile_config[truth_key][0]).GetValue()
        reco_hist = df_reco.Histo1D(self.hist_model, rootFile_config[reco_key][0]).GetValue()

        for i in range(1, truth_hist.GetNbinsX() + 1):
            if reco_hist.GetBinContent(i) > truth_hist.GetBinContent(i):
                print(f"Warning: Bin {i} has reco ({reco_hist.GetBinContent(i)}) > truth ({truth_hist.GetBinContent(i)})")

        # Calculate efficiency
        eff = ROOT.TEfficiency(reco_hist, truth_hist)
        eff.SetStatisticOption(ROOT.TEfficiency.kFCP)  # R.TEfficiency.kFCP
        eff.SetConfidenceLevel(0.683)
        h_efficiency = reco_hist.Clone("effi")
        for i in range(1, reco_hist.GetNbinsX() + 1):
            h_efficiency.SetBinContent(i, eff.GetEfficiency(i))
            h_efficiency.SetBinError(i, eff.GetEfficiencyErrorLow(i))

        h_efficiency.GetYaxis().SetTitle("#varepsilon")
        h_efficiency.GetXaxis().SetTitle("#sqrt{s'} [GeV]")
        
        return h_efficiency

    def divide_hist(self, h_numerator: ROOT.TH1, h_denominator: ROOT.TH1, name: str = "divided") -> ROOT.TH1:
        """
        Divide two histograms bin-by-bin and return the result.

        Parameters:
        -----------
        h_numerator : ROOT.TH1
            Numerator histogram
        h_denominator : ROOT.TH1
            Denominator histogram
        name : str
            Name for the output histogram

        Returns:
        --------
        ROOT.TH1
            Resulting histogram after division
        """
        h_result = h_numerator.Clone(name)
        h_result.Sumw2()
        h_result.Divide(h_numerator, h_denominator, 1, 1, "B")
        h_result.GetYaxis().SetTitle("Ratio")
        h_result.GetXaxis().SetTitle(h_denominator.GetXaxis().GetTitle())
        return h_result
    
    def getNsigHist(self, nsig_txt: str) -> Optional[ROOT.TH1F]:
        """
        Read signal data from text file and create histogram
        
        Parameters:
        -----------
        nsig_txt : str
            Path to text file containing signal data
            
        Returns:
        --------
        Optional[ROOT.TH1F]
            Histogram with signal data or None if file doesn't exist
        """
        if not os.path.isfile(nsig_txt):
            print(f"Error: Unable to open file {nsig_txt}")
            return None
        
        # Read data from file
        xValues, xErrors, yValues, yErrors = [], [], [], []
        with open(nsig_txt, 'r') as file:
            for line in file:
                try:
                    x, xErr, y, yErr, other = map(float, line.split())
                    xValues.append(x)
                    xErrors.append(xErr)
                    yValues.append(y)
                    yErrors.append(yErr)
                except ValueError:
                    print(f"Error: Invalid line format in file {nsig_txt}")
        
        # Create histogram
        h_nsig = ROOT.TH1F("nsig", "Nsig;#sqrt{s'}[GeV];Nsig", self.nbin_tot, self.bins)
        
        # Fill histogram with data
        for i in range(self.nbin_tot):
            h_nsig.SetBinContent(i + 1, yValues[i])
            h_nsig.SetBinError(i + 1, yErrors[i])
        
        return h_nsig
        
    def Wx2nd_noVP(self, s, x):
        """Radiator correction up to second order"""
        alpha = 1./137.
        pi = math.pi
        me = 0.511 * 0.001  # e mass
        xi2 = 1.64493407
        xi3 = 1.2020569
        sme = math.sqrt(s)/me
        L = 2*math.log(sme)
        beta = 2*alpha/pi*(L-1)
        delta2 = (9./8.-2*xi2)*L*L - (45./16.-5.5*xi2-3*xi3)*L-6./5.*xi2*xi2-4.5*xi3-6*xi2*math.log(2.)+3./8.*xi2+57./12.
        ap = alpha/pi
        Delta = 1 + ap *(1.5*L + 1./3.*pi*pi-2) + ap*ap *delta2
        wsx = Delta * beta * math.pow(x, beta-1) - 0.5*beta*(2-x)
        wsx2 = (2-x)*( 3*math.log(1-x)-4*math.log(x) ) - 4*math.log(1-x)/x - 6 + x
        wsx = wsx + beta*beta/8. *wsx2
        return wsx
        
    def calculateISRLumino(self, luminosities : Optional[dict] =None):
        """Calculate ISR luminosity"""
        luminosities = {10.58: 365.55} if luminosities is None else luminosities
     
        h_lum = ROOT.TH1F("lumino", ";#sqrt{s^{`}}[GeV];L_{eff} [pb^{-1}]", self.nbin_tot, self.bins)       
        
        for i in range(h_lum.GetNbinsX()):
            bin_left = h_lum.GetBinLowEdge(i + 1) 
            bin_right = h_lum.GetBinLowEdge(i + 2)
            sqrts = h_lum.GetBinCenter(i + 1)
            sqrts_sqr = sqrts * sqrts

            eff_lum = 0
            for Ecm, lum_value in luminosities.items():
                Ecm_sqr = Ecm * Ecm 
                x = 1 - sqrts_sqr / Ecm_sqr 
                xstep = (1 - pow(bin_left, 2)/Ecm_sqr) - (1 - pow(bin_right, 2)/Ecm_sqr)
                eff_lum += lum_value * self.Wx2nd_noVP(Ecm_sqr, x) * xstep 

            h_lum.SetBinContent(i + 1, eff_lum * 1000)  # *1000 convert to pb^-1
            h_lum.SetBinError(i+1,0)
        
        return h_lum
    

    def calculate_cross_section(self, 
                                h_nsig: ROOT.TH1,
                                h_eff: ROOT.TH1,
                                h_lum: ROOT.TH1,
                                h_subtract: Optional[List[ROOT.TH1]] = None,
                                branching_fraction: Tuple[float, float] = None) -> ROOT.TH1F:
        """
        Calculate cross section using the formula: (nsig - nsubtract) / (effi * lumio * Bsub)
        
        Parameters:
        -----------
        h_nsig : ROOT.TH1
            Signal histogram
        h_eff : ROOT.TH1
            Efficiency histogram
        h_lum : ROOT.TH1
            Luminosity histogram
        h_subtract : Optional[List[ROOT.TH1]]
            List of histograms to subtract from signal (e.g., backgrounds)
        branching_fraction : Optional[Tuple[float, float]]
            Tuple containing (value, error) for the branching fraction
            
        Returns:
        --------
        ROOT.TH1F
            Cross section histogram with properly propagated errors
        """
        # Default branching fraction if not provided (1.0 ± 0.0)
        if branching_fraction is None:
            branching_fraction = (1.0, 0.0)
        
        bf_value, bf_error = branching_fraction
        
        h_cross = h_nsig.Clone("cross_section")
        h_cross.SetTitle("Cross Section;#sqrt{s'}[GeV];#sigma [pb]")
        
        if h_subtract is not None and len(h_subtract) > 0:
            for h in h_subtract:
                h_cross.Add(h, -1.0)  # Subtract histogram
        
        # Create temporary histogram for denominator (efficiency * luminosity * BF)
        h_denom = h_eff.Clone("denominator")
        h_denom.Multiply(h_lum)
        
        # Divide by denominator
        h_cross.Divide(h_denom)
        
        # Apply branching fraction and propagate its error
        for i in range(1, h_cross.GetNbinsX() + 1):
            bin_value = h_cross.GetBinContent(i)
            bin_error = h_cross.GetBinError(i)
            
            # Calculate final value divided by branching fraction
            final_value = bin_value / bf_value
            
            # Propagate errors
            # For f = A/B, (σf/f)² = (σA/A)² + (σB/B)²
            rel_error_bin = bin_error / bin_value
            rel_error_bf = bf_error / bf_value
            final_error = final_value * math.sqrt(rel_error_bin**2 + rel_error_bf**2)
                
            # Update bin value and error
            h_cross.SetBinContent(i, final_value)
            h_cross.SetBinError(i, final_error)
        
        # Optional: Add systematic error handling here if needed
        
        return h_cross

    def combine_results(self, histograms: List[ROOT.TH1]) -> ROOT.TH1:
        """
        Combine multiple histograms using a weighted average based on their errors.
        
        Parameters:
        -----------
        histograms : List[ROOT.TH1]
            List of histograms to combine
            
        Returns:
        --------
        ROOT.TH1
            Combined histogram with weighted average values
            
        Raises:
        -------
        ValueError
            If the histogram list is empty or contains only one histogram
        """
        if not histograms:
            raise ValueError("No histograms provided to combine")
        
        if len(histograms) == 1:
            return histograms[0].Clone("combined")
        
        # Use the first histogram as a template
        h_combined = histograms[0].Clone("combined")
        h_combined.Reset()
        
        for i in range(1, h_combined.GetNbinsX() + 1):
            sum_weights = 0.0
            weighted_sum = 0.0
            
            # Calculate weighted sum for this bin across all histograms
            for hist in histograms:
                bin_value = hist.GetBinContent(i)
                bin_error = hist.GetBinError(i)
                
                # Skip bins with zero error to avoid division by zero
                if bin_error <= 0:
                    continue
                
                weight = 1.0 / (bin_error * bin_error)
                weighted_sum += bin_value * weight
                sum_weights += weight
            
            # Set bin content and error
            if sum_weights > 0:
                h_combined.SetBinContent(i, weighted_sum / sum_weights)
                h_combined.SetBinError(i, math.sqrt(1.0 / sum_weights))
            else:
                h_combined.SetBinContent(i, 0)
                h_combined.SetBinError(i, 0)
        
        return h_combined

    def calculate_Jpsi_fraction(self):
        return 0.0
