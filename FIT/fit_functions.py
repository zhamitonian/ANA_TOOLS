import ROOT
from ROOT import RooFit as rf
from ROOT import RooStats
import ctypes  # Add this import for c_double
import sys
import contextlib
import os
import time
import math
from contextlib import contextmanager
from typing import Optional, Union
from array import array
from DRAW import style_draw
from PHY_CALCULATOR import PhysicsCalculator
from .fit_tools import FIT_UTILS, QUICK_FIT

from math import sqrt

def perform_resonance_fit(input_tree:ROOT.TTree, output_dir:str, log_file:Optional[str]=None, bin_fit_range:Optional[str]=None, binned_fit = False,joint_fit:bool=False, **kwargs):
    """
    Perform sPlot analysis for digamma to diphi process
    
    Args:
        input_tree: Input ROOT TTree with the data
        output_dir: Output directory for plots and results
        log_file: Optional file to save logs (stdout and stderr)
        bin_fit_range: Optional string representing the bin range for fitting
        binned_fit: Whether to perform binned fit or unbinned fit
        **kwargs: Additional keyword arguments:
            branches_name: List of branch names to combine (e.g., ["phi1_M", "phi2_M"])
    """
    #reso, mass, width = "phi", 1.0195, 0.004249
    reso, mass, width = kwargs.get("particle_config", ("phi", 1.0195, 0.004249))
    x_min, x_max, nbin = kwargs.get("fit_config", (1, 1.06, 60))
    #x_min, x_max, nbin = 0.99, 1.15, 100 # ralf siedl  
    #x_min, x_max, nbin = 1, 1.06, 60

    var_config = [(f"{reso}_M", x_min, x_max)]  
    tools = FIT_UTILS(log_file=log_file, var_config= var_config)

    branches_name = kwargs.get('branches_name', None)
    print(f"binned_fit: { binned_fit}")
    # redirect log
    with tools.redirect_output():
        # Add some useful information at the beginning of the log
        print(f"=== Starting sPlot Analysis ===")
        print(f"Output file: {output_dir}")
        if branches_name is not None:
            print(f"Branch names to combine: {branches_name}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"===============================")
        
        w = ROOT.RooWorkspace("w", "workspace")
        
        dataset = tools.handle_dataset(input_tree, w, branches_name, binned_fit, nbin )
        # Save dataset to workspace
        w.Import(dataset, ROOT.RooFit.Rename("dataset"))

        print(input_tree.GetEntries())
        print(f"Dataset created with {dataset.numEntries()} entries")
        
        # BW 
        w.factory(f"BreitWigner::reso_bw({reso}_M, {mass}, {width})")
        #w.factory(f"BreitWigner::reso_bw({reso}_M, mass[{mass}, 3.09, 3.11], width[{width},1e-5, 0.01])")
        #w.factory(f"BreitWigner::sig_pdf({reso}_M, {mass}, {width})")
        #w.factory(f"BreitWigner::reso_bw({reso}_M, mass[{mass}, 1.01, 1.03], width[{width}, 0.001, 0.01])")
        #w.factory(f"BreitWigner::sig_pdf({reso}_M, mass[{mass}, 1.01, 1.03], width[{width}, 0.001, 0.01])")

        # rela BW
        #formula = f"({mass} * {width}) / (pow({reso}_M*{reso}_M - {mass}*{mass}, 2) + pow({mass}*{width}, 2))"
        #w.factory(f"RooGenericPdf::reso_bw('{formula}', {{{reso}_M}})")

        #from .relativistic_breit_wigner import define_relativistic_breit_wigner
        #define_relativistic_breit_wigner(w, reso, mass, width)

        #from .__helper_functions import define_relativistic_breit_wigner
        #define_relativistic_breit_wigner(w, reso, mass, width)

        #w.factory(f"Gaussian::smear({reso}_M, smear_mean[0, -0.0001, 0.0001], smear_sigma[0.0008, 0.0002, 0.0014])")
        #w.factory(f"Gaussian::smear({reso}_M, smear_mean[0], smear_sigma[0.00083, 0.0002, 0.0014])")
        w.factory(f"Gaussian::smear({reso}_M, smear_mean[0], smear_sigma[0.0002, 0.001, 0.02])") # for jpsi fitting
        w.factory(f"FCONV::sig_pdf({reso}_M, reso_bw, smear)")
        
        # Polynomial, Chebychev, ArgusBG, or reversed Argus background PDF selection
        which_bkg = kwargs.get("which_bkg", 0)  # 0: Chebychev, 1: Polynomial, 2: ArgusBG, 3: reversed Argus 
        bkg_order = kwargs.get("bkg_order", 2)  # Order of polynomial/Chebychev (default: 1)
        bkg_funcs = ["Chebychev", "Polynomial", "ArgusBG", "revArgus", "custom", "custom2"]
        bkg_func = bkg_funcs[which_bkg]

        if bkg_func in ["Chebychev", "Polynomial"]:
            # Dynamically build coefficient list for the chosen order
            coef_list = ", ".join([f"b_{i}[-10, 10]" for i in range(bkg_order + 1)])
            w.factory(f"{bkg_func}::bkg_pdf({reso}_M, {{{coef_list}}})")
        elif bkg_func == "ArgusBG":
            # Standard Argus background
            w.factory(f"ArgusBG::bkg_pdf({reso}_M, arg_m0[{x_max}], arg_c[-20, -100, -0.01], arg_p[0.5, 0, 1])")
        elif bkg_func == "revArgus":
            # Reversed Argus: (m0-m) instead of (m-m0)
            argus_formula = f"({reso}_M / rev_m0) * pow(1 - pow(rev_m0/{reso}_M, 2), rev_p) * exp(rev_c * (1 - pow(rev_m0/{reso}_M, 2)))"
            w.factory(f"rev_m0[{x_min}]")
            w.factory("rev_c[-20, -100, -0.01]")
            w.factory("rev_p[0.5, 0, 1]")
            w.factory(f"RooGenericPdf::bkg_pdf('{argus_formula}', {{{reso}_M, rev_m0, rev_c, rev_p}})")
        elif bkg_func == "custom":
            formula = f"a*pow({reso}_M - m0, b)* exp(c * ({reso}_M - m0))"
            #w.factory(f"m0[{x_min}]")
            w.factory(f"m0[{x_min}]")
            w.factory("a[1, -10000, 100000]")
            w.factory("b[0.5, -10, 10]")
            w.factory("c[-1, -100, 100]")
            w.factory(f"RooGenericPdf::bkg_pdf('{formula}', {{{reso}_M, m0, a, b, c}})")
        elif bkg_func == "custom2":
            formula = f"pow(a + b*({reso}_M - m0), c + d*({reso}_M - m0))"
            w.factory(f"m0[{x_min}]")
            w.factory("a[1, -10000, 100000]")
            w.factory("b[0.5, -10, 10]")
            w.factory("c[1, -100, 100]")
            w.factory("d[0.1, -10, 10]")
            w.factory(f"RooGenericPdf::bkg_pdf('{formula}', {{{reso}_M, m0, a, b, c, d}})")

        w.factory("SUM::model(nsig[20000, 0, 400000] * sig_pdf, nbkg[45000, 0, 200000] * bkg_pdf)")

        # Perform fit
        model = w.pdf("model")
        if not dataset:
            print("Error: dataset is None")
            return
    
        if not model:
            print("Error: model is None")
            return

        print(f"Dataset entries: {dataset.numEntries()}")
        if dataset.numEntries() == 0:
            print("Error: dataset is empty")
            return
        
        if joint_fit:
            return w 

        result = model.fitTo(dataset, 
                             rf.Save(True), 
                             rf.NumCPU(4), 
                             rf.PrintLevel(0),
                             rf.Strategy(1))
                             #rf.Minimizer("Minuit2"))


        # --------------------------------------- splot ---------------------------------------
        # sPlot only for unbinned fits
        if not binned_fit:
            # Create sPlot for visualization using phi_M  as discriminating variables
            sData = RooStats.SPlot("sData", "sPlot", dataset, model,
                                ROOT.RooArgList(w.var("nsig"), w.var("nbkg")))
            for i in range(dataset.numEntries()):
                row = dataset.get(i)
                nsig_sw = row.getRealValue("nsig_sw")

            # Print the yields with their uncertainties
            print(f"Signal yield: {w.var('nsig').getVal()} ± {w.var('nsig').getError()}")
            print(f"Background yield: {w.var('nbkg').getVal()} ± {w.var('nbkg').getError()}")
            
            # Create sWeighted datasets
            sWeighted_data = ROOT.RooDataSet("sWeighted_data", "Data with sWeights", 
                                            dataset, dataset.get(), "", "nsig_sw")
            bkg_weighted_data = ROOT.RooDataSet("bkg_weighted_data", "Data with background sWeights", 
                                            dataset, dataset.get(), "", "nbkg_sw")
            
            out_file = ROOT.TFile(output_dir + "splot_output.root", "RECREATE")
            sWeighted_data.Write("signal_weighted_data")
            bkg_weighted_data.Write("background_weighted_data")

            out_file.Close()
        
        #----------------------------- plot distribution  ------------------------
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


        # First, create temporary frames to get the maximum values for y-axis range
        frame_reso = w.var(f"{reso}_M").frame()
        dataset.plotOn(frame_reso, rf.MarkerColor(ROOT.kBlack), rf.MarkerStyle(20), rf.Binning(nbin))
        model.plotOn(frame_reso, rf.LineColor(4))
        
        def plot(var_name:str):
            canvas = ROOT.TCanvas("c","c", 1600, 1080)
            canvas.Divide(1, 2)

            pad1 = canvas.cd(1)
            pad1.SetPad(0, 0.3, 1, 1)
            pad1.SetBottomMargin(0.01)
            pad1.SetLeftMargin(0.15)
            pad1.SetRightMargin(0.05)
            pad1.Draw()

            framex = w.var(var_name).frame()
            #framex.GetXaxis().SetRangeUser(0.986, 1.06)
            dataset.plotOn(framex, rf.Name("data"), rf.MarkerColor(ROOT.kBlack), rf.MarkerStyle(20), rf.Binning(nbin))
            model.plotOn(framex, rf.Name("sum"), rf.LineColor(4))
            model.plotOn(framex, rf.Components("sig_pdf"), rf.Name("signal"), rf.LineColor(2), rf.LineStyle(4), rf.LineWidth(3))
            model.plotOn(framex, rf.Components("bkg_pdf"), rf.Name("bkg"), rf.LineColor(ROOT.kGreen+2), rf.LineStyle(7), rf.LineWidth(3))

            framex.SetTitle("")
            #framex.GetXaxis().SetTitle("#phi Mass (GeV/c^{2}")
            framex.GetYaxis().SetTitle(f"Candidates / ({((x_max - x_min) / nbin * 1000):.3f} MeV/c^{{2}})")
            framex.GetYaxis().SetTitleOffset(1)
            
            # Set unified y-axis range
            #framex.SetMaximum(y_max)
            #framex.SetMinimum(y_min)
            
            framex.Draw()

            leg_ysize = 7
            if bin_fit_range:
                ranges = bin_fit_range.split(';')
                leg_ysize += len(ranges)

            #leg = ROOT.TLegend(0.75, 0.9 - 0.05*5, 0.95, 0.9)
            leg = ROOT.TLegend(0.7, 0.9 - 0.05*leg_ysize, 0.95, 0.9)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)
            leg.SetFillColor(0)
            leg.SetTextFont(22)
            
            # Create simple entries with correct colors
            data_entry = ROOT.TMarker(0, 0, 20)
            data_entry.SetMarkerColor(ROOT.kBlack)
            data_entry.SetMarkerStyle(20)
            
            model_entry = ROOT.TLine()
            model_entry.SetLineColor(4)  # Blue
            
            signal_entry = ROOT.TLine()
            signal_entry.SetLineColor(2)  # Red
            signal_entry.SetLineStyle(4)
            signal_entry.SetLineWidth(3)
            
            bkg_entry = ROOT.TLine()
            bkg_entry.SetLineColor(ROOT.kGreen+2)  # Match color with bkg1_curve
            bkg_entry.SetLineStyle(7)
            bkg_entry.SetLineWidth(3)

            leg.AddEntry(data_entry, "Data", "pe")
            leg.AddEntry(model_entry, "Total fit", "l")
            leg.AddEntry(signal_entry, "Signal", "l")
            leg.AddEntry(bkg_entry, "Background", "l")
            
            # Get chi-square and degrees of freedom properly
            chi2 = framex.chiSquare("sum", "data")  # Get reduced chi-square
            
            data_hist = framex.getHist("data")
            nBins = data_hist.GetN()
            nPars = result.floatParsFinal().getSize()
            ndf = nBins - nPars
            
            # Calculate chi-square
            chi2_val = chi2 * ndf
            
            # Show appropriate goodness-of-fit measure
            if binned_fit:
                # For binned fit, show chi2/ndf
                leg.AddEntry(0, "#chi^{2}/ndf = " + f"{chi2_val:.1f}/{ndf}", "")
            else:
                # For unbinned fit, show both (chi2 is less meaningful but still shown)
                chi2_val = chi2 * ndf
                leg.AddEntry(0, "#chi^{2}/ndf = " + f"{chi2_val:.1f}/{ndf}", "")
                # Could also show NLL for unbinned:
                # nll = result.minNll()
                # leg.AddEntry(0, f"NLL = {nll:.1f}", "")

            leg.AddEntry(0, "N_{sig} = " + f"{w.var('nsig').getVal():.1f} #pm {w.var('nsig').getError():.1f}", "")
            #leg.AddEntry(0, "#sigma_{gauss} = " + f"{w.var('smear_sigma').getVal():.2e} #pm {w.var('smear_sigma').getError():.2e}", "")
            leg.AddEntry(0, "#sigma_{gauss} = " + f"{(w.var('smear_sigma').getVal()*1000):.2f} #pm {(w.var('smear_sigma').getError()*1000):.2f} MeV", "")

            if bin_fit_range:
                # Add each dimension on a separate line
                for range_str in ranges:
                    range_str = range_str.strip()
                    leg.AddEntry(0, f"{range_str}", "")
            leg.Draw()

            pad2 = canvas.cd(2)
            pad2.SetPad(0, 0, 1, 0.3)
            pad2.SetTopMargin(0.01)
            pad2.SetBottomMargin(0.3)
            pad2.SetLeftMargin(0.15)
            pad2.SetRightMargin(0.05)
            pad2.Draw()

            framex_pull = w.var(var_name).frame()
            framex_pull.SetTitle("")
            framex_pull.GetYaxis().SetTitle("Pull")
            framex_pull.GetYaxis().SetTitleOffset(0.35)
            framex_pull.GetYaxis().SetTitleSize(0.1)
            framex_pull.GetYaxis().CenterTitle()
            framex_pull.GetYaxis().SetRangeUser(-5, 5)
            framex_pull.GetYaxis().SetLabelSize(0.1)
            framex_pull.GetXaxis().SetLabelSize(0.13)
            x_title = "M_{K^{+}K^{-}} (GeV/c^{2})" if var_name == "phi_M" else "M_{reso}"
            framex_pull.GetXaxis().SetTitle(x_title)
            framex_pull.GetXaxis().CenterTitle()
            framex_pull.GetXaxis().SetTitleSize(0.12)
            framex_pull.GetXaxis().SetTitleOffset(1.1)
            #framex_pull.GetXaxis().SetRangeUser(0.986, 1.06)

            pullhist = framex.pullHist("data", "sum")
            framex_pull.addObject(pullhist, "P")
            framex_pull.Draw()
            canvas.SaveAs(output_dir + f"_{var_name}.png")

        plot(f"{reso}_M")

        #return w, result, sData
        nsig = w.var("nsig").getVal()
        nsig_err = w.var("nsig").getError()
                
        # Add summary at the end of the log
        print(f"=== Analysis Complete ===")
        print(f"Fit type: {'Binned (histogram)' if binned_fit else 'Unbinned (tree)'}")
        print(f"Signal yield: {nsig:.2f} ± {nsig_err:.2f}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"========================")

        # Print the names of PDFs and coefficients in the model
        pdf_names = [model.pdfList().at(i).GetName() for i in range(model.pdfList().getSize())]
        coef_names = [model.coefList().at(i).GetName() for i in range(model.coefList().getSize())]
        print("PDFs in model:", pdf_names)
        print("Coefficients in model:", coef_names)
                
    return result, nsig, nsig_err


def perform_2dfit(tree:ROOT.TTree, output_dir:dir, log_file = None, bin_fit_range:Optional[str]=None, **kwargs):
    """
    Perform sPlot analysis for digamma to diphi process
    
    Args:
        tree_file: Input ROOT file with the data
        output_file: Output file for plots and results
        log_file: Optional file to save logs (stdout and stderr)
        bin_fit_range: Optional bin information for labeling
    
    Returns:
        tuple: (result, nsig, nsig_err)
    """

    reso1, reso2 = "phi1", "phi2"
    mass, width = 1.0195, 0.004249    
    x_min, x_max, nbin = 1, 1.06, 60
    ''' 
    reso1, reso2 = "omega1", "omega2"
    mass , width = 0.78265, 0.00849
    x_min , x_max, nbin = 0.68265 , 0.88265, 200
    '''
    '''
    reso1, reso2 = "Kstar", "antiKstar"
    mass, width = 0.89555, 0.0473
    x_min , x_max, nbin = 0.75, 1.1, 175
    '''

    var_config = [("phi1_M",x_min, x_max),("phi2_M",x_min, x_max),
                  (f"PHI_{reso1}_{reso2}", 0, math.pi*2),
                  ("vpho_M", 2, sqrt(12)), 
                  ("vpho_m2Recoil", 0, 80), 
                  ("vpho_ee_cms_pt", 0, 1)]
     
    tools = FIT_UTILS(log_file=log_file ,var_config=var_config)

    # redirect log
    with tools.redirect_output():
        # Add some useful information at the beginning of the log
        print(f"=== Starting sPlot Analysis ===")
        print(f"Output file: {output_dir}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"===============================")
        
        w = ROOT.RooWorkspace("w", "workspace")
        
        gauss_width = 0.0005
        w.factory(f"{reso1}_M[{x_min}, {x_max}]")  
        w.factory(f"{reso2}_M[{x_min}, {x_max}]")  

        dataset = tools.handle_dataset(tree, w)

        print(tree.GetEntries())
        print(f"Dataset created with {dataset.numEntries()} entries")
        
        # Use Breit-Wigner (RooBreitWigner) with PDG values for phi meson
        w.factory(f"BreitWigner::reso1_bw({reso1}_M, {mass}, {width})")
        w.factory(f"BreitWigner::reso2_bw({reso2}_M, {mass}, {width})")
        
        #w.factory(f"Gaussian::smear1({reso1}_M, smear_mean1[0,-0.001,0.001], smear_sigma1[0.0008, 5e-04, 0.005])")
        #w.factory(f"Gaussian::smear2({reso2}_M, smear_mean2[0,-0.001,0.001], smear_sigma1[0.0008, 5e-04, 0.005])")
        w.factory(f"Gaussian::smear1({reso1}_M, smear_mean1[0], smear_sigma1[0.0008, 5e-04, 0.005])")
        w.factory(f"Gaussian::smear2({reso2}_M, smear_mean2[0], smear_sigma1[0.0008, 5e-04, 0.005])")
        
        w.factory(f"FCONV::reso1_sig_pdf({reso1}_M, reso1_bw, smear1)")
        w.factory(f"FCONV::reso2_sig_pdf({reso2}_M, reso2_bw, smear2)")
        
        w.factory("PROD::sig_pdf(reso1_sig_pdf, reso2_sig_pdf)")
        #w.factory("PROD::sig_pdf(reso1_bw, reso2_bw)")
        
        # Polynomial , Chebychev 
        bkg_func = "Chebychev"  
        bkg_func = "Polynomial"  
        #w.factory(f"{bkg_func}::reso1_bkg_pdf({reso1}_M, {{b1_0[-10, 10], b1_1[-10, 10], b1_2[-10, 10], b1_3[-10, 10]}})")
        #w.factory(f"{bkg_func}::reso2_bkg_pdf({reso2}_M, {{b2_0[-10, 10], b2_1[-10, 10], b2_2[-10, 10], b2_3[-10, 10]}})")
        #w.factory(f"{bkg_func}::reso1_bkg_pdf({reso1}_M, {{b1_0[-10, 10], b1_1[-10, 10], b1_2[-10, 10]}})")
        #w.factory(f"{bkg_func}::reso2_bkg_pdf({reso2}_M, {{b2_0[-10, 10], b2_1[-10, 10], b2_2[-10, 10]}})")
        #w.factory(f"{bkg_func}::reso1_bkg_pdf({reso1}_M, {{b1_0[-10, 10]}})")
        #w.factory(f"{bkg_func}::reso2_bkg_pdf({reso2}_M, {{b2_0[-10, 10]}})")
        
        w.factory(f"{bkg_func}::reso1_bkg_pdf({reso1}_M, {{b1_0[-10, 10]}})")
        w.factory(f"{bkg_func}::reso2_bkg_pdf({reso2}_M, {{b2_0[-10, 10]}})")
        
        w.factory("PROD::bkg1(reso1_bkg_pdf, reso2_bkg_pdf)")
        w.factory("PROD::bkg2(reso1_sig_pdf, reso2_bkg_pdf)")
        w.factory("PROD::bkg3(reso1_bkg_pdf, reso2_sig_pdf)")

        #w.factory("SUM::bkg_pdf(nbkg1[50, 0, 700] * bkg1, nbkg2[50, 0, 700] * bkg2, nbkg3[50, 0, 700] * bkg3)")
        w.factory("SUM::model(nsig[100, 0, 3000] * sig_pdf, nbkg1[50, 0, 2000] * bkg1, nbkg2[50, 0, 2000] * bkg2, nbkg3[50, 0, 2000] * bkg3)")
        #w.factory("SUM::model(nsig[100, 0, 2000] * sig_pdf, bkg_pdf)")

        # Perform fit
        model = w.pdf("model")
        if not dataset:
            print("Error: dataset is None")
            return
    
        if not model:
            print("Error: model is None")
            return
    
        print(f"Dataset entries: {dataset.numEntries()}")
        if dataset.numEntries() == 0:
            print("Error: dataset is empty")
            return
        
        result = model.fitTo(dataset, 
                             rf.Save(True), 
                             rf.NumCPU(4), 
                             rf.PrintLevel(0),
                             rf.Strategy(1))
                             #rf.Minimizer("Minuit2"))

        
        # --------------------------------------- splot ---------------------------------------

        # Create sPlot for visualization using phi1_M and phi2_M as discriminating variables
        sData = RooStats.SPlot("sData", "sPlot", dataset, model,
                               ROOT.RooArgList(w.var("nsig"), w.var("nbkg1"), w.var("nbkg2"), w.var("nbkg3")))
        for i in range(dataset.numEntries()):
            row = dataset.get(i)
            nsig_sw = row.getRealValue("nsig_sw")
            #print(f"Entry {i}: nsig_sw = {nsig_sw}")

        # Print the yields with their uncertainties
        print(f"Signal yield: {w.var('nsig').getVal()} ± {w.var('nsig').getError()}")
        print(f"Background yield: {w.var('nbkg1').getVal()} ± {w.var('nbkg1').getError()}")
        
        # Create sWeighted datasets
        sWeighted_data = ROOT.RooDataSet("sWeighted_data", "Data with sWeights", 
                                        dataset, dataset.get(), "", "nsig_sw")
        bkg_weighted_data = ROOT.RooDataSet("bkg_weighted_data", "Data with background sWeights", 
                                          dataset, dataset.get(), "", "nbkg1_sw")
        
        phi_hist_sig = sWeighted_data.createHistogram(f"h_PHI_{reso1}_{reso2}_sig", w.var(f"PHI_{reso1}_{reso2}"), ROOT.RooFit.Binning(30))
        phi_hist_sig.GetXaxis().SetTitle("#phi")

        phi_hist = dataset.createHistogram(f"h_PHI_{reso1}_{reso2}", w.var(f"PHI_{reso1}_{reso2}"), ROOT.RooFit.Binning(30))
        
        phi_hist_bkg = bkg_weighted_data.createHistogram(f"h_PHI_{reso1}_{reso2}_bkg", w.var(f"PHI_{reso1}_{reso2}"), ROOT.RooFit.Binning(30))
        phi_hist_bkg.SetTitle("#phi with Background Weights")

        style_draw([phi_hist_sig,phi_hist_bkg,phi_hist], output_dir + "/phi.png")
        style_draw([phi_hist_sig], output_dir + "sWeighted_phi.png")
        
        sqrts_hist_sig = sWeighted_data.createHistogram(f"vpho_M", w.var("vpho_M"), ROOT.RooFit.Binning(100))
        sqrts_hist_sig.GetXaxis().SetTitle("#sqrt{s'}")

        out_file = ROOT.TFile(output_dir + "splot_output.root", "RECREATE")
        sWeighted_data.Write("signal_weighted_data")
        bkg_weighted_data.Write("background_weighted_data")

        sqrts_hist_sig.Write("h_sqrts")
        phi_hist_sig.Write("h_phi")
        out_file.Close()
        
        
        sqrts_hist_bkg = bkg_weighted_data.createHistogram(f"vpho_M", w.var(f"PHI_{reso1}_{reso2}"), ROOT.RooFit.Binning(50))
        sqrts_hist_bkg.SetTitle("#phi with Background Weights")

        style_draw([sqrts_hist_sig], output_dir + "sqrts.png")
        
        #----------------------------- plot profile distribution  ------------------------
        ROOT.gStyle.SetLabelSize(0.04,"xyz")
        ROOT.gStyle.SetPadTopMargin(.1)
        ROOT.gStyle.SetPadLeftMargin(.14)
        ROOT.gStyle.SetPadRightMargin(.07)
        ROOT.gStyle.SetPadBottomMargin(.14)
        ROOT.gStyle.SetTitleSize(0.04,"xyz")
        ROOT.gStyle.SetOptTitle(0)
        ROOT.gStyle.SetMarkerSize(0.5)
        ROOT.gStyle.SetLabelFont(12,"XYZ")
        ROOT.gStyle.SetTitleFont(12,"XYZ")
        ROOT.gStyle.SetCanvasDefH(1080)
        ROOT.gStyle.SetCanvasDefW(1600)
        ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
        ROOT.gStyle.SetPadTickX(1)
        ROOT.gStyle.SetPadTickY(1)
        ROOT.gStyle.SetPadGridX(1)
        ROOT.gStyle.SetPadGridY(1)


        # First, create temporary frames to get the maximum values for y-axis range
        frame_phi1 = w.var(f"{reso1}_M").frame()
        dataset.plotOn(frame_phi1, rf.MarkerColor(ROOT.kBlack), rf.MarkerStyle(20), rf.Binning(nbin))
        model.plotOn(frame_phi1, rf.LineColor(4))
        
        frame_phi2 = w.var(f"{reso2}_M").frame()
        dataset.plotOn(frame_phi2, rf.MarkerColor(ROOT.kBlack), rf.MarkerStyle(20), rf.Binning(nbin))
        model.plotOn(frame_phi2, rf.LineColor(4))
        
        # Get maximum values and set unified y-axis range
        max_phi1 = frame_phi1.GetMaximum()
        max_phi2 = frame_phi2.GetMaximum()
        y_max = max(max_phi1, max_phi2) * 1.1  # Add 10% margin
        y_min = 0

        def plot(var_name:str):
            canvas = ROOT.TCanvas("c","c", 1600, 1080)
            canvas.Divide(1, 2)

            pad1 = canvas.cd(1)
            pad1.SetPad(0, 0.3, 1, 1)
            pad1.SetBottomMargin(0.01)
            pad1.SetLeftMargin(0.15)
            pad1.SetRightMargin(0.05)
            pad1.Draw()


            framex = w.var(var_name).frame()
            dataset.plotOn(framex, rf.Name("data"), rf.MarkerColor(ROOT.kBlack), rf.MarkerStyle(20), rf.Binning(nbin))
            model.plotOn(framex, rf.Name("sum"), rf.LineColor(4))
            model.plotOn(framex, rf.Components("sig_pdf"), rf.Name("signal"), rf.LineColor(2), rf.LineStyle(4), rf.LineWidth(3))
            model.plotOn(framex, rf.Components("bkg1"), rf.Name("bkg1_curve"), rf.LineColor(ROOT.kGreen+2), rf.LineStyle(7), rf.LineWidth(3))
            model.plotOn(framex, rf.Components("bkg2"), rf.Name("bkg2_curve"), rf.LineColor(ROOT.kMagenta+1), rf.LineStyle(5), rf.LineWidth(3))
            model.plotOn(framex, rf.Components("bkg3"), rf.Name("bkg3_curve"), rf.LineColor(7), rf.LineStyle(2), rf.LineWidth(3))

            framex.SetTitle("")
            framex.GetXaxis().SetTitle("#phi_{1} Mass (GeV/c^{2}")
            framex.GetYaxis().SetTitle(f"Candidates / ({((x_max - x_min) / nbin * 1000):.3f} MeV/c^{{2}})")
            framex.GetYaxis().SetTitleOffset(1)
            
            # Set unified y-axis range
            framex.SetMaximum(y_max)
            framex.SetMinimum(y_min)
            
            framex.Draw()

            leg = ROOT.TLegend(0.75, 0.9 - 0.05*5, 0.95, 0.9)
            leg.SetBorderSize(1)
            leg.SetFillStyle(0)
            
            # Create simple entries with correct colors
            data_entry = ROOT.TMarker(0, 0, 20)
            data_entry.SetMarkerColor(ROOT.kBlack)
            data_entry.SetMarkerStyle(20)
            
            model_entry = ROOT.TLine()
            model_entry.SetLineColor(4)  # Blue
            
            signal_entry = ROOT.TLine()
            signal_entry.SetLineColor(2)  # Red
            signal_entry.SetLineStyle(4)
            signal_entry.SetLineWidth(3)
            
            bkg1_entry = ROOT.TLine()
            bkg1_entry.SetLineColor(ROOT.kGreen+2)  # Match color with bkg1_curve
            bkg1_entry.SetLineStyle(7)
            bkg1_entry.SetLineWidth(3)

            bkg2_entry = ROOT.TLine()
            bkg2_entry.SetLineColor(ROOT.kMagenta+1)  # Match color with bkg2_curve
            bkg2_entry.SetLineStyle(5)
            bkg2_entry.SetLineWidth(3)
            
            bkg3_entry = ROOT.TLine()
            bkg3_entry.SetLineColor(7)  # Match color with bkg3_curve
            bkg3_entry.SetLineStyle(2)
            bkg3_entry.SetLineWidth(3)

            leg.AddEntry(data_entry, "Data", "pe")
            leg.AddEntry(model_entry, "Total fit", "l")
            leg.AddEntry(signal_entry, "Signal", "l")
            leg.AddEntry(bkg1_entry, "Bkg1", "l")
            leg.AddEntry(bkg2_entry, "Bkg2", "l")
            leg.AddEntry(bkg3_entry, "Bkg3", "l")
            
            # Get chi-square and degrees of freedom properly
            chi2 = framex.chiSquare("sum", "data")  # Get reduced chi-square
            
            data_hist = framex.getHist("data")
            nBins = data_hist.GetN()
            nPars = result.floatParsFinal().getSize()
            ndf = nBins - nPars
            
            # Calculate chi-square
            chi2_val = chi2 * ndf
            
            unbinned = kwargs.get('unbinned', True)
            if unbinned:
                # 对于unbinned拟合，显示NLL值
                nll = result.minNll()
                leg.AddEntry(0, f"NLL = {nll:.1f}", "")
            else:
                # 对于binned拟合，保留原有的χ²/ndof显示
                chi2 = framex.chiSquare("sum", "data")  # Get reduced chi-square
                data_hist = framex.getHist("data")
                nBins = data_hist.GetN()
                nPars = result.floatParsFinal().getSize()
                ndf = nBins - nPars
                chi2_val = chi2 * ndf
                leg.AddEntry(0, "#chi^{2}/ndf = " + f"{chi2_val:.1f}/{ndf}", "")

            leg.AddEntry(0, "N_{sig}=" + f"{w.var('nsig').getVal():.1f} #pm {w.var('nsig').getError():.1f}", "")
            if var_name == "phi1_M":
                leg.AddEntry(0, "#sigma_{gauss} = " + f"{w.var('smear_sigma1').getVal():.2e} #pm {w.var('smear_sigma1').getError():.2e}", "")
            elif var_name == "phi2_M":
                leg.AddEntry(0, "#sigma_{gauss} = " + f"{w.var('smear_sigma1').getVal():.2e} #pm {w.var('smear_sigma1').getError():.2e}", "")

            if bin_fit_range:
                leg.AddEntry(0, f"Bin: {bin_fit_range}", "")
            leg.Draw()

            pad2 = canvas.cd(2)
            pad2.SetPad(0, 0, 1, 0.3)
            pad2.SetTopMargin(0.01)
            pad2.SetBottomMargin(0.3)
            pad2.SetLeftMargin(0.15)
            pad2.SetRightMargin(0.05)
            pad2.Draw()

            framex_pull = w.var(var_name).frame()
            framex_pull.SetTitle("")
            framex_pull.GetYaxis().SetTitle("Pull")
            framex_pull.GetYaxis().SetTitleOffset(0.35)
            framex_pull.GetYaxis().SetTitleSize(0.1)
            framex_pull.GetYaxis().CenterTitle()
            framex_pull.GetYaxis().SetRangeUser(-5, 5)
            framex_pull.GetYaxis().SetLabelSize(0.1)
            framex_pull.GetXaxis().SetLabelSize(0.13)
            x_title = "M_{K^{+}K^{-}}1 (GeV/c^{2})" if var_name == "phi1_M" else "M_{K^{+}K^{-}}2 (GeV/c^{2})"
            framex_pull.GetXaxis().SetTitle(x_title)
            #framex_pull.GetXaxis().SetTitle("M_{K^{+}K^{-}} (GeV/c^{2})")
            framex_pull.GetXaxis().CenterTitle()
            framex_pull.GetXaxis().SetTitleSize(0.12)
            framex_pull.GetXaxis().SetTitleOffset(1.1)

            pullhist = framex.pullHist("data", "sum")
            framex_pull.addObject(pullhist, "P")
            framex_pull.Draw()
            canvas.SaveAs(output_dir + f"_{var_name}.png")

        plot(f"{reso1}_M")
        plot(f"{reso2}_M")


        #----- plot 3d distribution ------
        nsig = w.var("nsig").getVal()
        nsig_err = w.var("nsig").getError()
                
        # Add summary at the end of the log
        print(f"=== Analysis Complete ===")
        print(f"Signal yield: {nsig:.2f} ± {nsig_err:.2f}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"========================")
                
    return result, nsig, nsig_err

def perform_chisq_fit(tree:ROOT.TTree, output_dir:str, log_file:Optional[str]=None, bin_fit_range:Optional[str]=None, joint_fit:bool=False, **kwargs):
    """
    Perform chi-squared fit using signal MC template (unbinned fit)
    
    Args:
        tree: Input ROOT TTree with the data
        output_dir: Output directory for plots and results
        log_file: Optional file to save logs (stdout and stderr)
        bin_fit_range: Optional string representing the bin range for fitting
        **kwargs: Additional keyword arguments:
            sMC_path: signal MC path which contain data for template
            chisq_var: Name of chi-squared variable (default: "chisq")
            chisq_range: Tuple for chi-squared range (default: (0, 50))
            chisq_bins: Number of bins for chi-squared histogram (default: 50)
            unbinned: Boolean flag for unbinned fit (default: True)
    """
    # Get additional parameters from kwargs
    sMC_path = kwargs.get('sMC_path', None)
    sMC_file = ROOT.TFile(sMC_path) if sMC_path else None
    sMC_tree = sMC_file.Get("event") if sMC_file else None
    
    chisq_var = kwargs.get('chisq_var', 'chisq')
    chisq_range = kwargs.get('chisq_range', (0, 40))
    chisq_bins = kwargs.get('chisq_bins', 80)
    unbinned = kwargs.get('unbinned',True)

    var_config = [(chisq_var, chisq_range[0], chisq_range[1]), ("vpho_M", 2, 3)]  
    tools = FIT_UTILS(log_file=log_file, var_config=var_config)

    #print(kwargs)
    branches_name = kwargs.get('branches_name', None)
    #print(f"Branch names {branches_name}")

    # redirect log
    with tools.redirect_output():
        print(f"=== Starting Chi-squared Template Fit (Unbinned) ===")
        print(f"Output file: {output_dir}")
        if branches_name:
            print(f"Branch names to combine: {branches_name}")
        if sMC_tree:
            print(f"Using signal MC template for chi-squared: {chisq_var}")
        print(f"Unbinned fit: {unbinned}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"===============================")
        
        w = ROOT.RooWorkspace("w", "workspace")
        
        # Handle data dataset
        dataset = tools.handle_dataset(tree, w, branches_name, False)
        w.Import(dataset, ROOT.RooFit.Rename("dataset"))

        print(f"Data tree entries: {tree.GetEntries()}")
        print(f"Dataset created with {dataset.numEntries()} entries")
        
        # Create signal MC template for chi-squared if provided
        chisq_template_pdf = None
        if sMC_tree and unbinned:
            print(f"Creating unbinned chi-squared template from signal MC...")
            
            # Create signal MC dataset for unbinned template
            mc_dataset = tools.handle_dataset(sMC_tree, w, branches_name, False)
            w.Import(mc_dataset, ROOT.RooFit.Rename("mc_dataset"))
            
            # For unbinned fit, use RooKeysPdf (kernel density estimation)
            # This creates a smooth PDF from the MC data points
            w.factory(f"RooKeysPdf::sig_pdf({chisq_var}, mc_dataset, RooKeysPdf::MirrorBoth, 2.0)")
            chisq_template_pdf = w.pdf("sig_pdf")
            
            print(f"Signal MC unbinned template created with {mc_dataset.numEntries()} entries")
            
        elif sMC_tree and not unbinned:
            print(f"Creating binned chi-squared template from signal MC...")
            
            # Create signal MC dataset
            mc_dataset = tools.handle_dataset(sMC_tree, w, branches_name, False)
            w.Import(mc_dataset, ROOT.RooFit.Rename("mc_dataset"))
            
            # Create histogram template from signal MC chi-squared distribution
            chisq_hist = mc_dataset.createHistogram("h_chisq_template", 
                                                   w.var(chisq_var), 
                                                   ROOT.RooFit.Binning(chisq_bins))
            
            # Convert histogram to RooDataHist and then to RooHistPdf
            chisq_datahist = ROOT.RooDataHist("chisq_dataHist", 
                                            "Chi-squared template from MC", 
                                            ROOT.RooArgList(w.var(chisq_var)), 
                                            chisq_hist)
            w.Import(chisq_datahist)
            
            # Create template PDF
            w.factory(f"RooHistPdf::sig_pdf({chisq_var}, chisq_dataHist)")
            chisq_template_pdf = w.pdf("sig_pdf")

            print(f"Signal MC binned template created with {mc_dataset.numEntries()} entries")

        # Alternative unbinned approaches if no MC is provided
        if not chisq_template_pdf:
            print("Warning: No signal MC provided, using analytical PDF")
            if unbinned:
                # Use analytical functions for unbinned fit
                # Gamma distribution often works well for chi-squared-like variables
                w.factory(f"Gamma::sig_pdf({chisq_var}, gamma_shape[2, 0.5, 10], gamma_rate[0.5, 0.1, 2], 0)")
            else:
                # Use chi-squared distribution
                w.factory(f"RooChiSquarePdf::sig_pdf({chisq_var}, ndof[5, 1, 20])")
        
        # Background chi-squared PDF (typically exponential)
        w.factory(f"Exponential::bkg_pdf({chisq_var}, exp_slope[-0.1, -10, 10])")

        w.factory("bkg_p0[122.504]")  # Fixed normalization of exponential term
        w.factory("bkg_p1[59.8483]")  # Fixed decay constant of exponential term
        w.factory("bkg_p2[40.2843]")  # Fixed normalization of power term
        w.factory("bkg_p3[1.75904]")  # Fixed power exponent
        w.factory("bkg_p4[2.04382]")  # Fixed decay constant of power term
        #w.factory(f"RooGenericPdf::bkg_pdf('bkg_p0*exp(-{chisq_var}/bkg_p1) + bkg_p2*pow({chisq_var},bkg_p3)*exp(-{chisq_var}/bkg_p4)', {{{chisq_var}, bkg_p0, bkg_p1, bkg_p2, bkg_p3, bkg_p4}})")
        
        # Total model
        w.factory("SUM::model(nsig[20000, 0, 40000] * sig_pdf, nbkg[45000, 0, 200000] * bkg_pdf)")

        # Perform fit
        model = w.pdf("model")
        if not dataset or not model:
            print("Error: dataset or model is None")
            return

        print(f"Dataset entries: {dataset.numEntries()}")
        if dataset.numEntries() == 0:
            print("Error: dataset is empty")
            return
        
        if joint_fit:
            return w 

        # Unbinned maximum likelihood fit
        result = model.fitTo(dataset, 
                             rf.Save(True), 
                             rf.NumCPU(4), 
                             rf.PrintLevel(1),  # Slightly more verbose for unbinned
                             rf.Strategy(2),    # More robust strategy for unbinned
                             rf.Extended(True)) # Extended likelihood for yield determination

        # --------------------------------------- splot ---------------------------------------

        # Create sPlot
        sData = RooStats.SPlot("sData", "sPlot", dataset, model,
                               ROOT.RooArgList(w.var("nsig"), w.var("nbkg")))

        # Print results
        print(f"Signal yield: {w.var('nsig').getVal()} ± {w.var('nsig').getError()}")
        print(f"Background yield: {w.var('nbkg').getVal()} ± {w.var('nbkg').getError()}")
        
        # Create sWeighted datasets
        sWeighted_data = ROOT.RooDataSet("sWeighted_data", "Data with sWeights", 
                                        dataset, dataset.get(), "", "nsig_sw")
        bkg_weighted_data = ROOT.RooDataSet("bkg_weighted_data", "Data with background sWeights", 
                                          dataset, dataset.get(), "", "nbkg_sw")
        
        # Save results
        out_file = ROOT.TFile(output_dir + "splot_output.root", "RECREATE")
        sWeighted_data.Write("signal_weighted_data")
        bkg_weighted_data.Write("background_weighted_data")
        if sMC_tree:
            if unbinned:
                # For unbinned, create a histogram for visualization
                mc_hist = ROOT.RooDataSet("mc_dataset", "", ROOT.RooArgList(w.var(chisq_var))).createHistogram(
                    "h_chisq_template", w.var(chisq_var), ROOT.RooFit.Binning(chisq_bins))
                mc_hist.Write("h_chisq_template_unbinned")
            else:
                chisq_hist.Write("h_chisq_template")
        out_file.Close()
        
        #----------------------------- plot distribution  ------------------------
        ROOT.gStyle.SetLabelSize(0.04,"xyz")
        ROOT.gStyle.SetPadTopMargin(.1)
        ROOT.gStyle.SetPadLeftMargin(.14)
        ROOT.gStyle.SetPadRightMargin(.07)
        ROOT.gStyle.SetPadBottomMargin(.14)
        ROOT.gStyle.SetTitleSize(0.04,"xyz")
        ROOT.gStyle.SetOptTitle(0)
        ROOT.gStyle.SetMarkerSize(0.5)
        ROOT.gStyle.SetLabelFont(12,"XYZ")
        ROOT.gStyle.SetTitleFont(12,"XYZ")
        ROOT.gStyle.SetCanvasDefH(1080)
        ROOT.gStyle.SetCanvasDefW(1600)
        ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
        ROOT.gStyle.SetPadTickX(1)
        ROOT.gStyle.SetPadTickY(1)
        ROOT.gStyle.SetPadGridX(1)
        ROOT.gStyle.SetPadGridY(1)

        def plot():
            # Create plot for chi-squared variable
            canvas = ROOT.TCanvas("c_chisq", "Chi-squared unbinned fit", 1600, 1080)
            canvas.Divide(1, 2)
            pad1 = canvas.cd(1)
            pad1.SetPad(0, 0.3, 1, 1)
            pad1.SetBottomMargin(0.01)
            pad1.SetLeftMargin(0.15)
            pad1.SetRightMargin(0.05)
            pad1.Draw()

            frame_chisq = w.var(chisq_var).frame()
            
            # For unbinned fit, use adaptive binning or fixed binning for visualization
            dataset.plotOn(frame_chisq, rf.Name("data"), rf.MarkerColor(ROOT.kBlack), rf.MarkerStyle(20), rf.Binning(chisq_bins))
            model.plotOn(frame_chisq, rf.Name("total"), rf.LineColor(4), rf.Normalization(1.0, ROOT.RooAbsReal.RelativeExpected))
            model.plotOn(frame_chisq, rf.Components("sig_pdf"), rf.Name("signal"), rf.LineColor(2), rf.LineStyle(4), rf.LineWidth(3), rf.Normalization(1.0, ROOT.RooAbsReal.RelativeExpected))
            model.plotOn(frame_chisq, rf.Components("bkg_pdf"), rf.Name("bkg"), rf.LineColor(ROOT.kGreen+2), rf.LineStyle(7), rf.LineWidth(3), rf.Normalization(1.0, ROOT.RooAbsReal.RelativeExpected))

            frame_chisq.SetTitle("")
            frame_chisq.GetXaxis().SetTitle("#chi^{2}")
            frame_chisq.GetYaxis().SetTitle("Candidates")
            frame_chisq.GetYaxis().CenterTitle()
            frame_chisq.Draw()
            
            leg = ROOT.TLegend(0.75, 0.9 - 0.05*5, 0.95, 0.9)
            leg.SetBorderSize(0)
            leg.SetFillStyle(0)

            # Create legend entries
            data_entry = ROOT.TMarker(0, 0, 20)
            data_entry.SetMarkerColor(ROOT.kBlack)
            data_entry.SetMarkerStyle(20)   

            model_entry = ROOT.TLine()
            model_entry.SetLineColor(4)

            signal_entry = ROOT.TLine()
            signal_entry.SetLineColor(2)

            background_entry = ROOT.TLine()
            background_entry.SetLineColor(ROOT.kGreen+2)
            background_entry.SetLineStyle(7)
            background_entry.SetLineWidth(3)

            leg.AddEntry(data_entry, "Data", "pe")
            leg.AddEntry(model_entry, "Total fit", "l")
            leg.AddEntry(signal_entry, "Signal", "l")
            leg.AddEntry(background_entry, "Background", "l")
            leg.SetTextFont(12)

            # Calculate chi-square for visualization (note: less meaningful for unbinned fits)
            #if not unbinned:
            chi2 = frame_chisq.chiSquare("total", "data")
            data_hist = frame_chisq.getHist("data")
            nBins = data_hist.GetN()
            nPars = result.floatParsFinal().getSize()
            ndf = nBins - nPars
            chi2_val = chi2 * ndf
            leg.AddEntry(0, "#chi^{2}/ndf = " + f"{chi2_val:.1f}/{ndf}", "")
            #else:
            # For unbinned fits, show the negative log likelihood
            nll = result.minNll()
            leg.AddEntry(0, f"-log L = {nll:.1f}", "")
            
            leg.AddEntry(0, "N_{sig}=" + f"{w.var('nsig').getVal():.1f} #pm {w.var('nsig').getError():.1f}", "")
            if bin_fit_range:
                leg.AddEntry(0, f"Bin: {bin_fit_range}", "")
            leg.Draw()

            pad2 = canvas.cd(2)
            pad2.SetPad(0, 0, 1, 0.3)
            pad2.SetTopMargin(0.01)
            pad2.SetBottomMargin(0.3)
            pad2.SetLeftMargin(0.15)
            pad2.SetRightMargin(0.05)
            pad2.Draw()

            frame_chisq_pull = w.var(chisq_var).frame()
            frame_chisq_pull.SetTitle("")
            frame_chisq_pull.GetYaxis().SetTitle("Pull")
            frame_chisq_pull.GetYaxis().SetTitleOffset(0.35)
            frame_chisq_pull.GetYaxis().SetTitleSize(0.1)
            frame_chisq_pull.GetYaxis().CenterTitle()
            frame_chisq_pull.GetYaxis().SetRangeUser(-5, 5)
            frame_chisq_pull.GetYaxis().SetLabelSize(0.1)
            frame_chisq_pull.GetXaxis().SetLabelSize(0.13)
            frame_chisq_pull.GetXaxis().SetTitle("#chi^{2}")
            frame_chisq_pull.GetXaxis().CenterTitle()
            frame_chisq_pull.GetXaxis().SetTitleSize(0.12)
            frame_chisq_pull.GetXaxis().SetTitleOffset(1.1)

            pullhist = frame_chisq.pullHist("data", "total")
            frame_chisq_pull.addObject(pullhist, "P")
            frame_chisq_pull.Draw()
            canvas.SaveAs(output_dir + f"_chisq.png")

        plot()

        nsig = w.var("nsig").getVal()
        nsig_err = w.var("nsig").getError()
                
        print(f"=== Analysis Complete ===")
        print(f"Signal yield: {nsig:.2f} ± {nsig_err:.2f}")
        print(f"Fit method: {'Unbinned' if unbinned else 'Binned'} maximum likelihood")
        if chisq_template_pdf:
            print(f"Chi-squared template method used")
        print(f"Minimum NLL: {result.minNll():.2f}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"========================")
                
    return result, nsig, nsig_err


def get_effCurve(h_eff: ROOT.TH1, plot_path: str) -> ROOT.TH1:
    """
    Fits a polynomial to an efficiency histogram and returns a histogram with the fit results.
    Also plots the efficiency histogram with the fitted curve if plot_path is provided.
    
    Args:
        h_eff: ROOT TH1 histogram containing efficiency data
        plot_path: Optional file path to save the efficiency plot
    
    Returns:
        ROOT.TH1: Histogram with same binning as input, containing fit values and errors
    """
    h_eff_update = h_eff.Clone("h_eff_fit")
    # Create a polynomial fit function (5th order polynomial)
    fit_func = ROOT.TF1("eff_fit_func", "pol3", h_eff.GetXaxis().GetXmin(), h_eff.GetXaxis().GetXmax())
    
    # Fit the histogram
    fit_result = h_eff.Fit(fit_func, "SQR")  # S to return fit result, Q for quiet, R for range

    def plot_efficiency():
        # Apply styling for the plot
        ROOT.gStyle.SetOptStat(0)
        ROOT.gStyle.SetOptFit(0)
        
        # Create canvas
        c = ROOT.TCanvas("c_eff", "Efficiency", 1600, 1080)
        c.SetLeftMargin(0.15)
        c.SetBottomMargin(0.14)
        c.SetRightMargin(0.05)
        c.SetTopMargin(0.1)
        c.SetGridx(True)
        c.SetGridy(True)
        
        # Draw histogram
        h_eff.SetMarkerStyle(20)
        h_eff.SetMarkerColor(ROOT.kBlack)
        h_eff.SetLineColor(ROOT.kBlack)
        h_eff.SetTitle("")
        h_eff.GetXaxis().SetTitleSize(0.045)
        h_eff.GetYaxis().SetTitleSize(0.045)
        h_eff.GetXaxis().SetLabelSize(0.04)
        h_eff.GetYaxis().SetLabelSize(0.04)
        h_eff.GetYaxis().SetTitle("Efficiency")
        h_eff.SetMinimum(0)
        h_eff.Draw("PE")
        
        # Draw fit function
        fit_func.SetLineColor(ROOT.kRed)
        fit_func.SetLineWidth(3)
        fit_func.Draw("SAME")
        
        # Create legend
        legend = ROOT.TLegend(0.7, 0.5, 0.9, 0.6)
        legend.SetBorderSize(1)
        legend.SetFillStyle(0)
        legend.AddEntry(h_eff, "Efficiency data", "PE")
        legend.AddEntry(fit_func, "Polynomial fit", "L")
        legend.Draw()
        
        # Add chi-square information
        chi2 = fit_func.GetChisquare()
        ndf = fit_func.GetNDF()
        chi2_label = ROOT.TLatex(0.7, 0.45, f"#chi^{{2}}/ndf = {chi2:.1f}/{ndf}")
        chi2_label.SetNDC()
        chi2_label.SetTextSize(0.035)
        chi2_label.Draw()
    
        # Use ROOT's TVirtualFitter to get accurate confidence intervals
        n_points = 100
        error_band = ROOT.TGraphErrors(n_points)
        for i in range(n_points):
            x = h_eff.GetXaxis().GetXmin() + i * (h_eff.GetXaxis().GetXmax() - h_eff.GetXaxis().GetXmin()) / (n_points - 1)
            error_band.SetPoint(i, x, 0)  # y value will be overwritten by GetConfidenceIntervals
            
        # Use ROOT's TVirtualFitter to calculate confidence intervals (as in C++ examples)
        fitter = ROOT.TVirtualFitter.GetFitter()
        fitter.GetConfidenceIntervals(error_band, 0.683)  # 68.3% confidence level (1σ)
        fitter.GetConfidenceIntervals(h_eff_update, 0.683)  # 68.3% confidence level (1σ)
        
        error_band.SetFillColor(ROOT.kRed-9)
        error_band.SetFillStyle(3001)
        error_band.Draw("3 SAME")
        
        # Update legend with error band
        legend.AddEntry(error_band, "1 #sigma error band", "f")
        legend.Draw()

        c.SaveAs(plot_path)
    
    plot_efficiency()
    return h_eff_update



