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
from typing import Optional
from DRAW import style_draw
from PHY_CALCULATOR import PhysicsCalculator
from .fit_tools import FIT_UTILS, QUICK_FIT

from math import sqrt

def perform_resonance_fit(tree:ROOT.TTree, output_dir:str, log_file=None, bin_fit_range:Optional[str]=None, **kwargs):
    """
    Perform sPlot analysis for digamma to diphi process
    
    Args:
        tree: Input ROOT TTree with the data
        output_dir: Output directory for plots and results
        log_file: Optional file to save logs (stdout and stderr)
        bin_fit_range: Optional string representing the bin range for fitting
        **kwargs: Additional keyword arguments:
            branches_name: List of branch names to combine (e.g., ["phi1_M", "phi2_M"])
    """
    reso, mass, width = "phi", 1.0195, 0.004249
    x_min, x_max, nbin = 1, 1.06, 60

    var_config = [("phi_M", x_min, x_max),("vpho_M", 2, 3)]  
    tools = FIT_UTILS(log_file=log_file, var_config= var_config)

    print(kwargs)
    branches_name = kwargs['branches_name']
    print(f"Branch names {branches_name}")

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
        
        dataset = tools.handle_dataset(tree, w, branches_name, True)

        print(tree.GetEntries())
        print(f"Dataset created with {dataset.numEntries()} entries")
        
        # Use Breit-Wigner (RooBreitWigner) with PDG values for phi meson
        w.factory(f"BreitWigner::reso_bw({reso}_M, {mass}, {width})")
        w.factory(f"Gaussian::smear({reso}_M, smear_mean[0], smear_sigma[0.0008, 5e-05, 0.001])")
        w.factory(f"FCONV::sig_pdf({reso}_M, reso_bw, smear)")
        
        # Polynomial , Chebychev 
        bkg_func = "Chebychev"  
        bkg_func = "Polynomial"  
        w.factory(f"{bkg_func}::bkg_pdf({reso}_M, {{b_0[-10, 10], b_1[-10, 10], b_2[-10, 10], b_3[-10, 10]}})")
        #w.factory(f"{bkg_func}::bkg_pdf({reso}_M, {{b_0[-10, 10], b_1[-10, 10], b_2[-10, 10]}})")
        #w.factory(f"{bkg_func}::bkg_pdf({reso}_M, {{b_0[-10, 10], b_1[-10, 10]}})")
        #w.factory(f"{bkg_func}::bkg_pdf({reso}_M, {{b_0[-10, 10]}})")
        
        w.factory("SUM::model(nsig[20000, 0, 40000] * sig_pdf, nbkg[45000, 0, 200000] * bkg_pdf)")

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

        # Create sPlot for visualization using phi_M  as discriminating variables
        sData = RooStats.SPlot("sData", "sPlot", dataset, model,
                               ROOT.RooArgList(w.var("nsig"), w.var("nbkg")))
        for i in range(dataset.numEntries()):
            row = dataset.get(i)
            nsig_sw = row.getRealValue("nsig_sw")
            #print(f"Entry {i}: nsig_sw = {nsig_sw}")

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
        ROOT.gStyle.SetLabelFont(42,"XYZ")
        ROOT.gStyle.SetTitleFont(42,"XYZ")
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
            dataset.plotOn(framex, rf.Name("data"), rf.MarkerColor(ROOT.kBlack), rf.MarkerStyle(20), rf.Binning(nbin))
            model.plotOn(framex, rf.Name("sum"), rf.LineColor(4))
            model.plotOn(framex, rf.Components("sig_pdf"), rf.Name("signal"), rf.LineColor(2), rf.LineStyle(4), rf.LineWidth(3))
            model.plotOn(framex, rf.Components("bkg_pdf"), rf.Name("bkg"), rf.LineColor(ROOT.kGreen+2), rf.LineStyle(7), rf.LineWidth(3))

            framex.SetTitle("")
            framex.GetXaxis().SetTitle("#phi Mass (GeV/c^{2}")
            framex.GetYaxis().SetTitle(f"Candidates / ({((x_max - x_min) / nbin * 1000):.3f} MeV/c^{{2}})")
            framex.GetYaxis().SetTitleOffset(1)
            
            # Set unified y-axis range
            #framex.SetMaximum(y_max)
            #framex.SetMinimum(y_min)
            
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
            
            # print(f"Chi2: {chi2_val:.2f}, ndf: {ndf}, chi2/ndf: {chi2:.4f}")
            
            #leg.AddEntry(0, "#chi^{2}/ndf = " + f"{chi2:.4f} ({chi2_val:.1f}/{ndf})", "")
            leg.AddEntry(0, "#chi^{2}/ndf = " + f"{chi2_val:.1f}/{ndf}", "")
            leg.AddEntry(0, "N_{sig}=" + f"{w.var('nsig').getVal():.1f} #pm {w.var('nsig').getError():.1f}", "")
            leg.AddEntry(0, "#sigma_{gauss} = " + f"{w.var('smear_sigma').getVal():.2e} #pm {w.var('smear_sigma').getError():.2e}", "")

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
            x_title = "M_{K^{+}K^{-}} (GeV/c^{2})" if var_name == "phi_M" else "M_{reso}"
            framex_pull.GetXaxis().SetTitle(x_title)
            #framex_pull.GetXaxis().SetTitle("M_{K^{+}K^{-}} (GeV/c^{2})")
            framex_pull.GetXaxis().CenterTitle()
            framex_pull.GetXaxis().SetTitleSize(0.12)
            framex_pull.GetXaxis().SetTitleOffset(1.1)

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
        print(f"Signal yield: {nsig:.2f} ± {nsig_err:.2f}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"========================")
                
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
        #w.factory(f"{bkg_func}::reso1_bkg_pdf({reso1}_M, {{b1_0[-10, 10], b1_1[-10, 10]}})")
        #w.factory(f"{bkg_func}::reso2_bkg_pdf({reso2}_M, {{b2_0[-10, 10], b2_1[-10, 10]}})")
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
        ROOT.gStyle.SetLabelFont(42,"XYZ")
        ROOT.gStyle.SetTitleFont(42,"XYZ")
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
            
            # print(f"Chi2: {chi2_val:.2f}, ndf: {ndf}, chi2/ndf: {chi2:.4f}")
            
            #leg.AddEntry(0, "#chi^{2}/ndf = " + f"{chi2:.4f} ({chi2_val:.1f}/{ndf})", "")
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

