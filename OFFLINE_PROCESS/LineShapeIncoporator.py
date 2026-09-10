#!/usr/bin/env python3

import ROOT as R
import random
import numpy as np
from array import array
from typing import Tuple, List

"""
Functions for incorporating lineshape into ROOT trees
version : v1.0.0
Date : 2026-09-10
Author : Zheng Wang
"""


def sampling_flat_dist(target_lineshape: callable,
                       rootFiles: List[str],
                       var_name: str,
                       var_range: Tuple[float, float],
                       output_rootFile: str = None,
                       tree_names: List[str] = ["truth", "event"]) -> None:
    """
    Sampling the flat distribution of a variable to match a given lineshape function in one tree (in generate level),
    then synchronizing this sampling to other trees (in detect level) by exp run event index matching.

    Args:
        target_lineshape: Callable function that evaluates probability based on input variable
        rootFiles: List of paths to ROOT files containing the trees
        var_name: Variable name in truth tree to be sampled
        var_range: Tuple of (min, max) values for the variable range
        output_rootFile: Path to output ROOT file for sampled trees (if None, a default name will be used)
        tree_names: List containing (truth_tree_name, reco_tree_name)
    """
    tree0_tfile = R.TFile(rootFiles[0], "READ")
    tree0 = tree0_tfile.Get(tree_names[0])

    # Sampling tree0
    if output_rootFile is None:
        output_rootFile = rootFiles[0].replace(".root", "_sampled.root")
        print(f"Output ROOT file not specified. Using {output_rootFile}")

    output_tfile = R.TFile(output_rootFile, "RECREATE")
    sampled_tree0 = tree0.CloneTree(0)
    sampled_tree0.SetDirectory(output_tfile)
    # tree0_tfile.Close()

    # Show some lineshape information
    xs = np.linspace(var_range[0], var_range[1], 10000)
    ys = np.array([target_lineshape(x) for x in xs])
    max_x = xs[np.argmax(ys)]
    max_value = target_lineshape(max_x)
    print(f"Max value in range {var_range}: {max_value} at x = {max_x}")

    # Start sampling
    random.seed()
    event_idx_set = set()
    for i, entry in enumerate(tree0):
        if i % 1000 == 0:
            print(f"Processing entry {i}/{tree0.GetEntries()}")

        var_value = getattr(entry, var_name)
        experiment = getattr(entry, "__experiment__")
        run = getattr(entry, "__run__")
        event = getattr(entry, "__event__")

        if not (var_range[0] <= var_value <= var_range[1]):
            continue
        f_val = target_lineshape(var_value) / max_value
        dice = random.random()

        if f_val > dice:
            sampled_tree0.Fill()

            # Store event identifier
            event_id = (experiment, run, event)
            event_idx_set.add(event_id)

    # Write truth tree to output file
    output_tfile.cd()
    sampled_tree0.Write(tree_names[0])

    # sync to other trees
    if len(rootFiles) > 1:
        for tree_name, root_file in zip(tree_names[1:], rootFiles[1:]):
            tfile = R.TFile(root_file, "READ")
            tree = tfile.Get(tree_name)
            output_tfile.cd()
            cloned_tree = tree.CloneTree(0)
            cloned_tree.SetDirectory(output_tfile)
            # tfile.Close()
            for i, entry in enumerate(tree):
                experiment = getattr(entry, "__experiment__")
                run = getattr(entry, "__run__")
                event = getattr(entry, "__event__")
                event_id = (experiment, run, event)
                if event_id in event_idx_set:
                    cloned_tree.Fill()
            cloned_tree.Write(tree_name)

    print(
        f"Done! Selected {len(event_idx_set)} events out of {tree0.GetEntries()} {tree_names[0]} entries.")
    print(f"Output saved to: {output_rootFile}")

    output_tfile.Close()


def _build_input_density(tree0, var_name, var_range, nbins=200):
    """
    Estimate the underlying generation density g(x) of var_name in tree0
    using a normalized histogram. Returns a callable g(x).
    """
    hist = R.TH1D(
        "h_input_density",
        "input density",
        nbins,
        var_range[0],
        var_range[1])
    tree0.Draw(f"{var_name}>>h_input_density", "", "goff")

    n_total = hist.Integral()
    if n_total <= 0:
        raise ValueError(
            f"No entries found for {var_name} in range {var_range}")

    hist.Scale(1.0 / n_total / hist.GetBinWidth(1))  # normalize to a PDF

    def g(x):
        bin_idx = hist.FindBin(x)
        val = hist.GetBinContent(bin_idx)
        if val <= 0:
            print(f"Warning: g(x) is non-positive at x={x} with g(x)={val}")
        return val if val > 0 else 1e-12  # avoid div-by-zero

    return g, hist


def general_sampling(target_lineshape: callable,
                     rootFiles: List[str],
                     var_name: str,
                     var_range: Tuple[float, float],
                     output_rootFile: str = None,
                     tree_names: List[str] = ["truth", "event"],
                     nbins: int = 200) -> None:
    """
    Ssampling an MC sample whose generation distribution g(x) is
    NOT flat, so that the output sample follows target_lineshape(x).

    Uses acceptance-rejection with:
        w(x) = target_lineshape(x) / g(x)
        accept if random() < w(x) / max(w)

    where g(x) is estimated from the input tree's histogram of var_name.

    Args:
        target_lineshape: desired output distribution f(x)
        rootFiles: list of ROOT files (truth tree first, others synced by event id)
        var_name: variable used for reweighting
        var_range: (min, max) range to consider
        output_rootFile: output path
        tree_names: tree names corresponding to rootFiles
        nbins: number of bins used to estimate g(x)
    """
    tree0_tfile = R.TFile(rootFiles[0], "READ")
    tree0 = tree0_tfile.Get(tree_names[0])

    if output_rootFile is None:
        output_rootFile = rootFiles[0].replace(".root", "_reweighted.root")
        print(f"Output ROOT file not specified. Using {output_rootFile}")

    output_tfile = R.TFile(output_rootFile, "RECREATE")
    sampled_tree0 = tree0.CloneTree(0)
    sampled_tree0.SetDirectory(output_tfile)

    # Step 1: estimate input generation density g(x)
    g, hist = _build_input_density(tree0, var_name, var_range, nbins)

    # Step 2: compute weight function w(x) = target(x) / g(x) on a grid
    # not taking endpoint, since this will find the last + 1 bin of hist
    xs = np.linspace(var_range[0], var_range[1], 1000, endpoint=False)
    target_ys = np.array([target_lineshape(x) for x in xs])
    g_ys = np.array([g(x) for x in xs])
    w_ys = target_ys / g_ys
    max_w = np.max(w_ys)

    print(f"Max weight w(x) = target(x)/g(x) in range {var_range}: {max_w}")

    # Step 3: acceptance-rejection sampling using w(x)/max_w
    random.seed()
    event_idx_set = set()
    n_entries = tree0.GetEntries()

    for i, entry in enumerate(tree0):
        var_value = getattr(entry, var_name)
        if not (var_range[0] <= var_value <= var_range[1]):
            continue

        target_val = target_lineshape(var_value)
        g_val = g(var_value)
        w = target_val / g_val
        accept_prob = w / max_w

        if i % 1000 == 0:
            print(
                f"Processing entry {i}/{n_entries}: var_value = {var_value}, target_val = {target_val}, g_val = {g_val}, w = {w}, accept_prob = {accept_prob}")

        if random.random() < accept_prob:
            sampled_tree0.Fill()
            experiment = getattr(entry, "__experiment__")
            run = getattr(entry, "__run__")
            event = getattr(entry, "__event__")
            event_idx_set.add((experiment, run, event))

    output_tfile.cd()
    sampled_tree0.Write(tree_names[0])

    # sync to other trees
    if len(rootFiles) > 1:
        for tree_name, root_file in zip(tree_names[1:], rootFiles[1:]):
            tfile = R.TFile(root_file, "READ")
            tree = tfile.Get(tree_name)
            output_tfile.cd()
            cloned_tree = tree.CloneTree(0)
            cloned_tree.SetDirectory(output_tfile)
            for entry in tree:
                experiment = getattr(entry, "__experiment__")
                run = getattr(entry, "__run__")
                event = getattr(entry, "__event__")
                event_id = (experiment, run, event)
                if event_id in event_idx_set:
                    cloned_tree.Fill()
            cloned_tree.Write(tree_name)

    print(
        f"Done! Selected {len(event_idx_set)} events out of {n_entries} {tree_names[0]} entries.")
    print(f"Output saved to: {output_rootFile}")

    output_tfile.Close()


def get_lineshape_weight(
        target_lineshape: callable,
        rootFiles: List[str],
        var_name: str,
        var_range: Tuple[float, float],
        output_rootFile: str = None,
        tree_names: List[str] = ["truth", "event"],
        nbins: int = 200,
        weight_name: str = "lineshape_weight") -> None:
    """
    Keep all events and write target/g to a new weight branch.
    """
    tree0_tfile = R.TFile(rootFiles[0], "READ")
    tree0 = tree0_tfile.Get(tree_names[0])

    if output_rootFile is None:
        output_rootFile = rootFiles[0].replace(".root", "_reweighted.root")
        print(f"Output ROOT file not specified. Using {output_rootFile}")

    # Step 1: estimate input generation density g(x)
    g, hist = _build_input_density(tree0, var_name, var_range, nbins)

    # Step 2: compute weight function w(x) = target(x) / g(x) on a grid
    # not taking endpoint, since this will find the last + 1 bin of hist
    xs = np.linspace(var_range[0], var_range[1], 1000, endpoint=False)
    target_ys = np.array([target_lineshape(x) for x in xs])
    g_ys = np.array([g(x) for x in xs])
    w_ys = target_ys / g_ys
    max_w = np.max(w_ys)

    print(f"Max weight w(x) = target(x)/g(x) in range {var_range}: {max_w}")

    # Step 3: get weight as w(x)/max_w
    random.seed()
    event_weights = {}
    n_entries = tree0.GetEntries()

    # Copy the truth tree and add the weight branch.
    output_tfile = R.TFile(output_rootFile, "RECREATE")
    output_tfile.cd()
    output_tree0 = tree0.CloneTree(0)
    output_tree0.SetDirectory(output_tfile)
    weight = array("d", [0.0])
    output_tree0.Branch(weight_name, weight, f"{weight_name}/D")

    for i, entry in enumerate(tree0):
        var_value = getattr(entry, var_name)
        if not (var_range[0] <= var_value <= var_range[1]):
            continue

        target_val = target_lineshape(var_value)
        g_val = g(var_value)
        w = target_val / g_val
        weight[0] = w / max_w

        if i % 1000 == 0:
            print(
                f"Processing entry {i}/{n_entries}: var_value = {var_value}, target_val = {target_val}, g_val = {g_val}, w = {w}, weight = {weight}")

        event_id = (
            getattr(entry, "__experiment__"),
            getattr(entry, "__run__"),
            getattr(entry, "__event__"),
        )

        event_weights[event_id] = weight[0]
        output_tree0.Fill()

    output_tfile.cd()
    output_tree0.Write(tree_names[0])

    # sync to other trees
    for tree_name, root_file in zip(tree_names[1:], rootFiles[1:]):
        tfile = R.TFile(root_file, "READ")
        input_tree = tfile.Get(tree_name)
        output_tfile.cd()
        cloned_tree = input_tree.CloneTree(0)
        cloned_tree.SetDirectory(output_tfile)

        weight = array("d", [0.0])
        cloned_tree.Branch(weight_name, weight, f"{weight_name}/D")

        for entry in cloned_tree:
            event_id = (
                getattr(entry, "__experiment__"),
                getattr(entry, "__run__"),
                getattr(entry, "__event__"),
            )

            weight[0] = event_weights.get(event_id, 0.0)
            cloned_tree.Fill()

        cloned_tree.Write(tree_name)

    output_tfile.Close()

    print(f"Weighted file saved to: {output_rootFile}")
