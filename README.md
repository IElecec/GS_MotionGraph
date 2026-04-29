# GS_MotionGraph

Minimal motion-graph pipeline for Gaussian sequence data.

## Overview

This repo currently contains three command-line steps:

1. Generate sample Gaussian `.ply` sequences under `data/char1`
2. Precompute pairwise similarity matrices
3. Build a motion graph from the saved similarity matrices

## Data Layout

The motion database is organized as:

```text
data/
  char1/
    walk/
      anim_1/
        frame_0.ply
        frame_1.ply
        ...
    attack_1/
      anim_1/
      anim2/
    attack_2/
      anim_1/
```

Each animation directory contains frame-wise Gaussian `.ply` files named as `frame_<index>.ply`.

## Requirements

The scripts assume the Python environment can import the dependencies already used in this repo, especially:

- `torch`
- `plyfile`
- `tqdm`

## Quick Start

### 1. Generate sample files

This will create a small demo character dataset under `data/char1`:

```bash
python generate_sample_char1.py
```

Optional arguments:

```bash
python generate_sample_char1.py --output-dir data/char1 --sh-degree 3
```

### 2. Compute similarity matrices

```bash
python compute_similarity.py -m data/char1 -o outputs --window 10 --min-gap 6 --sh-degree 3
```

This writes files like:

```text
outputs/
  similarity_matrices/
    walk/
      anim_1/
        attack_1/
          anim_1/
            similarity.pt
```

Each `similarity.pt` contains:

- `distance_matrix`
- `angle_matrix`
- `window_size`
- `min_gap`
- matrix index metadata used by the motion-graph builder

### 3. Build the motion graph

```bash
python build_motion_graph.py -m data/char1 -s outputs -o outputs/motion_graph.json --distance-threshold 0.5 --top-k 5
```

Useful options:

- `--distance-threshold`: only keep transitions below this distance
- `--top-k`: keep at most top-k local-minimum transitions for each animation pair
- `--keep-dead-ends`: disable pruning to the largest strongly connected component

The output is a JSON file containing:

- `nodes`: frame references
- `edges`: sequence edges and transition edges
- `transitions`: extracted transition candidates

## Notes

- `generate_sample_char1.py` creates synthetic demo data intended for pipeline testing, not realistic production Gaussians.
- `compute_similarity.py` currently compares fixed-length windows between animation sequences.
- `build_motion_graph.py` uses the saved similarity matrices plus the original database structure to build the graph.
