# GS_MotionGraph

Minimal motion-graph pipeline for Gaussian sequence data.

## Overview

This repo currently contains a small end-to-end command-line pipeline:

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

If you also want an HTML visualization at build time:

```bash
python build_motion_graph.py -m data/char1 -s outputs -o outputs/motion_graph.json --visualization outputs/motion_graph.html --distance-threshold 0.5 --top-k 5
```

Useful options:

- `--distance-threshold`: only keep transitions below this distance
- `--top-k`: keep at most top-k local-minimum transitions for each animation pair
- `--keep-dead-ends`: disable pruning to the largest strongly connected component
- `--visualization`: also export a standalone HTML visualization
- `--max-visualized-transitions`: limit cross-action transition edges in that HTML view

The output is a JSON file containing:

- `nodes`: frame references
- `edges`: sequence edges and transition edges
- `transitions`: extracted transition candidates

This step only finds transition relationships. It does not synthesize any new Gaussian point clouds.

### 3.5. Visualize an existing motion graph

If you already have `motion_graph.json`, you can render it separately:

```bash
python visualize_motion_graph.py -g outputs/motion_graph.json -o outputs/motion_graph.html
```

Useful options:

- `--frame-spacing`: horizontal spacing between frame nodes
- `--lane-spacing`: vertical spacing between animation lanes
- `--max-transition-edges`: limit the number of rendered cross-action transition edges

The HTML view is split into per-action panels plus one cross-action panel:

- each action keeps its frame sequence paths and only the smallest set of same-action transitions needed to connect its internal components
- transitions across different actions are summarized separately as directed action-to-action links

The generated HTML is standalone and can be opened directly in a browser.

### 4. Fill transition frames

After `motion_graph.json` is generated, you can synthesize in-between Gaussian point clouds for its transitions:

```bash
python fill_transition_frames.py -g outputs/motion_graph.json -o outputs/transitions --num-transition-frames 4 --sh-degree 3
```

Useful options:

- `--database`: override the database directory stored in `motion_graph.json`
- `--num-transition-frames`: number of synthesized in-between frames for each transition
- `--max-transitions`: export only the first N transitions
- `--sh-degree`: SH degree used when loading Gaussian frames

This writes one directory per transition, for example:

```text
outputs/
  transitions/
    0000_walk_anim_1_0002__attack_1_anim_1_0004/
      transition.json
      frame_0.ply
      frame_1.ply
      frame_2.ply
      frame_3.ply
```

## Notes

- `generate_sample_char1.py` creates synthetic demo data intended for pipeline testing, not realistic production Gaussians.
- `compute_similarity.py` currently compares fixed-length windows between animation sequences.
- `build_motion_graph.py` uses the saved similarity matrices plus the original database structure to build the graph.
- `fill_transition_frames.py` uses direct interpolation of position, opacity, scaling, SH features, and quaternion rotation to synthesize transition Gaussians.
