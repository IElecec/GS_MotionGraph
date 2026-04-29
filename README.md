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
      anim_2/
    attack_2/
      anim_1/
```

Each animation directory contains frame-wise Gaussian `.ply` files named as `point_cloud_<index>.ply`.

## Requirements

The scripts assume the Python environment can import the dependencies already used in this repo, especially:

- `torch`
- `plyfile`
- `tqdm`

## Quick Start

### 1. Compute similarity matrices

```bash
python compute_similarity.py -m data/jyf -o outputs/jyf --window 10 --min-gap 60 --sh-degree 3
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

### 2. Build the motion graph

```bash
python build_motion_graph.py -m data/char1 -s outputs -o outputs/motion_graph.json --distance-threshold 0.5 --top-k 5
```

If you also want an HTML visualization at build time:

```bash
python build_motion_graph.py -m data/jyf -s outputs/jyf -o outputs/jyf/motion_graph.json --visualization outputs/jyf/motion_graph.html --distance-threshold 0.5 --top-k 5
```

Useful options:

- `--distance-threshold`: only keep transitions below this distance
- `--top-k`: keep at most top-k local-minimum transitions for each animation pair
- `--keep-dead-ends`: disable pruning to the largest strongly connected component
- `--visualization`: also export a standalone HTML visualization
- `--max-visualized-transitions`: limit cross-action transition edges in that HTML view
- `--visualization-fps`: logical playback fps used by the random walker in that HTML view

The output is a JSON file containing:

- `nodes`: only frame references that participate in at least one transition
- `edges`: compressed sequence edges between consecutive transition nodes, plus transition edges
- `transitions`: extracted transition candidates

For each transition:

- `source.frame` is the source window start frame
- `target.frame` is the target window end frame

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
- `--fps`: logical playback fps used by the random walker

The generated HTML is standalone and can be opened directly in a browser.

The random walker uses logical graph time instead of SVG pixel distance:

- traversal time of one edge = `edge.length / fps`
- for example, if `fps = 30` and `edge.length = 9`, the walker spends `0.3s` on that edge

The visualization is rendered as one large SVG:

- transition nodes from the same `action` are grouped into one cluster
- each cluster keeps its internal sequence edges and same-action bridge transitions
- cross-action transitions are drawn directly between clusters in the same figure
- a small animated ball performs a random walk over the visible graph edges
- the HTML page includes an FPS slider / numeric input so you can change walker speed interactively

### 3. Fill transition frames

After `motion_graph.json` is generated, you can synthesize in-between Gaussian point clouds for its transitions:

```bash
python fill_transition_frames.py -g outputs/jyf/motion_graph.json -o outputs/jyf/transitions --sh-degree 3
```

Useful options:

- `--database`: override the database directory stored in `motion_graph.json`
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

The number of synthesized in-between frames is no longer set manually here. It is derived from `motion_graph.json`:

- `transition_edge_length = window_size - 1`
- `num_transition_frames = window_size - 2`

## Notes

- `generate_sample_char1.py` creates synthetic demo data intended for pipeline testing, not realistic production Gaussians.
- `compute_similarity.py` currently compares fixed-length windows between animation sequences.
- `build_motion_graph.py` uses the saved similarity matrices plus the original database structure to build the graph.
- `fill_transition_frames.py` uses direct interpolation of position, opacity, scaling, SH features, and quaternion rotation to synthesize transition Gaussians.
