# GS_MotionGraph

从 Gaussian 动作序列构建 motion graph，并用 Flask 在网页里浏览随机游走、动作切换和渲染后的帧预览。

## 依赖

常用环境依赖：

- `torch`
- `plyfile`
- `tqdm`
- `flask`

## 常用流程

### 1. 计算相似度矩阵

```bash
python compute_similarity.py -m "D:\Files\Data\char1\anim_base\char1" -o outputs/char1 --window 10 --min-gap 60 --sh-degree 3
```

### 2. 构建 motion graph，并导出 shortest path

```bash
python build_motion_graph.py -m "D:\Files\Data\char1\anim_base\char1" -s outputs/char1 -o outputs/char1/motion_graph.json --shortest-path --distance-threshold 0.5 --top-k-intra-sequence 4 --top-k-inter-animation 5 --top-k-inter-sequence 4
```

其中：

- `--top-k-intra-sequence` 控制同一个 `action/animation` 内部候选 transition 的保留数
- `--top-k-inter-animation` 控制同一个 `action` 下不同 `animation` 之间候选 transition 的保留数
- `--top-k-inter-sequence` 控制不同 `action` 之间候选 transition 的保留数
- 如果你只传 `--top-k`，它会作为以上三类情况的默认值
- 如果你不传 `--top-k-inter-animation`，它会默认继承 `--top-k-inter-sequence`

这一步会生成：

- `outputs/char1/motion_graph.json`
- `outputs/char1/motion_graph_before_prune.svg`
- `outputs/char1/motion_graph_after_prune.svg`
- `outputs/char1/shortest_path.json`

`shortest_path.json` 会被可视化页面用来驱动 `To <action>` 按钮。
`motion_graph_before_prune.svg` 是裁剪前的 motion graph 静态图片。
`motion_graph_after_prune.svg` 是裁剪后的 motion graph 静态图片。

### 3. 补全 transition 中间帧

```bash
python fill_transition_frames.py -g outputs/char1/motion_graph.json -o outputs/char1/transitions --sh-degree 3
```

### 4. 渲染图片库

```bash
python render_image_library.py \
  -m "D:\Files\Data\char1\anim_base\char1" \
  -t "outputs\char1\transitions" \
  -o "outputs\char1\rendered_images_front" \
  --camera-position "-2.9414925575256348, 1.4234471321105957, -1.035473346710205" \
  --camera-target "0.2563316226005554, 0.7723545432090759, 0.8107913136482239"

python render_image_library.py \
  -m "D:\Files\Data\char1\anim_base\char1" \
  -t "outputs\char1\transitions" \
  -o "outputs\char1\rendered_images_right" \
  --camera-position "-1.589932918548584, 1.4234471321105957, 4.008615493774414" \
  --camera-target "0.2563316226005554, 0.7723545432090759, 0.8107913136482239"
```

这一步会生成 `outputs/char1/rendered_images/manifest.json`。

### 5. 启动可视化

```bash
python visualize_motion_graph.py \
  -g "outputs\char1\motion_graph.json" \
  --mode image \
  --port 8765 \
  --image-manifest "outputs\char1\rendered_images\manifest.json"
```
```bash
python visualize_motion_graph.py \
  -g "outputs\char1\motion_graph.json" \
  --mode image \
  --port 8765 \
  --image-manifest "outputs\char1\rendered_images_front\manifest.json" \
  --image-manifest "outputs\char1\rendered_images_right\manifest.json"
```
```bash
python visualize_motion_graph.py \
  -g "outputs\char1\motion_graph.json" \
  --mode graph \
  --port 8765 
```

在 `image` 模式下，重复传 `--image-manifest` 可以同时播放多个图片库；当前页面会把它们并排放在上方，motion graph 放在下方。

然后打开 `http://127.0.0.1:8765`。
1
页面里最重要的交互有：

- `To <action>`：临时中断 random walk，按 shortest path 切换到目标动作
- `Stay Within Current Action`：只在当前动作内部移动，直到你主动切换到别的动作
- `1` 到 `9`：按顶部动作按钮的显示顺序，直接切换到对应 action

## 附加工具

### Render scan view

如果你想对单个 Gaussian 帧做一组环绕视角渲染，可以运行：

```bash
python render_scan_view.py --frame-path "D:\Files\Data\char1\anim_base\char1\attack_1\point_cloud_150.ply" --output-dir outputs/char1/scan_view --scan-distance-scale 2.0
```

这会在 `outputs/char1/scan_view` 下生成一组 `view_az_*.png`。这个工具依赖 CUDA。

如果你想直接拿某个角度对应的 `--camera-position` 和 `--camera-target`，可以运行：

```bash
python render_scan_view.py \
  --frame-path "D:\Files\Data\char1\anim_base\char1\attack_1\point_cloud_150.ply" \
  --scan-distance-scale 2.0 \
  --print-camera-at-azimuth 330 \
  --print-only
```

它会直接输出可用于 `render_image_library.py` 的参数；如果需要，也可以重复传多个 `--print-camera-at-azimuth`。
