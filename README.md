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
python build_motion_graph.py -m "D:\Files\Data\char1\anim_base\char1" -s outputs/char1 -o outputs/char1/motion_graph.json --shortest-path --distance-threshold 0.5 --top-k 5
```

这一步会生成：

- `outputs/char1/motion_graph.json`
- `outputs/char1/shortest_path.json`

`shortest_path.json` 会被可视化页面用来驱动 `To <action>` 按钮。

### 3. 补全 transition 中间帧

```bash
python fill_transition_frames.py -g outputs/char1/motion_graph.json -o outputs/char1/transitions --sh-degree 3
```

### 4. 渲染图片库

```bash
python render_image_library.py \
  -m "D:\Files\Data\char1\anim_base\char1" \
  -t "outputs\char1\transitions" \
  -o "outputs\char1\rendered_images" \
  --camera-position "-1.5602173805236816, 1.2468605041503906, -1.820025086402893" \
  --camera-target "-0.10451453924179077, 0.7335010170936584, 0.7013264894485474"
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
  --mode graph \
  --port 8765 
```

然后打开 `http://127.0.0.1:8765`。

页面里最重要的交互有：

- `To <action>`：临时中断 random walk，按 shortest path 切换到目标动作
- `Stay Within Current Action`：只在当前动作内部移动，直到你主动切换到别的动作

## 附加工具

### Render scan view

如果你想对单个 Gaussian 帧做一组环绕视角渲染，可以运行：

```bash
python render_scan_view.py --frame-path "D:\Files\Data\char1\anim_base\char1\attack_1\point_cloud_60.ply" --output-dir outputs/char1/scan_view --scan-distance-scale 1.0
```

这会在 `outputs/char1/scan_view` 下生成一组 `view_az_*.png`。这个工具依赖 CUDA。

如果你想直接拿某个角度对应的 `--camera-position` 和 `--camera-target`，可以运行：

```bash
python render_scan_view.py \
  --frame-path "D:\Files\Data\char1\anim_base\char1\attack_1\point_cloud_60.ply" \
  --scan-distance-scale 1.5 \
  --print-camera-at-azimuth 210 \
  --print-only
```

它会直接输出可用于 `render_image_library.py` 的参数；如果需要，也可以重复传多个 `--print-camera-at-azimuth`。
