import html
import json
import math
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

NodeKey = Tuple[str, str, int]
LaneKey = Tuple[str, str]

FRAME_GAP = 78
LANE_GAP = 88
CLUSTER_PAD_X = 84
CLUSTER_PAD_Y = 70
CLUSTER_GAP_X = 34
CLUSTER_GAP_Y = 36
OUTER_PAD = 28
NODE_RADIUS = 7
EDGE_START_PAD = NODE_RADIUS + 2
EDGE_END_PAD = NODE_RADIUS + 2

SVG_IMAGE_STYLE = """
<style>
  svg {
    background: #f5efe4;
  }
  text {
    font-family: Georgia, "Times New Roman", serif;
  }
  .cluster-box {
    fill: #f8f0e6;
    stroke: #d8c7b1;
    stroke-width: 1.3;
  }
  .cluster-title {
    fill: #1b1b1b;
    font-size: 19px;
    font-weight: 600;
  }
  .cluster-meta {
    fill: #7b7265;
    font-size: 11px;
  }
  .lane-line {
    stroke: #decfbc;
    stroke-width: 1;
    stroke-dasharray: 4 6;
  }
  .lane-label {
    fill: #7b7265;
    font-size: 11px;
  }
  .edge {
    fill: none;
    stroke-width: 2.2;
    opacity: 0.74;
  }
  .edge-sequence { stroke: #5f8f9b; }
  .edge-bridge { stroke: #d56a3a; }
  .edge-jump { stroke: #aa4058; opacity: 0.56; }
  .node {
    fill: #164e59;
    stroke: #fff7e9;
    stroke-width: 1.2;
  }
  .node-label {
    fill: #7b7265;
    font-size: 9px;
    text-anchor: middle;
  }
</style>
""".strip()

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Motion Graph Viewer</title>
  <style>
    :root {
      --bg: #f5efe4;
      --panel: #fffdf9;
      --ink: #1b1b1b;
      --muted: #7b7265;
      --border: #d8c7b1;
      --cluster: #f8f0e6;
      --sequence: #5f8f9b;
      --bridge: #d56a3a;
      --jump: #aa4058;
      --walker: #f2b134;
      --accent: #164e59;
      --accent-soft: #e0f0f2;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: linear-gradient(180deg, #efe5d8 0%, var(--bg) 100%);
    }
    .page {
      max-width: 1700px;
      margin: 0 auto;
      padding: 18px;
    }
    .hero {
      display: grid;
      gap: 14px;
      margin-bottom: 14px;
    }
    .hero-main {
      padding: 18px 20px;
      border: 1px solid var(--border);
      border-radius: 20px;
      background: var(--panel);
      box-shadow: 0 12px 32px rgba(63, 45, 28, 0.08);
    }
    h1 {
      margin: 0;
      font-size: 30px;
      line-height: 1.1;
    }
    .summary {
      margin-top: 8px;
      color: var(--muted);
      font-size: 14px;
    }
    .topbar {
      position: sticky;
      top: 14px;
      z-index: 5;
      display: grid;
      gap: 12px;
      padding: 16px 18px;
      border: 1px solid var(--border);
      border-radius: 22px;
      background: rgba(255, 253, 249, 0.96);
      backdrop-filter: blur(8px);
      box-shadow: 0 14px 36px rgba(63, 45, 28, 0.1);
    }
    .topbar-row {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      align-items: center;
    }
    .topbar-label {
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
      min-width: 88px;
    }
    .button {
      appearance: none;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: #fff8ef;
      color: var(--ink);
      padding: 10px 16px;
      font: inherit;
      font-size: 14px;
      cursor: pointer;
      transition: transform 120ms ease, background 120ms ease, border-color 120ms ease, color 120ms ease;
    }
    .button:hover:not(:disabled) {
      transform: translateY(-1px);
      border-color: #c89446;
      background: #fde9c9;
    }
    .button:disabled {
      cursor: default;
      opacity: 0.42;
    }
    .button.primary {
      border-color: #cf8f08;
      background: #f2b134;
      color: #2d2108;
      font-weight: 600;
      box-shadow: 0 10px 24px rgba(242, 177, 52, 0.28);
    }
    .button.primary.active,
    .button.active {
      border-color: #0f5663;
      background: var(--accent);
      color: #f5fbfc;
      box-shadow: 0 10px 24px rgba(22, 78, 89, 0.2);
    }
    .action-buttons {
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
    }
    .status {
      min-height: 22px;
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
    }
    .speed {
      display: inline-flex;
      align-items: center;
      gap: 10px;
      margin-left: auto;
      padding: 8px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
    }
    .speed input[type="range"] {
      width: 180px;
    }
    .speed-readout {
      font-size: 13px;
      min-width: 48px;
      text-align: right;
    }
    .content {
      display: grid;
      grid-template-columns: __CONTENT_COLUMNS__;
      gap: 18px;
      align-items: start;
      margin-top: 16px;
    }
    .preview-strip {
      display: grid;
      gap: 18px;
      align-items: start;
    }
    .card {
      border: 1px solid var(--border);
      border-radius: 20px;
      background: var(--panel);
      box-shadow: 0 12px 32px rgba(63, 45, 28, 0.08);
    }
    .preview {
      padding: 16px;
      min-width: 0;
    }
    .preview h2,
    .graph h2 {
      margin: 0 0 12px 0;
      font-size: 18px;
    }
    .preview-stage {
      min-height: 420px;
      border-radius: 16px;
      border: 1px solid var(--border);
      background: #efe5d8;
      overflow: hidden;
    }
    .preview-frame {
      width: 100%;
      display: block;
      min-height: 420px;
      object-fit: contain;
    }
    .preview-caption {
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
      word-break: break-word;
    }
    .graph {
      padding: 16px;
      overflow: auto;
    }
    .legend {
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .legend span::before {
      content: "";
      display: inline-block;
      width: 10px;
      height: 10px;
      margin-right: 7px;
      border-radius: 999px;
      vertical-align: middle;
      background: currentColor;
    }
    svg {
      width: 100%;
      height: auto;
      display: block;
    }
    .cluster-box {
      fill: var(--cluster);
      stroke: var(--border);
      stroke-width: 1.3;
    }
    .cluster-title {
      fill: var(--ink);
      font-size: 19px;
      font-weight: 600;
    }
    .cluster-meta {
      fill: var(--muted);
      font-size: 11px;
    }
    .lane-line {
      stroke: #decfbc;
      stroke-width: 1;
      stroke-dasharray: 4 6;
    }
    .lane-label {
      fill: var(--muted);
      font-size: 11px;
    }
    .edge {
      fill: none;
      stroke-width: 2.2;
      opacity: 0.74;
    }
    .edge-sequence { stroke: var(--sequence); }
    .edge-bridge { stroke: var(--bridge); }
    .edge-jump { stroke: var(--jump); opacity: 0.56; }
    .edge-active {
      stroke: var(--walker) !important;
      stroke-width: 3.6;
      opacity: 0.95 !important;
    }
    .node {
      fill: var(--accent);
      stroke: #fff7e9;
      stroke-width: 1.2;
    }
    .node-label {
      fill: var(--muted);
      font-size: 9px;
      text-anchor: middle;
    }
    .walker {
      fill: var(--walker);
      stroke: #fff7e9;
      stroke-width: 1.7;
      filter: drop-shadow(0 0 5px rgba(242, 177, 52, 0.55));
    }
    @media (max-width: 1180px) {
      .content {
        grid-template-columns: 1fr;
      }
      .preview-strip {
        grid-template-columns: 1fr !important;
      }
      .speed {
        margin-left: 0;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div class="hero-main">
        <h1>Motion Graph Viewer</h1>
        <div class="summary">__SUMMARY__</div>
      </div>
      <div class="topbar">
        <div class="topbar-row">
          <div class="topbar-label">Main Controls</div>
          <button id="lock-button" class="button primary" type="button">Stay Within Current Action</button>
          <div class="speed">
            <span>Speed</span>
            <input id="speed-slider" type="range" min="1" max="60" step="1" value="__FPS__">
            <span id="speed-readout" class="speed-readout"></span>
          </div>
        </div>
        <div class="topbar-row">
          <div class="topbar-label">To Action</div>
          <div id="action-buttons" class="action-buttons"></div>
        </div>
        <div id="status" class="status">__STATUS__</div>
      </div>
    </section>
    <section class="content">
      __PREVIEW_BLOCK__
      <div class="card graph">
        <h2>Graph</h2>
        <div class="legend">
          <span style="color: var(--sequence);">sequence edge</span>
          <span style="color: var(--bridge);">intra-action transition</span>
          <span style="color: var(--jump);">cross-action transition</span>
          <span style="color: var(--walker);">walker</span>
        </div>
        __SVG__
      </div>
    </section>
  </div>
  <script>
    const data = __DATA__;

    const walker = document.getElementById("walker");
    const previewPanels = Array.from(document.querySelectorAll("[data-preview-index]")).map((panel) => {
      const image = panel.querySelector(".preview-frame");
      const caption = panel.querySelector(".preview-caption");
      return {
        index: Number(panel.dataset.previewIndex || 0),
        image,
        caption,
        shownPath: null,
        wantedPath: null,
      };
    });
    const actionButtons = document.getElementById("action-buttons");
    const status = document.getElementById("status");
    const lockButton = document.getElementById("lock-button");
    const speedSlider = document.getElementById("speed-slider");
    const speedReadout = document.getElementById("speed-readout");

    if (walker && data.edge_ids.length > 0) {
      let fps = Math.max(1, Number(data.fps) || 24);
      let currentEdgeId = null;
      let currentPath = null;
      let edgeStart = 0;
      let edgeLength = 1;
      let edgeDuration = 1000 / fps;
      let activePath = null;
      let walkMode = "random";
      let guidedRoute = null;
      let routeTarget = null;
      let routeQueue = [];
      let lockEnabled = false;
      let lockAction = null;
      const assetNonce = String(Date.now());
      const imageCache = new Map();
      const edgesByAction = {};

      function setStatus(text) {
        if (status) {
          status.textContent = text;
        }
      }

      function setSpeed(nextFps) {
        fps = Math.max(1, Number(nextFps) || fps || 24);
        if (speedSlider) {
          speedSlider.value = String(Math.round(fps));
        }
        if (speedReadout) {
          speedReadout.textContent = `${Math.round(fps)} fps`;
        }
        if (currentEdgeId) {
          edgeDuration = logicalDuration(currentEdgeId);
        }
      }

      function logicalDuration(edgeId) {
        const edge = data.edges[edgeId];
        const frames = edge ? Math.max(1, Number(edge.length) || 1) : 1;
        return (1000 / fps) * frames;
      }

      function nodeInfo(nodeId) {
        return data.nodes[nodeId] || null;
      }

      function nodeAction(nodeId) {
        const info = nodeInfo(nodeId);
        return info ? info.action : null;
      }

      function currentNodeId() {
        const edge = data.edges[currentEdgeId];
        return edge ? edge.target : null;
      }

      function pick(list) {
        return list[Math.floor(Math.random() * list.length)];
      }

      function assetUrl(path) {
        if (!path) {
          return path;
        }
        return path.includes("?") ? `${path}&v=${assetNonce}` : `${path}?v=${assetNonce}`;
      }

      function preload(path) {
        if (!path) {
          return null;
        }
        let record = imageCache.get(path);
        if (record) {
          return record;
        }
        const img = new Image();
        img.decoding = "async";
        record = { image: img, ready: false };
        img.onload = () => {
          record.ready = true;
          previewPanels.forEach((panel) => {
            if (panel.wantedPath === path) {
              showPreview(panel, path);
            }
          });
        };
        img.src = assetUrl(path);
        imageCache.set(path, record);
        return record;
      }

      function showPreview(panel, path) {
        if (!panel.image || !path) {
          return;
        }
        if (panel.shownPath !== path) {
          panel.image.src = assetUrl(path);
          panel.shownPath = path;
        }
        panel.image.hidden = false;
      }

      function edgeImages(edgeId, previewIndex) {
        const edge = data.edges[edgeId];
        return edge ? (edge.image_path_sets?.[previewIndex] || []) : [];
      }

      function warmEdge(edgeId) {
        previewPanels.forEach((panel) => {
          edgeImages(edgeId, panel.index).forEach((path) => preload(path));
        });
      }

      function warmOutgoing(nodeId, limit = 6) {
        (data.outgoing[nodeId] || []).slice(0, limit).forEach((edgeId) => warmEdge(edgeId));
      }

      function previewPath(progress, previewIndex) {
        const edge = data.edges[currentEdgeId];
        if (!edge) {
          return null;
        }
        const paths = edge.image_path_sets?.[previewIndex] || [];
        if (paths.length > 0) {
          const index = Math.min(paths.length - 1, Math.floor(progress * paths.length));
          return paths[index];
        }
        const target = nodeInfo(edge.target);
        return target ? (target.image_paths?.[previewIndex] || null) : null;
      }

      function updatePreview(progress) {
        const edge = data.edges[currentEdgeId];
        previewPanels.forEach((panel) => {
          const path = previewPath(progress, panel.index);
          if (panel.caption) {
            panel.caption.textContent = edge ? edge.label : "";
          }

          if (!panel.image) {
            return;
          }
          if (!path) {
            panel.wantedPath = null;
            panel.image.hidden = true;
            return;
          }
          panel.wantedPath = path;
          const record = preload(path);
          if (record && record.ready) {
            showPreview(panel, path);
          }
        });
      }

      function actionEdgePool(action) {
        if (!edgesByAction[action]) {
          edgesByAction[action] = data.edge_ids.filter((edgeId) => {
            const edge = data.edges[edgeId];
            return edge && nodeAction(edge.source) === action && nodeAction(edge.target) === action;
          });
        }
        return edgesByAction[action];
      }

      function preferNonTransition(edgeIds) {
        const preferred = edgeIds.filter((edgeId) => {
          const edge = data.edges[edgeId];
          return edge && edge.kind !== "transition";
        });
        return preferred.length > 0 ? preferred : edgeIds;
      }

      function localActionEdges(fromNodeId, action) {
        return (data.outgoing[fromNodeId] || []).filter((edgeId) => {
          const edge = data.edges[edgeId];
          return edge && nodeAction(edge.source) === action && nodeAction(edge.target) === action;
        });
      }

      function forwardSequenceEdges(fromNodeId) {
        const source = nodeInfo(fromNodeId);
        if (!source) {
          return [];
        }
        return (data.outgoing[fromNodeId] || []).filter((edgeId) => {
          const edge = data.edges[edgeId];
          const target = edge ? nodeInfo(edge.target) : null;
          return (
            edge &&
            edge.kind === "sequence" &&
            target &&
            source.action === target.action &&
            source.animation === target.animation &&
            Number(target.frame) >= Number(source.frame)
          );
        });
      }

      function sequenceStartRoute(nodeId) {
        if (!nodeId) {
          return null;
        }
        const routes = data.routes?.sequence_start_routes || {};
        return routes[nodeId] || null;
      }

      function actionRoute(action) {
        const startNode = currentNodeId();
        if (!startNode) {
          return null;
        }
        if (nodeAction(startNode) === action) {
          return sequenceStartRoute(startNode);
        }
        const sourceRoutes = (data.routes && data.routes.source_routes[startNode]) || {};
        return sourceRoutes[action] || null;
      }

      function beginGuidedRoute(route, metadata, statusText) {
        guidedRoute = metadata;
        routeTarget = metadata.routeTarget || null;
        routeQueue = Array.isArray(route?.edge_ids) ? [...route.edge_ids] : [];
        walkMode = "guided";
        routeQueue.forEach((edgeId) => warmEdge(edgeId));
        setStatus(statusText);
        refreshActionButtons();
      }

      function maybeStartLockReturnRoute(fromNodeId) {
        if (!lockEnabled || !lockAction) {
          return null;
        }
        const source = nodeInfo(fromNodeId);
        const route = sequenceStartRoute(fromNodeId);
        if (!source || source.action !== lockAction || !route || !route.reachable || !route.edge_ids.length) {
          return null;
        }

        beginGuidedRoute(
          route,
          {
            kind: "lock-return",
            action: lockAction,
            animation: route.target_animation || source.animation,
            startFrame: Number(route.target_frame) || 0,
          },
          `Reached the end of ${source.animation}. Returning to frame ${route.target_frame} inside ${lockAction}.`
        );
        return routeQueue.shift() || null;
      }

      function nextRandomEdge(fromNodeId) {
        if (lockEnabled && lockAction) {
          const sequence = forwardSequenceEdges(fromNodeId);
          if (sequence.length > 0) {
            return pick(sequence);
          }
          const returnEdge = maybeStartLockReturnRoute(fromNodeId);
          if (returnEdge) {
            return returnEdge;
          }
          const local = localActionEdges(fromNodeId, lockAction);
          if (local.length > 0) {
            return pick(preferNonTransition(local));
          }
          const fallback = actionEdgePool(lockAction);
          if (fallback.length > 0) {
            return pick(preferNonTransition(fallback));
          }
        }
        const outgoing = data.outgoing[fromNodeId];
        if (outgoing && outgoing.length > 0) {
          return pick(outgoing);
        }
        return pick(data.edge_ids);
      }

      function updateLockButton() {
        if (!lockButton) {
          return;
        }
        lockButton.classList.toggle("active", lockEnabled);
        if (lockEnabled && lockAction) {
          lockButton.textContent = `Stay Within ${lockAction}`;
        } else {
          lockButton.textContent = "Stay Within Current Action";
        }
      }

      function highlightEdge(edgeId) {
        if (activePath) {
          activePath.classList.remove("edge-active");
        }
        activePath = edgeId ? document.getElementById(edgeId) : null;
        if (activePath) {
          activePath.classList.add("edge-active");
        }
      }

      function refreshActionButtons() {
        if (!actionButtons) {
          return;
        }
        actionButtons.querySelectorAll("button").forEach((button) => {
          const action = button.dataset.action;
          const route = actionRoute(action);
          button.disabled = !route || !route.reachable;
          button.classList.toggle("active", walkMode === "guided" && routeTarget === action);
        });
        updateLockButton();
      }

      function routeToAction(action) {
        const startNode = currentNodeId();
        const currentAction = nodeAction(startNode);
        const route = actionRoute(action);
        if (!route || !route.reachable) {
          setStatus(
            currentAction === action
              ? `No precomputed route from the next ${action} node back to its sequence start.`
              : `No route from the current node to ${action}.`
          );
          refreshActionButtons();
          return;
        }

        if (currentAction === action) {
          beginGuidedRoute(
            route,
            {
              kind: "same-action-return",
              action,
              animation: route.target_animation || nodeInfo(startNode)?.animation || action,
              startFrame: Number(route.target_frame) || 0,
              routeTarget: action,
            },
            route.edge_ids.length > 0
              ? `After the next ${action} node, returning to frame ${route.target_frame} of ${route.target_animation}.`
              : `The next ${action} node is already frame ${route.target_frame} of ${route.target_animation}.`
          );
          return;
        }

        beginGuidedRoute(
          route,
          {
            kind: "action",
            action,
            routeTarget: action,
          },
          route.edge_ids.length > 0
            ? `Following shortest path to ${action}.`
            : `Already entering ${action}. Random walk will resume after this edge.`
        );
      }

      function buildActionButtons() {
        if (!actionButtons) {
          return;
        }
        if (!data.routes || !data.routes.actions.length) {
          actionButtons.innerHTML = "<span style='color: var(--muted);'>Build with --shortest-path to enable action routing and return-to-start routing.</span>";
          return;
        }

        actionButtons.innerHTML = "";
        data.routes.actions.forEach((action, index) => {
          const button = document.createElement("button");
          button.type = "button";
          button.className = "button";
          button.dataset.action = action;
          const hotkey = index < 9 ? `${index + 1}. ` : "";
          button.textContent = `${hotkey}To ${action}`;
          button.addEventListener("click", () => {
            routeToAction(action);
          });
          actionButtons.appendChild(button);
        });
        refreshActionButtons();
      }

      function isEditableTarget(target) {
        if (!target || !(target instanceof Element)) {
          return false;
        }
        if (target.isContentEditable) {
          return true;
        }
        const tagName = target.tagName;
        if (tagName === "TEXTAREA" || tagName === "SELECT") {
          return true;
        }
        if (tagName !== "INPUT") {
          return false;
        }
        const inputType = String(target.getAttribute("type") || "text").toLowerCase();
        return [
          "text",
          "search",
          "url",
          "tel",
          "password",
          "email",
          "number",
        ].includes(inputType);
      }

      function handleHotkey(event) {
        if (!data.routes || !Array.isArray(data.routes.actions) || !data.routes.actions.length) {
          return;
        }
        if (event.defaultPrevented || event.repeat || event.altKey || event.ctrlKey || event.metaKey) {
          return;
        }
        if (isEditableTarget(event.target)) {
          return;
        }
        if (!/^[1-9]$/.test(event.key)) {
          return;
        }
        const index = Number(event.key) - 1;
        const action = data.routes.actions[index];
        if (!action) {
          return;
        }
        event.preventDefault();
        routeToAction(action);
      }

      function toggleLock() {
        if (!lockEnabled) {
          const action = nodeAction(currentNodeId());
          if (!action) {
            setStatus("The lock will apply after the walker reaches a node.");
            return;
          }
          lockEnabled = true;
          lockAction = action;
          setStatus(`Single-action mode enabled for ${lockAction}.`);
        } else {
          lockEnabled = false;
          lockAction = null;
          setStatus("Single-action mode disabled.");
        }
        refreshActionButtons();
      }

      function activate(edgeId, now) {
        currentEdgeId = edgeId;
        currentPath = document.getElementById(edgeId);
        if (!currentPath) {
          return;
        }
        edgeStart = now;
        edgeLength = Math.max(currentPath.getTotalLength(), 1);
        edgeDuration = logicalDuration(edgeId);
        highlightEdge(edgeId);
        warmEdge(edgeId);
        const edge = data.edges[edgeId];
        if (edge) {
          warmOutgoing(edge.target);
        }
        updatePreview(0);
        refreshActionButtons();
      }

      function step(now) {
        if (!currentEdgeId) {
          activate(pick(data.edge_ids), now);
        }
        if (!currentPath) {
          requestAnimationFrame(step);
          return;
        }

        const progress = Math.min((now - edgeStart) / edgeDuration, 1);
        const point = currentPath.getPointAtLength(progress * edgeLength);
        walker.setAttribute("cx", point.x.toFixed(2));
        walker.setAttribute("cy", point.y.toFixed(2));
        updatePreview(progress);

        if (progress >= 1) {
          const edge = data.edges[currentEdgeId];
          if (walkMode === "guided") {
            if (routeQueue.length > 0) {
              activate(routeQueue.shift(), now);
            } else {
              const completedRoute = guidedRoute;
              const reached = routeTarget;
              walkMode = "random";
              guidedRoute = null;
              routeTarget = null;
              if (lockEnabled && completedRoute?.kind === "action") {
                lockAction = reached || nodeAction(edge.target) || lockAction;
              }
              if (completedRoute?.kind === "lock-return" || completedRoute?.kind === "same-action-return") {
                setStatus(
                  lockEnabled
                    ? `Returned to frame ${completedRoute.startFrame} of ${completedRoute.animation}. Continuing inside ${completedRoute.action}.`
                    : `Returned to frame ${completedRoute.startFrame} of ${completedRoute.animation}. Random walk resumed.`
                );
              } else {
                setStatus(
                  lockEnabled && lockAction
                    ? `Arrived at ${reached}. Continuing inside ${lockAction}.`
                    : `Arrived at ${reached}. Random walk resumed.`
                );
              }
              activate(nextRandomEdge(edge.target), now);
            }
          } else {
            if (lockEnabled && !lockAction) {
              lockAction = nodeAction(edge.target);
            }
            activate(nextRandomEdge(edge.target), now);
          }
        }

        requestAnimationFrame(step);
      }

      if (lockButton) {
        lockButton.addEventListener("click", toggleLock);
      }
      if (speedSlider) {
        speedSlider.addEventListener("input", (event) => setSpeed(event.target.value));
      }
      document.addEventListener("keydown", handleHotkey);
      buildActionButtons();
      setSpeed(data.fps);
      if (!status.textContent) {
        setStatus("Random walk is running.");
      }
      requestAnimationFrame(step);
    }
  </script>
</body>
</html>
"""


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def node_key(node: Dict[str, Any]) -> NodeKey:
    return (node["action"], node["animation"], int(node["frame"]))


def lane_key(node: Dict[str, Any]) -> LaneKey:
    return (node["action"], node["animation"])


def node_id(node: Dict[str, Any]) -> str:
    return f"{node['action']}|{node['animation']}|{int(node['frame'])}"


def edge_signature(edge: Dict[str, Any]) -> str:
    return "|".join(
        [
            node_id(edge["source"]),
            node_id(edge["target"]),
            str(edge.get("kind", "sequence")),
            str(int(edge.get("length", 0))),
            f"{float(edge.get('distance', 0.0)):.12g}",
            f"{float(edge.get('theta', 0.0)):.12g}",
        ]
    )


def transition_folder(index: int, edge: Dict[str, Any]) -> str:
    source = edge["source"]
    target = edge["target"]
    return (
        f"{index:04d}_"
        f"{source['action']}_{source['animation']}_{int(source['frame']):04d}"
        f"__"
        f"{target['action']}_{target['animation']}_{int(target['frame']):04d}"
    )


def default_transition_length(payload: Dict[str, Any]) -> int:
    if payload.get("transition_edge_length") is not None:
        return max(0, int(payload["transition_edge_length"]))
    if payload.get("window_size") is not None:
        return max(0, int(payload["window_size"]) - 1)
    return 0


def load_graph(graph_path: Path) -> Dict[str, Any]:
    payload = load_json(graph_path)
    transition_len = default_transition_length(payload)

    transition_meta: Dict[str, Dict[str, Any]] = {}
    for index, transition in enumerate(payload.get("transitions", [])):
        edge = {
            "source": transition["source"],
            "target": transition["target"],
            "kind": "transition",
            "length": transition_len,
            "distance": float(transition.get("distance", 0.0)),
            "theta": float(transition.get("theta", 0.0)),
        }
        transition_meta[edge_signature(edge)] = {
            "transition_index": index,
            "transition_folder": transition_folder(index, transition),
        }

    edges: List[Dict[str, Any]] = []
    for edge in payload.get("edges", []):
        item = dict(edge)
        item["kind"] = item.get("kind", "sequence")
        item["length"] = int(item.get("length", transition_len if item["kind"] == "transition" else 0))
        item["distance"] = float(item.get("distance", 0.0))
        item["theta"] = float(item.get("theta", 0.0))
        if item["kind"] == "transition":
            item.update(transition_meta.get(edge_signature(item), {}))
        edges.append(item)

    payload["edges"] = edges
    payload["nodes"] = list(payload.get("nodes", []))
    return payload


def register_asset(
    files: Dict[str, Path],
    ids: Dict[str, str],
    path: Path,
) -> Optional[str]:
    resolved = path.resolve()
    if not resolved.exists():
        return None
    key = str(resolved)
    asset_id = ids.get(key)
    if asset_id is None:
        asset_id = f"asset_{len(files):06d}"
        ids[key] = asset_id
        files[asset_id] = resolved
    return f"/assets/{asset_id}"


def load_image_library(
    graph_path: Path,
    manifest_path: Path,
    files: Dict[str, Path],
    ids: Dict[str, str],
    index: int,
    required: bool,
) -> Optional[Dict[str, Any]]:
    if not manifest_path.exists():
        if required:
            raise FileNotFoundError(f"Image manifest not found: {manifest_path}")
        return None

    raw = load_json(manifest_path)
    normal: Dict[str, str] = {}
    transitions: Dict[str, List[str]] = {}

    for key, rel_path in raw.get("normal_frames", {}).items():
        url = register_asset(files, ids, manifest_path.parent / rel_path)
        if url:
            normal[key] = url

    for folder, rel_paths in raw.get("transition_frames", {}).items():
        transitions[folder] = []
        for rel_path in rel_paths:
            url = register_asset(files, ids, manifest_path.parent / rel_path)
            if url:
                transitions[folder].append(url)

    label = str(raw.get("label") or manifest_path.parent.name or f"View {index + 1}")
    return {
        "label": label,
        "normal": normal,
        "transitions": transitions,
    }


def load_images(
    graph_path: Path,
    manifest_paths: Optional[List[Path]],
    required: bool,
) -> Optional[Dict[str, Any]]:
    if not manifest_paths:
        manifest_paths = [graph_path.parent / "rendered_images" / "manifest.json"]

    files: Dict[str, Path] = {}
    ids: Dict[str, str] = {}
    libraries: List[Dict[str, Any]] = []

    for index, manifest_path in enumerate(manifest_paths):
        library = load_image_library(
            graph_path=graph_path,
            manifest_path=manifest_path,
            files=files,
            ids=ids,
            index=index,
            required=required,
        )
        if library is not None:
            libraries.append(library)

    if required and not libraries:
        raise FileNotFoundError("No image manifests could be loaded.")
    if not libraries:
        return None

    return {
        "files": files,
        "libraries": libraries,
    }


def load_routes(graph_path: Path) -> Optional[Dict[str, Any]]:
    path = graph_path.parent / "shortest_path.json"
    if not path.exists():
        return None

    raw = load_json(path)
    routes: Dict[str, Dict[str, Any]] = {}
    sequence_start_routes: Dict[str, Dict[str, Any]] = {}

    if "source_nodes" in raw:
        for source in raw.get("source_nodes", []):
            source_id = source["source_id"]
            routes[source_id] = {
                route["target_action"]: route
                for route in source.get("paths_to_other_actions", [])
                if route.get("target_action")
            }
            sequence_route = source.get("path_to_sequence_start")
            if isinstance(sequence_route, dict):
                sequence_start_routes[source_id] = sequence_route
        actions = list(raw.get("available_actions", []))
    else:
        actions = list(raw.get("actions", []))
        for source_id, entry in raw.get("sources", {}).items():
            routes[source_id] = dict(entry.get("routes", {}))
            sequence_route = entry.get("path_to_sequence_start")
            if isinstance(sequence_route, dict):
                sequence_start_routes[source_id] = sequence_route

    return {
        "actions": actions,
        "source_routes": routes,
        "sequence_start_routes": sequence_start_routes,
    }


def group_actions(nodes: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for node in nodes:
        grouped.setdefault(node["action"], []).append(node)
    return grouped


def layout_graph(payload: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[NodeKey, Tuple[float, float]], int, int]:
    grouped = group_actions(payload["nodes"])
    actions = sorted(grouped)
    clusters: List[Dict[str, Any]] = []
    positions: Dict[NodeKey, Tuple[float, float]] = {}

    for action in actions:
        lanes: Dict[LaneKey, List[Dict[str, Any]]] = {}
        for node in grouped[action]:
            lanes.setdefault(lane_key(node), []).append(node)

        lane_order = sorted(lanes)
        local: Dict[NodeKey, Tuple[float, float]] = {}
        max_count = 1
        for lane_index, lane in enumerate(lane_order):
            lane_nodes = sorted(lanes[lane], key=lambda item: int(item["frame"]))
            max_count = max(max_count, len(lane_nodes))
            for frame_index, node in enumerate(lane_nodes):
                local[node_key(node)] = (
                    CLUSTER_PAD_X + frame_index * FRAME_GAP,
                    CLUSTER_PAD_Y + lane_index * LANE_GAP,
                )

        clusters.append(
            {
                "action": action,
                "nodes": grouped[action],
                "lanes": lane_order,
                "local": local,
                "width": CLUSTER_PAD_X * 2 + max(1, max_count - 1) * FRAME_GAP + 52,
                "height": CLUSTER_PAD_Y * 2 + max(0, len(lane_order) - 1) * LANE_GAP + 42,
            }
        )

    columns = 1 if len(clusters) <= 1 else 2 if len(clusters) <= 4 else 3
    widths = [0] * columns
    heights = [0] * ((len(clusters) + columns - 1) // columns)

    for index, cluster in enumerate(clusters):
        row = index // columns
        col = index % columns
        widths[col] = max(widths[col], cluster["width"])
        heights[row] = max(heights[row], cluster["height"])

    x_offsets = [OUTER_PAD]
    for width in widths[:-1]:
        x_offsets.append(x_offsets[-1] + width + CLUSTER_GAP_X)
    y_offsets = [OUTER_PAD]
    for height in heights[:-1]:
        y_offsets.append(y_offsets[-1] + height + CLUSTER_GAP_Y)

    for index, cluster in enumerate(clusters):
        row = index // columns
        col = index % columns
        cluster["x"] = x_offsets[col]
        cluster["y"] = y_offsets[row]
        cluster["positions"] = {}
        for key, (x, y) in cluster["local"].items():
            point = (cluster["x"] + x, cluster["y"] + y)
            cluster["positions"][key] = point
            positions[key] = point

    width = OUTER_PAD + sum(widths) + CLUSTER_GAP_X * max(0, len(widths) - 1) + OUTER_PAD
    height = OUTER_PAD + sum(heights) + CLUSTER_GAP_Y * max(0, len(heights) - 1) + OUTER_PAD
    return clusters, positions, width, height


def trim_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    start: float,
    end: float,
) -> Tuple[float, float, float, float]:
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length <= start + end:
        return x1, y1, x2, y2
    ux = dx / length
    uy = dy / length
    return (
        x1 + ux * start,
        y1 + uy * start,
        x2 - ux * end,
        y2 - uy * end,
    )


def trim_quadratic(
    x1: float,
    y1: float,
    cx: float,
    cy: float,
    x2: float,
    y2: float,
    start: float,
    end: float,
) -> Tuple[float, float, float, float]:
    start_dx = cx - x1
    start_dy = cy - y1
    start_length = math.hypot(start_dx, start_dy)
    if start_length > 1e-6:
        x1 += start_dx / start_length * start
        y1 += start_dy / start_length * start

    end_dx = x2 - cx
    end_dy = y2 - cy
    end_length = math.hypot(end_dx, end_dy)
    if end_length > 1e-6:
        x2 -= end_dx / end_length * end
        y2 -= end_dy / end_length * end

    return x1, y1, x2, y2


def line_path(source: Tuple[float, float], target: Tuple[float, float]) -> str:
    x1, y1, x2, y2 = trim_line(*source, *target, EDGE_START_PAD, EDGE_END_PAD)
    return f"M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f}"


def curve_path(
    source: Tuple[float, float],
    target: Tuple[float, float],
    lift: float,
    *,
    upward: bool = True,
) -> str:
    x1, y1 = source
    x2, y2 = target
    cx = (x1 + x2) / 2
    cy = min(y1, y2) - lift if upward else max(y1, y2) + lift
    x1, y1, x2, y2 = trim_quadratic(
        x1,
        y1,
        cx,
        cy,
        x2,
        y2,
        EDGE_START_PAD,
        EDGE_END_PAD,
    )
    return f"M {x1:.2f} {y1:.2f} Q {cx:.2f} {cy:.2f} {x2:.2f} {y2:.2f}"


def render_svg(payload: Dict[str, Any]) -> Tuple[str, Dict[str, str]]:
    clusters, positions, width, height = layout_graph(payload)
    path_ids: Dict[str, str] = {}

    cluster_parts: List[str] = []
    lane_parts: List[str] = []
    edge_parts: List[str] = []
    node_parts: List[str] = []

    for cluster in clusters:
        cluster_parts.append(
            (
                f'<rect class="cluster-box" x="{cluster["x"]}" y="{cluster["y"]}" '
                f'width="{cluster["width"]}" height="{cluster["height"]}" rx="18"></rect>'
                f'<text class="cluster-title" x="{cluster["x"] + 20}" y="{cluster["y"] + 28}">{html.escape(cluster["action"])}</text>'
                f'<text class="cluster-meta" x="{cluster["x"] + 20}" y="{cluster["y"] + 48}">{len(cluster["nodes"])} nodes</text>'
            )
        )
        for lane_index, lane in enumerate(cluster["lanes"]):
            y = cluster["y"] + CLUSTER_PAD_Y + lane_index * LANE_GAP
            lane_parts.append(
                f'<line class="lane-line" x1="{cluster["x"] + 26}" y1="{y}" x2="{cluster["x"] + cluster["width"] - 26}" y2="{y}"></line>'
            )
            lane_parts.append(
                f'<text class="lane-label" x="{cluster["x"] + 22}" y="{y - 10}">{html.escape(lane[1])}</text>'
            )

    for index, edge in enumerate(payload["edges"]):
        source = positions[node_key(edge["source"])]
        target = positions[node_key(edge["target"])]
        edge_id = f"edge_{index}"
        path_ids[edge_signature(edge)] = edge_id

        if edge["kind"] == "sequence":
            path_d = line_path(source, target)
            css = "edge edge-sequence"
            marker = "arrow-sequence"
        elif edge["source"]["action"] == edge["target"]["action"]:
            source_frame = int(edge["source"]["frame"])
            target_frame = int(edge["target"]["frame"])
            path_d = curve_path(
                source,
                target,
                34 + abs(target[0] - source[0]) * 0.18,
                upward=target_frame >= source_frame,
            )
            css = "edge edge-bridge"
            marker = "arrow-bridge"
        else:
            path_d = curve_path(source, target, 86)
            css = "edge edge-jump"
            marker = "arrow-jump"

        edge_parts.append(
            (
                f'<path id="{edge_id}" class="{css}" d="{path_d}" marker-end="url(#{marker})">'
                f'<title>{html.escape(node_id(edge["source"]))} -> {html.escape(node_id(edge["target"]))}</title>'
                f'</path>'
            )
        )

    for node in payload["nodes"]:
        x, y = positions[node_key(node)]
        node_parts.append(f'<circle class="node" cx="{x:.2f}" cy="{y:.2f}" r="{NODE_RADIUS}"></circle>')
        node_parts.append(
            f'<text class="node-label" x="{x:.2f}" y="{y + 20:.2f}">{int(node["frame"])}</text>'
        )

    svg = (
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">'
        '<defs>'
        '<marker id="arrow-sequence" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M 0 0 L 12 6 L 0 12 z" fill="#5f8f9b"></path></marker>'
        '<marker id="arrow-bridge" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M 0 0 L 12 6 L 0 12 z" fill="#d56a3a"></path></marker>'
        '<marker id="arrow-jump" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="userSpaceOnUse">'
        '<path d="M 0 0 L 12 6 L 0 12 z" fill="#aa4058"></path></marker>'
        '</defs>'
        + "".join(cluster_parts)
        + "".join(lane_parts)
        + "".join(edge_parts)
        + "".join(node_parts)
        + '<circle id="walker" class="walker" cx="-20" cy="-20" r="6.5"></circle>'
        + "</svg>"
    )
    return svg, path_ids


def render_svg_image(payload: Dict[str, Any]) -> str:
    svg, _ = render_svg(payload)
    return svg.replace(
        ">",
        f">{SVG_IMAGE_STYLE}<rect x=\"0\" y=\"0\" width=\"100%\" height=\"100%\" fill=\"#f5efe4\"></rect>",
        1,
    )


def save_graph_svg(payload: Dict[str, Any], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_svg_image(payload), encoding="utf-8")
    return output_path


def frame_key(action: str, animation: str, frame: int) -> str:
    return f"{action}|{animation}|{int(frame)}"


def edge_images(edge: Dict[str, Any], images: Optional[Dict[str, Any]]) -> List[str]:
    if images is None:
        return []

    if edge["kind"] == "sequence":
        source = edge["source"]
        target = edge["target"]
        if source["action"] == target["action"] and source["animation"] == target["animation"]:
            step = 1 if int(target["frame"]) >= int(source["frame"]) else -1
            paths: List[str] = []
            for frame in range(int(source["frame"]) + step, int(target["frame"]) + step, step):
                path = images["normal"].get(frame_key(source["action"], source["animation"], frame))
                if path:
                    paths.append(path)
            return paths

    folder = edge.get("transition_folder")
    paths = list(images["transitions"].get(str(folder), [])) if folder else []
    target_path = images["normal"].get(
        frame_key(
            edge["target"]["action"],
            edge["target"]["animation"],
            int(edge["target"]["frame"]),
        )
    )
    if target_path:
        paths.append(target_path)
    return paths


def build_routes(
    routes: Optional[Dict[str, Any]],
    payload: Dict[str, Any],
    path_ids: Dict[str, str],
) -> Optional[Dict[str, Any]]:
    if routes is None:
        return None

    def resolve_route(route: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if route is None:
            return None

        edge_ids: List[str] = []
        reachable = bool(route.get("reachable", False))
        if reachable:
            for edge in route.get("edges", []):
                edge_id = path_ids.get(edge_signature(edge))
                if edge_id is None:
                    reachable = False
                    edge_ids = []
                    break
                edge_ids.append(edge_id)

        target = route.get("target") or {}
        target_frame = route.get("target_frame", target.get("frame", 0))
        return {
            "reachable": reachable,
            "target_id": route.get("target_id"),
            "target_action": route.get("target_action", target.get("action")),
            "target_animation": route.get("target_animation", target.get("animation")),
            "target_frame": int(target_frame),
            "total_cost": float(route.get("total_cost", 0.0)),
            "total_frames": int(route.get("total_frames", 0)),
            "num_transitions": int(route.get("num_transitions", 0)),
            "edge_ids": edge_ids,
        }

    source_routes: Dict[str, Dict[str, Any]] = {}
    for source_id, actions in routes["source_routes"].items():
        source_routes[source_id] = {}
        for action, route in actions.items():
            source_routes[source_id][action] = resolve_route(route)

    sequence_start_routes = {
        source_id: resolved
        for source_id, route in routes.get("sequence_start_routes", {}).items()
        if (resolved := resolve_route(route)) is not None
    }

    return {
        "actions": list(routes["actions"]),
        "source_routes": source_routes,
        "sequence_start_routes": sequence_start_routes,
    }


def build_data(
    payload: Dict[str, Any],
    images: Optional[Dict[str, Any]],
    routes: Optional[Dict[str, Any]],
    path_ids: Dict[str, str],
    fps: float,
) -> Dict[str, Any]:
    edges: Dict[str, Any] = {}
    outgoing: Dict[str, List[str]] = {}
    nodes: Dict[str, Any] = {}
    libraries = [] if images is None else images.get("libraries", [])

    for node in payload["nodes"]:
        nodes[node_id(node)] = {
            "action": node["action"],
            "animation": node["animation"],
            "frame": int(node["frame"]),
            "image_paths": [
                library["normal"].get(frame_key(node["action"], node["animation"], int(node["frame"])))
                for library in libraries
            ],
        }

    edge_ids: List[str] = []
    for index, edge in enumerate(payload["edges"]):
        edge_id = f"edge_{index}"
        edge_ids.append(edge_id)
        edges[edge_id] = {
            "source": node_id(edge["source"]),
            "target": node_id(edge["target"]),
            "kind": edge["kind"],
            "length": int(edge["length"]),
            "distance": float(edge.get("distance", 0.0)),
            "image_path_sets": [edge_images(edge, library) for library in libraries],
            "label": f'{node_id(edge["source"])} -> {node_id(edge["target"])}',
        }
        outgoing.setdefault(node_id(edge["source"]), []).append(edge_id)

    return {
        "fps": float(fps),
        "edge_ids": edge_ids,
        "edges": edges,
        "nodes": nodes,
        "outgoing": outgoing,
        "routes": build_routes(routes, payload, path_ids),
    }


def render_page(
    payload: Dict[str, Any],
    images: Optional[Dict[str, Any]],
    routes: Optional[Dict[str, Any]],
    mode: str,
    fps: float,
) -> str:
    svg, path_ids = render_svg(payload)
    data = build_data(payload, images, routes, path_ids, fps)

    preview_block = ""
    content_columns = "minmax(0, 1fr)"
    status = "Random walk is running."
    if mode == "image":
        libraries = [] if images is None else images.get("libraries", [])
        columns = max(1, len(libraries))
        cards: List[str] = []
        for index, library in enumerate(libraries):
            label = html.escape(str(library.get("label", f"View {index + 1}")))
            cards.append(
                f"""
      <div class="card preview" data-preview-index="{index}">
        <h2>{label}</h2>
        <div class="preview-stage">
          <img class="preview-frame" alt="{label} preview" hidden>
        </div>
        <div class="preview-caption"></div>
      </div>
""".rstrip()
            )
        preview_block = (
            f'<div class="preview-strip" style="grid-template-columns: repeat({columns}, minmax(0, 1fr));">'
            + "".join(cards)
            + "</div>"
        )

    summary = html.escape(
        f"actions={len(group_actions(payload['nodes']))}, nodes={len(payload['nodes'])}, "
        f"edges={len(payload['edges'])}, transitions={len(payload.get('transitions', []))}"
    )
    if routes is None:
        status = "Random walk is running. Build with --shortest-path to enable action routing and return-to-start routing."

    return (
        HTML_TEMPLATE
        .replace("__CONTENT_COLUMNS__", content_columns)
        .replace("__SUMMARY__", summary)
        .replace("__STATUS__", status)
        .replace("__PREVIEW_BLOCK__", preview_block)
        .replace("__SVG__", svg)
        .replace("__FPS__", str(int(round(fps))))
        .replace("__DATA__", json.dumps(data, ensure_ascii=True))
    )


def port_error(host: str, port: int) -> Optional[OSError]:
    family = socket.AF_INET6 if ":" in host and host != "0.0.0.0" else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        sock.bind((host, port))
    except OSError as exc:
        return exc
    finally:
        sock.close()
    return None


def suggest_ports(host: str, port: int) -> List[int]:
    ports: List[int] = []
    for candidate in range(port + 1, port + 10):
        if port_error(host, candidate) is None:
            ports.append(candidate)
        if len(ports) >= 3:
            break
    return ports


def create_app(
    *,
    graph_path: Path,
    mode: str = "graph",
    image_manifests: Optional[List[Path]] = None,
    fps: float = 24.0,
) -> Any:
    from flask import Flask, abort, make_response, send_file

    payload = load_graph(graph_path)
    images = load_images(graph_path, image_manifests, required=mode == "image")
    routes = load_routes(graph_path)
    page = render_page(payload, images, routes, mode=mode, fps=fps)
    asset_files = {} if images is None else images["files"]

    app = Flask(__name__)

    def disable_cache(response: Any) -> Any:
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.route("/")
    def index() -> Any:
        return disable_cache(make_response(page))

    @app.route("/assets/<asset_id>")
    def asset(asset_id: str) -> Any:
        path = asset_files.get(asset_id)
        if path is None or not path.exists():
            abort(404)
        return disable_cache(send_file(path, conditional=False, max_age=0))

    return app


def serve(
    *,
    graph_path: Path,
    host: str = "127.0.0.1",
    port: int = 8765,
    mode: str = "graph",
    image_manifests: Optional[List[Path]] = None,
    fps: float = 24.0,
) -> None:
    error = port_error(host, port)
    if error is not None:
        options = suggest_ports(host, port)
        suffix = ""
        if options:
            suffix = " Try " + ", ".join(f"--port {item}" for item in options) + "."
        raise SystemExit(f"Cannot start viewer on {host}:{port}: {error}.{suffix}")

    app = create_app(
        graph_path=graph_path.resolve(),
        mode=mode,
        image_manifests=None if image_manifests is None else [path.resolve() for path in image_manifests],
        fps=fps,
    )
    visible_host = "127.0.0.1" if host == "0.0.0.0" else host
    print(f"Serving motion graph visualization at http://{visible_host}:{port}")
    app.run(host=host, port=port, debug=False, use_reloader=False)
