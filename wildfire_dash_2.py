"""Wildfire Dash dashboard with a cleaner three-column layout.

This version keeps the wildfire model implementation from the previous Dash app,
but updates the page styling to match the cleaner dashboard layout:
- left sidebar with landscape and parameter controls
- top action buttons
- center area with title, map, and burned-proportion graph
- right panel with current settings and brief help text
- click on the map to move the ignition point

Run with:
    python wildfire_dash_2.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from PIL import Image
from skimage.transform import resize

from dash import Dash, Input, Output, State, callback_context, dcc, html
from dash.exceptions import PreventUpdate
import plotly.graph_objects as go


# ============================================================
# Configuration
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LANDSCAPES = {
    "California": os.path.join(BASE_DIR, "california.png"),
    "Australia": os.path.join(BASE_DIR, "australia.png"),
    "Spain": os.path.join(BASE_DIR, "spain.png"),
}
GRID_SHAPE = (150, 150)


# ============================================================
# Image processing
# ============================================================

def build_beta_map(image_path: str, target_shape: Tuple[int, int] = GRID_SHAPE):
    """Return a vegetation-based beta map and the resized RGB background."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}\n"
            "Put the landscape image next to this script or edit LANDSCAPES."
        )

    img = Image.open(image_path).convert("RGB")
    img_array = np.array(img).astype(float)

    R = img_array[:, :, 0]
    G = img_array[:, :, 1]
    B = img_array[:, :, 2]

    veg = G - (R + B) / 2
    vmin = np.percentile(veg, 5)
    vmax = np.percentile(veg, 95)
    veg = np.clip(veg, vmin, vmax)

    veg = 2 * G - R - B
    vmin = np.percentile(veg, 5)
    vmax = np.percentile(veg, 95)
    veg = np.clip(veg, vmin, vmax)

    beta = (veg - veg.min()) / (veg.max() - veg.min() + 1e-12)
    beta = np.interp(beta, [0.0, 0.35, 0.7, 1.0], [0.0, 0.08, 0.45, 1.0])
    beta = np.clip(beta, 0, 1)

    beta_resized = resize(beta, target_shape, anti_aliasing=True)
    background = np.array(img.resize((target_shape[1], target_shape[0])))
    return beta_resized.astype(float), background.astype(np.uint8)


# ============================================================
# Fire spread model
# ============================================================

def shift_no_wrap(a: np.ndarray, di: int, dj: int) -> np.ndarray:
    b = np.roll(a, shift=(di, dj), axis=(0, 1))

    if di > 0:
        b[:di, :] = 0
    elif di < 0:
        b[di:, :] = 0

    if dj > 0:
        b[:, :dj] = 0
    elif dj < 0:
        b[:, dj:] = 0

    return b


def burning_neighbors(grid: np.ndarray) -> np.ndarray:
    burning = (grid == 1).astype(int)
    return (
        shift_no_wrap(burning, -1, 0)
        + shift_no_wrap(burning, 1, 0)
        + shift_no_wrap(burning, 0, -1)
        + shift_no_wrap(burning, 0, 1)
        + shift_no_wrap(burning, -1, -1)
        + shift_no_wrap(burning, -1, 1)
        + shift_no_wrap(burning, 1, -1)
        + shift_no_wrap(burning, 1, 1)
    )


def directional_burning_neighbors(
    grid: np.ndarray, wind_strength: float = 1.0, wind_dir: str = "right"
) -> np.ndarray:
    """Directional neighborhood weight.

    wind_dir means the side the wind COMES FROM:
      right -> wind blows left
      left  -> wind blows right
      up    -> wind blows down
      down  -> wind blows up
    """
    burning = (grid == 1).astype(float)

    north = shift_no_wrap(burning, -1, 0)
    south = shift_no_wrap(burning, 1, 0)
    west = shift_no_wrap(burning, 0, -1)
    east = shift_no_wrap(burning, 0, 1)
    nw = shift_no_wrap(burning, -1, -1)
    ne = shift_no_wrap(burning, -1, 1)
    sw = shift_no_wrap(burning, 1, -1)
    se = shift_no_wrap(burning, 1, 1)

    base = north + south + west + east + 0.5 * (nw + ne + sw + se)

    if wind_dir == "right":
        downwind = west + 0.5 * (nw + sw)
        upwind = east + 0.5 * (ne + se)
    elif wind_dir == "left":
        downwind = east + 0.5 * (ne + se)
        upwind = west + 0.5 * (nw + sw)
    elif wind_dir == "up":
        downwind = south + 0.5 * (sw + se)
        upwind = north + 0.5 * (nw + ne)
    elif wind_dir == "down":
        downwind = north + 0.5 * (nw + ne)
        upwind = south + 0.5 * (sw + se)
    else:
        raise ValueError(f"Invalid wind_dir: {wind_dir}")

    N_eff = base + wind_strength * (1.5 * downwind - 1.0 * upwind)
    return np.clip(N_eff, 0, None)


def step(
    grid: np.ndarray,
    burn_age: np.ndarray,
    beta: np.ndarray,
    gamma_map: np.ndarray,
    min_burn_time: int = 20,
    k: float = 0.20,
    p_cap: float = 0.80,
    rng: np.random.Generator | None = None,
    use_wind: bool = True,
    wind_strength: float = 1.0,
    wind_dir: str = "right",
) -> Tuple[np.ndarray, np.ndarray]:
    if rng is None:
        rng = np.random.default_rng()

    new_grid = grid.copy()
    new_burn_age = burn_age.copy()

    if use_wind and wind_strength > 0:
        N = directional_burning_neighbors(
            grid, wind_strength=wind_strength, wind_dir=wind_dir
        )
    else:
        N = burning_neighbors(grid)

    susceptible = grid == 0
    beta_eff = beta ** 3
    p_ignite = 1.0 - np.exp(-k * beta_eff * N)
    p_ignite = np.minimum(p_ignite, p_cap)
    ignite = susceptible & (rng.random(grid.shape) < p_ignite)
    new_grid[ignite] = 1
    new_burn_age[ignite] = 0

    burning = grid == 1
    new_burn_age[burning] += 1

    eligible = burning & (burn_age >= min_burn_time)
    burn_out = eligible & (rng.random(grid.shape) < gamma_map)
    new_grid[burn_out] = 2
    new_burn_age[burn_out] = 0

    return new_grid, new_burn_age


# ============================================================
# State
# ============================================================

@dataclass
class SimState:
    landscape_name: str
    beta: np.ndarray
    background: np.ndarray
    gamma_map: np.ndarray
    nrows: int
    ncols: int
    grid: np.ndarray
    burn_age: np.ndarray
    seed_row: int
    seed_col: int
    running: bool
    finished: bool
    t: int
    time_history: list
    burned_history: list
    rng: np.random.Generator
    use_wind: bool = True
    wind_strength: float = 0.70
    wind_dir: str = "right"
    K: float = 0.35
    MIN_BURN_TIME: int = 20
    P_CAP: float = 0.80
    RNG_SEED: int = 42


def load_state(landscape_name: str = "California") -> SimState:
    beta, background = build_beta_map(LANDSCAPES[landscape_name])
    nrows, ncols = beta.shape
    gamma_map = np.clip(0.10 * (1 - beta), 0.01, 0.10)
    grid = np.zeros((nrows, ncols), dtype=int)
    burn_age = np.zeros((nrows, ncols), dtype=int)
    seed_row = nrows // 2
    seed_col = ncols // 2
    grid[seed_row, seed_col] = 1
    return SimState(
        landscape_name=landscape_name,
        beta=beta,
        background=background,
        gamma_map=gamma_map,
        nrows=nrows,
        ncols=ncols,
        grid=grid,
        burn_age=burn_age,
        seed_row=seed_row,
        seed_col=seed_col,
        running=False,
        finished=False,
        t=0,
        time_history=[0],
        burned_history=[0.0],
        rng=np.random.default_rng(42),
    )


SIM = load_state()


# ============================================================
# Helpers
# ============================================================

def reset_simulation(keep_controls: bool = True) -> None:
    global SIM

    beta = SIM.beta
    background = SIM.background
    gamma_map = SIM.gamma_map
    nrows, ncols = SIM.nrows, SIM.ncols
    landscape_name = SIM.landscape_name

    seed_row = int(np.clip(SIM.seed_row, 0, nrows - 1))
    seed_col = int(np.clip(SIM.seed_col, 0, ncols - 1))

    grid = np.zeros((nrows, ncols), dtype=int)
    burn_age = np.zeros((nrows, ncols), dtype=int)
    grid[seed_row, seed_col] = 1

    SIM = SimState(
        landscape_name=landscape_name,
        beta=beta,
        background=background,
        gamma_map=gamma_map,
        nrows=nrows,
        ncols=ncols,
        grid=grid,
        burn_age=burn_age,
        seed_row=seed_row,
        seed_col=seed_col,
        running=True,
        finished=False,
        t=0,
        time_history=[0],
        burned_history=[0.0],
        rng=np.random.default_rng(SIM.RNG_SEED),
        use_wind=SIM.use_wind if keep_controls else True,
        wind_strength=SIM.wind_strength if keep_controls else 0.70,
        wind_dir=SIM.wind_dir if keep_controls else "right",
        K=SIM.K,
        MIN_BURN_TIME=SIM.MIN_BURN_TIME,
        P_CAP=SIM.P_CAP,
        RNG_SEED=SIM.RNG_SEED,
    )


def load_landscape(landscape_name: str) -> None:
    global SIM
    beta, background = build_beta_map(LANDSCAPES[landscape_name])
    nrows, ncols = beta.shape
    gamma_map = np.clip(0.10 * (1 - beta), 0.01, 0.10)

    seed_row = nrows // 2
    seed_col = ncols // 2
    grid = np.zeros((nrows, ncols), dtype=int)
    burn_age = np.zeros((nrows, ncols), dtype=int)
    grid[seed_row, seed_col] = 1

    SIM = SimState(
        landscape_name=landscape_name,
        beta=beta,
        background=background,
        gamma_map=gamma_map,
        nrows=nrows,
        ncols=ncols,
        grid=grid,
        burn_age=burn_age,
        seed_row=seed_row,
        seed_col=seed_col,
        running=False,
        finished=False,
        t=0,
        time_history=[0],
        burned_history=[0.0],
        rng=np.random.default_rng(42),
        use_wind=True,
        wind_strength=0.70,
        wind_dir="right",
        K=0.35,
    )


def compose_rgba() -> np.ndarray:
    base = SIM.background.astype(np.uint8)
    h, w, _ = base.shape
    comp = np.zeros((h, w, 4), dtype=np.uint8)
    comp[..., :3] = base
    comp[..., 3] = 255

    burning = SIM.grid == 1
    burned = SIM.grid == 2

    comp[burning, :3] = np.array([255, 120, 0], dtype=np.uint8)
    comp[burned, :3] = np.array([20, 20, 20], dtype=np.uint8)
    return comp


def make_map_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Image(z=compose_rgba()))
    fig.add_trace(
        go.Scatter(
            x=[SIM.seed_col],
            y=[SIM.seed_row],
            mode="markers",
            hoverinfo="skip",
            marker=dict(
                symbol="x",
                size=14,
                color="white",
                line=dict(width=2, color="black"),
            ),
            name="Ignition point",
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=35, b=0),
        title=f"Map: {SIM.landscape_name}",
        dragmode=False,
        clickmode="event+select",
        showlegend=False,
        uirevision="keep",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#2d3142"),
    )
    fig.update_xaxes(visible=False, range=[-0.5, SIM.ncols - 0.5], constrain="domain")
    fig.update_yaxes(
        visible=False,
        range=[SIM.nrows - 0.5, -0.5],
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def make_burn_figure() -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=SIM.time_history,
            y=SIM.burned_history,
            mode="lines",
            line=dict(width=2),
            name="Burned fraction",
        )
    )
    fig.update_layout(
        margin=dict(l=45, r=20, t=35, b=45),
        title="Burned proportion",
        xaxis_title="Time step",
        yaxis_title="Burned fraction",
        yaxis=dict(range=[0, 1]),
        uirevision="keep",
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="#2d3142"),
    )
    return fig


def status_text() -> str:
    wind = "ON" if SIM.use_wind else "OFF"
    direction = {
        "right": "Right",
        "left": "Left",
        "up": "Up",
        "down": "Down",
    }[SIM.wind_dir]
    run_state = "Running" if SIM.running else "Paused"
    if SIM.finished:
        run_state = "Finished"
    return run_state, wind, direction


def set_seed(row: int, col: int) -> None:
    SIM.seed_row = int(np.clip(row, 0, SIM.nrows - 1))
    SIM.seed_col = int(np.clip(col, 0, SIM.ncols - 1))


def step_once() -> None:
    SIM.grid, SIM.burn_age = step(
        SIM.grid,
        SIM.burn_age,
        SIM.beta,
        SIM.gamma_map,
        min_burn_time=SIM.MIN_BURN_TIME,
        k=SIM.K,
        p_cap=SIM.P_CAP,
        rng=SIM.rng,
        use_wind=SIM.use_wind,
        wind_strength=SIM.wind_strength,
        wind_dir=SIM.wind_dir,
    )
    SIM.t += 1
    SIM.time_history.append(SIM.t)
    SIM.burned_history.append(float(np.mean(SIM.grid == 2)))
    if np.sum(SIM.grid == 1) == 0:
        SIM.finished = True
        SIM.running = False


def stat_card(label: str, value: str) -> html.Div:
    return html.Div(
        style={"marginBottom": "22px"},
        children=[
            html.Div(label, style={"fontSize": "13px", "color": "#6c7280", "marginBottom": "3px"}),
            html.Div(value, style={"fontSize": "34px", "fontWeight": "500", "lineHeight": "1.1", "color": "#2d3142"}),
        ],
    )


def control_block(title: str, children, helper: str | None = None) -> html.Div:
    items = [html.Div(title, style={"fontSize": "17px", "fontWeight": "600", "marginBottom": "10px", "color": "#2d3142"})]
    if helper:
        items.append(html.Div(helper, style={"fontSize": "13px", "lineHeight": "1.6", "color": "#7a8190", "marginBottom": "14px"}))
    items.append(children)
    return html.Div(style={"marginBottom": "28px"}, children=items)


# ============================================================
# App layout
# ============================================================

SIDEBAR = {
    "width": "210px",
    "minWidth": "210px",
    "maxWidth": "210px",
    "background": "#f3f5f9",
    "borderRight": "1px solid #e3e7ef",
    "padding": "22px 16px",
    "boxSizing": "border-box",
    "minHeight": "100vh",
}

PAGE = {
    "display": "flex",
    "minHeight": "100vh",
    "background": "#ffffff",
    "color": "#2d3142",
    "fontFamily": 'Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif',
}

MAIN = {
    "flex": "1",
    "padding": "16px 18px 24px 18px",
    "boxSizing": "border-box",
}

TOPBAR = {
    "display": "grid",
    "gridTemplateColumns": "repeat(3, 1fr)",
    "gap": "10px",
    "marginBottom": "14px",
}

TOPBTN = {
    "height": "28px",
    "borderRadius": "6px",
    "border": "1px solid #d9dee8",
    "background": "#ffffff",
    "color": "#a0a6b4",
    "fontSize": "12px",
    "cursor": "pointer",
    "boxShadow": "0 1px 0 rgba(0,0,0,0.02)",
}

CARD = {
    "background": "#ffffff",
    "borderRadius": "12px",
}

app = Dash(__name__)
app.layout = html.Div(
    style=PAGE,
    children=[
        html.Div(
            style=SIDEBAR,
            children=[
                html.Div("Wildfire controls", style={"fontSize": "20px", "fontWeight": "600", "marginBottom": "6px", "color": "#2d3142"}),
                html.Div("Same model, cleaner interface.", style={"fontSize": "12px", "color": "#8a90a0", "marginBottom": "16px"}),

                control_block(
                    "Landscape",
                    dcc.Dropdown(
                        id="landscape",
                        options=[{"label": k, "value": k} for k in LANDSCAPES.keys()],
                        value=SIM.landscape_name,
                        clearable=False,
                        style={"fontSize": "13px"},
                    ),
                ),

                control_block(
                    "Ignition point",
                    html.Div(
                        [
                            html.Div(
                                "Click anywhere on the map to choose where the fire starts. The sliders below stay available as well.",
                                style={"fontSize": "13px", "lineHeight": "1.6", "color": "#7a8190", "marginBottom": "14px"},
                            ),
                            html.Div("Seed row", style={"fontSize": "12px", "color": "#9aa1af", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="seed-row",
                                min=0,
                                max=SIM.nrows - 1,
                                step=1,
                                value=SIM.seed_row,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div("Seed col", style={"fontSize": "12px", "color": "#9aa1af", "marginTop": "14px", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="seed-col",
                                min=0,
                                max=SIM.ncols - 1,
                                step=1,
                                value=SIM.seed_col,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]
                    ),
                ),

                control_block(
                    "Spread intensity",
                    html.Div(
                        [
                            html.Div(
                                "K controls how easily a susceptible cell ignites. Larger K means faster spread.",
                                style={"fontSize": "13px", "lineHeight": "1.6", "color": "#7a8190", "marginBottom": "14px"},
                            ),
                            html.Div("K", style={"fontSize": "12px", "color": "#9aa1af", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="k-slider",
                                min=0.05,
                                max=0.80,
                                step=0.01,
                                value=SIM.K,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ]
                    ),
                ),

                control_block(
                    "Wind",
                    html.Div(
                        [
                            html.Div(
                                "Turn wind on or off, choose its strength, and select the direction the fire is pushed toward.",
                                style={"fontSize": "13px", "lineHeight": "1.6", "color": "#7a8190", "marginBottom": "14px"},
                            ),
                            dcc.Checklist(
                                id="wind-toggle",
                                options=[{"label": " Wind on", "value": "on"}],
                                value=["on"] if SIM.use_wind else [],
                                style={"fontSize": "14px", "color": "#6c7280", "marginBottom": "14px"},
                            ),
                            html.Div("Wind strength", style={"fontSize": "12px", "color": "#9aa1af", "marginBottom": "4px"}),
                            dcc.Slider(
                                id="wind-strength",
                                min=0.0,
                                max=1.0,
                                step=0.01,
                                value=SIM.wind_strength,
                                marks=None,
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                            html.Div("Wind direction", style={"fontSize": "12px", "color": "#9aa1af", "marginTop": "14px", "marginBottom": "8px"}),
                            dcc.RadioItems(
                                id="wind-direction",
                                options=[
                                    {"label": " Right", "value": "right"},
                                    {"label": " Left", "value": "left"},
                                    {"label": " Up", "value": "up"},
                                    {"label": " Down", "value": "down"},
                                ],
                                value=SIM.wind_dir,
                                labelStyle={"display": "block", "marginBottom": "6px", "fontSize": "14px", "color": "#6c7280"},
                            ),
                        ]
                    ),
                ),

                html.Hr(style={"border": "none", "borderTop": "1px solid #dde2ea", "margin": "18px 0"}),
                html.Div("How to read the parameters", style={"fontSize": "12px", "fontWeight": "600", "color": "#7a8190", "marginBottom": "8px"}),
                html.Ul(
                    style={"paddingLeft": "16px", "margin": 0, "fontSize": "12px", "lineHeight": "1.8", "color": "#6c7280"},
                    children=[
                        html.Li("Seed row / Seed col: initial ignition location in the grid."),
                        html.Li("K: global spread intensity."),
                        html.Li("Wind strength: how strongly the wind biases propagation."),
                        html.Li("Wind direction: direction in which the fire is pushed."),
                    ],
                ),
            ],
        ),

        html.Div(
            style=MAIN,
            children=[
                html.Div(style=TOPBAR, children=[
                    html.Button("Run simulation", id="btn-start", n_clicks=0, style=TOPBTN),
                    html.Button("Pause", id="btn-pause", n_clicks=0, style=TOPBTN),
                    html.Button("Restart", id="btn-restart", n_clicks=0, style=TOPBTN),
                ]),

                html.Div(
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "minmax(0, 1fr) 360px",
                        "gap": "36px",
                        "alignItems": "start",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    "Wildfire Simulation Dashboard",
                                    style={"fontSize": "40px", "fontWeight": "700", "letterSpacing": "-0.02em", "marginBottom": "16px", "color": "#2d3142"},
                                ),
                                html.Div(
                                    "Explore the effect of landscape, ignition point, vegetation, and wind on fire spread.",
                                    style={"fontSize": "14px", "color": "#8a90a0", "marginBottom": "34px"},
                                ),

                                html.Div(
                                    f"Map: {SIM.landscape_name}",
                                    style={"fontSize": "24px", "fontWeight": "600", "marginBottom": "14px", "color": "#2d3142"},
                                ),
                                html.Div(
                                    "Click directly on the map to choose the ignition cell.",
                                    style={"fontSize": "13px", "color": "#8a90a0", "marginBottom": "16px"},
                                ),
                                dcc.Graph(
                                    id="map-graph",
                                    figure=make_map_figure(),
                                    config={"displayModeBar": False},
                                    style={"height": "540px"},
                                ),

                                html.Div(
                                    style={"marginTop": "18px"},
                                    children=[
                                        html.Div("Burned proportion", style={"fontSize": "22px", "fontWeight": "600", "marginBottom": "10px", "color": "#2d3142"}),
                                        dcc.Graph(
                                            id="burn-graph",
                                            figure=make_burn_figure(),
                                            config={"displayModeBar": False},
                                            style={"height": "240px"},
                                        ),
                                    ],
                                ),

                                html.Div(
                                    style={
                                        "display": "grid",
                                        "gridTemplateColumns": "repeat(4, 1fr)",
                                        "gap": "18px",
                                        "marginTop": "8px",
                                        "paddingRight": "12px",
                                    },
                                    children=[
                                        html.Div([html.Div("Time step", style={"fontSize": "12px", "color": "#7a8190"}), html.Div(str(SIM.t), style={"fontSize": "28px", "fontWeight": "500", "marginTop": "6px"})]),
                                        html.Div([html.Div("Burned fraction", style={"fontSize": "12px", "color": "#7a8190"}), html.Div(f"{SIM.burned_history[-1]:.3f}", id="burn-fraction-readout", style={"fontSize": "28px", "fontWeight": "500", "marginTop": "6px"})]),
                                        html.Div([html.Div("Burning cells", style={"fontSize": "12px", "color": "#7a8190"}), html.Div(str(int(np.sum(SIM.grid == 1))), id="burning-count", style={"fontSize": "28px", "fontWeight": "500", "marginTop": "6px"})]),
                                        html.Div([html.Div("Burned cells", style={"fontSize": "12px", "color": "#7a8190"}), html.Div(str(int(np.sum(SIM.grid == 2))), id="burned-count", style={"fontSize": "28px", "fontWeight": "500", "marginTop": "6px"})]),
                                    ],
                                ),
                            ],
                        ),

                        html.Div(
                            children=[
                                html.Div("Current settings", style={"fontSize": "26px", "fontWeight": "600", "marginBottom": "18px", "color": "#2d3142"}),
                                html.Div(id="settings-panel", children=[]),
                                html.Hr(style={"border": "none", "borderTop": "1px solid #dde2ea", "margin": "18px 0 20px 0"}),
                                html.Div("What the controls do", style={"fontSize": "16px", "fontWeight": "600", "marginBottom": "14px", "color": "#2d3142"}),
                                html.Ul(
                                    style={"paddingLeft": "18px", "margin": 0, "fontSize": "13px", "lineHeight": "1.8", "color": "#3b4252"},
                                    children=[
                                        html.Li([html.B("Run simulation"), " starts the animation."]),
                                        html.Li([html.B("Pause"), " stops it where it is."]),
                                        html.Li([html.B("Restart"), " resets the fire with the current settings."]),
                                        html.Li([html.B("K"), " changes the global ignition intensity."]),
                                        html.Li([html.B("Wind strength"), " changes how directional the spread is."]),
                                        html.Li([html.B("Wind direction"), " changes where the fire is pushed."]),
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        dcc.Interval(id="interval", interval=250, n_intervals=0, disabled=True),
    ],
)


# ============================================================
# Callbacks
# ============================================================

@app.callback(
    Output("map-graph", "figure"),
    Output("burn-graph", "figure"),
    Output("seed-row", "value"),
    Output("seed-col", "value"),
    Output("wind-toggle", "value"),
    Output("wind-direction", "value"),
    Output("k-slider", "value"),
    Output("wind-strength", "value"),
    Output("interval", "disabled"),
    Output("settings-panel", "children"),
    Output("burn-fraction-readout", "children"),
    Output("burning-count", "children"),
    Output("burned-count", "children"),
    Input("interval", "n_intervals"),
    Input("btn-start", "n_clicks"),
    Input("btn-pause", "n_clicks"),
    Input("btn-restart", "n_clicks"),
    Input("wind-toggle", "value"),
    Input("wind-direction", "value"),
    Input("seed-row", "value"),
    Input("seed-col", "value"),
    Input("k-slider", "value"),
    Input("wind-strength", "value"),
    Input("map-graph", "clickData"),
    Input("landscape", "value"),
    State("seed-row", "value"),
    State("seed-col", "value"),
    prevent_initial_call=False,
)
def update_app(
    _interval,
    _start,
    _pause,
    _restart,
    wind_toggle_value,
    wind_direction_value,
    seed_row_value,
    seed_col_value,
    k_value,
    wind_strength_value,
    click_data,
    landscape_value,
    seed_row_state,
    seed_col_state,
):
    global SIM

    triggered = callback_context.triggered_id

    # Landscape change reloads the full map and resets the simulation.
    if triggered == "landscape" and landscape_value and landscape_value != SIM.landscape_name:
        load_landscape(landscape_value)

    # Control values update the persistent state.
    if k_value is not None:
        SIM.K = float(k_value)
    if wind_strength_value is not None:
        SIM.wind_strength = float(wind_strength_value)
    if wind_direction_value is not None:
        SIM.wind_dir = wind_direction_value
    if wind_toggle_value is not None:
        SIM.use_wind = "on" in wind_toggle_value

    if triggered == "btn-start":
        SIM.running = True
        SIM.finished = False
    elif triggered == "btn-pause":
        SIM.running = False
    elif triggered == "btn-restart":
        reset_simulation()
    elif triggered == "wind-toggle":
        reset_simulation()
    elif triggered == "wind-direction":
        reset_simulation()
    elif triggered == "seed-row" or triggered == "seed-col":
        if seed_row_value is not None and seed_col_value is not None:
            if int(seed_row_value) != SIM.seed_row or int(seed_col_value) != SIM.seed_col:
                set_seed(int(seed_row_value), int(seed_col_value))
                reset_simulation()
    elif triggered == "map-graph" and click_data:
        point = click_data.get("points", [{}])[0]
        if "x" in point and "y" in point:
            col = int(round(point["x"]))
            row = int(round(point["y"]))
            set_seed(row, col)
            reset_simulation()
    elif triggered == "interval":
        if SIM.running and not SIM.finished:
            step_once()

    # Keep bounds valid after any update.
    SIM.seed_row = int(np.clip(SIM.seed_row, 0, SIM.nrows - 1))
    SIM.seed_col = int(np.clip(SIM.seed_col, 0, SIM.ncols - 1))

    run_state, wind_state, direction_state = status_text()
    settings_children = [
        stat_card("K", f"{SIM.K:.2f}"),
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "24px"},
            children=[
                stat_card("Wind", wind_state),
                stat_card("Direction", direction_state),
            ],
        ),
        stat_card("Seed location", f"({SIM.seed_row}, {SIM.seed_col})"),
        stat_card("Minimum burn time", f"{SIM.MIN_BURN_TIME} steps"),
    ]

    return (
        make_map_figure(),
        make_burn_figure(),
        SIM.seed_row,
        SIM.seed_col,
        ["on"] if SIM.use_wind else [],
        SIM.wind_dir,
        SIM.K,
        SIM.wind_strength,
        not SIM.running,
        settings_children,
        f"{SIM.burned_history[-1]:.3f}",
        int(np.sum(SIM.grid == 1)),
        int(np.sum(SIM.grid == 2)),
    )

server = app.server

if __name__ == "__main__":
    app.run(debug=True)