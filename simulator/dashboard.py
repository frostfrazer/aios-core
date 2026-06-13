"""
AIOS-Core :: Live Dashboard
-----------------------------
Rich-based terminal UI. Shows the orchestrator's decisions in real time:
  - Per-process feature heatmap
  - AI tuning decisions (nice_delta, preempt_ms, cpu/io weights)
  - Reward signal trending
  - System-wide CPU/MEM overview
  - Tick rate and model update counter
"""

import time
import threading
import psutil
import numpy as np
from collections import deque
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.columns import Columns
from rich.progress import BarColumn, Progress, TextColumn
from rich import box
from rich.text import Text
from rich.layout import Layout
from rich.align import Align


console = Console()

_NICE_COLORS = {
    range(-20, -10): "bold red",
    range(-10,   0): "red",
    range(  0,   1): "white",
    range(  1,  10): "green",
    range( 10,  20): "bold green",
}

def _nice_color(delta: int) -> str:
    for r, color in _NICE_COLORS.items():
        if delta in r:
            return color
    return "white"

def _bar(value: float, width: int = 10) -> str:
    """ASCII fill bar for 0..1 values."""
    filled = int(round(value * width))
    return "█" * filled + "░" * (width - filled)

def _reward_color(r: float) -> str:
    if r is None:   return "dim white"
    if r >  0.1:    return "bold green"
    if r >  0:      return "green"
    if r > -0.1:    return "yellow"
    return "red"


class Dashboard:
    """
    Runs in its own thread, refreshing the terminal every `refresh_ms`.
    Reads decision data from the orchestrator via callback injection.
    """

    HISTORY_DEPTH = 60     # ticks to keep for sparkline
    MAX_ROWS      = 20     # max process rows shown

    def __init__(self, refresh_ms: int = 500):
        self.refresh_ms = refresh_ms
        self._lock      = threading.Lock()
        self._latest_decisions: dict = {}
        self._tick_num   = 0
        self._reward_hist: deque = deque(maxlen=self.HISTORY_DEPTH)
        self._cpu_hist:    deque = deque(maxlen=self.HISTORY_DEPTH)
        self._model_updates = 0
        self._start_time    = time.time()

    # ── Called by orchestrator each tick ──────────────────────────────────────

    def on_tick(self, tick_num: int, decisions: dict):
        with self._lock:
            self._tick_num = tick_num
            self._latest_decisions = decisions

            rewards = [d.reward for d in decisions.values() if d.reward is not None]
            if rewards:
                self._reward_hist.append(float(np.mean(rewards)))
                self._model_updates += len(rewards)

            self._cpu_hist.append(psutil.cpu_percent(interval=None))

    # ── Rendering ─────────────────────────────────────────────────────────────

    def _make_header(self) -> Panel:
        uptime = time.time() - self._start_time
        cpu    = psutil.cpu_percent(interval=None)
        mem    = psutil.virtual_memory()
        avg_r  = (np.mean(self._reward_hist) if self._reward_hist else 0.0)

        cpu_bar = _bar(cpu / 100.0, width=8)
        parts = [
            f"[bold cyan]AIOS-Core[/bold cyan]  v0.1.0-dev",
            f"  [dim]|[/dim]  tick [yellow]#{self._tick_num}[/yellow]",
            f"  [dim]|[/dim]  uptime [white]{uptime:.0f}s[/white]",
            f"  [dim]|[/dim]  CPU [{'red' if cpu > 80 else 'green'}]{cpu:5.1f}% {cpu_bar}[/]",
            f"  [dim]|[/dim]  MEM [white]{mem.percent:.1f}%[/white]",
            f"  [dim]|[/dim]  avg reward [{_reward_color(avg_r)}]{avg_r:+.3f}[/]",
            f"  [dim]|[/dim]  model updates [magenta]{self._model_updates}[/magenta]",
        ]
        return Panel(
            " ".join(parts),
            style="on grey7",
            padding=(0, 1),
        )

    def _make_process_table(self) -> Table:
        with self._lock:
            decisions = dict(self._latest_decisions)

        t = Table(
            box=box.SIMPLE_HEAD,
            show_header=True,
            header_style="bold cyan",
            row_styles=["", "dim"],
            expand=True,
            padding=(0, 1),
        )
        t.add_column("PID",       min_width=7,  justify="right",  no_wrap=True)
        t.add_column("Name",      min_width=16, justify="left",   no_wrap=True, ratio=2)
        t.add_column("CPU%",      min_width=6,  justify="right",  no_wrap=True)
        t.add_column("cpu_w",     min_width=12, justify="left",   no_wrap=True)
        t.add_column("io_w",      min_width=12, justify="left",   no_wrap=True)
        t.add_column("nice \u0394", min_width=7, justify="center", no_wrap=True)
        t.add_column("pmpt ms",   min_width=8,  justify="right",  no_wrap=True)
        t.add_column("reward",    min_width=8,  justify="right",  no_wrap=True)
        t.add_column("status",    min_width=10, justify="left",   no_wrap=True)

        # Sort by abs(nice_delta) desc so most-affected procs rise
        rows = sorted(
            decisions.items(),
            key=lambda kv: abs(kv[1].params["nice_delta"]),
            reverse=True,
        )[: self.MAX_ROWS]

        for pid, dec in rows:
            p      = dec.params
            feat   = dec.features
            cpu_pct = feat[0] * 100
            status  = "?"
            name    = "?"
            try:
                proc   = psutil.Process(pid)
                name   = proc.name()[:17]
                status = proc.status()
            except Exception:
                pass

            nd = p["nice_delta"]
            t.add_row(
                str(pid),
                name,
                f"{cpu_pct:5.1f}",
                f"[cyan]{_bar(p['cpu_weight'])}[/cyan]",
                f"[yellow]{_bar(p['io_weight'])}[/yellow]",
                f"[{_nice_color(nd)}]{nd:+d}[/]",
                f"{p['preempt_ms']:.1f}",
                (f"[{_reward_color(dec.reward)}]{dec.reward:+.3f}[/]"
                 if dec.reward is not None else "[dim]…[/dim]"),
                f"[dim]{status}[/dim]",
            )

        return t

    def _make_sparklines(self) -> Panel:
        def spark(hist, label, color="green"):
            if not hist:
                return f"[dim]{label}: no data[/dim]"
            vals = list(hist)
            mn, mx = min(vals), max(vals)
            span   = mx - mn or 1
            bars   = " ".join(
                ["▁▂▃▄▅▆▇█"[int((v - mn) / span * 7)] for v in vals[-30:]]
            )
            return f"[dim]{label}[/dim] [{color}]{bars}[/] [{color}]{vals[-1]:.2f}[/]"

        lines = "\n".join([
            spark(self._cpu_hist,    "CPU%  ", "cyan"),
            spark(self._reward_hist, "reward", "magenta"),
        ])
        return Panel(lines, title="[bold]Trends[/bold]", border_style="dim white")

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(self._make_header(),        name="header", size=3),
            Layout(self._make_process_table(), name="table"),
            Layout(self._make_sparklines(),    name="sparks", size=5),
        )
        return layout

    # ── Run ───────────────────────────────────────────────────────────────────

    def run(self, stop_event: threading.Event):
        with Live(
            self.render(),
            console=console,
            refresh_per_second=1000 // self.refresh_ms,
            screen=True,
        ) as live:
            while not stop_event.is_set():
                live.update(self.render())
                time.sleep(self.refresh_ms / 1000.0)
