from __future__ import annotations
import json, html
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

@dataclass
class RunReport:
    title: str
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    finished_at: str | None = None
    config: Dict[str, Any] = field(default_factory=dict)
    partitions: Dict[str, Any] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    dataset: Dict[str, Any] = field(default_factory=dict)
    targets: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def add_step(self, name: str, seconds: float, extra: Dict[str, Any] | None = None):
        d = {"name": name, "seconds": round(seconds, 2)}
        if extra: d["extra"] = extra
        self.steps.append(d)

    def set_config(self, **cfg): self.config.update(cfg)
    def set_partitions(self, **d): self.partitions.update(d)
    def set_dataset(self, **d): self.dataset.update(d)
    def add_target_diag(self, horizon: int, cols: List[str], null_rates: Dict[str, float]):
        self.targets.append({
            "horizon": horizon,
            "columns": cols,
            "null_rates": {k: round(v*100, 2) for k, v in null_rates.items()},
        })
    def add_note(self, txt: str): self.notes.append(txt)

    def finish(self): self.finished_at = datetime.now().isoformat(timespec="seconds")

    # --- writers ---
    def _html_table(self, rows: List[Tuple[str, Any]]) -> str:
        out = ['<table class="kv"><tbody>']
        for k, v in rows:
            out.append(f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>")
        out.append("</tbody></table>")
        return "\n".join(out)

    def _html_steps(self) -> str:
        rows = "".join(
            f"<tr><td>{html.escape(s['name'])}</td><td class='num'>{s['seconds']:.2f}s</td>"
            f"<td class='muted'>{html.escape(json.dumps(s.get('extra', {})) if s.get('extra') else '')}</td></tr>"
            for s in self.steps
        )
        return f"<table class='steps'><thead><tr><th>Step</th><th>Time</th><th>Details</th></tr></thead><tbody>{rows}</tbody></table>"

    def _html_targets(self) -> str:
        parts = []
        for t in self.targets:
            cols = ", ".join(t["columns"]) if t["columns"] else "—"
            nulls = ", ".join(f"{k}: {v:.2f}%" for k, v in t["null_rates"].items()) or "—"
            parts.append(f"<tr><td class='num'>{t['horizon']}</td><td>{html.escape(cols)}</td><td class='muted'>{html.escape(nulls)}</td></tr>")
        head = "<thead><tr><th>H</th><th>Columns</th><th>Null %</th></tr></thead>"
        return f"<table class='targets'>{head}<tbody>{''.join(parts)}</tbody></table>"

    def to_html(self) -> str:
        cfg_tbl = self._html_table(list(self.config.items()))
        part_tbl = self._html_table(list(self.partitions.items()))
        ds_tbl = self._html_table(list(self.dataset.items()))
        steps_tbl = self._html_steps()
        targets_tbl = self._html_targets()
        notes = "".join(f"<li>{html.escape(n)}</li>" for n in self.notes)

        return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8"/>
<title>{html.escape(self.title)}</title>
<style>
body {{ font: 13px/1.45 system-ui, -apple-system, Segoe UI, Roboto, Arial; margin: 24px; color:#111; }}
h1 {{ font-size: 20px; margin: 0 0 12px; }}
h2 {{ font-size: 16px; margin: 24px 0 8px; }}
.mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }}
.wrap {{ max-width: 1200px; }}
.kv th {{ text-align:left; padding:6px 10px; background:#f7f7f7; white-space:nowrap; }}
.kv td {{ padding:6px 10px; }}
.kv {{ border-collapse: collapse; border:1px solid #eee; }}
.steps, .targets {{ border-collapse: collapse; width:100%; }}
.steps th, .steps td, .targets th, .targets td {{ border-bottom:1px solid #eee; padding:8px 10px; vertical-align:top; }}
.num {{ text-align:right; font-variant-numeric: tabular-nums; }}
.muted {{ color:#666; }}
.badge {{ display:inline-block; background:#eef; border:1px solid #dde; padding:2px 6px; border-radius:10px; font-size:12px; }}
.footer {{ color:#666; margin-top:24px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>{html.escape(self.title)} <span class="badge">started {html.escape(self.started_at)}</span></h1>

  <h2>Config</h2>
  {cfg_tbl}

  <h2>Partitions</h2>
  {part_tbl}

  <h2>Steps & Timings</h2>
  {steps_tbl}

  <h2>Dataset</h2>
  {ds_tbl}

  <h2>Targets</h2>
  {targets_tbl}

  {"<h2>Notes</h2><ul>"+notes+"</ul>" if self.notes else ""}

  <div class="footer">Finished {html.escape(self.finished_at or '')}</div>
</div>
</body>
</html>"""

    def write(self, out_dir: Path, basename: str = "features") -> Path:
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = out_dir / f"{basename}_{ts}"
        # JSON
        (base.with_suffix(".json")).write_text(json.dumps(asdict(self), indent=2))
        # HTML
        html_path = base.with_suffix(".html")
        html_path.write_text(self.to_html(), encoding="utf-8")
        return html_path
