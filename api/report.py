"""SHANK Analysis Report generator.

Produces song-breakdown reports from a completed task payload in three formats:
    JSON  – structured data (all analysis fields)
    HTML  – self-contained single-file page with inline SVG charts
    PDF   – multi-page PDF produced via matplotlib (optional; falls back to
             raising ImportError when matplotlib is not installed)
"""

from __future__ import annotations

import html
import io
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# JSON report
# ---------------------------------------------------------------------------

def build_report_json(task: dict[str, Any]) -> dict[str, Any]:
    """Return a structured report dict assembled from *task* payload."""
    analysis = task.get('analysis') or {}
    full_mix: dict[str, Any] = analysis.get('full_mix') or {}
    stems_analysis: dict[str, Any] = analysis.get('stems') or {}

    bpm = full_mix.get('bpm') or task.get('bpm')
    key = full_mix.get('key') or task.get('key')
    duration = full_mix.get('duration_seconds') or task.get('duration_seconds')
    lufs = full_mix.get('lufs')

    # Chords
    chords_raw = full_mix.get('chords') or task.get('chords') or {}
    chords: dict[str, Any] = {
        'segments': chords_raw.get('segments') or [],
        'progression': chords_raw.get('progression') or [],
    }

    # Structure / Sections / Cue points
    structure = full_mix.get('structure') or []
    sections = full_mix.get('sections') or []
    cue_points = full_mix.get('cue_points') or []

    # Graph data
    waveform = full_mix.get('waveform') or []
    energy = full_mix.get('energy_over_time') or []
    loudness = full_mix.get('loudness_curve') or []

    # Beat / grid data
    beatgrid = full_mix.get('beatgrid') or task.get('beatgrid') or {}

    # MIDI / stems
    mt3_data = task.get('mt3') or {}
    stems_paths = task.get('stems') or {}

    stems_summary: dict[str, Any] = {}
    for stem_name, stem_data in stems_analysis.items():
        if not isinstance(stem_data, dict):
            continue
        stems_summary[stem_name] = {
            'bpm': stem_data.get('bpm'),
            'key': stem_data.get('key'),
            'duration_seconds': stem_data.get('duration_seconds'),
            'lufs': stem_data.get('lufs'),
            'file': stems_paths.get(stem_name),
        }

    midi_summary: dict[str, Any] = {}
    if isinstance(mt3_data, dict):
        full_mix_mt3 = mt3_data.get('full_mix') or {}
        if isinstance(full_mix_mt3, dict):
            midi_summary['full_mix'] = {
                'midi_path': full_mix_mt3.get('midi_path'),
                'note_count': len(full_mix_mt3.get('notes') or []),
                'status': mt3_data.get('status'),
            }
        mt3_stems = mt3_data.get('stems') or {}
        if isinstance(mt3_stems, dict):
            for stem_name, stem_midi in mt3_stems.items():
                if not isinstance(stem_midi, dict):
                    continue
                midi_summary[stem_name] = {
                    'midi_path': stem_midi.get('midi_path'),
                    'note_count': len(stem_midi.get('notes') or []),
                }

    source = task.get('source') or task.get('youtube', {}).get('title') or 'Unknown'
    title = _derive_title(task)

    return {
        'report_version': '1.0',
        'generated_at': datetime.now(timezone.utc).isoformat(),
        'task_id': task.get('task_id'),
        'title': title,
        'source': source,
        'summary': {
            'bpm': bpm,
            'key': key,
            'duration_seconds': duration,
            'lufs': lufs,
        },
        'structure': structure,
        'sections': sections,
        'cue_points': cue_points,
        'chords': chords,
        'beatgrid': beatgrid,
        'waveform': waveform,
        'energy': energy,
        'loudness': loudness,
        'stems': stems_summary,
        'midi': midi_summary,
    }


def _derive_title(task: dict[str, Any]) -> str:
    yt = task.get('youtube')
    if isinstance(yt, dict) and yt.get('title'):
        return str(yt['title'])
    source = task.get('source')
    if isinstance(source, str) and source:
        # Strip common path prefixes so only the filename remains.
        import os
        return os.path.splitext(os.path.basename(source))[0]
    return task.get('task_id') or 'SHANK Analysis'


# ---------------------------------------------------------------------------
# SVG helpers (server-side, no external deps)
# ---------------------------------------------------------------------------

def _svg_polyline(
    values: list[float],
    *,
    width: int = 800,
    height: int = 120,
    color: str = '#4ea1ff',
    fill: str = 'none',
    label: str = '',
    x_labels: list[str] | None = None,
) -> str:
    """Return a minimal inline SVG graph of *values*."""
    if not values:
        return (
            f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
            f'xmlns="http://www.w3.org/2000/svg">'
            f'<rect width="{width}" height="{height}" fill="#1a1d24"/>'
            f'<text x="{width//2}" y="{height//2}" fill="#667" font-size="12" '
            f'text-anchor="middle" dominant-baseline="middle">No data</text>'
            f'</svg>'
        )

    pad_left = 10
    pad_right = 10
    pad_top = 10
    pad_bottom = 20 if x_labels else 10

    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom

    min_v = min(values)
    max_v = max(values)
    span = max_v - min_v if max_v != min_v else 1.0

    n = len(values)
    points: list[str] = []
    for i, v in enumerate(values):
        x = pad_left + i / max(n - 1, 1) * plot_w
        y = pad_top + (1 - (v - min_v) / span) * plot_h
        points.append(f'{x:.1f},{y:.1f}')

    pts_str = ' '.join(points)

    # Optional filled area under the curve
    fill_path = ''
    if fill != 'none' and points:
        first_x = pad_left
        last_x = pad_left + plot_w
        bottom_y = pad_top + plot_h
        fill_path = (
            f'<polyline points="{first_x},{bottom_y} {pts_str} {last_x},{bottom_y}" '
            f'fill="{fill}" stroke="none" opacity="0.2"/>'
        )

    label_el = ''
    if label:
        label_el = (
            f'<text x="{width - pad_right}" y="{pad_top + 10}" '
            f'fill="#667" font-size="10" text-anchor="end">{html.escape(label)}</text>'
        )

    # X-axis tick labels (e.g. time markers)
    tick_labels_el = ''
    if x_labels:
        tick_step = max(1, len(x_labels) // 6)
        ticks = []
        for i in range(0, len(x_labels), tick_step):
            x = pad_left + i / max(len(x_labels) - 1, 1) * plot_w
            ticks.append(
                f'<text x="{x:.1f}" y="{height - 3}" fill="#556" font-size="9" '
                f'text-anchor="middle">{html.escape(x_labels[i])}</text>'
            )
        tick_labels_el = '\n'.join(ticks)

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<rect width="{width}" height="{height}" fill="#1a1d24" rx="4"/>'
        f'{fill_path}'
        f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="1.5" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'{label_el}'
        f'{tick_labels_el}'
        f'</svg>'
    )


def _time_labels(n_bins: int, duration: float | None) -> list[str] | None:
    if not duration or duration <= 0 or n_bins < 2:
        return None
    step = duration / n_bins
    labels: list[str] = []
    for i in range(n_bins):
        t = i * step
        m = int(t // 60)
        s = int(t % 60)
        labels.append(f'{m}:{s:02d}')
    return labels


def _fmt_seconds(t: float | None) -> str:
    if not isinstance(t, (int, float)):
        return '—'
    t = float(t)
    m = int(t // 60)
    s = t % 60
    return f'{m}:{s:05.2f}'


def _esc(v: Any) -> str:
    return html.escape(str(v)) if v is not None else '—'


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { background: #0d0f14; color: #c8d3e0; font-family: system-ui, sans-serif;
       font-size: 14px; line-height: 1.5; padding: 24px; }
h1 { font-size: 1.5rem; color: #e2e8f0; margin-bottom: 4px; }
h2 { font-size: 1.05rem; color: #94a3b8; margin: 20px 0 8px; letter-spacing: .04em;
     text-transform: uppercase; }
.meta { color: #667; font-size: 0.82rem; margin-bottom: 20px; }
.cards { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }
.card { background: #161921; border: 1px solid #232b38; border-radius: 8px;
        padding: 12px 18px; min-width: 120px; }
.card-label { font-size: 0.75rem; color: #667; text-transform: uppercase; letter-spacing:.06em; }
.card-value { font-size: 1.4rem; color: #e2e8f0; font-weight: 600; margin-top: 2px; }
.chart-wrap { background: #1a1d24; border-radius: 6px; padding: 8px; margin-bottom: 16px;
              overflow-x: auto; }
.chart-title { font-size: 0.78rem; color: #556; margin-bottom: 4px; }
table { width: 100%; border-collapse: collapse; margin-bottom: 16px; }
th { background: #161921; color: #94a3b8; font-size: 0.78rem; text-transform: uppercase;
     letter-spacing: .04em; padding: 6px 10px; text-align: left; border-bottom: 1px solid #232b38; }
td { padding: 6px 10px; border-bottom: 1px solid #1c2130; font-size: 0.85rem; }
tr:last-child td { border-bottom: none; }
.badge { display: inline-block; background: #1e2e48; color: #7ab8f5; border-radius: 4px;
         padding: 1px 7px; font-size: 0.75rem; }
.empty { color: #445; font-style: italic; padding: 8px 0; }
@media (max-width: 600px) { .cards { flex-direction: column; } }
@media print {
  body { background: #fff; color: #111; }
  .card { background: #f5f5f5; border-color: #ccc; }
  th { background: #eee; }
}
"""


def build_report_html(task: dict[str, Any]) -> str:
    """Return a self-contained HTML report string for *task*."""
    report = build_report_json(task)
    title = _esc(report['title'])
    task_id = _esc(report['task_id'] or '')
    generated = _esc(report['generated_at'])
    summary = report['summary']

    bpm_str = f"{summary['bpm']:.2f}" if isinstance(summary['bpm'], (int, float)) else '—'
    key_str = _esc(summary['key'] or '—')
    dur_str = _fmt_seconds(summary['duration_seconds'])
    lufs_str = f"{summary['lufs']:.1f} LUFS" if isinstance(summary['lufs'], (int, float)) else '—'

    duration = summary.get('duration_seconds')

    # ── Charts ──────────────────────────────────────────────────────────────
    waveform_svg = _svg_polyline(
        report['waveform'],
        width=780,
        height=100,
        color='#4ea1ff',
        fill='#4ea1ff',
        label='Waveform',
        x_labels=_time_labels(len(report['waveform']), duration),
    )
    energy_svg = _svg_polyline(
        report['energy'],
        width=780,
        height=80,
        color='#22c55e',
        fill='#22c55e',
        label='Energy',
        x_labels=_time_labels(len(report['energy']), duration),
    )
    loudness_svg = _svg_polyline(
        report['loudness'],
        width=780,
        height=80,
        color='#f97316',
        fill='#f97316',
        label='Loudness (RMS)',
        x_labels=_time_labels(len(report['loudness']), duration),
    )

    # ── Structure table ──────────────────────────────────────────────────────
    structure_rows = ''
    for entry in report['structure']:
        if not isinstance(entry, dict):
            continue
        structure_rows += (
            f'<tr><td>{_esc(entry.get("label"))}</td>'
            f'<td>{_esc(entry.get("timestamp") or _fmt_seconds(entry.get("start_seconds")))}</td>'
            f'<td>{_fmt_seconds(entry.get("start_seconds"))}</td>'
            f'<td>{_fmt_seconds(entry.get("end_seconds"))}</td></tr>'
        )
    structure_html = (
        '<table><tr><th>Section</th><th>Timestamp</th><th>Start</th><th>End</th></tr>'
        + (structure_rows or '<tr><td colspan="4" class="empty">No structure data</td></tr>')
        + '</table>'
    )

    # ── Sections table ───────────────────────────────────────────────────────
    sections_rows = ''
    for entry in report['sections']:
        if not isinstance(entry, dict):
            continue
        sections_rows += (
            f'<tr><td>{_esc(entry.get("label"))}</td>'
            f'<td>{_fmt_seconds(entry.get("start_seconds"))}</td>'
            f'<td>{_fmt_seconds(entry.get("end_seconds"))}</td></tr>'
        )
    sections_html = (
        '<table><tr><th>Label</th><th>Start</th><th>End</th></tr>'
        + (sections_rows or '<tr><td colspan="3" class="empty">No sections data</td></tr>')
        + '</table>'
    )

    # ── Cue points table ─────────────────────────────────────────────────────
    cue_rows = ''
    for entry in report['cue_points']:
        if not isinstance(entry, dict):
            continue
        cue_rows += (
            f'<tr><td>{_esc(entry.get("name"))}</td>'
            f'<td>{_fmt_seconds(entry.get("time_seconds"))}</td></tr>'
        )
    cue_html = (
        '<table><tr><th>Name</th><th>Time</th></tr>'
        + (cue_rows or '<tr><td colspan="2" class="empty">No cue points</td></tr>')
        + '</table>'
    )

    # ── Chords table ─────────────────────────────────────────────────────────
    chord_rows = ''
    for entry in report['chords']['segments']:
        if not isinstance(entry, dict):
            continue
        symbol = _esc(entry.get('symbol') or entry.get('chord') or '?')
        conf = entry.get('confidence')
        conf_str = f'{conf:.0%}' if isinstance(conf, (int, float)) else '—'
        chord_rows += (
            f'<tr><td>{symbol}</td>'
            f'<td>{_fmt_seconds(entry.get("start_seconds"))}</td>'
            f'<td>{_fmt_seconds(entry.get("end_seconds"))}</td>'
            f'<td>{conf_str}</td></tr>'
        )
    chord_progression = report['chords'].get('progression') or []
    prog_html = ''
    if chord_progression:
        prog_html = (
            '<p style="margin:6px 0 10px;font-size:0.85rem;color:#94a3b8;">'
            '<strong>Progression:</strong> '
            + ' → '.join(_esc(c) for c in chord_progression[:24])
            + ('…' if len(chord_progression) > 24 else '')
            + '</p>'
        )
    chords_html = (
        prog_html
        + '<table><tr><th>Chord</th><th>Start</th><th>End</th><th>Confidence</th></tr>'
        + (chord_rows or '<tr><td colspan="4" class="empty">No chord data</td></tr>')
        + '</table>'
    )

    # ── Stems section ────────────────────────────────────────────────────────
    stems_rows = ''
    for stem_name, stem_data in report['stems'].items():
        if not isinstance(stem_data, dict):
            continue
        sbpm = f"{stem_data['bpm']:.2f}" if isinstance(stem_data.get('bpm'), (int, float)) else '—'
        skey = _esc(stem_data.get('key') or '—')
        sdur = _fmt_seconds(stem_data.get('duration_seconds'))
        slugs = f"{stem_data['lufs']:.1f}" if isinstance(stem_data.get('lufs'), (int, float)) else '—'
        stems_rows += (
            f'<tr><td><span class="badge">{_esc(stem_name)}</span></td>'
            f'<td>{sbpm}</td><td>{skey}</td><td>{sdur}</td><td>{slugs}</td></tr>'
        )
    stems_html = (
        '<table><tr><th>Stem</th><th>BPM</th><th>Key</th><th>Duration</th><th>LUFS</th></tr>'
        + (stems_rows or '<tr><td colspan="5" class="empty">No stem analysis</td></tr>')
        + '</table>'
    )

    # ── MIDI section ─────────────────────────────────────────────────────────
    midi_rows = ''
    for track_name, midi_data in report['midi'].items():
        if not isinstance(midi_data, dict):
            continue
        note_count = midi_data.get('note_count') or 0
        status = _esc(midi_data.get('status') or '—')
        midi_rows += (
            f'<tr><td><span class="badge">{_esc(track_name)}</span></td>'
            f'<td>{note_count}</td><td>{status}</td></tr>'
        )
    midi_html = (
        '<table><tr><th>Track</th><th>Notes</th><th>Status</th></tr>'
        + (midi_rows or '<tr><td colspan="3" class="empty">No MIDI data</td></tr>')
        + '</table>'
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SHANK Report — {title}</title>
<style>{_HTML_CSS}</style>
</head>
<body>
<h1>🎵 {title}</h1>
<p class="meta">SHANK Analysis Report · Task {task_id} · Generated {generated}</p>

<div class="cards">
  <div class="card"><div class="card-label">BPM</div><div class="card-value">{bpm_str}</div></div>
  <div class="card"><div class="card-label">Key</div><div class="card-value">{key_str}</div></div>
  <div class="card"><div class="card-label">Duration</div><div class="card-value">{dur_str}</div></div>
  <div class="card"><div class="card-label">Loudness</div><div class="card-value">{lufs_str}</div></div>
</div>

<h2>Waveform</h2>
<div class="chart-wrap">{waveform_svg}</div>

<h2>Energy</h2>
<div class="chart-wrap">{energy_svg}</div>

<h2>Loudness (RMS)</h2>
<div class="chart-wrap">{loudness_svg}</div>

<h2>Song Structure</h2>
{structure_html}

<h2>Sections</h2>
{sections_html}

<h2>Cue Points</h2>
{cue_html}

<h2>Chords</h2>
{chords_html}

<h2>Stems</h2>
{stems_html}

<h2>MIDI / Transcription</h2>
{midi_html}

</body>
</html>"""


# ---------------------------------------------------------------------------
# PDF report (requires matplotlib)
# ---------------------------------------------------------------------------

def build_report_pdf(task: dict[str, Any]) -> bytes:
    """Return a PDF report as raw bytes.

    Requires ``matplotlib``.  Raises ``ImportError`` if it is unavailable.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'matplotlib is required for PDF report generation. '
            'Install it with: pip install matplotlib'
        ) from exc

    report = build_report_json(task)
    title = report['title']
    summary = report['summary']

    bpm_str = f"{summary['bpm']:.2f}" if isinstance(summary['bpm'], (int, float)) else 'N/A'
    key_str = summary['key'] or 'N/A'
    dur_str = _fmt_seconds(summary['duration_seconds'])
    lufs_str = f"{summary['lufs']:.1f} LUFS" if isinstance(summary['lufs'], (int, float)) else 'N/A'
    duration = summary.get('duration_seconds') or 1.0

    buf = io.BytesIO()

    _DARK_BG = '#0d0f14'
    _TEXT = '#c8d3e0'
    _MUTED = '#667788'
    _BORDER = '#232b38'

    plt.rcParams.update({
        'figure.facecolor': _DARK_BG,
        'axes.facecolor': '#161921',
        'axes.edgecolor': _BORDER,
        'axes.labelcolor': _MUTED,
        'text.color': _TEXT,
        'xtick.color': _MUTED,
        'ytick.color': _MUTED,
        'grid.color': '#232b38',
        'grid.alpha': 0.5,
        'font.size': 8,
    })

    def _time_axis(n: int) -> list[float]:
        return [i / max(n - 1, 1) * duration for i in range(n)] if n else []

    def _plot_curve(ax: Any, values: list[float], color: str, label: str) -> None:
        if not values:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                    transform=ax.transAxes, color=_MUTED)
            ax.set_yticks([])
            return
        xs = _time_axis(len(values))
        ys = values
        ax.plot(xs, ys, color=color, linewidth=0.8)
        ax.fill_between(xs, ys, alpha=0.15, color=color)
        ax.set_xlim(0, duration)
        ax.set_ylabel(label, fontsize=7)
        ax.grid(True, linewidth=0.4)

    with PdfPages(buf) as pdf:
        # ── Page 1: Summary + charts ────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor(_DARK_BG)

        gs = gridspec.GridSpec(5, 1, figure=fig,
                               top=0.88, bottom=0.08, hspace=0.7)

        # Title block
        fig.text(0.08, 0.93, f'[SHANK]  {title}', fontsize=16, color='#e2e8f0', fontweight='bold')
        fig.text(0.08, 0.90, f'Task {report["task_id"]}  ·  {report["generated_at"]}',
                 fontsize=8, color=_MUTED)

        # Summary card row (text only)
        fig.text(0.08, 0.86,
                 f'BPM: {bpm_str}     Key: {key_str}     Duration: {dur_str}     Loudness: {lufs_str}',
                 fontsize=10, color='#94a3b8')

        ax_wave = fig.add_subplot(gs[0])
        ax_energy = fig.add_subplot(gs[1])
        ax_loud = fig.add_subplot(gs[2])
        ax_struct = fig.add_subplot(gs[3])
        ax_cue = fig.add_subplot(gs[4])

        _plot_curve(ax_wave, report['waveform'], '#4ea1ff', 'Waveform')
        ax_wave.set_xticklabels([])

        _plot_curve(ax_energy, report['energy'], '#22c55e', 'Energy')
        ax_energy.set_xticklabels([])

        _plot_curve(ax_loud, report['loudness'], '#f97316', 'RMS Loudness')
        ax_loud.set_xlabel('Time (s)', fontsize=7)

        # Structure bar chart
        structure = report['structure']
        if structure:
            STRUCT_COLORS = {
                'Intro': '#3b82f6',
                'Verse': '#22c55e',
                'Chorus': '#f97316',
                'Bridge': '#a855f7',
                'Breakdown': '#ec4899',
                'Outro': '#64748b',
            }
            ax_struct.set_facecolor('#161921')
            for entry in structure:
                if not isinstance(entry, dict):
                    continue
                start = entry.get('start_seconds') or 0
                end = entry.get('end_seconds') or start
                lbl = entry.get('label', '')
                color = STRUCT_COLORS.get(lbl, '#94a3b8')
                ax_struct.barh(0, end - start, left=start, height=0.6,
                               color=color, alpha=0.85)
                mid = (start + end) / 2
                ax_struct.text(mid, 0, lbl, ha='center', va='center',
                               fontsize=6, color='white')
            ax_struct.set_xlim(0, duration)
            ax_struct.set_yticks([])
            ax_struct.set_xlabel('Time (s)', fontsize=7)
            ax_struct.set_ylabel('Structure', fontsize=7)
            ax_struct.grid(False)
        else:
            ax_struct.text(0.5, 0.5, 'No structure data', ha='center', va='center',
                           transform=ax_struct.transAxes, color=_MUTED)
            ax_struct.set_yticks([])

        # Cue points scatter
        cue_points = report['cue_points']
        if cue_points:
            ax_cue.set_facecolor('#161921')
            times = [c.get('time_seconds', 0) for c in cue_points if isinstance(c, dict)]
            names = [c.get('name', '') for c in cue_points if isinstance(c, dict)]
            ax_cue.scatter(times, [0.5] * len(times), color='#f97316', s=30, zorder=3)
            for t, n in zip(times, names):
                ax_cue.text(t, 0.65, n, ha='center', fontsize=6, color='#94a3b8', rotation=30)
            ax_cue.set_xlim(0, duration)
            ax_cue.set_ylim(0, 1)
            ax_cue.set_yticks([])
            ax_cue.set_xlabel('Time (s)', fontsize=7)
            ax_cue.set_ylabel('Cue Points', fontsize=7)
            ax_cue.grid(False)
        else:
            ax_cue.text(0.5, 0.5, 'No cue points', ha='center', va='center',
                        transform=ax_cue.transAxes, color=_MUTED)
            ax_cue.set_yticks([])

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── Page 2: Chords + Stems + MIDI ───────────────────────────────────
        fig2, axes2 = plt.subplots(3, 1, figsize=(11, 8.5))
        fig2.patch.set_facecolor(_DARK_BG)
        fig2.subplots_adjust(top=0.92, bottom=0.06, hspace=0.7)
        fig2.text(0.08, 0.96, 'Chords · Stems · MIDI', fontsize=13,
                  color='#e2e8f0', fontweight='bold')

        # Chord progression timeline
        ax_chords = axes2[0]
        ax_chords.set_facecolor('#161921')
        chord_segs = report['chords'].get('segments') or []
        if chord_segs:
            palette = [
                '#3b82f6', '#22c55e', '#f97316', '#a855f7',
                '#ec4899', '#06b6d4', '#eab308', '#64748b',
            ]
            seen: dict[str, int] = {}
            for entry in chord_segs:
                if not isinstance(entry, dict):
                    continue
                sym = entry.get('symbol') or entry.get('chord') or '?'
                start = entry.get('start_seconds') or 0
                end = entry.get('end_seconds') or start
                if sym not in seen:
                    seen[sym] = len(seen) % len(palette)
                color = palette[seen[sym]]
                ax_chords.barh(0, end - start, left=start, height=0.6,
                               color=color, alpha=0.85)
                if (end - start) > duration * 0.04:
                    ax_chords.text((start + end) / 2, 0, sym, ha='center',
                                   va='center', fontsize=6, color='white')
            ax_chords.set_xlim(0, duration)
            ax_chords.set_yticks([])
            ax_chords.set_xlabel('Time (s)', fontsize=7)
            ax_chords.set_ylabel('Chords', fontsize=7)
            ax_chords.grid(False)
        else:
            ax_chords.text(0.5, 0.5, 'No chord data', ha='center', va='center',
                           transform=ax_chords.transAxes, color=_MUTED)
            ax_chords.set_yticks([])
            ax_chords.set_xticks([])

        # Stems table
        ax_stems = axes2[1]
        ax_stems.set_facecolor('#161921')
        ax_stems.axis('off')
        stems = report.get('stems') or {}
        if stems:
            col_labels = ['Stem', 'BPM', 'Key', 'Duration', 'LUFS']
            cell_data = []
            for stem_name, stem_data in stems.items():
                sbpm = f"{stem_data['bpm']:.2f}" if isinstance(stem_data.get('bpm'), (int, float)) else '—'
                skey = stem_data.get('key') or '—'
                sdur = _fmt_seconds(stem_data.get('duration_seconds'))
                slugs = f"{stem_data['lufs']:.1f}" if isinstance(stem_data.get('lufs'), (int, float)) else '—'
                cell_data.append([stem_name, sbpm, skey, sdur, slugs])
            t = ax_stems.table(
                cellText=cell_data,
                colLabels=col_labels,
                loc='center',
                cellLoc='left',
            )
            t.auto_set_font_size(False)
            t.set_fontsize(8)
            for key_cell in t._cells:
                cell = t._cells[key_cell]
                cell.set_facecolor('#161921' if key_cell[0] > 0 else '#0d1929')
                cell.set_edgecolor(_BORDER)
                cell.set_text_props(color=_TEXT if key_cell[0] > 0 else '#94a3b8')
            ax_stems.set_title('Stems', color='#94a3b8', fontsize=9, loc='left', pad=6)
        else:
            ax_stems.text(0.5, 0.5, 'No stem analysis', ha='center', va='center',
                          transform=ax_stems.transAxes, color=_MUTED)

        # MIDI table
        ax_midi = axes2[2]
        ax_midi.set_facecolor('#161921')
        ax_midi.axis('off')
        midi = report.get('midi') or {}
        if midi:
            col_labels_midi = ['Track', 'Notes', 'Status']
            cell_midi = []
            for track_name, midi_data in midi.items():
                if not isinstance(midi_data, dict):
                    continue
                cell_midi.append([
                    track_name,
                    str(midi_data.get('note_count') or 0),
                    midi_data.get('status') or '—',
                ])
            if cell_midi:
                t2 = ax_midi.table(
                    cellText=cell_midi,
                    colLabels=col_labels_midi,
                    loc='center',
                    cellLoc='left',
                )
                t2.auto_set_font_size(False)
                t2.set_fontsize(8)
                for key_cell in t2._cells:
                    cell = t2._cells[key_cell]
                    cell.set_facecolor('#161921' if key_cell[0] > 0 else '#0d1929')
                    cell.set_edgecolor(_BORDER)
                    cell.set_text_props(color=_TEXT if key_cell[0] > 0 else '#94a3b8')
            ax_midi.set_title('MIDI / Transcription', color='#94a3b8', fontsize=9, loc='left', pad=6)
        else:
            ax_midi.text(0.5, 0.5, 'No MIDI data', ha='center', va='center',
                         transform=ax_midi.transAxes, color=_MUTED)

        pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig2)

    plt.rcdefaults()

    buf.seek(0)
    return buf.read()
