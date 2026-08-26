"""
Shared matplotlib styling so Python-generated figures match the R `common_theme`
specification used elsewhere in this project. Import this module (or import
anything from it) before creating any figures — it applies the shared rcParams
as a side effect. Always finish a figure with `save_plot_standard()` rather
than calling `plt.savefig()` / `fig.savefig()` directly.
"""
import math
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import pandas as pd
from matplotlib.lines import Line2D

TEXTWIDTH = 6.27  # inches, LaTeX \textwidth for A4 with 1in margins
DEFAULT_ASPECT_RATIO = 5.02 / TEXTWIDTH  # matches the R Rmarkdown chunk default (fig.width=6.27, fig.height=5.02)

# Inches of figure height per row of panels, independent of column count --
# the shared convention for every multi-row grid/stacked figure, so a 3-row
# figure is always the same height whether it has 1 column or 4, and a
# 2-row figure is always shorter than a 3-row one.
# ROW_HEIGHT is the default rate, used by every multi-row figure. ROW_HEIGHT_3
# is a shorter override used only by the plot_wage_comparison_dev /
# _dev_error / _comoment family (set manually, 2026-07-02).
ROW_HEIGHT = 3
ROW_HEIGHT_3 = 2.2

_available_fonts = {f.name for f in fm.fontManager.ttflist}
FONT_FAMILY = 'Latin Modern Roman' if 'Latin Modern Roman' in _available_fonts else 'DejaVu Serif'

MAJOR_GRID_COLOR = '#D9D9D9'
MAJOR_GRID_WIDTH = 0.4
MINOR_GRID_COLOR = '#EBEBEB'
MINOR_GRID_WIDTH = 0.2

mpl.rcParams.update({
    'font.family':           FONT_FAMILY,
    'font.size':             9,
    # matplotlib's mathtext engine ignores font.family and defaults to DejaVu
    # Sans for $...$ math regardless of the surrounding text font -- 'cm'
    # (Computer Modern) is the closest built-in match to Latin Modern Roman
    # (a direct descendant of Computer Modern), so math and text read as the
    # same typeface instead of visibly clashing.
    'mathtext.fontset':      'cm',
    'axes.titlesize':        11,
    'axes.titleweight':      'bold',
    'figure.titlesize':      11,
    'figure.titleweight':    'bold',
    'axes.labelsize':        10,
    'xtick.labelsize':       9,
    'ytick.labelsize':       9,
    'legend.fontsize':       9,
    'legend.title_fontsize': 10,
    'figure.facecolor':      'white',
    'axes.facecolor':        'white',
    'axes.grid':             True,
    'grid.color':            MAJOR_GRID_COLOR,
    'grid.linewidth':        MAJOR_GRID_WIDTH,
    'axes.axisbelow':        True,
    'xtick.color':           '#333333',
    'ytick.color':           '#333333',
    'figure.dpi':            300,
    'axes.spines.top':       False,
    'axes.spines.right':     False,
    'lines.linewidth':       1.0,
})


def apply_minor_gridlines(ax):
    """
    Enable gridlines matching the R common_theme spec: major gridlines on both
    axes, but minor gridlines on x only. A numeric y-axis's AutoMinorLocator
    typically inserts 3-4 minor ticks per major interval, which multiplies the
    horizontal gridline count several-fold into visual noise; the x-axis
    (dates, one major tick per year) doesn't have the same density problem,
    so minor vertical gridlines are kept.
    """
    ax.set_axisbelow(True)
    ax.minorticks_on()
    ax.grid(which='major', axis='both', color=MAJOR_GRID_COLOR, linewidth=MAJOR_GRID_WIDTH)
    ax.grid(which='minor', axis='y', visible=False)
    ax.grid(which='minor', axis='x', color=MINOR_GRID_COLOR, linewidth=MINOR_GRID_WIDTH)


def _bold_legend_titles(fig):
    """Bold every legend title on `fig`, matching R's legend.title = element_text(face='bold')."""
    legends = list(fig.legends)
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            legends.append(leg)
    for leg in legends:
        title = leg.get_title()
        if title is not None and title.get_text():
            title.set_fontweight('bold')


def add_shared_legend(fig, colors, extra_handles=None, ncol=3, title=None, min_fontsize=6):
    """
    Add one legend for the whole figure, placed below all subplots with
    reserved vertical space so it never overlaps axis labels/ticks.

    `colors` : dict {model_name: color} — one line-swatch handle per entry,
        in the given order (should match panel color usage in the figure).
    `extra_handles` : optional list of extra Line2D/Patch handles (e.g. a
        black dotted 'Observed' reference line). Placed at the END of
        column 1 (its last row(s)), not scattered elsewhere — matplotlib
        fills legend columns column-major (top-to-bottom within a column,
        then next column), and column 1 is always the "full" column
        (gets ceil(total/ncol) entries) when the entry count doesn't
        divide evenly by ncol.
    `ncol` : number of legend columns (e.g. 3 -> two rows of 3 for 6 models).
    `min_fontsize` : floor for the auto-shrink below, so very long labels
        never grow the saved canvas past TEXTWIDTH (see below) — matplotlib's
        bbox_inches='tight' will otherwise widen the whole saved image to fit
        an over-wide legend, silently breaking the fixed-width guarantee.

    Bottom margin is not a fixed heuristic — it's measured from the actual
    rendered extent of each axes' tick labels/xlabel (which varies with
    date-tick rotation, offset text, number of rows, etc.) and grown
    iteratively until nothing overlaps. A fixed guess silently under-reserves
    whenever an axis's real content is taller than assumed.

    If the figure has a `fig.supxlabel(...)` (a shared x-label spanning
    multiple subplots, used instead of repeating a per-axes xlabel), it is
    re-anchored directly below the axes on every iteration of the loop below
    — unlike `ax.set_xlabel()`, `fig.supxlabel()`'s position is fixed at
    creation time and does not track the axes as `subplots_adjust(bottom=…)`
    moves them, so left alone it silently drifts out of sync and ends up
    colliding with the legend (or floating in an oversized gap) instead of
    sitting snugly between the axes and the legend.

    Call this AFTER all subplots are drawn (titles/labels finalized) and
    BEFORE save_plot_standard, since it measures the figure's current size
    and rendered content to compute how much bottom margin to reserve.
    """
    model_handles = [Line2D([0], [0], color=c, linewidth=2, label=name) for name, c in colors.items()]
    extra_handles = list(extra_handles) if extra_handles else []
    total = len(model_handles) + len(extra_handles)
    nrows = math.ceil(total / ncol) if total else 0
    insert_at = max(0, nrows - len(extra_handles))
    handles = model_handles[:insert_at] + extra_handles + model_handles[insert_at:]

    fontsize = mpl.rcParams['legend.fontsize']
    # Start small and grow only as much as actually needed (measured below),
    # rather than starting at a fixed fraction of figure height -- a fixed
    # starting fraction (e.g. 0.14) is a much bigger *absolute* margin on a
    # tall multi-row figure than on a short one, leaving a large unused gap
    # above the legend whenever the starting guess already happened to be
    # more than sufficient.
    bottom = 0.02

    legend = None
    for _ in range(20):
        fig.subplots_adjust(bottom=bottom)
        if legend is not None:
            legend.remove()
        legend = fig.legend(handles=handles, loc='lower center', ncol=ncol, title=title,
                             bbox_to_anchor=(0.5, 0.01), frameon=True, fontsize=fontsize)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        # Width: shrink the legend's own font if it's wider than the figure,
        # instead of letting bbox_inches='tight' widen the whole saved canvas.
        fig_w_px = fig.get_size_inches()[0] * fig.dpi
        leg_bbox = legend.get_window_extent(renderer)
        if leg_bbox.width > fig_w_px * 0.98 and fontsize > min_fontsize:
            fontsize -= 0.5
            for text in legend.get_texts():
                text.set_fontsize(fontsize)
            if legend.get_title().get_text():
                legend.get_title().set_fontsize(fontsize + 1)
            continue

        # Height: make sure no axes' real rendered content (tick labels,
        # xlabel, offset text) dips into the legend's vertical extent, with a
        # visible gap rather than just zero overlap -- a couple of px reads as
        # "touching" once rendered, so require a real gap (one full legend
        # font line-height, converted to px at the figure's dpi -- bumped up
        # from 1/3 of that after the legend read as too close to the x-axis
        # across every figure using this helper).
        _gap_px = fontsize * fig.dpi / 72
        # get_tightbbox() returns None for an invisible axes (e.g. a leftover
        # grid cell hidden via ax.set_visible(False) when the model/group
        # count doesn't evenly fill the grid) -- skip those rather than
        # crashing on None.y0.
        min_axes_bottom_px = min(
            tb.y0
            for ax in fig.axes if ax.get_label() != '<colorbar>'
            for tb in [ax.get_tightbbox(renderer)] if tb is not None
        )
        # fig.supxlabel() is a figure-level Text -- unlike ax.set_xlabel(),
        # its position is fixed at creation time and does NOT track the axes
        # as subplots_adjust(bottom=...) moves them each iteration. Re-anchor
        # it directly below the axes' current position on every iteration
        # (not just once beforehand) so its bbox reflects reality when we
        # check for a legend collision below -- otherwise it silently drifts
        # out of sync with the axes and ends up colliding with the legend.
        _supxlabel = getattr(fig, '_supxlabel', None)
        if _supxlabel is not None:
            _fig_h_px = fig.get_size_inches()[1] * fig.dpi
            _supxlabel.set_verticalalignment('top')
            _supxlabel.set_position((0.5, (min_axes_bottom_px - _gap_px / 2) / _fig_h_px))
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            min_axes_bottom_px = _supxlabel.get_window_extent(renderer).y0
        if min_axes_bottom_px < leg_bbox.y1 + _gap_px and bottom < 0.6:
            # Step by the exact measured deficit (floored at 1px of forward
            # progress, not an arbitrary 2% of figure height) -- since
            # subplots_adjust(bottom=...) moves the axes block by a fixed
            # number of *pixels* per unit of `bottom` regardless of figure
            # height, a coarse minimum step overshoots far more on a short
            # figure than a tall one, leaving an inconsistent final gap
            # across figures of different heights.
            fig_h_px = fig.get_size_inches()[1] * fig.dpi
            bottom += max(1 / fig_h_px, (leg_bbox.y1 + _gap_px - min_axes_bottom_px) / fig_h_px)
            continue

        break


def add_suptitle_spacing(fig, gap_pt=None):
    """
    Ensure a consistent physical gap between fig.suptitle() and the topmost
    rendered axes content (panel titles, ticks), regardless of the figure's
    height -- mirrors add_shared_legend's measured-gap approach for the
    bottom margin. A fixed `top=` fraction (or tight_layout()'s own spacing
    heuristic) produces a bigger absolute gap on a tall multi-row figure
    than a short one; measuring and adjusting `top` in pixel terms keeps it
    the same physical size everywhere.

    No-op if the figure has no suptitle. Call AFTER fig.suptitle(...) and
    any tight_layout()/gridspec spacing is finalized (this will override a
    tight_layout()-computed top, which is expected), and before or after
    add_shared_legend -- the two margins are independent.
    """
    suptitle = fig._suptitle
    if suptitle is None:
        return
    if gap_pt is None:
        gap_pt = mpl.rcParams['figure.titlesize'] / 3

    top = fig.subplotpars.top
    for _ in range(20):
        fig.subplots_adjust(top=top)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        gap_px = gap_pt * fig.dpi / 72

        st_bbox = suptitle.get_window_extent(renderer)
        # get_tightbbox() returns None for an invisible axes (e.g. a leftover
        # grid cell hidden via ax.set_visible(False)) -- skip those.
        topmost_ax_top = max(
            tb.y1
            for ax in fig.axes if ax.get_label() != '<colorbar>'
            for tb in [ax.get_tightbbox(renderer)] if tb is not None
        )
        gap = st_bbox.y0 - topmost_ax_top
        # Bidirectional: a figure that skips tight_layout() (e.g. custom
        # gridspec_kw) starts from matplotlib's default top, which can leave
        # a much *bigger* gap than the target, not just a smaller one -- so
        # this must be able to shrink an oversized gap, not just grow one.
        deficit_px = gap_px - gap
        if abs(deficit_px) < 1:
            break
        fig_h_px = fig.get_size_inches()[1] * fig.dpi
        top = min(0.98, max(0.5, top - deficit_px / fig_h_px))


def add_supylabel_spacing(fig, gap_pt=None):
    """
    Reposition fig.supylabel(...) to sit a consistent physical gap to the
    left of the leftmost rendered axes content (y-tick labels), regardless
    of how wide those tick labels happen to render.

    fig.supylabel()'s x position defaults to a fixed figure-fraction (0.02)
    at creation time and does NOT track how wide the y-tick labels actually
    are (which varies with the data's value range/decimal places) -- left
    alone, the label can end up overlapping wide tick labels, or floating
    with an inconsistent gap next to narrow ones. This also means two
    different figures with different tick-label widths won't have their
    plot areas start at the same fraction of image width once both are
    bbox_inches='tight'-cropped and scaled to the same size in a document
    (e.g. two figures stacked as LaTeX subfigures, expected to have their
    y-axes visually align).

    No-op if the figure has no supylabel. Call AFTER all subplots are drawn
    and tick labels finalized -- mirrors add_suptitle_spacing/the supxlabel
    handling in add_shared_legend, but for the left margin. Unlike those,
    a single measurement pass suffices here: repositioning the label doesn't
    change the axes region itself (no subplots_adjust involved), so there's
    no feedback loop to iterate.
    """
    supylabel = getattr(fig, '_supylabel', None)
    if supylabel is None:
        return
    if gap_pt is None:
        gap_pt = mpl.rcParams['axes.labelsize'] / 3

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    gap_px = gap_pt * fig.dpi / 72
    fig_w_px = fig.get_size_inches()[0] * fig.dpi

    lefts = [
        tb.x0
        for ax in fig.axes if ax.get_label() != '<colorbar>'
        for tb in [ax.get_tightbbox(renderer)] if tb is not None
    ]
    if not lefts:
        return
    axes_left_px = min(lefts)
    supylabel.set_horizontalalignment('right')
    supylabel.set_position(((axes_left_px - gap_px) / fig_w_px, 0.5))


def shade_out_of_sample(ax, calib_end_date, color='grey', alpha=0.15):
    """
    Shade the region of `ax` after `calib_end_date` (a date string/Timestamp)
    to mark the out-of-sample period, leaving everything before it unshaded.

    Call AFTER the data is plotted and any explicit `ax.set_xlim(...)` is
    set, since it reads (and restores) the axes' current x-limits to find
    the right edge of the shaded region.
    """
    if calib_end_date is None:
        return
    xmin, xmax = ax.get_xlim()
    boundary = mdates.date2num(pd.to_datetime(calib_end_date))
    if boundary < xmax:
        ax.axvspan(max(boundary, xmin), xmax, color=color, alpha=alpha, zorder=0)
    ax.set_xlim(xmin, xmax)


def shade_out_of_sample_year(ax, calib_end_year, color='grey', alpha=0.15):
    """
    Same as `shade_out_of_sample`, for axes with a plain integer Year x-axis
    (e.g. annual time series) rather than a calendar-date axis.
    """
    if calib_end_year is None:
        return
    xmin, xmax = ax.get_xlim()
    if calib_end_year < xmax:
        ax.axvspan(max(calib_end_year, xmin), xmax, color=color, alpha=alpha, zorder=0)
    ax.set_xlim(xmin, xmax)


def save_plot_standard(fig, path, aspect_ratio=DEFAULT_ASPECT_RATIO, width_frac=1.0, apply_grid=True):
    """
    Resize and save a figure to match the shared R/Python figure style.

    `aspect_ratio` defaults to the R Rmarkdown chunk default (fig.width=6.27,
    fig.height=5.02, i.e. 5.02/6.27) so a single-panel Python figure prints at
    the same size as a default R figure. Multi-panel figures should pass an
    explicit larger `aspect_ratio` (increasing height, same convention as
    manually bumping fig.height per-chunk in R) rather than relying on this
    default.

    `width_frac` is a multiple of TEXTWIDTH (6.27in). Pass width_frac=1.0 for
    a normal single full-width figure. For dense multi-panel grids, pass a
    larger width_frac (e.g. derived from the original figsize width /
    TEXTWIDTH) so panels stay legible instead of being crushed to 6.27in.

    Set apply_grid=False for figures where gridlines don't make sense
    (heatmaps, network diagrams, colorbars).
    """
    w = TEXTWIDTH * width_frac
    h = w * aspect_ratio
    fig.set_size_inches(w, h)
    if apply_grid:
        # A twin axis (.twinx()/.twiny()) occupies the exact same bounding box
        # as its parent axes but has its own independent y (or x) scale/ticks.
        # Gridding it too draws a second set of gridlines at different pixel
        # positions than the parent's, overlapping into a noisy double-grid --
        # so only the first axes seen at a given position gets gridded. The
        # skipped (twin) axes must be explicitly degridded, not just left
        # alone: rcParams['axes.grid']=True makes matplotlib turn gridlines on
        # for *every* axes the moment it's created (verified -- a bare
        # ax.twinx() with no .grid() call anywhere already has visible
        # gridlines), so simply not calling apply_minor_gridlines on it
        # leaves that default-styled grid showing through untouched.
        _seen_positions = set()
        for ax in fig.axes:
            if ax.get_label() == '<colorbar>':
                continue
            pos_key = tuple(round(v, 6) for v in ax.get_position().bounds)
            if pos_key in _seen_positions:
                ax.grid(False)
                continue
            _seen_positions.add(pos_key)
            try:
                apply_minor_gridlines(ax)
            except (AttributeError, NotImplementedError):
                pass
    _bold_legend_titles(fig)
    fig.savefig(path, dpi=300, bbox_inches='tight')
