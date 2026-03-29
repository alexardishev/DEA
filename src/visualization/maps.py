"""Optional geographic visualization for Austrian banks."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def plot_efficiency_map(df, score_col: str = "stage2_efficiency") -> str:
    """Build map if latitude/longitude exist, otherwise return explanatory note."""
    required = {"latitude", "longitude"}
    if not required.issubset(df.columns):
        return "Map skipped: latitude/longitude columns are not available."

    try:
        import geopandas as gpd

        gdf = gpd.GeoDataFrame(df.copy(), geometry=gpd.points_from_xy(df["longitude"], df["latitude"]), crs="EPSG:4326")
        fig, ax = plt.subplots(figsize=(8, 8))
        gdf.plot(column=score_col, cmap="viridis", legend=True, markersize=25, alpha=0.9, ax=ax)
        ax.set_title("Banks in Austria colored by efficiency")
        out = Path("outputs/maps/bank_efficiency_map.png")
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return f"Map created: {out}"
    except Exception as exc:  # noqa: BLE001
        return f"Map skipped due to plotting/geodata issue: {exc}"
