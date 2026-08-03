#!/usr/bin/env python3
"""Generates fig_esp32_internal.{pdf,png} - a self-authored internal-component block
diagram of the classic ESP32 to replace the copyrighted board photo (Fig 2). Focuses on
the internal components relevant to the paper: dual LX6 cores (no vector unit), internal
SRAM with the measured regions and the ~150 kB usable contiguous block used as the tensor
arena, internal ROM, the on-chip WiFi/BT radio (disabled during inference), and the
external 4 MB flash that holds the int8 weights - with no PSRAM."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams.update({"font.size": 11, "font.family": "sans-serif"})
fig, ax = plt.subplots(figsize=(7.4, 5.2))
ax.set_xlim(0, 100); ax.set_ylim(0, 72); ax.axis("off")

def box(x, y, w, h, label, fc, ec="#222", sub=None, fs=10, subfs=8.5, lw=1.3, tc="#111",
        title=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.5",
                                fc=fc, ec=ec, lw=lw))
    if title:  # container: label sits as a title near the top edge
        ax.text(x + w/2, y + h - 2.4, label, ha="center", va="center", fontsize=fs,
                weight="bold", color=tc)
        return
    ax.text(x + w/2, y + h - (h*0.30 if sub else h/2), label, ha="center",
            va="center", fontsize=fs, weight="bold", color=tc)
    if sub:
        ax.text(x + w/2, y + h*0.28, sub, ha="center", va="center", fontsize=subfs, color="#333")

# ---- SoC outer boundary ----
ax.add_patch(FancyBboxPatch((3, 5), 66, 63, boxstyle="round,pad=0.3,rounding_size=2",
                            fc="#f7f9fc", ec="#2b3a55", lw=2.2))
ax.text(36, 65.5, "Classic ESP32 SoC (ESP32-D0WD-V3), no PSRAM", ha="center",
        va="center", fontsize=11.5, weight="bold", color="#2b3a55")

# CPU cores
box(7, 52, 26, 9, "Xtensa LX6 core 0", "#dbe8ff", sub="240 MHz, scalar (no vector unit)")
box(38, 52, 26, 9, "Xtensa LX6 core 1", "#dbe8ff", sub="240 MHz, scalar (no vector unit)")

# Internal SRAM (the star of the memory story) - container with title at top
box(7, 22, 57, 27, "Internal SRAM  (520 kB total)", "#eaf6ea", ec="#2f7d32", lw=1.7,
    fs=10.2, title=True)
# sub-regions inside SRAM (from measured heap_init)
box(9.5, 34.0, 25, 9.0, "DRAM 180 kB", "#ffffff", ec="#7aa77c", fs=8.6,
    sub="data / activations", subfs=7.6)
box(36.5, 34.0, 25, 9.0, "D/IRAM  14+111 kB", "#ffffff", ec="#7aa77c", fs=8.6,
    sub="data + instructions", subfs=7.6)
box(9.5, 23.5, 52, 8.6, "largest free block ~150 kB = TENSOR ARENA",
    "#fff2cc", ec="#c9a227", fs=8.8, sub="holds all int8 activations at inference", subfs=7.6)

# ROM + radio
box(7, 8, 26, 12, "Internal ROM", "#efeaf6", sub="boot / libraries")
box(38, 8, 26, 12, "WiFi / BT radio", "#f2e6e6", ec="#9c5b5b",
    sub="(disabled during inference)", tc="#7a3b3b")

# ---- external components ----
box(76, 40, 21, 15, "External flash", "#e9e9e9", ec="#555",
    sub="4 MB\nint8 model weights\n(const byte array)", subfs=8.0)
# no PSRAM marker
box(76, 16, 21, 13, "PSRAM", "#f6f6f6", ec="#b03030", lw=1.6, tc="#b03030",
    sub="ABSENT on this board", subfs=8.2)
ax.plot([77.5, 95.5], [17.5, 27.5], color="#b03030", lw=2.0)
ax.plot([77.5, 95.5], [27.5, 17.5], color="#b03030", lw=2.0)

# arrows: flash weights -> SoC (SRAM), radio note
ar = dict(arrowstyle="-|>", color="#444", lw=1.6, mutation_scale=14)
ax.add_patch(FancyArrowPatch((76, 47.5), (64.5, 40), connectionstyle="arc3,rad=-0.15", **ar))
ax.text(70.5, 49.5, "weights\n(flash, not SRAM)", ha="center", va="center", fontsize=7.8, color="#444")

ax.text(36, 1.7, "Weights live in external flash; only int8 activations occupy internal SRAM. "
        "There is no PSRAM.", ha="center", va="center", fontsize=8.6, color="#333")

fig.tight_layout(pad=0.4)
for ext in ("pdf", "png"):
    fig.savefig(f"fig_esp32_internal.{ext}", dpi=200, bbox_inches="tight")
print("wrote fig_esp32_internal.pdf / .png")
