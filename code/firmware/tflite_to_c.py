#!/usr/bin/env python3
"""Emit a C source array for a .tflite model (xxd -i replacement, fixed symbol name).
Usage: python tflite_to_c.py <model.tflite> <out.cc>  [symbol]
Default symbol is model_data_tflite, matching csi_bench/main/main.cc.
"""
import sys
from pathlib import Path
src=Path(sys.argv[1]); out=Path(sys.argv[2])
sym=sys.argv[3] if len(sys.argv)>3 else "model_data_tflite"
b=src.read_bytes()
lines=[f"// auto-generated from {src.name} ({len(b)} bytes)",
       "// 'extern const' forces external linkage so main.cc can see the symbol",
       "// (a plain 'const' global has internal linkage in C++).",
       f"extern const unsigned char {sym}[];",
       f"extern const unsigned int {sym}_len;",
       f"const unsigned char {sym}[] = {{"]
for i in range(0,len(b),16):
    lines.append("  "+", ".join(f"0x{x:02x}" for x in b[i:i+16])+",")
lines.append("};")
lines.append(f"const unsigned int {sym}_len = {len(b)};")
out.write_text("\n".join(lines)+"\n")
print(f"wrote {out} symbol={sym} bytes={len(b)}")
