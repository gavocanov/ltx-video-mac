#!/usr/bin/env python3
"""Test harness: extract the Swift-generated Python script from LTXBridge.swift,
substitute Swift interpolations with representative values, then compile + run --help."""
import re, subprocess, sys, tempfile, os

SRC = 'LTXVideoGenerator/Sources/Services/LTXBridge.swift'
lines = open(SRC).read().split('\n')
# Main generation script body: lines 276..582 (0-indexed 275..581)
body = '\n'.join(lines[275:582])

# Representative values for each Swift interpolation.
vals = {
    'logFile': '/tmp/ltx_generation.log',
    'modelRepo': "'black-forest-labs/FLUX.1-dev'",
    'textEncoderRepo': "'google/t5-v1_1-xxl'",
    'useLocalMlxVideoRepoPref ? "True" : "False"': 'False',
    'escapedImagePath': '',
    'escapedPrompt': 'a cat',
    'escapedNegativePrompt': '',
    'genWidth': '512',
    'genHeight': '320',
    'params.numFrames': '33',
    'seed': '42',
    'request.disableAudio ? "True" : "False"': 'False',
    'resourcesPath': "'/tmp/resources'",
    'params.fps': '24',
    'params.numInferenceSteps': '30',
    'params.guidanceScale': '3.0',
    'outputPath': "'/tmp/out.mp4'",
    'effectiveTilingMode': 'auto',
    'previewEvery': '3',
    'previewDir': "'/tmp/preview'",
    'params.imageStrength': '0.5',
    'saveAudioTrackSeparately ? "True" : "False"': 'False',
}

def substitute(text):
    def repl(m):
        key = m.group(1)
        if key in vals:
            return vals[key]
        raise SystemExit(f"UNKNOWN INTERPOLATION: {key}")
    return re.sub(r'\\\(([^)]*)\)', repl, text)

py = substitute(body)

# Assert no leftover Swift interpolation artifacts remain in the generated Python.
leftover = re.findall(r'\\\([^)]*\)', py)
if leftover:
    print("LEFTOVER INTERPOLATION ARTIFACTS:", leftover)
    sys.exit(1)
print("NO LEFTOVER INTERPOLATION ARTIFACTS")

# Verify the preview args are wired into the cmd.
if '--preview-every' in py and '--preview-dir' in py:
    print("PREVIEW ARGS PRESENT IN CMD")
else:
    print("WARNING: preview args missing from cmd")

# Write to temp file
fd, path = tempfile.mkstemp(suffix='.py')
os.write(fd, py.encode())
os.close(fd)

# 1) Compile check
try:
    compile(py, path, 'exec')
    print("COMPILE OK")
except SyntaxError as e:
    print(f"SYNTAX ERROR: {e}")
    # print surrounding lines
    ln = e.lineno or 0
    for i in range(max(0,ln-3), min(len(py.split('\n')), ln+2)):
        print(f"{i+1}: {py.split(chr(10))[i]}")
    sys.exit(1)

# 2) Run --help to verify argparse works (use a python with mlx if available)
pyexe = sys.argv[1] if len(sys.argv) > 1 else sys.executable
r = subprocess.run([pyexe, path, '--help'], capture_output=True, text=True)
print("--- --help exit:", r.returncode)
if r.returncode != 0:
    print(r.stderr[-2000:])
else:
    # show preview args present
    for line in r.stdout.split('\n'):
        if 'preview' in line.lower():
            print(line.strip())

os.unlink(path)
