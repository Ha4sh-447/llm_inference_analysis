#!/usr/bin/env python3
import sys
import subprocess
import os

def main():
    # Find the python executable
    python_exe = sys.executable
    
    # Path to run_performance.py
    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_script = os.path.join(script_dir, "run_performance.py")
    
    # Forward all command line arguments
    cmd = [python_exe, target_script] + sys.argv[1:]
    
    print(f"Running performance benchmark (filtering for Prefill & Decode metrics)...")
    
    # Launch the subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )
    
    # Stream stdout and filter lines
    for line in process.stdout:
        line_strip = line.strip()
        # Filter for lines containing prefill/decode metrics
        if any(keyword in line_strip for keyword in ["Prefill", "Decode"]):
            print(line_strip)
            
    process.wait()
    sys.exit(process.returncode)

if __name__ == "__main__":
    main()
