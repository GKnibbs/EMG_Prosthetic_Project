# Config.py: Central config for dataset windowing

# Trimming parameters (ms) to reduce ambiguous transition frames at start/end of subject streams
# Trimming helps remove frames where gesture transitions may not be clean, improving training quality
trim_head_ms = 0  # Default: no trim
trim_tail_ms = 0  # Default: no trim

# ...other config options...
