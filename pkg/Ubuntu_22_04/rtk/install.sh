curl -fsSL https://raw.githubusercontent.com/rtk-ai/rtk/refs/heads/master/install.sh | sh
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.paths

# rtk init -g                     # Claude Code / Copilot (default)
# rtk init -g --gemini            # Gemini CLI
# rtk init -g --codex             # Codex (OpenAI)
# rtk init -g --agent cursor      # Cursor
# rtk init -g --agent windsurf    # Windsurf
# rtk init --agent cline          # Cline / Roo Code
# rtk init --agent kilocode       # Kilo Code
# rtk init --agent antigravity    # Google Antigravity
# rtk init -g --agent pi          # Pi
# rtk init --agent hermes         # Hermes
