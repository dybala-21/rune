# Records the honest-agent demo into docs/assets/honest-agent-demo.{gif,png}.
#
#   brew install vhs        # needs ttyd + ffmpeg
#   vhs scripts/demo_honest_agent.tape     (run from the repo root)
#
# Requires ollama serving qwen2.5-coder:7b; an ANTHROPIC_API_KEY in
# ~/.rune/.env enables the escalation ("rescued") half. The minutes of
# model work are time-cut with Hide/Show — every byte of output is from
# the real run happening inside the recording.
Output docs/assets/honest-agent-demo.gif

Set FontSize 15
Set Width 1000
Set Height 660
Set Padding 12
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

Type "uv run python scripts/demo_honest_agent.py"
Enter
Hide
Wait+Screen@20m /reproduce:/
Show
Sleep 8s
Screenshot docs/assets/honest-agent-demo.png
Sleep 2s
