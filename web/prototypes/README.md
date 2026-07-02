# RUNE UI prototypes

Self-contained HTML mockups that are the **design source of truth** for the
`feat/adaptive-workbench-ui` work. Open any file directly in a browser (no build,
no external assets — everything is inline).

| File | What it shows |
|------|---------------|
| `identity.html` | Full identity + app screens: glacier-on-basalt palette, line icons, typographic craft, first-run onboarding, verify / Guardian / memory pillars, and the reactive 8-bit ice-wolf mascot. |
| `chat-workbench.html` | The **mixed-session** interaction model: the conversation is the spine; a coding task opens a terminal-flavored **workbench panel** beside it (Cursor / Canvas style). Task cards link to their workbench; collapse/reopen any time. Mode is contextual, not a global toggle. |
| `pixel-wolf.html` | The mascot studio: the sitting ice-wolf with a **brow rune** that colour-codes state (idle / working / passed / failed / context-warning), size ramp, and blink/bob. |

## Direction (locked)

- **Brand:** friendly 8-bit ice-wolf mascot (crafted, not the AI-generated raster),
  brow rune, on a restrained glacier-on-basalt dark palette. Anti-AI-slop:
  line icons (no emoji), no neon/glow/gradient buttons, elevation by tone.
- **Form factor:** one conversation; chat bubbles for general work, a terminal
  workbench that opens for coding tasks. Never a whole-app mode swap.
- **Switch logic:** driven by the backend LLM classifier (`classify_goal`), not
  keyword matching. Reversible via an in-thread banner / task card.

## Implementation status on this branch

- [x] Design tokens landed in `web/src/styles/globals.css` (glacier palette, no Inter,
      no colored glow, wolf `--rune-*` state colors).
- [ ] Pixel-wolf React component (sprite renderer + state).
- [ ] Chat thread + collapsible workbench layout.
- [ ] Wire `classify_goal` → workbench open + SSE → pet state.

The pixel-wolf sprite map and renderer live inline in `pixel-wolf.html` /
`chat-workbench.html` — lift them into the component in the next step.
