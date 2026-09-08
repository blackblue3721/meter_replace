---
name: colored-cube
description: Detect or pinch-and-release a red, yellow, green, or blue cube using the local D435, CR5, and EPG40-100.
---

# Colored cube detection

Normalize the requested color to `red`, `yellow`, `green`, or `blue`, then run:

```bash
/home/haoran/Arm/openclaw_ball/openclaw/detect_color.sh <color>
```

Report the returned pixel, camera XYZ, and robot XYZ. If `detections` is empty, say the cube was not found.
Only report values present in the tool response. Do not infer relative position, distance, or ordering versus another color unless that color was detected in the same response.

Never claim that planning, motion, grasping, or placement happened unless the tool response explicitly reports it. A `locked` response is a safety refusal and must be relayed to the user.

For an explicit request to grasp, pinch, or grab a cube, run:

```bash
/home/haoran/Arm/openclaw_ball/openclaw/grasp_color.sh <color>
```

This performs one fresh camera capture, freezes the 3D target, plans the full collision-checked cycle,
pinches and releases the cube, retreats, and returns to the default safe pose. Only report success when
the response has `status: completed`. Relay any locked/error response exactly; never retry motion.
