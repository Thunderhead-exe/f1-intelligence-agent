# Track Status Codes

FastF1 track status values can contain one or more encoded status digits. Interpret them cautiously and use race-control messages when possible.

Common codes:

- 1: track clear or green.
- 2: yellow flag.
- 4: safety car.
- 5: red flag.
- 6: virtual safety car deployed.
- 7: virtual safety car ending.

Multiple digits can appear when statuses occur close together. Non-green status should usually be treated as session context, especially when multiple drivers show unusual laps at the same time.

