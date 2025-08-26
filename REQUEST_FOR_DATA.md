
# Request for Gesture Keyboard Validation Data

## What We Need
We need real human gesture keyboard traces to validate our synthetic generation system.

## Data Format Required
Each trace should contain:
1. **Word**: The intended word (ground truth)
2. **Trace Points**: Array of (x, y, timestamp) coordinates
3. **User ID**: Anonymous identifier for the user
4. **Device Info**: Screen size, DPI, keyboard layout used

## How to Collect
1. Use any gesture keyboard app that can export traces
2. Type 20-50 common words
3. Export the trace data in JSON or CSV format
4. Ensure consistent sampling (>60Hz preferred)

## Privacy
- No personal information needed
- User IDs can be anonymous (user1, user2, etc.)
- Only the trace coordinates and intended words are required

## Suggested Words
- Common: the, and, you, that, with
- Medium: hello, world, computer, typing
- Long: keyboard, beautiful, wonderful

## Data Submission
Please save traces in this format:
```json
{
  "traces": [
    {
      "word": "hello",
      "points": [[x1,y1,t1], [x2,y2,t2], ...],
      "user": "user1",
      "device": "phone"
    }
  ]
}
```

## Contact
Submit data or questions to: [project repository]
