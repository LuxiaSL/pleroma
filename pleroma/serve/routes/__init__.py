"""The loom's routes, by family, as mixins of ``pleroma.serve.context.ServeContext``.

chat      /chat /undo /truncate /reroll /edit
loom      /loom
probe     /probe (+ the probe run /loom can trigger)
wear      /wear /wear_code
sessions  persistence at the request boundary, /restore, the GET payloads
"""
