# Wifi and CSI Drone navigation

## Research & Reflections

### SpotFi — Kotaru et al.
SpotFi is an indoor localization system built on commodity WiFi
- No extra hardware
-  ~40 cm median accuracy ( better than 6-8 antennas or rotating antenna.)

**How it works:**
Standard WiFi localization either uses RSSI (signal strength), which tops out at 2-4m accuracy, or Angle of Arrival (AoA) estimation, which needs lots of antennas to work well.  SpotFi instead of estimating AoA alone, it jointly estimates AoA *and* Time of Flight (ToF) for each signal path. 
ToF introduces phase shifts across subcarriers (not just across antennas), so by combining subcarrier + antenna data, SpotFi effectively creates a virtual sensor array much larger than the 3 physical antennas. This is fed into a modified MUSIC algorithm (a super-resolution spectral estimation technique) which can then disentangle multiple reflected paths even with limited hardware.

Now, we need to figure out how to identify direct path and indirect path. We do this by comparing statbility, direct path are much much stable, given enough datatset we can identify the stable signal.
But the only issue is drone is constatly moving and vibrating identifyting a stable path is not possible when device itself is under motion.
Finally, it combines direct-path AoA estimates from multiple APs with RSSI measurements to triangulate the target's location.

**What makes sense:**
The insight of using AOA and TOF with reflection is genuinely good. It uses elredy present hardware so no additional cost. The direct path likelihood method is also practical using consistency across packets as a proxy for "this is the real path" is a reasonable heuristic in a static environment.

**Where it breaks down for a moving drone:**
SpotFi's whole direct path identification strategy relies on AoA and ToF estimates being *stable across packets*. That stability assumption completely falls whn the receiver is moving. A drone in flight is constantly changing position, tilting, and vibrating. What SpotFi would see is every path looking "unstable", making it impossible to distinguish the direct path from reflections.

Beyond that, the multipath profile itself changes as the drone moves — reflectors that were relevant a second ago might not be relevant now. The environment is no longer static relative to the receiver, which is the core assumption the whole system is built on.

---

### WiFi Sensing Survey — Ma et al.


---

### DeepFi — Wang et al.


---

## Experimentation

---

## Code
```/code```
