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

CSI is a 3D matrix of complex values (amplitude + phase) across transmit antennas, receive antennas, and subcarriers. It captures how the wireless channel changes due to effects form reflections, diffraction, absorption, scattering. It also changes when people or objects move in the environment.
But he raw measured CSI isn't clean though. It has phase offsets . A lot of the signal processing work in this space is just dealing with those before you can extract anything meaningful.

**The full application landscape:**

The survey groups CSI applications into three buckets:

- **Detection** (binary): Is someone there? Did someone fall? Is there motion?
  Usually threshold based or simple one-class SVM. Doesn't need super clean data.
- **Recognition** (multi-class): What activity? Which gesture? Who is this person?
  Almost always learning-based — SVM, kNN, DTW, CNN, LSTM.
- **Estimation** (continuous values): Where exactly? What direction? Breathing rate?
  Almost always modeling-based — AoA, ToF, Doppler, Fresnel Zone model, MUSIC. These are the most sensitive to noise and require the most signal processing.

Localization falls in the estimation bucket, which is the hardest one. It needs accurate phase and timing information, which is exactly what gets corrupted by hardware offsets and relevant to us by the receiver moving around.

**What makes sense:**
The taxonomy is useful. The paper makes clear that the harder the task (detection → recognition → estimation), the more you depend on clean, well-calibrated signals, and the more fragile things get when conditions change. Localization is at the hardest end.

The survey also notes something directly relevant: most of these systems assume the WiFi device and surrounding environment are either both static, or at most one is moving (a person walking past fixed APs). Nobody has really tackled the case where the *receiver itself* is moving through space.

**Where it breaks down for a moving drone:**

The survey actually mentions drones briefly — as a future opportunity for cross-device WiFi sensing. But that optimism glosses over a real problem. Basically every localization and estimation technique surveyed assumes a static receiver. The
signal processing pipeline (phase offset removal, AoA/ToF estimation, clustering) is designed around the idea that the channel changes because the *environment* changes, not because the receiver itself is flying through it.

On a drone, you get Doppler shifts from the drone's own motion on top of any environmental multipath, vibration from the motors introducing high-frequency noise into the CSI measurements, and the antenna orientation changing continuously as the drone tilts. The signal you're trying to extract and the noise you're trying to remove become very hard to separate.


---

### DeepFi — Wang et al.


---

## Experimentation

---

## Code
```/code```
