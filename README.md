# Spawn Along Curve
Spawn objects along a bezier curve

## Spawn Type: Count
Spawn a specific number of objects along a bezier curve.
 - Options:
   - Spawn objects at the end of the curve
## Spawn Type: Distance
Spawn objects with a set distance between them along a bezier curve

## To Use
1. Create a bezier curve
    - Only bezier curves are supported currently
2. With the bezier curve selected changing any of the settings or hitting the 'Spawn Along Curve' button will spawn a cube along the curve
![Spawn objects use example](/exts/tvfx.curve.spawnOnCurve/docs/spawn_example.gif)
3. To change the object that is being spawned:
    - Drag an object from the stage panel into the 'Spawn Object' input area
     - That object's path should now appear in the 'Spawn Object' input area
    - Reselect your curve and change any of the settings/hit the 'Spawn Along Curve' button to see the new object along the curve.
![Spawn objects use example](/exts/tvfx.curve.spawnOnCurve/docs/change_spawn_object_example.gif)


## Known Limitations
 - Selecting an object when dragging to update the Spawn Object stops any automatic updates since the curve is no longer selected
 - Spawned object location doesn't update when editing the curve
 - Tight curves give weird results when spacing objects
 - Only Bezier Curves are supported

## Adding This Extension
To add this extension to your Omniverse app:
1. Go into: Extension Manager -> Gear Icon -> Extension Search Path
2. Add this as a search path: `git://github.com/True-VFX/spawn-on-curve.git?branch=main&dir=exts`