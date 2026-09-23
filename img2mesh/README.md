# img2mesh

ROS 2 bridge from a prepared portrait to a textured Tripo GLB model.

In the Ghost Game launch, `flux_image_editor` publishes the exact generated
PNG on `/ghost/reconstruction/image_png`. `img2mesh` uploads those PNG bytes
unchanged, and every new image automatically starts one Tripo image-to-model
task. Progress is published as JSON on
`/ghost/reconstruction/mesh_status`; the completed signed GLB URL is published
on `/ghost/reconstruction/model_url` and is consumed by the Web monitor.

Set the private API key in the process environment:

```bash
export TRIPO_API_KEY='...'
```

Alternatively copy `config/local_api.example.yaml` to
`config/local_api.yaml`. That file is ignored by Git and must not be committed.

Build and launch through the unified application:

```bash
colcon build --symlink-install --packages-up-to ghost_game
source install/setup.bash
ros2 launch ghost_game ghost_game.launch.py
```

The detailed `ghost_game_interfaces/msg/MeshResult` stream remains available
on `/ghost/reconstruction/mesh_result`. The JSON status deliberately omits the
signed model URL; only the dedicated URL topic carries it.

The Ghost Game configuration uses Tripo's low-latency texture path:
`P1-20260311`, 4,000 faces, texture v3.5 `fast`, no PBR and no delight pass.
The generic `img2mesh.yaml` keeps the standard-quality defaults. This makes
the event profile reversible without changing the standalone package.
