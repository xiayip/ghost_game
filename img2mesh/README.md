# img2mesh

ROS 2 bridge from a prepared portrait to a textured Tripo GLB model.

In the Ghost Game launch, `flux_image_editor` publishes one prepared portrait
on `/ghost/reconstruction/image`. Every new image automatically starts one
Tripo image-to-model task. Progress is published as JSON on
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
