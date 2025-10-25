import Engine from "./core/engine"
import { Sphere } from "./scene/sphere"
import { Vector3 } from "./types/vector3";

await Engine.init();

let ball = new Sphere(1, 64, 64);
ball.cframe.add(new Vector3(0, 0, -2));

Engine.meshes.push(ball);

await Engine.loop();
