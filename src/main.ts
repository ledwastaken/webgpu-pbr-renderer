import { engine } from "./core/engine"
import { Sphere } from "./scene/sphere"

await engine.init();

let ball = new Sphere();
engine.meshes.push(ball);

await engine.loop();
