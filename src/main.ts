import { Engine } from "./core/engine";

const canvas = document.getElementById("GLCanvas") as HTMLCanvasElement
const engine = Engine.getSingleton()

await engine.init(canvas)
engine.loop()
