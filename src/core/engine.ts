import { Mesh } from "../scene/mesh";
import { pbrPipeline } from "../gfx/pbr-pipeline";

class Engine {
    canvas: HTMLCanvasElement;
    device!: GPUDevice;
    context!: GPUCanvasContext;
    format!: GPUTextureFormat;

    meshes: Array<Mesh>;

    constructor() {
        this.canvas = document.getElementById("GLCanvas") as HTMLCanvasElement;
        this.meshes = new Array<Mesh>();

        const devicePixelRatio = window.devicePixelRatio || 1;
        this.canvas.width = this.canvas.clientWidth * devicePixelRatio;
        this.canvas.height = this.canvas.clientHeight * devicePixelRatio;
    }

    async init() {
        const adapter = await navigator.gpu?.requestAdapter();
        if (!adapter)
            throw new Error("Browser does not support WebGPU");

        this.device = await adapter?.requestDevice();
        if (!this.device)
            throw new Error("Browser does not support WebGPU");

        this.context = this.canvas.getContext("webgpu")!;
        if (!this.context)
            throw new Error("Failed to get canvas context");

        this.format = navigator.gpu.getPreferredCanvasFormat();
        this.context.configure({
            device: this.device,
            format: this.format
        });

        pbrPipeline.init();
    }

    async loop() {
        for (let mesh of this.meshes) {
            pbrPipeline.draw(mesh);
        }

        requestAnimationFrame(() => this.loop());
    }
}

export let engine = new Engine();
