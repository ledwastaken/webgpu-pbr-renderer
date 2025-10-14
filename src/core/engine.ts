import vertexWGSL from "../shaders/vertex.wgsl?raw";
import fragmentWGSL from "../shaders/fragment.wgsl?raw";

export class Engine {
    private static instance: Engine
    private device!: GPUDevice
    private context!: GPUCanvasContext
    private pipeline!: GPURenderPipeline

    private constructor() {
    }

    static getSingleton(): Engine {
        this.instance = this.instance || new Engine();

        return this.instance;
    }

    public async init(canvas: HTMLCanvasElement) {
        const adapter = await navigator.gpu?.requestAdapter()
        const device = await adapter?.requestDevice()
        if (!device)
            throw new Error("Browser does not support WebGPU")

        const context = canvas.getContext("webgpu")
        const format = navigator.gpu.getPreferredCanvasFormat();
        if (!context)
            throw new Error("Failed to get canvas context")

        context.configure({
            device: device,
            format: format
        })

        const pipeline = device.createRenderPipeline({
            layout: "auto",
            vertex: {
                module: device.createShaderModule({ code: vertexWGSL }),
                entryPoint: "main",
            },
            fragment: {
                module: device.createShaderModule({ code: fragmentWGSL }),
                entryPoint: "main",
                targets: [{ format }],
            },
            primitive: { topology: "triangle-list" },
        });

        this.device = device
        this.context = context
        this.pipeline = pipeline
    }

    public loop() {
        const encoder = this.device.createCommandEncoder();
        const pass = encoder.beginRenderPass({
            colorAttachments: [{
                view: this.context.getCurrentTexture().createView(),
                clearValue: { r: 0.5, g: 0.05, b: 0.08, a: 1 },
                loadOp: "clear",
                storeOp: "store",
            }],
        });

        pass.setPipeline(this.pipeline);
        pass.draw(3);
        pass.end();
        this.device.queue.submit([encoder.finish()]);

        requestAnimationFrame(() => this.loop())
    }
}
