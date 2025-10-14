export class Renderer {
    private device!: GPUDevice
    private context!: GPUCanvasContext
    private presentationFormat!: GPUTextureFormat
    private vertexShader!: GPUShaderModule
    private fragmentShader!: GPUShaderModule
    private pipeline!: GPURenderPipeline
    private renderPassDescriptor!: GPURenderPassDescriptor
    private canvas: HTMLCanvasElement

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas
    }

    public async init() {
        await this.getGPUDevice()
        this.configCanvas()
        this.loadShaders()
        this.configurePipeline()
        this.configureRenderPassDescriptor()
        this.render()
    }

    private async getGPUDevice() {
        var adapter = await navigator.gpu?.requestAdapter()
        const device = await adapter?.requestDevice()
        if (!device) {
            this.fail("Browser does not support WebGPU")
            return
        }

        this.device = device
    }

    private fail(msg: string) {
        document.body.innerHTML = `<H1>${msg}</H1>`
    }

    private configCanvas() {
        var context = this.canvas.getContext("webgpu")
        if (!context) {
            this.fail("Failed to get canvas context")
            return
        }
        this.context = context

        this.presentationFormat = navigator.gpu.getPreferredCanvasFormat()
        context.configure({
            device: this.device,
            format: this.presentationFormat
        })
    }

    private loadShaders() {
        this.loadVertexShader()
        this.loadFragmentShader()
    }

    private loadVertexShader() {
        this.vertexShader = this.device.createShaderModule({
            label: "Vertex Shader",
            code: /* wgsl */ `
            // data structure to store output of vertex function
            struct VertexOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f
            };

            // process the points of the triangle
            @vertex 
            fn vs(
                @builtin(vertex_index) vertexIndex : u32
            ) -> VertexOut {
                let pos = array(
                    vec2f(   0,  0.8),  // top center
                    vec2f(-0.8, -0.8),  // bottom left
                    vec2f( 0.8, -0.8)   // bottom right
                );

                let color = array(
                    vec4f(1.0, .0, .0, .0),
                    vec4f( .0, 1., .0, .0),
                    vec4f( .0, .0, 1., .0)
                );

                var out: VertexOut;
                out.pos = vec4f(pos[vertexIndex], 0.0, 1.0);
                out.color = color[vertexIndex];

                return out;
            }
         `
        })
    }

    private loadFragmentShader() {
        this.fragmentShader = this.device.createShaderModule({
            label: "Fragment Shader",
            code: /* wgsl */ `
            // data structure to input to fragment shader
            struct VertexOut {
                @builtin(position) pos: vec4f,
                @location(0) color: vec4f
            };

            // set the colors of the area within the triangle
            @fragment 
            fn fs(in: VertexOut) -> @location(0) vec4f {
                return in.color;
            }
            `
        })
    }

    private configurePipeline() {
        this.pipeline = this.device.createRenderPipeline({
            label: "Render Pipeline",
            layout: "auto",
            vertex: { module: this.vertexShader },
            fragment: {
                module: this.fragmentShader,
                targets: [{ format: this.presentationFormat }]
            }
        })
    }

    private configureRenderPassDescriptor() {
        this.renderPassDescriptor = {
            label: "Render Pass Description",
            colorAttachments: [{
                clearValue: [0.0, 0.8, 0.0, 1.0],
                loadOp: "clear",
                storeOp: "store",
                view: this.context.getCurrentTexture().createView()
            }]
        }
    }

    private render() {

        (this.renderPassDescriptor.colorAttachments as any)[0].view = this.context.getCurrentTexture().createView()

        const encoder = this.device.createCommandEncoder({ label: "render encoder" })

        const pass = encoder.beginRenderPass(this.renderPassDescriptor)
        pass.setPipeline(this.pipeline)
        pass.draw(6)
        pass.end()

        this.device.queue.submit([encoder.finish()])
    }
}
