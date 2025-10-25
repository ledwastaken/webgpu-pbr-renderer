import vertexWGSL from "../shaders/skybox-vertex.wgsl?raw";
import fragmentWGSL from "../shaders/skybox-fragment.wgsl?raw";
import Engine from "../core/engine";

class SkyboxPipeline {
    pipeline!: GPURenderPipeline;
    uniformBuffer!: GPUBuffer;
    uniformBindGroup!: GPUBindGroup;
    depthTexture!: GPUTexture;

    cubeVertexBuffer!: GPUBuffer;

    public async init() {
        const layout = Engine.device.createPipelineLayout({
            bindGroupLayouts: [
                Engine.device.createBindGroupLayout({
                    entries: [
                        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }
                    ],
                }),
                Engine.device.createBindGroupLayout({
                    entries: [
                        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                        {
                            binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {
                                viewDimension: 'cube',
                            }
                        },
                    ],
                }),
            ],
        });

        this.pipeline = Engine.device.createRenderPipeline({
            layout,
            vertex: {
                module: Engine.device.createShaderModule({ code: vertexWGSL }),
                entryPoint: "main",
                buffers: [
                    {
                        arrayStride: (4) * 4,
                        attributes: [
                            { shaderLocation: 0, offset: 0, format: "float32x4" },
                        ],
                    },
                ],
            },
            fragment: {
                module: Engine.device.createShaderModule({ code: fragmentWGSL }),
                entryPoint: "main",
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
            },
            primitive: { topology: "triangle-list", cullMode: "none" },
            multisample: {
                count: 4,
            },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        const vertices = new Float32Array([
            1, -1, 1, 1,
            -1, -1, 1, 1,
            -1, -1, -1, 1,
            1, -1, -1, 1,
            1, -1, 1, 1,
            -1, -1, -1, 1,

            1, 1, 1, 1,
            1, -1, 1, 1,
            1, -1, -1, 1,
            1, 1, -1, 1,
            1, 1, 1, 1,
            1, -1, -1, 1,

            -1, 1, 1, 1,
            1, 1, 1, 1,
            1, 1, -1, 1,
            -1, 1, -1, 1,
            -1, 1, 1, 1,
            1, 1, -1, 1,

            -1, -1, 1, 1,
            -1, 1, 1, 1,
            -1, 1, -1, 1,
            -1, -1, -1, 1,
            -1, -1, 1, 1,
            -1, 1, -1, 1,

            1, 1, 1, 1,
            -1, 1, 1, 1,
            -1, -1, 1, 1,
            -1, -1, 1, 1,
            1, -1, 1, 1,
            1, 1, 1, 1,

            1, -1, -1, 1,
            -1, -1, -1, 1,
            -1, 1, -1, 1,
            1, 1, -1, 1,
            1, -1, -1, 1,
            -1, 1, -1, 1,
        ]);

        this.cubeVertexBuffer = Engine.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        new Float32Array(this.cubeVertexBuffer.getMappedRange()).set(vertices);
        this.cubeVertexBuffer.unmap();

        this.uniformBuffer = Engine.device.createBuffer({
            size: 128,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.uniformBindGroup = Engine.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
        });
        this.depthTexture = Engine.device.createTexture({
            size: [800, 600],
            sampleCount: 4,
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    public async draw(commandEncoder: GPUCommandEncoder, textureBindGroup: GPUBindGroup, view: Float32Array, proj: Float32Array) {
        Engine.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array(view));
        Engine.device.queue.writeBuffer(this.uniformBuffer, 64, new Float32Array(proj));

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: Engine.msaaColorTexture.createView(),
                resolveTarget: Engine.context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 0.0 },
                loadOp: "clear",
                storeOp: "store",
            }],
            depthStencilAttachment: {
                view: this.depthTexture.createView(),
                depthClearValue: 1.0,
                depthLoadOp: 'clear',
                depthStoreOp: 'store',
            },
        });

        renderPass.setPipeline(this.pipeline);
        renderPass.setBindGroup(0, this.uniformBindGroup);
        renderPass.setBindGroup(1, textureBindGroup);
        renderPass.setVertexBuffer(0, this.cubeVertexBuffer);
        renderPass.draw(36);
        renderPass.end();
    }
}

export default new SkyboxPipeline();
