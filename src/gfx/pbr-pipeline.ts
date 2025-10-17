import * as Engine from "../core/engine";
import vertexWGSL from "../shaders/pbr-vertex.wgsl?raw";
import fragmentWGSL from "../shaders/pbr-fragment.wgsl?raw";
import type { Mesh } from "../scene/mesh";

export class PBRPipeline {
    private pipeline: GPURenderPipeline;
    private uniformBuffer: GPUBuffer;
    private uniformBindGroup: GPUBindGroup;
    private depthTexture: GPUTexture;

    public constructor() {
        const layout = Engine.device.createPipelineLayout({
            bindGroupLayouts: [
                Engine.device.createBindGroupLayout({
                    entries: [
                        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }
                    ],
                }),
                // Engine.device.createBindGroupLayout({
                //     entries: [
                //         { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                //         { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
                //         { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                //         { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
                //     ],
                // }),
            ],
        });

        this.pipeline = Engine.device.createRenderPipeline({
            layout,
            vertex: {
                module: Engine.device.createShaderModule({ code: vertexWGSL }),
                entryPoint: "main",
                buffers: [
                    {
                        arrayStride: 44,
                        attributes: [
                            { shaderLocation: 0, offset: 0, format: "float32x3" },
                            { shaderLocation: 1, offset: 12, format: "float32x3" },
                            { shaderLocation: 2, offset: 24, format: "float32x2" },
                            { shaderLocation: 3, offset: 32, format: "float32x3" },
                        ],
                    },
                ],
            },
            fragment: {
                module: Engine.device.createShaderModule({ code: fragmentWGSL }),
                entryPoint: "main",
                targets: [{ format: navigator.gpu.getPreferredCanvasFormat() }],
            },
            primitive: { topology: "triangle-list", cullMode: "front" },
            depthStencil: {
                format: "depth24plus",
                depthWriteEnabled: true,
                depthCompare: "less",
            },
        });

        this.uniformBuffer = Engine.device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.uniformBindGroup = Engine.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
        });

        this.depthTexture = Engine.device.createTexture({
            size: [800, 600],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    public draw(mesh: Mesh) {
        const commandEncoder = Engine.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: Engine.context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 },
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
        renderPass.setVertexBuffer(0, mesh.vertexBuffer);
        renderPass.setIndexBuffer(mesh.indexBuffer, "uint16");
        renderPass.setBindGroup(0, this.uniformBindGroup);
        renderPass.drawIndexed(mesh.indicesCount);
        renderPass.end();
        Engine.device.queue.submit([commandEncoder.finish()]);
    }
}
