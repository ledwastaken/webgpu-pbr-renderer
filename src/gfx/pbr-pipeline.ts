import Engine from "../core/engine";
import vertexWGSL from "../shaders/pbr-vertex.wgsl?raw";
import fragmentWGSL from "../shaders/pbr-fragment.wgsl?raw";
import type { Mesh } from "../scene/mesh";

class PBRPipeline {
    public pipeline!: GPURenderPipeline;
    private uniformBuffer!: GPUBuffer;
    private uniformBindGroup!: GPUBindGroup;

    private depthTexture!: GPUTexture;
    private albedoSampler!: GPUSampler;
    private albedoTexture!: GPUTexture;
    private roughnessSampler!: GPUSampler;
    private roughnessTexture!: GPUTexture;
    private textureBindGroup!: GPUBindGroup;

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
                        { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: "uniform" } }
                    ],
                }),
                Engine.device.createBindGroupLayout({
                    entries: [
                        { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                        { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
                        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: {} },
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
            multisample: {
                count: 4,
            },
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

        [this.roughnessTexture, this.roughnessSampler] = await this.loadTexture("texture/Metal046B_2K-JPG_Roughness.jpg");
        [this.albedoTexture, this.albedoSampler] = await this.loadTexture("texture/Metal046B_2K-JPG_Color.jpg");

        this.textureBindGroup = Engine.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(2),
            entries: [
                { binding: 0, resource: this.albedoSampler },
                { binding: 1, resource: this.albedoTexture.createView() },
                { binding: 2, resource: this.roughnessSampler },
                { binding: 3, resource: this.roughnessTexture.createView() },
            ]
        });
        this.depthTexture = Engine.device.createTexture({
            size: [800, 600],
            sampleCount: 4,
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    private async loadTexture(url: string): Promise<[GPUTexture, GPUSampler]> {
        const img = document.createElement('img');

        img.src = url;
        await img.decode();

        let imageBitmap = await createImageBitmap(img);
        let texture = Engine.device.createTexture({
            size: [imageBitmap.width, imageBitmap.height, 1],
            format: 'rgba8unorm',
            usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
        });
        let sampler = Engine.device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

        Engine.device.queue.copyExternalImageToTexture(
            { source: imageBitmap },
            { texture: texture },
            [imageBitmap.width, imageBitmap.height]
        );

        return [texture, sampler];
    }

    public draw(commandEncoder: GPUCommandEncoder, mesh: Mesh, view: Float32Array, proj: Float32Array, fragmentBindGroup: GPUBindGroup, skyboxBindGroup: GPUBindGroup) {
        const model = mat4_rotationY(0);

        Engine.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array(model));
        Engine.device.queue.writeBuffer(this.uniformBuffer, 64, new Float32Array(view));
        Engine.device.queue.writeBuffer(this.uniformBuffer, 128, new Float32Array(proj));

        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: Engine.msaaColorTexture.createView(),
                resolveTarget: Engine.context.getCurrentTexture().createView(),
                clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 },
                loadOp: "load",
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
        renderPass.setBindGroup(1, fragmentBindGroup);
        renderPass.setBindGroup(2, this.textureBindGroup);
        renderPass.setBindGroup(3, skyboxBindGroup);
        renderPass.drawIndexed(mesh.indicesCount);
        renderPass.end();
    }
}

function mat4_rotationY(angle: number): Float32Array {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return new Float32Array([
        c, 0, s, 0,
        0, 1, 0, 0,
        -s, 0, c, 0,
        0, 0, 0, 1,
    ]);
}

export default new PBRPipeline();
