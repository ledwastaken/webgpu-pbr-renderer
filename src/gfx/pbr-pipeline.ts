import { engine } from "../core/engine";
import vertexWGSL from "../shaders/pbr-vertex.wgsl?raw";
import fragmentWGSL from "../shaders/pbr-fragment.wgsl?raw";
import type { Mesh } from "../scene/mesh";

let startTime = performance.now();

class PBRPipeline {
    private pipeline!: GPURenderPipeline;
    private uniformBuffer!: GPUBuffer;
    private uniformBindGroup!: GPUBindGroup;
    private depthTexture!: GPUTexture;

    public init() {
        const layout = engine.device.createPipelineLayout({
            bindGroupLayouts: [
                engine.device.createBindGroupLayout({
                    entries: [
                        { binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }
                    ],
                }),
            ],
        });

        this.pipeline = engine.device.createRenderPipeline({
            layout,
            vertex: {
                module: engine.device.createShaderModule({ code: vertexWGSL }),
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
                module: engine.device.createShaderModule({ code: fragmentWGSL }),
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

        this.uniformBuffer = engine.device.createBuffer({
            size: 256,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.uniformBindGroup = engine.device.createBindGroup({
            layout: this.pipeline.getBindGroupLayout(0),
            entries: [{ binding: 0, resource: { buffer: this.uniformBuffer } }],
        });

        this.depthTexture = engine.device.createTexture({
            size: [800, 600],
            format: 'depth24plus',
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });
    }

    public draw(mesh: Mesh) {
        const now = (performance.now() - startTime) / 1000;
        const aspect = 800.0 / 600.0;
        const fov = Math.PI / 2;
        const near = 0.1;
        const far = 100;
        const proj = mat4_perspective(fov, aspect, near, far);
        const view = mat4_lookAt([1.5, 0, 1.5], [0, 0, 0], [0, 1, 0]);
        const model = mat4_rotationY(now * 0.3);

        engine.device.queue.writeBuffer(this.uniformBuffer, 0, new Float32Array(model));
        engine.device.queue.writeBuffer(this.uniformBuffer, 64, new Float32Array(view));
        engine.device.queue.writeBuffer(this.uniformBuffer, 128, new Float32Array(proj));

        const commandEncoder = engine.device.createCommandEncoder();
        const renderPass = commandEncoder.beginRenderPass({
            colorAttachments: [{
                view: engine.context.getCurrentTexture().createView(),
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
        engine.device.queue.submit([commandEncoder.finish()]);
    }
}

export let pbrPipeline = new PBRPipeline();

function mat4_perspective(fov: number, aspect: number, near: number, far: number): Float32Array {
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    return new Float32Array([
        f / aspect, 0, 0, 0,
        0, f, 0, 0,
        0, 0, (far + near) * nf, -1,
        0, 0, 2 * far * near * nf, 0,
    ]);
}

function mat4_lookAt(eye: number[], center: number[], up: number[]): Float32Array {
    const [ex, ey, ez] = eye;
    const [cx, cy, cz] = center;
    const [ux, uy, uz] = up;
    let zx = ex - cx, zy = ey - cy, zz = ez - cz;
    const zlen = Math.hypot(zx, zy, zz); zx /= zlen; zy /= zlen; zz /= zlen;
    let xx = uy * zz - uz * zy, xy = uz * zx - ux * zz, xz = ux * zy - uy * zx;
    const xlen = Math.hypot(xx, xy, xz); xx /= xlen; xy /= xlen; xz /= xlen;
    const yx = zy * xz - zz * xy, yy = zz * xx - zx * xz, yz = zx * xy - zy * xx;
    return new Float32Array([
        xx, yx, zx, 0,
        xy, yy, zy, 0,
        xz, yz, zz, 0,
        -(xx * ex + xy * ey + xz * ez),
        -(yx * ex + yy * ey + yz * ez),
        -(zx * ex + zy * ey + zz * ez),
        1,
    ]);
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
