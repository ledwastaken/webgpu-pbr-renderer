import vertexWGSL from "../shaders/vertex.wgsl?raw";
import fragmentWGSL from "../shaders/fragment.wgsl?raw";
import * as Geometry from "../misc/geometry";

let canvas: HTMLCanvasElement;
let device: GPUDevice;
let context: GPUCanvasContext;
let format: GPUTextureFormat;
let pipeline: GPURenderPipeline;
let depthTexture: GPUTexture;

let vertexBuffer: GPUBuffer;
let indexBuffer: GPUBuffer;
let uniformBuffer: GPUBuffer;
let uniformBindGroup: GPUBindGroup;
let startTime = performance.now();

let texture: GPUTexture;
let sampler: GPUSampler;
let textureBindGroup: GPUBindGroup;

const sphere = Geometry.generateSphere();

const vertices = sphere.vertices;
const indices = sphere.indices;

export async function init() {
    canvas = document.getElementById("GLCanvas") as HTMLCanvasElement;
    const adapter = await navigator.gpu?.requestAdapter();
    if (!adapter)
        throw new Error("Browser does not support WebGPU");

    device = await adapter?.requestDevice();
    if (!device)
        throw new Error("Browser does not support WebGPU");

    context = canvas.getContext("webgpu")!;
    if (!context)
        throw new Error("Failed to get canvas context");

    const devicePixelRatio = window.devicePixelRatio || 1;

    canvas.width = canvas.clientWidth * devicePixelRatio;
    canvas.height = canvas.clientHeight * devicePixelRatio;

    format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
        device: device,
        format: format
    });

    depthTexture = device.createTexture({
        size: [canvas.width, canvas.height],
        format: 'depth24plus',
        usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });

    vertexBuffer = device.createBuffer({
        size: vertices.byteLength,
        usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(vertices);
    vertexBuffer.unmap();

    indexBuffer = device.createBuffer({
        size: indices.byteLength,
        usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
        mappedAtCreation: true,
    });
    new Uint16Array(indexBuffer.getMappedRange()).set(indices);
    indexBuffer.unmap();

    uniformBuffer = device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    const url = import.meta.env.BASE_URL + "texture/Metal046B_2K-JPG_Color.jpg";
    console.log(url)
    const img = document.createElement('img');

    img.src = url;
    await img.decode();

    const imageBitmap = await createImageBitmap(img);
    texture = device.createTexture({
        size: [imageBitmap.width, imageBitmap.height, 1],
        format: 'rgba8unorm',
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });

    sampler = device.createSampler({ magFilter: 'linear', minFilter: 'linear' });

    device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture },
        [imageBitmap.width, imageBitmap.height]
    );

    const pipelineLayout = device.createPipelineLayout({
        bindGroupLayouts: [
            device.createBindGroupLayout({
                entries: [{ binding: 0, visibility: GPUShaderStage.VERTEX, buffer: { type: "uniform" } }],
            }),
            device.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.FRAGMENT, sampler: {} },
                    { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: {} },
                ],
            }),
        ],
    });

    pipeline = device.createRenderPipeline({
        layout: pipelineLayout,
        vertex: {
            module: device.createShaderModule({ code: vertexWGSL }),
            entryPoint: "main",
            buffers: [
                {
                    arrayStride: 8 * 4,
                    attributes: [
                        { shaderLocation: 0, offset: 0, format: "float32x3" },
                        { shaderLocation: 1, offset: 12, format: "float32x3" },
                        { shaderLocation: 2, offset: 24, format: "float32x2" },
                    ],
                },
            ],
        },
        fragment: {
            module: device.createShaderModule({ code: fragmentWGSL }),
            entryPoint: "main",
            targets: [{ format }],
        },
        primitive: { topology: "triangle-list", cullMode: "front" },
        depthStencil: {
            format: "depth24plus",
            depthWriteEnabled: true,
            depthCompare: "less",
        },
    });

    uniformBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: uniformBuffer } }],
    });

    textureBindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(1),
        entries: [
            { binding: 0, resource: sampler },
            { binding: 1, resource: texture.createView() },
        ]
    });
}

export function loop() {
    const now = (performance.now() - startTime) / 1000;

    const aspect = 800.0 / 600.0;
    const fov = Math.PI / 2;
    const near = 0.1;
    const far = 100;
    const proj = mat4_perspective(fov, aspect, near, far);
    const view = mat4_lookAt([1.5, 0, 1.5], [0, 0, 0], [0, 1, 0]);
    const model = mat4_rotationY(now * 0.5);

    device.queue.writeBuffer(uniformBuffer, 0, new Float32Array(model));
    device.queue.writeBuffer(uniformBuffer, 64, new Float32Array(view));
    device.queue.writeBuffer(uniformBuffer, 128, new Float32Array(proj));

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginRenderPass({
        colorAttachments: [{
            view: context.getCurrentTexture().createView(),
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1 },
            loadOp: "clear",
            storeOp: "store",
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthClearValue: 1.0,
            depthLoadOp: 'clear',
            depthStoreOp: 'store',
        },
    });

    pass.setPipeline(pipeline);
    pass.setVertexBuffer(0, vertexBuffer);
    pass.setIndexBuffer(indexBuffer, "uint16");
    pass.setBindGroup(0, uniformBindGroup);
    pass.setBindGroup(1, textureBindGroup);
    pass.drawIndexed(indices.length);
    pass.end();
    device.queue.submit([encoder.finish()]);

    requestAnimationFrame(loop);
}

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
