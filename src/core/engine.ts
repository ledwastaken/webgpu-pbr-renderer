import { Mesh } from "../scene/mesh";
import { pbrPipeline } from "../gfx/pbr-pipeline";
import SkyboxPipeline from "../gfx/skybox-pipeline";
import { Camera } from "../scene/camera";

let startTime = performance.now();

class Engine {
    canvas: HTMLCanvasElement;
    device!: GPUDevice;
    context!: GPUCanvasContext;
    format!: GPUTextureFormat;
    msaaColorTexture!: GPUTexture;

    fragmentBuffer!: GPUBuffer;
    fragmentBindGroup!: GPUBindGroup;

    skyboxTexture!: GPUTexture;
    skyboxSampler!: GPUSampler;
    skyboxBindGroup!: GPUBindGroup;
    pbrSkyboxBindGroup!: GPUBindGroup;

    camera: Camera;
    meshes: Array<Mesh>;

    constructor() {
        this.canvas = document.getElementById("GLCanvas") as HTMLCanvasElement;
        this.meshes = new Array<Mesh>();
        this.camera = new Camera();

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

        const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

        this.msaaColorTexture = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height],
            sampleCount: 4,
            format: presentationFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT,
        });

        await pbrPipeline.init();
        await SkyboxPipeline.init();

        const bitmaps = await Promise.all([
            'texture/skybox/pos-x.jpg',
            'texture/skybox/neg-x.jpg',
            'texture/skybox/pos-y.jpg',
            'texture/skybox/neg-y.jpg',
            'texture/skybox/pos-z.jpg',
            'texture/skybox/neg-z.jpg',
        ].map(async path => {
            const response = await fetch(path);
            const blob = await response.blob();
            return await createImageBitmap(blob, { colorSpaceConversion: 'none' });
        }));

        const width = bitmaps[0].width;
        const height = bitmaps[0].height;
        const mipLevelCount = Math.floor(Math.log2(Math.max(width, height))) + 1;

        this.skyboxTexture = this.device.createTexture({
            size: [width, height, 6],
            mipLevelCount: mipLevelCount,
            format: 'rgba8unorm',
            dimension: '2d',
            usage: GPUTextureUsage.TEXTURE_BINDING |
                GPUTextureUsage.STORAGE_BINDING |
                GPUTextureUsage.COPY_DST |
                GPUTextureUsage.RENDER_ATTACHMENT
        });

        for (let face = 0; face < 6; face++) {
            this.device.queue.copyExternalImageToTexture(
                { source: bitmaps[face], flipY: false },
                { texture: this.skyboxTexture, mipLevel: 0, origin: [0, 0, face] },
                [width, height, 1]
            );
        }

        this.skyboxSampler = this.device.createSampler();

        const computeShader = this.device.createShaderModule({
            code: `
                @group(0) @binding(0) var srcTexture : texture_2d<f32>;
                @group(0) @binding(1) var dstTexture : texture_storage_2d<rgba8unorm, write>;

                @compute @workgroup_size(8, 8)
                fn main(@builtin(global_invocation_id) global_id : vec3u) {
                    let dstSize = textureDimensions(dstTexture);
                    if (global_id.x >= dstSize.x || global_id.y >= dstSize.y) {
                        return;
                    }

                    let srcCoord = vec2i(global_id.xy * 2u);

                    var color = textureLoad(srcTexture, srcCoord, 0);
                    color += textureLoad(srcTexture, srcCoord + vec2i(1, 0), 0);
                    color += textureLoad(srcTexture, srcCoord + vec2i(0, 1), 0);
                    color += textureLoad(srcTexture, srcCoord + vec2i(1, 1), 0);
                    color /= 4.0;

                    textureStore(dstTexture, vec2i(global_id.xy), color);
                }
            `
        });

        const computePipeline = this.device.createComputePipeline({
            layout: 'auto',
            compute: {
                module: computeShader,
                entryPoint: 'main',
            },
        });

        const commandEncoder = this.device.createCommandEncoder();

        for (let face = 0; face < 6; face++) {
            let currentWidth = width;
            let currentHeight = height;

            for (let mipLevel = 1; mipLevel < mipLevelCount; mipLevel++) {
                currentWidth = Math.max(1, Math.floor(currentWidth / 2));
                currentHeight = Math.max(1, Math.floor(currentHeight / 2));

                const srcView = this.skyboxTexture.createView({
                    dimension: '2d',
                    baseMipLevel: mipLevel - 1,
                    mipLevelCount: 1,
                    baseArrayLayer: face,
                    arrayLayerCount: 1,
                });

                const dstView = this.skyboxTexture.createView({
                    dimension: '2d',
                    baseMipLevel: mipLevel,
                    mipLevelCount: 1,
                    baseArrayLayer: face,
                    arrayLayerCount: 1,
                });

                const bindGroup = this.device.createBindGroup({
                    layout: computePipeline.getBindGroupLayout(0),
                    entries: [
                        { binding: 0, resource: srcView },
                        { binding: 1, resource: dstView }
                    ],
                });

                const passEncoder = commandEncoder.beginComputePass();
                passEncoder.setPipeline(computePipeline);
                passEncoder.setBindGroup(0, bindGroup);

                const workgroupCountX = Math.ceil(currentWidth / 8);
                const workgroupCountY = Math.ceil(currentHeight / 8);

                passEncoder.dispatchWorkgroups(workgroupCountX, workgroupCountY, 1);
                passEncoder.end();
            }
        }

        this.device.queue.submit([commandEncoder.finish()]);
        bitmaps.forEach(bitmap => bitmap.close());

        this.skyboxBindGroup = this.device.createBindGroup({
            layout: SkyboxPipeline.pipeline.getBindGroupLayout(1),
            entries: [
                { binding: 0, resource: this.skyboxSampler },
                { binding: 1, resource: this.skyboxTexture.createView({ dimension: 'cube' }) },
            ],
        });
        this.pbrSkyboxBindGroup = this.device.createBindGroup({
            layout: pbrPipeline.pipeline.getBindGroupLayout(3),
            entries: [
                { binding: 0, resource: this.skyboxSampler },
                { binding: 1, resource: this.skyboxTexture.createView({ dimension: 'cube' }) },
            ],
        });

        this.fragmentBuffer = engine.device.createBuffer({
            size: Float32Array.BYTES_PER_ELEMENT * 8,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        });
        this.fragmentBindGroup = engine.device.createBindGroup({
            layout: pbrPipeline.pipeline.getBindGroupLayout(1),
            entries: [{ binding: 0, resource: { buffer: this.fragmentBuffer } }],
        });
    }

    async loop() {
        let now = performance.now() - startTime;
        let x = Math.cos(now * 0.0003) * 2;
        let y = 1;
        let z = Math.sin(now * 0.0003) * 2;

        const aspect = 800.0 / 600.0;
        const fov = 100 * Math.PI / 180.0;
        const near = 0.1;
        const far = 100;
        const proj = mat4_perspective(fov, aspect, near, far);
        const view = mat4_lookAt([x, y, z], [0, 0, 0], [0, 1, 0]);

        engine.device.queue.writeBuffer(this.fragmentBuffer, 0, new Float32Array([x, y, z, 8, 4, 8]));

        const commandEncoder = this.device.createCommandEncoder();
        SkyboxPipeline.draw(commandEncoder, this.skyboxBindGroup, view, proj);

        for (let mesh of this.meshes) {
            pbrPipeline.draw(commandEncoder, mesh, view, proj, this.fragmentBindGroup, this.pbrSkyboxBindGroup);
        }

        engine.device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(() => this.loop());
    }
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

export let engine = new Engine();
