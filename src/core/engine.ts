import { Mesh } from "../scene/mesh";
import { pbrPipeline } from "../gfx/pbr-pipeline";
import SkyboxPipeline from "../gfx/skybox-pipeline";
import { Camera } from "../scene/camera";

async function createTextureFromImages(urls: string[]) {
    const bitmaps = await Promise.all(urls.map(async url => {
        return await loadImageBitmap(url);
    }));
    return createTextureFromSources(bitmaps);
}

async function loadImageBitmap(url: string) {
    const res = await fetch(url);
    const blob = await res.blob();
    return await createImageBitmap(blob, { colorSpaceConversion: 'none' });
}

function createTextureFromSources(sources: ImageBitmap[]) {
    const source = sources[0];
    const texture = engine.device.createTexture({
        format: 'rgba8unorm',
        size: [source.width, source.height, 6],
        usage: GPUTextureUsage.TEXTURE_BINDING |
            GPUTextureUsage.COPY_DST |
            GPUTextureUsage.RENDER_ATTACHMENT,
        dimension: '2d',
        mipLevelCount: 5
    });
    copySourcesToTexture(texture, sources);
    return texture;
}

function copySourcesToTexture(texture: GPUTexture, sources: Array<ImageBitmap>) {
    sources.forEach((source, layer) => {
        engine.device.queue.copyExternalImageToTexture(
            { source, },
            { texture, origin: [0, 0, layer] },
            { width: source.width, height: source.height },
        );
    });
}

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

        this.skyboxSampler = this.device.createSampler();
        this.skyboxTexture = await createTextureFromImages([
            'texture/skybox/pos-x.jpg',
            'texture/skybox/neg-x.jpg',
            'texture/skybox/pos-y.jpg',
            'texture/skybox/neg-y.jpg',
            'texture/skybox/pos-z.jpg',
            'texture/skybox/neg-z.jpg',
        ]);
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
