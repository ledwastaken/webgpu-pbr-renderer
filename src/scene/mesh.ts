import Engine from "../core/engine";
import { CFrame } from "../types/cframe";

export class Mesh {
    public cframe: CFrame;
    public indicesCount: number;
    public vertexBuffer: GPUBuffer;
    public indexBuffer: GPUBuffer;

    public constructor(vertices: Float32Array, indices: Int16Array) {
        this.cframe = new CFrame();
        this.indicesCount = indices.length;
        this.vertexBuffer = Engine.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
        this.vertexBuffer.unmap();

        this.indexBuffer = Engine.device.createBuffer({
            size: indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        new Int16Array(this.indexBuffer.getMappedRange()).set(indices);
        this.indexBuffer.unmap();
    }
}
