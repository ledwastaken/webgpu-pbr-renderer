import * as Engine from "../core/engine";

export class Mesh {
    private vertices: Float32Array;
    private indices: Int16Array;
    private vertexBuffer: GPUBuffer;
    private indexBuffer: GPUBuffer;

    public constructor(vertices: Float32Array, indices: Int16Array) {
        this.vertices = vertices;
        this.indices = indices;

        // Create vertex buffer
        this.vertexBuffer = Engine.device.createBuffer({
            size: vertices.byteLength,
            usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        // Load vertex data
        new Float32Array(this.vertexBuffer.getMappedRange()).set(vertices);
        this.vertexBuffer.unmap();

        // Create index buffer
        this.indexBuffer = Engine.device.createBuffer({
            size: indices.byteLength,
            usage: GPUBufferUsage.INDEX | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });

        // Load index data
        new Int16Array(this.indexBuffer.getMappedRange()).set(indices);
        this.indexBuffer.unmap();
    }
}
