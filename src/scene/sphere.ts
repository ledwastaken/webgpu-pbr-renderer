import { Mesh } from "./mesh";

export class Sphere extends Mesh {
    public constructor(radius = 1, rings = 32, segments = 32) {
        let vertices = new Float32Array((rings + 1) * (segments + 1) * 11);
        let indices = new Int16Array(rings * segments * 6);
        let vertexOffset = 0;
        let indexOffset = 0;

        for (let y = 0; y <= rings; y++) {
            const v = y / rings;
            const theta = v * Math.PI;
            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);

            for (let x = 0; x <= segments; x++) {
                const u = x / segments;
                const phi = u * Math.PI * 2;
                const cosPhi = Math.cos(phi);
                const sinPhi = Math.sin(phi);

                const px = radius * sinTheta * cosPhi;
                const py = radius * cosTheta;
                const pz = radius * sinTheta * sinPhi;

                const nx = px / radius;
                const ny = py / radius;
                const nz = pz / radius;

                const tx = -sinTheta * sinPhi;
                const ty = 0;
                const tz = sinTheta * cosPhi;

                vertices.set([px, py, pz, nx, ny, nz, u, 1 - v, tx, ty, tz], vertexOffset);
                vertexOffset += 11;
            }
        }

        for (let y = 0; y < rings; y++) {
            for (let x = 0; x < segments; x++) {
                const i0 = y * (segments + 1) + x;
                const i1 = i0 + segments + 1;

                indices.set([i0, i1, i0 + 1], indexOffset);
                indexOffset += 3;
                indices.set([i1, i1 + 1, i0 + 1], indexOffset);
                indexOffset += 3;
            }
        }

        super(vertices, indices);
    }
}
