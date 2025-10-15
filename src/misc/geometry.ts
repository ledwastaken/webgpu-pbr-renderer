export type SphereMesh = {
    vertices: Float32Array;
    indices: Uint32Array;
    vertexStride: number;
};

export function generateSphere(radius = 1, latSegments = 32, lonSegments = 32): SphereMesh {
    const vertexStride = 8;
    const vertices: number[] = [];
    const indices: number[] = [];

    for (let y = 0; y <= latSegments; y++) {
        const v = y / latSegments;
        const theta = v * Math.PI;

        for (let x = 0; x <= lonSegments; x++) {
            const u = x / lonSegments;
            const phi = u * Math.PI * 2;

            const sinTheta = Math.sin(theta);
            const cosTheta = Math.cos(theta);
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);

            const px = radius * sinTheta * cosPhi;
            const py = radius * cosTheta;
            const pz = radius * sinTheta * sinPhi;

            const nx = px / radius;
            const ny = py / radius;
            const nz = pz / radius;

            vertices.push(px, py, pz, nx, ny, nz, u, 1 - v);
        }
    }

    for (let y = 0; y < latSegments; y++) {
        for (let x = 0; x < lonSegments; x++) {
            const i0 = y * (lonSegments + 1) + x;
            const i1 = i0 + lonSegments + 1;
            indices.push(i0, i1, i0 + 1);
            indices.push(i1, i1 + 1, i0 + 1);
        }
    }

    return {
        vertices: new Float32Array(vertices),
        indices: new Uint32Array(indices),
        vertexStride: vertexStride * 4,
    };
}
