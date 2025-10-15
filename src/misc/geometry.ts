export type Mesh = {
    vertices: Float32Array;
    indices: Uint32Array;
    vertexStride: number;
};

export function generateSphere(radius = 1, latSegments = 32, lonSegments = 32): Mesh {
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

export function makeCube(size = 1): Mesh {
    const hs = size / 2; // half-size
    const vertices: number[] = [];
    const indices: number[] = [];

    // Cube faces: +X, -X, +Y, -Y, +Z, -Z
    const faceData: { normal: number[]; corners: number[][]; }[] = [
        { normal: [1, 0, 0], corners: [[hs, -hs, -hs], [hs, -hs, hs], [hs, hs, hs], [hs, hs, -hs]] },
        { normal: [-1, 0, 0], corners: [[-hs, -hs, hs], [-hs, -hs, -hs], [-hs, hs, -hs], [-hs, hs, hs]] },
        { normal: [0, 1, 0], corners: [[-hs, hs, -hs], [hs, hs, -hs], [hs, hs, hs], [-hs, hs, hs]] },
        { normal: [0, -1, 0], corners: [[-hs, -hs, hs], [hs, -hs, hs], [hs, -hs, -hs], [-hs, -hs, -hs]] },
        { normal: [0, 0, 1], corners: [[hs, -hs, hs], [-hs, -hs, hs], [-hs, hs, hs], [hs, hs, hs]] },
        { normal: [0, 0, -1], corners: [[-hs, -hs, -hs], [hs, -hs, -hs], [hs, hs, -hs], [-hs, hs, -hs]] },
    ];

    let vertOffset = 0;
    for (const face of faceData) {
        const [nx, ny, nz] = face.normal;
        const uvCorners = [
            [0, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ];

        for (let i = 0; i < 4; i++) {
            const [x, y, z] = face.corners[i];
            const [u, v] = uvCorners[i];
            vertices.push(x, y, z, nx, ny, nz, u, v);
        }

        // Two triangles per face
        indices.push(
            vertOffset, vertOffset + 1, vertOffset + 2,
            vertOffset, vertOffset + 2, vertOffset + 3
        );

        vertOffset += 4;
    }

    return {
        vertices: new Float32Array(vertices),
        indices: new Uint32Array(indices),
        vertexStride: 8 * 4, // 8 floats * 4 bytes
    };
}
