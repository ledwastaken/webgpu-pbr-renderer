export class Matrix4 {
    private data_: number[];

    constructor(data: number[]) {
        if (data.length !== 16)
            throw new Error("Matrix4 data array must have exactly 16 elements");
        this.data_ = [...data];
    }

    static identity(): Matrix4 {
        const data = new Array(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1
        );

        return new Matrix4(data);
    }

    static frustum(l: number, r: number, b: number, t: number, n: number, f: number): Matrix4 {
        const v0 = 2 * n;
        const v1 = r - l;
        const v2 = t - b;
        const v3 = f - n;

        const data = new Array(
            v0 / v1, 0, 0, 0,
            0, -v0 / v2, 0, 0,
            (r + l) / v1, (t + b) / v2, -(f + n) / v3, -1,
            0, 0, -v0 * f / v3, 0
        );

        return new Matrix4(data);
    }

    static perspective(fov: number, ratio: number, near: number, far: number): Matrix4 {
        const top = Math.tan(fov / 2.0) * near;
        const bottom = -top;
        const right = top * ratio;
        const left = -right;

        return this.frustum(left, right, bottom, top, near, far);
    }

    data(): readonly number[] {
        return [...this.data_];
    }

    dataMutable(): number[] {
        return [...this.data_];
    }

    get(row: number, col: number): number {
        if (row < 0 || row > 3 || col < 0 || col > 3) {
            throw new Error("Matrix index out of bounds");
        }
        return this.data_[row * 4 + col];
    }

    set(row: number, col: number, value: number): void {
        if (row < 0 || row > 3 || col < 0 || col > 3) {
            throw new Error("Matrix index out of bounds");
        }
        this.data_[row * 4 + col] = value;
    }

    toString(): string {
        const rows = [];
        for (let i = 0; i < 4; i++) {
            const rowData = this.data_.slice(i * 4, (i + 1) * 4);
            rows.push(`[${rowData.map(v => v.toFixed(4)).join(", ")}]`);
        }
        return `Matrix4(\n  ${rows.join(",\n  ")}\n)`;
    }

    multiply(other: Matrix4): Matrix4 {
        const a = this.data_;
        const b = other.data_;
        const c = new Array(16);

        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                let sum = 0;
                for (let k = 0; k < 4; k++) {
                    sum += a[i * 4 + k] * b[k * 4 + j];
                }
                c[i * 4 + j] = sum;
            }
        }

        return new Matrix4(c);
    }

    transpose(): Matrix4 {
        const result = new Array(16);
        for (let i = 0; i < 4; i++) {
            for (let j = 0; j < 4; j++) {
                result[j * 4 + i] = this.data_[i * 4 + j];
            }
        }
        return new Matrix4(result);
    }
}
