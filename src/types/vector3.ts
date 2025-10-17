export class Vector3 {
    x: number;
    y: number;
    z: number;

    constructor(x: number = 0, y: number = 0, z: number = 0) {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    magnitude(): number {
        return Math.sqrt(this.x * this.x + this.y * this.y + this.z * this.z);
    }

    dot(vec: Vector3): number {
        return this.x * vec.x + this.y * vec.y + this.z * vec.z;
    }

    unit(): Vector3 {
        const mag = this.magnitude();
        if (mag === 0) {
            return new Vector3(0, 0, 0);
        }
        return new Vector3(this.x / mag, this.y / mag, this.z / mag);
    }

    cross(vec: Vector3): Vector3 {
        return new Vector3(
            this.y * vec.z - this.z * vec.y,
            this.z * vec.x - this.x * vec.z,
            this.x * vec.y - this.y * vec.x
        );
    }

    negate(): Vector3 {
        return new Vector3(-this.x, -this.y, -this.z);
    }

    add(vec: Vector3): Vector3 {
        return new Vector3(this.x + vec.x, this.y + vec.y, this.z + vec.z);
    }

    subtract(vec: Vector3): Vector3 {
        return new Vector3(this.x - vec.x, this.y - vec.y, this.z - vec.z);
    }

    multiply(val: number): Vector3 {
        return new Vector3(this.x * val, this.y * val, this.z * val);
    }

    divide(val: number): Vector3 {
        if (val === 0) {
            throw new Error("Cannot divide by zero");
        }
        return new Vector3(this.x / val, this.y / val, this.z / val);
    }

    addAssign(vec: Vector3): Vector3 {
        this.x += vec.x;
        this.y += vec.y;
        this.z += vec.z;
        return this;
    }

    subtractAssign(vec: Vector3): Vector3 {
        this.x -= vec.x;
        this.y -= vec.y;
        this.z -= vec.z;
        return this;
    }

    multiplyAssign(val: number): Vector3 {
        this.x *= val;
        this.y *= val;
        this.z *= val;
        return this;
    }

    divideAssign(val: number): Vector3 {
        if (val === 0) {
            throw new Error("Cannot divide by zero");
        }
        this.x /= val;
        this.y /= val;
        this.z /= val;
        return this;
    }

    data(): readonly number[] {
        return [this.x, this.y, this.z];
    }

    toString(): string {
        return `Vector3(${this.x}, ${this.y}, ${this.z})`;
    }
}
