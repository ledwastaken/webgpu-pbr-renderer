import { Vector3 } from './vector3';
import { Matrix4 } from './matrix4';

export class CFrame {
    private r00: number = 1.0;
    private r01: number = 0.0;
    private r02: number = 0.0;
    private r10: number = 0.0;
    private r11: number = 1.0;
    private r12: number = 0.0;
    private r20: number = 0.0;
    private r21: number = 0.0;
    private r22: number = 1.0;
    private pos: Vector3 = new Vector3();

    toMatrix(): Matrix4 {
        const data = new Array(
            this.r00, this.r01, this.r02, this.pos.x,
            this.r10, this.r11, this.r12, this.pos.y,
            this.r20, this.r21, this.r22, this.pos.z,
            0, 0, 0, 1
        );

        return new Matrix4(data);
    }

    invert(): CFrame {
        let cf = new CFrame();

        cf.r00 = this.r00;
        cf.r01 = this.r10;
        cf.r02 = this.r20;
        cf.r10 = this.r01;
        cf.r11 = this.r11;
        cf.r12 = this.r21;
        cf.r20 = this.r02;
        cf.r21 = this.r12;
        cf.r22 = this.r22;

        cf.pos = new Vector3(
            -(this.r00 * this.pos.x + this.r01 * this.pos.y + this.r02 * this.pos.z),
            -(this.r10 * this.pos.x + this.r11 * this.pos.y + this.r12 * this.pos.z),
            -(this.r20 * this.pos.x + this.r21 * this.pos.y + this.r22 * this.pos.z)
        );

        return cf;
    }

    add(vec: Vector3): CFrame {
        let cf = new CFrame();
        cf.r00 = this.r00;
        cf.r01 = this.r01;
        cf.r02 = this.r02;
        cf.r10 = this.r10;
        cf.r11 = this.r11;
        cf.r12 = this.r12;
        cf.r20 = this.r20;
        cf.r21 = this.r21;
        cf.r22 = this.r22;
        cf.pos = this.pos.add(vec);

        return cf;
    }

    multiply(vec: Vector3): Vector3 {
        return new Vector3(
            this.r00 * vec.x + this.r01 * vec.y + this.r02 * vec.z + this.pos.x,
            this.r10 * vec.x + this.r11 * vec.y + this.r12 * vec.z + this.pos.y,
            this.r20 * vec.x + this.r21 * vec.y + this.r22 * vec.z + this.pos.z
        );
    }

    data(): readonly number[] {
        return [
            this.r00, this.r01, this.r02,
            this.r10, this.r11, this.r12,
            this.r20, this.r21, this.r22,
            this.pos.x, this.pos.y, this.pos.z
        ];
    }

    getPosition(): Vector3 {
        return new Vector3(this.pos.x, this.pos.y, this.pos.z);
    }

    getRight(): Vector3 {
        return new Vector3(this.r00, this.r10, this.r20);
    }

    getUp(): Vector3 {
        return new Vector3(this.r01, this.r11, this.r21);
    }

    getBack(): Vector3 {
        return new Vector3(this.r02, this.r12, this.r22);
    }

    toString(): string {
        return `CFrame(pos: ${this.pos.toString()}, rotation: [${this.r00}, ${this.r01}, ${this.r02}, ${this.r10}, ${this.r11}, ${this.r12}, ${this.r20}, ${this.r21}, ${this.r22}])`;
    }
}
