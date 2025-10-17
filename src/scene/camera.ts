import { CFrame } from "../types/cframe";
import { Matrix4 } from "../types/matrix4";

export class Camera {
    fov: number = 70.0;
    cframe: CFrame = new CFrame();

    getViewMatrix() {
        return this.cframe.invert().toMatrix();
    }

    getProjection(ratio: number) {
        return Matrix4.perspective(this.fov * Math.PI / 180.0, ratio, 0.1, 100.0);
    }
}
